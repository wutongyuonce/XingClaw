from __future__ import annotations

"""
IMService：平台无关的 IM -> Agent 服务层。

核心设计（对标 pi-mom）：
1) 每频道维护长期 AgentSession（缓存实例），而非每条消息新建；
2) 流式更新：先发占位消息，持续 PATCH 更新内容；
3) 成本统计：每次回复附加 token usage；
4) MEMORY.md：全局 + 频道级记忆注入。
"""

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.types import AssistantMessage, TextContent
from coding_agent import CreateAgentSessionOptions, create_agent_session
from coding_agent.agent_session import AgentSession
from coding_agent.command_registry import format_commands_for_help, resolve_registered_command
from coding_agent.extensions.types import ExtensionCommandContext

from .memory import load_merged_memory
from .session_router import SessionRouter
from .types import (
    IMAdapter,
    IMChannelInfo,
    IMIncomingMessage,
    IMOutgoingCard,
    IMOutgoingText,
    IMUserInfo,
)

logger = logging.getLogger("xingclaw.im.service")

_STREAM_UPDATE_INTERVAL = 1.5
_THINKING_PLACEHOLDER = "思考中..."
_STALE_EVENT_SECONDS = 60.0


@dataclass
class IMServiceConfig:
    workspace_dir: str | Path
    provider: str
    model_id: str
    read_only_mode: bool = False
    max_reply_chars: int = 4000
    channel_queue_limit: int = 20
    use_card_reply: bool = True
    show_cost_in_reply: bool = True
    stream_updates: bool = True
    session_idle_timeout: float = 3600.0


@dataclass
class _ChannelState:
    """每频道维护的长期状态。"""
    session: AgentSession
    session_id: str
    last_active: float
    user_cache: dict[str, IMUserInfo]
    channel_info: IMChannelInfo | None = None


class IMService:
    """平台无关的 IM -> Agent 服务层。"""

    def __init__(self, adapter: IMAdapter, config: IMServiceConfig, router: SessionRouter | None = None) -> None:
        self.adapter = adapter
        self.config = config
        self.router = router or SessionRouter(config.workspace_dir)
        self._processed_ids: set[str] = set()
        self._processed_id_order: deque[str] = deque()
        self._processed_id_limit = 2000
        self._channel_queues: dict[str, deque[IMIncomingMessage]] = {}
        self._channel_running: set[str] = set()
        self._channel_states: dict[str, _ChannelState] = {}

    async def handle_webhook(self, headers: dict[str, str], body: bytes) -> dict:
        parsed = self.adapter.handle_webhook(headers, body)
        for message in parsed.messages:
            await self.handle_incoming_message(message)
        return parsed.ack

    async def handle_incoming_message(self, message: IMIncomingMessage) -> None:
        if self._is_duplicate_message(message.message_id):
            logger.warning("skip duplicate message message_id=%s", message.message_id)
            return
        if self._is_stale_event(message):
            logger.warning(
                "skip stale event message_id=%s created_at=%s",
                message.message_id, message.created_at,
            )
            self._mark_processed(message.message_id)
            return
        channel_key = self._channel_key(message)
        queue = self._channel_queues.setdefault(channel_key, deque())
        if len(queue) >= self.config.channel_queue_limit:
            logger.warning("drop message due to queue limit key=%s", channel_key)
            self._mark_processed(message.message_id)
            return
        queue.append(message)
        if channel_key in self._channel_running:
            return
        self._channel_running.add(channel_key)
        try:
            while queue:
                current = queue.popleft()
                await self._handle_single_message(current)
        finally:
            self._channel_running.discard(channel_key)
            if not queue:
                self._channel_queues.pop(channel_key, None)

    async def _handle_single_message(self, message: IMIncomingMessage) -> None:
        logger.info(
            "processing message platform=%s channel=%s user=%s text=%r",
            message.platform, message.channel_id, message.user_id,
            message.text.strip()[:80],
        )
        if await self._handle_control_command(message):
            self._mark_processed(message.message_id)
            return

        session, session_id = self._get_or_create_channel_session(message)

        user_context = self._build_user_context(message)
        prompt_text = message.text
        if user_context:
            prompt_text = f"{user_context}\n\n{message.text}"

        if self.config.stream_updates:
            reply_text = await self._prompt_with_streaming(session, prompt_text, message)
        else:
            reply_text = await self._prompt_simple(session, prompt_text)

        if not reply_text:
            reply_text = "(empty)"

        cost_line = ""
        if self.config.show_cost_in_reply:
            cost_line = self._format_cost(session)

        if len(reply_text) > self.config.max_reply_chars:
            reply_text = reply_text[: self.config.max_reply_chars] + "\n...<truncated>..."

        full_reply = reply_text
        if cost_line:
            full_reply = f"{reply_text}\n\n{cost_line}"

        if not self.config.stream_updates:
            self._send_reply(message, full_reply)

        self._mark_processed(message.message_id)
        logger.info("reply sent channel=%s chars=%d", message.channel_id, len(full_reply))

    def _get_or_create_channel_session(self, message: IMIncomingMessage) -> tuple[AgentSession, str]:
        """获取或创建频道级长期 session。"""
        channel_key = self._channel_key(message)
        state = self._channel_states.get(channel_key)

        if state is not None:
            state.last_active = time.time()
            return state.session, state.session_id

        session_id = self.router.get_or_create_session_id(
            platform=message.platform,
            channel_id=message.channel_id,
            thread_id=message.thread_id,
        )

        memory_text = load_merged_memory(self.config.workspace_dir, message.channel_id)
        append_prompt = ""
        if memory_text:
            append_prompt = f"\n\n长期记忆（MEMORY）：\n{memory_text}"

        session = create_agent_session(
            CreateAgentSessionOptions(
                workspace_dir=self.config.workspace_dir,
                provider=self.config.provider,
                model_id=self.config.model_id,
                session_id=session_id,
                read_only_mode=self.config.read_only_mode,
                append_system_prompt=append_prompt if append_prompt else None,
            )
        )

        channel_info = None
        if hasattr(self.adapter, "get_chat_info"):
            channel_info = self.adapter.get_chat_info(message.channel_id)

        self._channel_states[channel_key] = _ChannelState(
            session=session,
            session_id=session_id,
            last_active=time.time(),
            user_cache={},
            channel_info=channel_info,
        )
        logger.info("channel session created key=%s session_id=%s", channel_key, session_id)
        self._evict_idle_channels()
        return session, session_id

    def _evict_idle_channels(self) -> None:
        """清理超时的频道 session。"""
        now = time.time()
        to_remove = [
            key for key, state in self._channel_states.items()
            if now - state.last_active > self.config.session_idle_timeout
        ]
        for key in to_remove:
            state = self._channel_states.pop(key)
            state.session.close()
            logger.info("evicted idle channel session key=%s", key)

    def _invalidate_channel_session(self, message: IMIncomingMessage) -> None:
        """强制清除频道 session（用于 /clear 等命令）。"""
        channel_key = self._channel_key(message)
        state = self._channel_states.pop(channel_key, None)
        if state:
            state.session.close()

    def _build_user_context(self, message: IMIncomingMessage) -> str:
        """构建用户上下文信息，注入到 prompt。"""
        parts: list[str] = []

        if hasattr(self.adapter, "get_user_info"):
            user_info = self.adapter.get_user_info(message.user_id)
            if user_info and user_info.name:
                parts.append(f"[发送者: {user_info.name}]")

        channel_key = self._channel_key(message)
        state = self._channel_states.get(channel_key)
        if state and state.channel_info and state.channel_info.name:
            parts.append(f"[频道: {state.channel_info.name}]")

        return " ".join(parts)

    async def _prompt_simple(self, session: AgentSession, text: str) -> str:
        """非流式：直接调用 prompt 并提取结果。"""
        try:
            await session.prompt(text)
            return self._extract_last_assistant_text(session)
        except Exception as exc:
            logger.exception("agent prompt failed: %s", exc)
            return f"[IM bridge error] {exc}"

    async def _prompt_with_streaming(
        self, session: AgentSession, text: str, message: IMIncomingMessage
    ) -> str:
        """流式：先发占位消息，持续 PATCH 更新。"""
        placeholder_id: str | None = None
        last_update_time = 0.0
        accumulated_text = ""

        try:
            placeholder_id = self.adapter.send_card(
                IMOutgoingCard(
                    channel_id=message.channel_id,
                    title="XingClaw",
                    markdown_content=_THINKING_PLACEHOLDER,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
        except Exception as exc:
            logger.warning("failed to send placeholder: %s", exc)

        async def _on_event(event: dict[str, Any]) -> None:
            nonlocal last_update_time, accumulated_text
            if event.get("type") != "message_update":
                return
            msg = event.get("message")
            if not isinstance(msg, AssistantMessage):
                return
            text_parts = [b.text for b in msg.content if isinstance(b, TextContent)]
            accumulated_text = "".join(text_parts)

            if not placeholder_id or not hasattr(self.adapter, "update_text"):
                return
            now = time.time()
            if now - last_update_time < _STREAM_UPDATE_INTERVAL:
                return
            last_update_time = now
            preview = accumulated_text[:self.config.max_reply_chars] if accumulated_text else _THINKING_PLACEHOLDER
            try:
                self.adapter.update_text(placeholder_id, preview)
            except Exception as exc:
                logger.warning("stream update failed: %s", exc)

        unsub = session.subscribe(_on_event)
        try:
            await session.prompt(text)
            final_text = self._extract_last_assistant_text(session) or accumulated_text
        except Exception as exc:
            logger.exception("agent prompt failed: %s", exc)
            final_text = f"[IM bridge error] {exc}"
        finally:
            unsub()

        if placeholder_id and hasattr(self.adapter, "update_text"):
            cost_line = ""
            if self.config.show_cost_in_reply:
                cost_line = self._format_cost(session)
            full = final_text
            if len(full) > self.config.max_reply_chars:
                full = full[: self.config.max_reply_chars] + "\n...<truncated>..."
            if cost_line:
                full = f"{full}\n\n{cost_line}"
            try:
                self.adapter.update_text(placeholder_id, full)
            except Exception as exc:
                logger.warning("final stream update failed: %s", exc)
        return final_text

    def _send_reply(self, message: IMIncomingMessage, text: str) -> None:
        """发送最终回复（非流式模式下使用）。"""
        if self.config.use_card_reply and hasattr(self.adapter, "send_card"):
            try:
                self.adapter.send_card(
                    IMOutgoingCard(
                        channel_id=message.channel_id,
                        title="XingClaw",
                        markdown_content=text,
                        thread_id=message.thread_id,
                        reply_to_message_id=message.message_id,
                    )
                )
                return
            except Exception as exc:
                logger.warning("card reply failed, fallback to text: %s", exc)

        self.adapter.send_text(
            IMOutgoingText(
                channel_id=message.channel_id,
                text=text,
                thread_id=message.thread_id,
                reply_to_message_id=message.message_id,
            )
        )

    @staticmethod
    def _format_cost(session: AgentSession) -> str:
        usage = session.last_usage
        if not usage:
            return ""
        tokens = usage.get("total_tokens", 0)
        cost = usage.get("cost", {})
        total_cost = cost.get("total", 0.0)
        if tokens <= 0:
            return ""
        parts = [f"tokens: {tokens}"]
        if total_cost > 0:
            parts.append(f"cost: ${total_cost:.4f}")
        return f"📊 {' | '.join(parts)}"

    async def _handle_control_command(self, message: IMIncomingMessage) -> bool:
        text = message.text.strip()
        if not text.startswith("/"):
            return False

        if text in {"/clear", "/new"}:
            self._invalidate_channel_session(message)
            new_session_id = self.router.rotate_session_id(
                platform=message.platform,
                channel_id=message.channel_id,
                thread_id=message.thread_id,
            )
            logger.info("session rotated by command=%s new_session_id=%s", text, new_session_id)
            self.adapter.send_text(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=f"已新建会话：`{new_session_id}`。后续对话将使用新上下文。",
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if text == "/session":
            session_id = self.router.get_or_create_session_id(
                platform=message.platform,
                channel_id=message.channel_id,
                thread_id=message.thread_id,
            )
            cum = ""
            channel_key = self._channel_key(message)
            state = self._channel_states.get(channel_key)
            if state:
                usage = state.session.cumulative_usage
                cum = f"\ntokens: {usage['total_tokens']} | cost: ${usage['total_cost']:.4f}"
            self.adapter.send_text(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=f"当前会话：`{session_id}`{cum}",
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if text == "/help":
            session, _ = self._get_or_create_channel_session(message)
            self.adapter.send_text(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=format_commands_for_help(session),
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        parts = text.lstrip("/").split()
        if not parts:
            return False
        cmd_name = parts[0]
        cmd_args = parts[1:]
        session, _ = self._get_or_create_channel_session(message)
        cmd = resolve_registered_command(session, cmd_name)
        if not cmd:
            return False
        result = cmd.handler(
            ExtensionCommandContext(
                name=cmd_name,
                args=cmd_args,
                raw_text=text,
                session=session,
                message=message,
            )
        )
        if inspect.isawaitable(result):
            result = await result
        if result:
            self.adapter.send_text(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=str(result),
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
        return True

    def _is_stale_event(self, message: IMIncomingMessage) -> bool:
        if message.created_at is None:
            return False
        age = time.time() - message.created_at
        return age > _STALE_EVENT_SECONDS

    def _is_duplicate_message(self, message_id: str | None) -> bool:
        if not message_id:
            return False
        return message_id in self._processed_ids

    def _mark_processed(self, message_id: str | None) -> None:
        if not message_id:
            return
        if message_id in self._processed_ids:
            return
        self._processed_ids.add(message_id)
        self._processed_id_order.append(message_id)
        while len(self._processed_id_order) > self._processed_id_limit:
            old = self._processed_id_order.popleft()
            self._processed_ids.discard(old)

    @staticmethod
    def _extract_last_assistant_text(session: AgentSession) -> str:
        final_assistant = next(
            (m for m in reversed(session.messages) if isinstance(m, AssistantMessage)),
            None,
        )
        if final_assistant is None:
            return ""
        return "".join(
            block.text for block in final_assistant.content if isinstance(block, TextContent)
        ).strip()

    @staticmethod
    def _channel_key(message: IMIncomingMessage) -> str:
        thread = message.thread_id or "_"
        return f"{message.platform}:{message.channel_id}:{thread}"

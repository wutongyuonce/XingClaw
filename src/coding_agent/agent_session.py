from __future__ import annotations

"""
AgentSession：面向应用层的会话编排对象。

职责：
1) 管理会话存储目录；
2) 把 agent_core 事件/消息写入持久层；
3) 提供稳定的 prompt / continue 调用入口；
4) 上下文溢出检测与 LLM 驱动压缩。
"""

from pathlib import Path
import asyncio
import inspect
import logging
from typing import Awaitable, Callable

from ai.overflow import estimate_context_tokens, is_context_overflow
from ai.stream import complete_simple
from ai.types import (
    AssistantMessage,
    Context,
    Message,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from agent_core import Agent, AgentEvent, AgentMessage, AgentOptions

from .extensions.types import ExtensionLifecycleContext
from .session_store import SessionStore, new_session_id
from .types import AgentSessionOptions

logger = logging.getLogger("xingclaw.coding_agent.session")

_COMPACTION_SYSTEM_PROMPT = """你是一个上下文压缩助手。请根据以下对话历史生成一份简明摘要。
要求：
1. 保留所有关键事实、决策和结论
2. 保留重要的文件路径、代码片段和技术细节
3. 保留用户的偏好和约束条件
4. 移除重复和冗余信息
5. 用简洁的要点形式输出
6. 使用中文"""


class AgentSession:
    def __init__(self, options: AgentSessionOptions) -> None:
        workspace_dir = Path(options.workspace_dir)
        self.workspace_dir = workspace_dir
        self.session_id = options.session_id or new_session_id()

        self.store = SessionStore(workspace_dir=workspace_dir, session_id=self.session_id)
        self.store.ensure_initialized(
            model_id=options.model.id,
            provider=options.model.provider,
            system_prompt=options.system_prompt,
        )

        persisted_messages = self.store.load_session_messages()
        if not persisted_messages:
            persisted_messages = self.store.load_context_messages()
        merged_messages = [*persisted_messages, *options.messages]

        agent_opts = AgentOptions(
            model=options.model,
            system_prompt=options.system_prompt,
            tools=options.tools,
            messages=merged_messages,
            thinking_level=options.thinking_level,
            tool_execution=options.tool_execution,
            before_tool_call=options.before_tool_call,
            after_tool_call=options.after_tool_call,
            session_id=self.session_id,
        )
        if options.convert_to_llm is not None:
            agent_opts.convert_to_llm = options.convert_to_llm
        self.agent = Agent(agent_opts)
        self.max_context_messages = options.max_context_messages
        self.max_context_tokens = options.max_context_tokens
        self.retain_recent_messages = options.retain_recent_messages
        self.summary_builder = options.summary_builder
        self.tool_execution = options.tool_execution
        self.retry_enabled = options.retry_enabled
        self.max_retries = options.max_retries
        self.retry_base_delay_ms = options.retry_base_delay_ms
        self.prompt_debug_sources = options.prompt_debug_sources
        self.mcp_servers = options.mcp_servers
        self.mcp_client = options.mcp_client
        self.extension_commands = dict(options.extension_commands)
        self.before_prompt_hooks = list(options.before_prompt_hooks)
        self.after_prompt_hooks = list(options.after_prompt_hooks)
        self.before_tool_call = options.before_tool_call
        self.after_tool_call = options.after_tool_call
        self._unsubscribe = self.agent.subscribe(self._on_agent_event)

    @property
    def messages(self) -> list[AgentMessage]:
        return self.agent.state.messages

    @property
    def last_usage(self) -> dict | None:
        """返回最近一次 AssistantMessage 的 usage 信息。"""
        for msg in reversed(self.agent.state.messages):
            if isinstance(msg, AssistantMessage):
                u = msg.usage
                return {
                    "input_tokens": u.input,
                    "output_tokens": u.output,
                    "total_tokens": u.total_tokens,
                    "cache_read": u.cache_read,
                    "cache_write": u.cache_write,
                    "cost": {
                        "input": u.cost.input,
                        "output": u.cost.output,
                        "total": u.cost.total,
                    },
                }
        return None

    @property
    def cumulative_usage(self) -> dict:
        """统计整个会话的累积 token 使用和成本。"""
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        for msg in self.agent.state.messages:
            if isinstance(msg, AssistantMessage):
                total_input += msg.usage.input
                total_output += msg.usage.output
                total_tokens += msg.usage.total_tokens
                total_cost += msg.usage.cost.total
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }

    async def prompt(self, text: str, *, images: list[str] | None = None) -> list[AgentMessage]:
        await self._run_lifecycle_hooks(text=text, is_continue=False, hooks=self.before_prompt_hooks)
        await self._check_and_compact_before_prompt()
        result = await self._run_with_retry(lambda: self.agent.prompt(text, images=images))
        await self._compact_context_if_needed()
        await self._run_lifecycle_hooks(text=text, is_continue=False, hooks=self.after_prompt_hooks)
        return result

    async def prompt_message(self, message: UserMessage) -> list[AgentMessage]:
        await self._check_and_compact_before_prompt()
        result = await self._run_with_retry(lambda: self.agent.prompt(message))
        await self._compact_context_if_needed()
        return result

    async def continue_run(self) -> list[AgentMessage]:
        await self._run_lifecycle_hooks(text="", is_continue=True, hooks=self.before_prompt_hooks)
        result = await self._run_with_retry(self.agent.continue_run)
        await self._compact_context_if_needed()
        await self._run_lifecycle_hooks(text="", is_continue=True, hooks=self.after_prompt_hooks)
        return result

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        return self.agent.subscribe(listener)

    def close(self) -> None:
        self._unsubscribe()

    def list_entry_ids(self) -> list[str]:
        return self.store.list_entry_ids()

    def list_entries(self) -> list[dict]:
        return self.store.list_entries()

    def get_leaf_id(self) -> str | None:
        return self.store.get_leaf_id()

    def get_entry_path(self, entry_id: str) -> list[str]:
        return self.store.get_entry_path(entry_id)

    def get_session_tree(self) -> list[dict]:
        return self.store.get_session_tree()

    def fork_session(self, from_entry_id: str | None = None) -> "AgentSession":
        new_id = new_session_id()
        fork_store = self.store.fork_to(new_id, from_entry_id=from_entry_id)
        meta = fork_store.read_meta() or {}
        model = self.agent.state.model
        system_prompt = str(meta.get("system_prompt", self.agent.state.system_prompt))
        return AgentSession(
            AgentSessionOptions(
                model=model,
                workspace_dir=self.workspace_dir,
                system_prompt=system_prompt,
                tools=list(self.agent.state.tools),
                session_id=new_id,
                thinking_level=self.agent.state.thinking_level,
                tool_execution=self.tool_execution,
                max_context_messages=self.max_context_messages,
                max_context_tokens=self.max_context_tokens,
                retain_recent_messages=self.retain_recent_messages,
                summary_builder=self.summary_builder,
                retry_enabled=self.retry_enabled,
                max_retries=self.max_retries,
                retry_base_delay_ms=self.retry_base_delay_ms,
                prompt_debug_sources=self.prompt_debug_sources,
                mcp_servers=self.mcp_servers,
                mcp_client=self.mcp_client,
                extension_commands=self.extension_commands,
                before_prompt_hooks=self.before_prompt_hooks,
                after_prompt_hooks=self.after_prompt_hooks,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
            )
        )

    def fork_from_entry(self, entry_id: str) -> "AgentSession":
        return self.fork_session(from_entry_id=entry_id)

    def switch_to_entry(self, entry_id: str) -> None:
        self.store.set_leaf(entry_id)
        restored = self.store.load_session_messages(leaf_id=entry_id)
        self.agent.set_messages(restored)
        self.store.append_event(
            {
                "type": "session_switch_entry",
                "session_id": self.session_id,
                "entry_id": entry_id,
            }
        )

    def switch_session(self, session_id: str) -> None:
        new_store = SessionStore(self.workspace_dir, session_id)
        meta = new_store.read_meta()
        if not meta:
            raise ValueError(f"Session not found: {session_id}")

        self.session_id = session_id
        self.store = new_store
        restored = new_store.load_session_messages()
        if not restored:
            restored = new_store.load_context_messages()
        self.agent.set_messages(restored)

    async def _on_agent_event(self, event: AgentEvent) -> None:
        self.store.append_event(event)
        if event["type"] == "message_end":
            message = event["message"]
            self.store.append_context_message(message)

    async def _run_lifecycle_hooks(
        self,
        *,
        text: str,
        is_continue: bool,
        hooks: list,
    ) -> None:
        if not hooks:
            return
        ctx = ExtensionLifecycleContext(
            session=self,
            text=text,
            is_continue=is_continue,
            message_count=len(self.agent.state.messages),
        )
        for hook in hooks:
            value = hook(ctx)
            if inspect.isawaitable(value):
                await value

    async def _check_and_compact_before_prompt(self) -> None:
        """调用 LLM 前检查上下文是否溢出，如溢出则先压缩。"""
        model = self.agent.state.model
        ctx = Context(
            messages=self.agent.state.messages,
            system_prompt=self.agent.state.system_prompt,
            tools=self.agent.state.tools,
        )
        if is_context_overflow(model, ctx):
            logger.warning(
                "context overflow detected before prompt, triggering compaction session_id=%s",
                self.session_id,
            )
            await self._compact_context_if_needed(force=True)

    async def _compact_context_if_needed(self, *, force: bool = False) -> None:
        max_messages = self.max_context_messages
        max_tokens = self.max_context_tokens
        over_message_limit = bool(max_messages and max_messages > 0 and len(self.agent.state.messages) > max_messages)
        estimated_tokens = estimate_context_tokens(self.agent.state.messages, self.agent.state.system_prompt)
        over_token_limit = bool(max_tokens and max_tokens > 0 and estimated_tokens > max_tokens)

        if not force and not over_message_limit and not over_token_limit:
            return

        messages = list(self.agent.state.messages)
        retain = max(2, min(self.retain_recent_messages, len(messages) - 1))
        if len(messages) <= retain:
            return

        older = messages[:-retain]
        recent = messages[-retain:]

        if self.summary_builder:
            summary_text = self.summary_builder(older).strip()
        else:
            summary_text = await self._llm_summary(older)

        if not summary_text:
            summary_text = self._fallback_summary(older)

        summary_message = UserMessage(
            content=[TextContent(text=f"[Context Summary]\n{summary_text}")],
        )
        compacted = [summary_message, *recent]

        self.agent.set_messages(compacted)
        self.store.rewrite_context_messages(compacted)
        self.store.append_event(
            {
                "type": "context_compacted",
                "sessionId": self.session_id,
                "before_count": len(messages),
                "after_count": len(compacted),
                "retained_recent": retain,
                "estimated_tokens_before": estimated_tokens,
                "reason": "overflow" if force else ("token_threshold" if over_token_limit else "message_threshold"),
            }
        )
        logger.info(
            "context compacted session_id=%s before=%d after=%d",
            self.session_id, len(messages), len(compacted),
        )

    async def _llm_summary(self, messages: list[Message]) -> str:
        """用 LLM 生成上下文摘要。"""
        formatted = self._format_messages_for_summary(messages)
        if not formatted.strip():
            return ""

        try:
            summary_context = Context(
                messages=[UserMessage(content=f"请压缩以下对话历史为简明摘要：\n\n{formatted}")],
                system_prompt=_COMPACTION_SYSTEM_PROMPT,
            )
            model = self.agent.state.model
            result = await complete_simple(
                model,
                summary_context,
                SimpleStreamOptions(max_tokens=2000),
            )
            text_parts = [b.text for b in result.content if isinstance(b, TextContent)]
            summary = "\n".join(text_parts).strip()
            if summary:
                logger.info("LLM compaction summary generated chars=%d", len(summary))
                return summary
        except Exception as exc:
            logger.warning("LLM compaction failed, using fallback: %s", exc)

        return ""

    @staticmethod
    def _format_messages_for_summary(messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages[-40:]:
            if isinstance(msg, UserMessage):
                text = _extract_text_from_user(msg)
                if text:
                    lines.append(f"User: {text}")
            elif isinstance(msg, AssistantMessage):
                text = _extract_text_from_assistant(msg)
                if text:
                    lines.append(f"Assistant: {text}")
            elif isinstance(msg, ToolResultMessage):
                text = _extract_text_from_tool_result(msg)
                if text:
                    lines.append(f"ToolResult({msg.tool_name}): {text}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages[-20:]:
            if isinstance(msg, UserMessage):
                text = _extract_text_from_user(msg)
                if text:
                    lines.append(f"- User: {text}")
            elif isinstance(msg, AssistantMessage):
                text = _extract_text_from_assistant(msg)
                if text:
                    lines.append(f"- Assistant: {text}")
            elif isinstance(msg, ToolResultMessage):
                text = _extract_text_from_tool_result(msg)
                if text:
                    lines.append(f"- ToolResult({msg.tool_name}): {text}")
        merged = "\n".join(lines).strip()
        if len(merged) > 3000:
            merged = merged[:3000] + "\n...<summary truncated>..."
        return merged

    async def _run_with_retry(self, op: Callable[[], Awaitable[list[AgentMessage]]]) -> list[AgentMessage]:
        attempts = self.max_retries + 1 if self.retry_enabled else 1
        last: list[AgentMessage] | None = None

        for attempt in range(attempts):
            messages = await op()
            last = messages

            final_assistant = next((m for m in reversed(self.agent.state.messages) if isinstance(m, AssistantMessage)), None)
            should_retry = self._should_retry(final_assistant)
            if not should_retry or attempt >= attempts - 1:
                return messages

            delay_ms = int(self.retry_base_delay_ms * (2**attempt))
            self.store.append_event(
                {
                    "type": "auto_retry_start",
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "delay_ms": delay_ms,
                    "error_message": final_assistant.error_message if final_assistant else "",
                }
            )
            await asyncio.sleep(delay_ms / 1000.0)

        return last or []

    @staticmethod
    def _should_retry(message: AssistantMessage | None) -> bool:
        if message is None:
            return False
        if message.stop_reason not in {"error", "aborted"}:
            return False
        error_text = (message.error_message or "").lower()
        if "invalid_api_key" in error_text or "authentication" in error_text or "unauthorized" in error_text:
            return False
        return True


def _extract_text_from_user(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content[:180]
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:180]


def _extract_text_from_assistant(message: AssistantMessage) -> str:
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:180]


def _extract_text_from_tool_result(message: ToolResultMessage) -> str:
    text = "".join(block.text for block in message.content if isinstance(block, TextContent))
    return text[:180]

from __future__ import annotations

"""
对外 Agent 封装：
提供 prompt/continue、状态管理、事件订阅、串行调度入口。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ai.types import ImageContent, Message, Model, TextContent, ThinkingLevel, UserMessage

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return messages


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _resolve_reasoning(thinking_level: str) -> ThinkingLevel | None:
    mapping = {
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }
    return mapping.get(thinking_level)  # type: ignore[return-value]


@dataclass
class AgentOptions:
    model: Model
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] = _default_convert_to_llm
    transform_context: Optional[
        Callable[[list[AgentMessage], Any | None], list[AgentMessage] | Awaitable[list[AgentMessage]]]
    ] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    session_id: Optional[str] = None


class Agent:
    def __init__(self, options: AgentOptions) -> None:
        self._state = AgentState(
            system_prompt=options.system_prompt,
            model=options.model,
            thinking_level=options.thinking_level,  # type: ignore[arg-type]
            tools=list(options.tools),
            messages=list(options.messages),
        )
        self._options = options
        self._listeners: list[AgentEventSink] = []

        self._stream_task: asyncio.Task[list[AgentMessage]] | None = None
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []

    @property
    def state(self) -> AgentState:
        return self._state

    def set_system_prompt(self, system_prompt: str) -> None:
        self._state.system_prompt = system_prompt

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = list(tools)

    def set_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def add_steering_message(self, message: AgentMessage) -> None:
        self._steering_queue.append(message)

    def add_follow_up_message(self, message: AgentMessage) -> None:
        self._follow_up_queue.append(message)

    def clear_error(self) -> None:
        self._state.error = None

    def subscribe(self, listener: AgentEventSink) -> Callable[[], None]:
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    async def prompt(self, message: str | UserMessage, images: list[str] | None = None) -> list[AgentMessage]:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")

        if isinstance(message, str):
            content: list[TextContent | ImageContent] = [TextContent(text=message)]
            for image in images or []:
                content.append(ImageContent(data=image))
            prompt = UserMessage(content=content)
        else:
            prompt = message

        return await self._start_run(prompts=[prompt], continue_mode=False)

    async def continue_run(self) -> list[AgentMessage]:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already running")
        return await self._start_run(prompts=[], continue_mode=True)

    async def wait_for_idle(self) -> None:
        if self._stream_task is not None:
            await self._stream_task

    def abort(self) -> None:
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()

    async def _start_run(self, prompts: list[AgentMessage], continue_mode: bool) -> list[AgentMessage]:
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        cfg = AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self._options.convert_to_llm,
            transform_context=self._options.transform_context,
            get_api_key=self._options.get_api_key,
            get_steering_messages=self._drain_steering_messages,
            get_follow_up_messages=self._drain_follow_up_messages,
            tool_execution=self._options.tool_execution,
            before_tool_call=self._options.before_tool_call,
            after_tool_call=self._options.after_tool_call,
            reasoning=_resolve_reasoning(self._state.thinking_level),
            session_id=self._options.session_id,
        )

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

        if continue_mode:
            coro = run_agent_loop_continue(context=context, config=cfg, emit=self._dispatch_event)
        else:
            coro = run_agent_loop(prompts=prompts, context=context, config=cfg, emit=self._dispatch_event)

        self._stream_task = asyncio.create_task(coro)
        try:
            new_messages = await self._stream_task
            self._state.messages.extend(new_messages)
            return new_messages
        except asyncio.CancelledError:
            self._state.error = "aborted"
            raise
        except Exception as exc:
            self._state.error = str(exc)
            raise
        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._stream_task = None

    async def _dispatch_event(self, event: AgentEvent) -> None:
        event_type = event.get("type")

        if event_type == "message_start":
            msg = event.get("message")
            self._state.stream_message = msg
        elif event_type == "message_update":
            self._state.stream_message = event.get("message")
        elif event_type == "message_end":
            self._state.stream_message = None
        elif event_type == "tool_execution_start":
            tool_call_id = event.get("toolCallId")
            if tool_call_id:
                self._state.pending_tool_calls.add(tool_call_id)
        elif event_type == "tool_execution_end":
            tool_call_id = event.get("toolCallId")
            if tool_call_id in self._state.pending_tool_calls:
                self._state.pending_tool_calls.remove(tool_call_id)
        elif event_type == "error":
            self._state.error = event.get("error", "unknown error")

        for listener in list(self._listeners):
            await _maybe_await(listener(event))

    async def _drain_steering_messages(self) -> list[AgentMessage]:
        items = list(self._steering_queue)
        self._steering_queue.clear()
        return items

    async def _drain_follow_up_messages(self) -> list[AgentMessage]:
        items = list(self._follow_up_queue)
        self._follow_up_queue.clear()
        return items

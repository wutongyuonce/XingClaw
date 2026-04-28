from __future__ import annotations

"""
Agent 主循环实现：
用户消息 -> LLM -> 工具调用 -> LLM -> ... -> 结束
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, cast

from ai.stream import stream_simple
from ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
)


StreamFn = Callable[[Any, Context, SimpleStreamOptions | None], Any | Awaitable[Any]]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


async def _emit(emit: AgentEventSink, event: dict[str, Any]) -> None:
    await _maybe_await(emit(cast(AgentEvent, event)))


def _with_event_schema(
    emit: AgentEventSink,
    session_id: str | None,
) -> AgentEventSink:
    """
    - runId: 本次 run 唯一 ID
    - turnId: 当前轮次（从 0 开始，turn_start 后 +1）
    - eventId: 本次 run 内递增事件 ID
    - timestamp: 事件毫秒时间戳
    - sessionId: 透传上层 session
    """

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    turn_id = 0
    event_seq = 0

    async def _wrapped(event: dict[str, Any]) -> None:
        nonlocal turn_id, event_seq
        event_type = event.get("type")
        if event_type == "turn_start":
            turn_id += 1

        event_seq += 1
        enriched = {
            **event,
            "runId": run_id,
            "turnId": turn_id,
            "eventId": f"{run_id}:{event_seq}",
            "timestamp": _now_ms(),
            "sessionId": session_id,
        }
        await _maybe_await(emit(cast(AgentEvent, enriched)))

    return _wrapped


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    emit = _with_event_schema(emit, config.session_id)
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})
    for prompt in prompts:
        await _emit(emit, {"type": "message_start", "message": prompt})
        await _emit(emit, {"type": "message_end", "message": prompt})

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    emit = _with_event_schema(emit, config.session_id)
    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> None:
    first_turn = True
    pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, {"type": "turn_start"})
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, {"type": "message_start", "message": message})
                    await _emit(emit, {"type": "message_end", "message": message})
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            assistant = await _stream_assistant_response(current_context, config, emit, signal, stream_fn)
            new_messages.append(assistant)

            if assistant.stop_reason in {"error", "aborted"}:
                await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": []})
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            tool_calls = [c for c in assistant.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0
            tool_results: list[ToolResultMessage] = []

            if has_more_tool_calls:
                tool_results = await _execute_tool_calls(current_context, assistant, config, emit, signal)
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": tool_results})
            pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []

        followups = await _maybe_await(config.get_follow_up_messages()) if config.get_follow_up_messages else []
        if followups:
            pending_messages = followups
            continue
        break

    await _emit(emit, {"type": "agent_end", "messages": new_messages})


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        messages = await _maybe_await(config.transform_context(messages, signal))

    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,  # AgentTool 与 ai.Tool 字段兼容
    )

    resolved_api_key = config.get_api_key and await _maybe_await(config.get_api_key(config.model.provider))
    options = SimpleStreamOptions(reasoning=config.reasoning, api_key=resolved_api_key, session_id=config.session_id)

    fn = stream_fn or stream_simple
    response_stream = await _maybe_await(fn(config.model, llm_context, options))

    partial: AssistantMessage | None = None
    added_partial = False

    async for event in response_stream:
        t = event.get("type")
        if t == "start":
            partial = event["partial"]
            context.messages.append(partial)
            added_partial = True
            await _emit(emit, {"type": "message_start", "message": partial})
        elif t in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial is not None:
                partial = event["partial"]
                context.messages[-1] = partial
                await _emit(emit, {"type": "message_update", "message": partial, "assistantMessageEvent": event})
        elif t in {"done", "error"}:
            final_message = await response_stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(emit, {"type": "message_start", "message": final_message})
            await _emit(emit, {"type": "message_end", "message": final_message})
            return final_message

    final_message = await response_stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, {"type": "message_start", "message": final_message})
    await _emit(emit, {"type": "message_end", "message": final_message})
    return final_message


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, Any]


@dataclass
class _ExecutedToolCall:
    result: AgentToolResult
    is_error: bool


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, emit, signal)
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, emit, signal)


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: Any | None,
) -> tuple[_PreparedToolCall | None, AgentToolResult, bool]:
    tool = next((t for t in current_context.tools if t.name == tool_call.name), None)
    if tool is None:
        return None, _error_tool_result(f"Tool {tool_call.name} not found"), True

    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

    if config.before_tool_call:
        before = await _maybe_await(
            config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=current_context,
                ),
                signal,
            )
        )
        if before and before.block:
            return None, _error_tool_result(before.reason or "Tool execution was blocked"), True

    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=args), AgentToolResult(content=[]), False


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    emit: AgentEventSink,
    signal: Any | None,
) -> _ExecutedToolCall:
    try:
        updates: list[Awaitable[Any] | Any] = []

        def _on_update(partial_result: AgentToolResult) -> None:
            updates.append(
                emit(
                    {
                        "type": "tool_execution_update",
                        "toolCallId": prepared.tool_call.id,
                        "toolName": prepared.tool_call.name,
                        "args": prepared.tool_call.arguments,
                        "partialResult": partial_result,
                    }
                )
            )

        raw_result = prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, _on_update)
        result = await _maybe_await(raw_result)

        for u in updates:
            await _maybe_await(u)
        return _ExecutedToolCall(result=result, is_error=False)
    except Exception as exc:
        return _ExecutedToolCall(result=_error_tool_result(str(exc)), is_error=True)


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCall,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> ToolResultMessage:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call:
        after = await _maybe_await(
            config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=current_context,
                ),
                signal,
            )
        )
        if after:
            if after.content is not None:
                result.content = after.content
            if after.details is not None:
                result.details = after.details
            if after.is_error is not None:
                is_error = after.is_error

    await _emit(
        emit,
        {
            "type": "tool_execution_end",
            "toolCallId": prepared.tool_call.id,
            "toolName": prepared.tool_call.name,
            "result": result,
            "isError": is_error,
        },
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=prepared.tool_call.id,
        tool_name=prepared.tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=_now_ms(),
    )
    await _emit(emit, {"type": "message_start", "message": tool_result_message})
    await _emit(emit, {"type": "message_end", "message": tool_result_message})
    return tool_result_message


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if prepared is None:
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                },
            )
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=True,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            results.append(msg)
            continue

        executed = await _execute_prepared_tool_call(prepared, emit, signal)
        results.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )
    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    immediate_results: list[ToolResultMessage] = []
    prepared_calls: list[_PreparedToolCall] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if prepared is None:
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                },
            )
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=True,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            immediate_results.append(msg)
        else:
            prepared_calls.append(prepared)

    tasks = [asyncio.create_task(_execute_prepared_tool_call(pc, emit, signal)) for pc in prepared_calls]
    executed_results = await asyncio.gather(*tasks)

    finalized: list[ToolResultMessage] = []
    for prepared, executed in zip(prepared_calls, executed_results):
        finalized.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )
    return [*immediate_results, *finalized]

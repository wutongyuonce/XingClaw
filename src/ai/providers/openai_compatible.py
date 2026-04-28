from __future__ import annotations

"""
OpenAI 标准 Chat Completions 流式 provider。

特点：
1) 使用 SSE data 行消费增量；
2) 统一输出 text/thinking/toolcall 事件；
3) 最终组装成 AssistantMessage。
"""

import asyncio
import json
from typing import Any

import httpx

from ..env_api_keys import get_env_api_key
from ..event_stream import AssistantMessageEventStream
from ..types import Context, Model, SimpleStreamOptions, StreamOptions, TextContent, ThinkingContent, ToolCall
from ._common import empty_assistant_message, parse_partial_json, to_openai_messages, to_openai_tools


def _map_stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "toolUse"
    if finish_reason == "length":
        return "length"
    return "stop"


def stream_openai_compatible(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    resolved_options = options or StreamOptions()

    async def _run() -> None:
        out = empty_assistant_message(api=model.api, provider=model.provider, model=model.id)
        try:
            # 允许调用参数覆盖环境变量。
            api_key = resolved_options.api_key or get_env_api_key(model.provider) or get_env_api_key("openai")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            # 模型级和调用级 headers 逐层覆盖。
            if model.headers:
                headers.update(model.headers)
            if resolved_options.headers:
                headers.update(resolved_options.headers)

            payload: dict[str, Any] = {
                "model": model.id,
                "messages": to_openai_messages(context),
                "stream": True,
            }
            if resolved_options.temperature is not None:
                payload["temperature"] = resolved_options.temperature
            if resolved_options.max_tokens is not None:
                payload["max_tokens"] = resolved_options.max_tokens
            tools = to_openai_tools(context.tools)
            if tools:
                payload["tools"] = tools

            timeout = resolved_options.timeout_seconds or None
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{model.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    stream.push({"type": "start", "partial": out})

                    current_text: TextContent | None = None
                    current_thinking: ThinkingContent | None = None
                    tool_call_index_map: dict[int, ToolCall] = {}
                    tool_call_partial_json: dict[int, str] = {}

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break

                        chunk = json.loads(raw)
                        if chunk.get("id"):
                            out.response_id = chunk["id"]
                        choice = (chunk.get("choices") or [{}])[0]

                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            out.stop_reason = _map_stop_reason(finish_reason)

                        delta = choice.get("delta") or {}

                        # 文本增量
                        text_delta = delta.get("content")
                        if text_delta:
                            if current_text is None:
                                current_text = TextContent(text="")
                                out.content.append(current_text)
                                stream.push({"type": "text_start", "contentIndex": len(out.content) - 1, "partial": out})
                            current_text.text += text_delta
                            stream.push(
                                {
                                    "type": "text_delta",
                                    "contentIndex": len(out.content) - 1,
                                    "delta": text_delta,
                                    "partial": out,
                                }
                            )

                        # reasoning 兼容字段（不同网关命名可能不同）
                        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning_delta:
                            if current_thinking is None:
                                current_thinking = ThinkingContent(thinking="")
                                out.content.append(current_thinking)
                                stream.push(
                                    {"type": "thinking_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )
                            current_thinking.thinking += reasoning_delta
                            stream.push(
                                {
                                    "type": "thinking_delta",
                                    "contentIndex": len(out.content) - 1,
                                    "delta": reasoning_delta,
                                    "partial": out,
                                }
                            )

                        # 工具调用增量（按 index 聚合）
                        for tc_delta in delta.get("tool_calls") or []:
                            index = tc_delta.get("index", 0)
                            tc = tool_call_index_map.get(index)
                            if tc is None:
                                tc = ToolCall(id=tc_delta.get("id", ""), name="", arguments={})
                                tool_call_index_map[index] = tc
                                tool_call_partial_json[index] = ""
                                out.content.append(tc)
                                stream.push({"type": "toolcall_start", "contentIndex": len(out.content) - 1, "partial": out})

                            if tc_delta.get("id"):
                                tc.id = tc_delta["id"]
                            fn = tc_delta.get("function") or {}
                            if fn.get("name"):
                                tc.name = fn["name"]
                            if fn.get("arguments"):
                                tool_call_partial_json[index] += fn["arguments"]
                                tc.arguments = parse_partial_json(tool_call_partial_json[index])
                                stream.push(
                                    {
                                        "type": "toolcall_delta",
                                        "contentIndex": out.content.index(tc),
                                        "delta": fn["arguments"],
                                        "partial": out,
                                    }
                                )

                        usage = chunk.get("usage")
                        if usage:
                            out.usage.input = usage.get("prompt_tokens", out.usage.input)
                            out.usage.output = usage.get("completion_tokens", out.usage.output)
                            out.usage.total_tokens = usage.get("total_tokens", out.usage.total_tokens)

                    # 收尾事件：把进行中的块发出 *_end
                    if current_text is not None:
                        stream.push(
                            {
                                "type": "text_end",
                                "contentIndex": out.content.index(current_text),
                                "content": current_text.text,
                                "partial": out,
                            }
                        )
                    if current_thinking is not None:
                        stream.push(
                            {
                                "type": "thinking_end",
                                "contentIndex": out.content.index(current_thinking),
                                "content": current_thinking.thinking,
                                "partial": out,
                            }
                        )
                    for tc in tool_call_index_map.values():
                        stream.push(
                            {
                                "type": "toolcall_end",
                                "contentIndex": out.content.index(tc),
                                "toolCall": tc,
                                "partial": out,
                            }
                        )

                    stream.push({"type": "done", "reason": out.stop_reason, "message": out})
                    stream.end(out)
        except Exception as exc:
            out.stop_reason = "error"
            out.error_message = str(exc)
            stream.push({"type": "error", "reason": "error", "error": out})
            stream.end(out)

    asyncio.create_task(_run())
    return stream


def stream_simple_openai_compatible(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    # 第一阶段实现：simple 接口直接复用标准 stream。
    return stream_openai_compatible(model, context, options)

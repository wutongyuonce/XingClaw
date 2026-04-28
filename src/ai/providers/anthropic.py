from __future__ import annotations

"""
Anthropic Messages API 流式 provider。

实现思路：
1) 读取 SSE 的 event/data；
2) 按 content block 组装 text/thinking/toolCall；
3) 映射 stop reason 并输出统一 done/error 事件。
"""

import asyncio
import json
from typing import Any

import httpx

from ..env_api_keys import get_env_api_key
from ..event_stream import AssistantMessageEventStream
from ..types import Context, Model, SimpleStreamOptions, StreamOptions, TextContent, ThinkingContent, ToolCall
from ._common import empty_assistant_message, parse_partial_json, to_anthropic_messages, to_anthropic_tools


def _map_stop_reason(reason: str | None) -> str:
    if reason == "tool_use":
        return "toolUse"
    if reason == "max_tokens":
        return "length"
    return "stop"


def stream_anthropic(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    resolved_options = options or StreamOptions()

    async def _run() -> None:
        out = empty_assistant_message(api=model.api, provider=model.provider, model=model.id)
        try:
            api_key = resolved_options.api_key or get_env_api_key(model.provider)
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY")

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            if model.headers:
                headers.update(model.headers)
            if resolved_options.headers:
                headers.update(resolved_options.headers)

            payload: dict[str, Any] = {
                "model": model.id,
                "max_tokens": resolved_options.max_tokens or model.max_tokens,
                "messages": to_anthropic_messages(context),
                "stream": True,
            }
            if context.system_prompt:
                payload["system"] = context.system_prompt
            if resolved_options.temperature is not None:
                payload["temperature"] = resolved_options.temperature
            tools = to_anthropic_tools(context.tools)
            if tools:
                payload["tools"] = tools

            timeout = resolved_options.timeout_seconds or None
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{model.base_url.rstrip('/')}/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    stream.push({"type": "start", "partial": out})

                    current_event: str | None = None
                    current_index: int | None = None
                    text_blocks: dict[int, TextContent] = {}
                    thinking_blocks: dict[int, ThinkingContent] = {}
                    tool_blocks: dict[int, ToolCall] = {}
                    tool_partial_json: dict[int, str] = {}

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue

                        if line.startswith("event:"):
                            current_event = line[len("event:") :].strip()
                            continue
                        if not line.startswith("data:"):
                            continue

                        data = json.loads(line[len("data:") :].strip())

                        if current_event == "message_start":
                            message = data.get("message", {})
                            out.response_id = message.get("id")
                            usage = message.get("usage", {})
                            out.usage.input = usage.get("input_tokens", out.usage.input)

                        elif current_event == "content_block_start":
                            current_index = data.get("index", 0)
                            block = data.get("content_block", {})
                            block_type = block.get("type")
                            if block_type == "text":
                                tb = TextContent(text="")
                                text_blocks[current_index] = tb
                                out.content.append(tb)
                                stream.push(
                                    {"type": "text_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )
                            elif block_type in {"thinking", "redacted_thinking"}:
                                th = ThinkingContent(thinking="", redacted=(block_type == "redacted_thinking"))
                                thinking_blocks[current_index] = th
                                out.content.append(th)
                                stream.push(
                                    {"type": "thinking_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )
                            elif block_type == "tool_use":
                                tc = ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments={})
                                tool_blocks[current_index] = tc
                                tool_partial_json[current_index] = ""
                                out.content.append(tc)
                                stream.push(
                                    {"type": "toolcall_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )

                        elif current_event == "content_block_delta":
                            idx = data.get("index", current_index if current_index is not None else 0)
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta" and idx in text_blocks:
                                text = delta.get("text", "")
                                text_blocks[idx].text += text
                                stream.push(
                                    {
                                        "type": "text_delta",
                                        "contentIndex": out.content.index(text_blocks[idx]),
                                        "delta": text,
                                        "partial": out,
                                    }
                                )
                            elif delta_type in {"thinking_delta", "signature_delta"} and idx in thinking_blocks:
                                text = delta.get("thinking", "")
                                if text:
                                    thinking_blocks[idx].thinking += text
                                    stream.push(
                                        {
                                            "type": "thinking_delta",
                                            "contentIndex": out.content.index(thinking_blocks[idx]),
                                            "delta": text,
                                            "partial": out,
                                        }
                                    )
                            elif delta_type == "input_json_delta" and idx in tool_blocks:
                                piece = delta.get("partial_json", "")
                                tool_partial_json[idx] += piece
                                tool_blocks[idx].arguments = parse_partial_json(tool_partial_json[idx])
                                stream.push(
                                    {
                                        "type": "toolcall_delta",
                                        "contentIndex": out.content.index(tool_blocks[idx]),
                                        "delta": piece,
                                        "partial": out,
                                    }
                                )

                        elif current_event == "content_block_stop":
                            idx = data.get("index", current_index if current_index is not None else 0)
                            if idx in text_blocks:
                                block = text_blocks[idx]
                                stream.push(
                                    {
                                        "type": "text_end",
                                        "contentIndex": out.content.index(block),
                                        "content": block.text,
                                        "partial": out,
                                    }
                                )
                            elif idx in thinking_blocks:
                                block = thinking_blocks[idx]
                                stream.push(
                                    {
                                        "type": "thinking_end",
                                        "contentIndex": out.content.index(block),
                                        "content": block.thinking,
                                        "partial": out,
                                    }
                                )
                            elif idx in tool_blocks:
                                block = tool_blocks[idx]
                                stream.push(
                                    {
                                        "type": "toolcall_end",
                                        "contentIndex": out.content.index(block),
                                        "toolCall": block,
                                        "partial": out,
                                    }
                                )

                        elif current_event == "message_delta":
                            delta = data.get("delta", {})
                            usage = data.get("usage", {})
                            out.stop_reason = _map_stop_reason(delta.get("stop_reason"))
                            out.usage.output = usage.get("output_tokens", out.usage.output)

                    stream.push({"type": "done", "reason": out.stop_reason, "message": out})
                    stream.end(out)
        except Exception as exc:
            out.stop_reason = "error"
            out.error_message = str(exc)
            stream.push({"type": "error", "reason": "error", "error": out})
            stream.end(out)

    asyncio.create_task(_run())
    return stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    # 第一阶段实现：simple 接口复用标准 stream。
    return stream_anthropic(model, context, options)

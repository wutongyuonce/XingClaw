from __future__ import annotations

"""
provider 共享工具函数：
1) 通用消息转换（Context -> provider payload）
2) 流式 JSON 片段解析
3) 空 AssistantMessage 初始化
"""

import json
import time
from typing import Any

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_partial_json(raw: str) -> dict[str, Any]:
    """
    解析流式工具参数（可能是半截 JSON）。
    解析失败时返回 {}，让上层保持稳态。
    """
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def empty_assistant_message(api: str, provider: str, model: str) -> AssistantMessage:
    """创建一个最小可用的 AssistantMessage，用于边流式边填充。"""
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model,
        usage=Usage(),
        timestamp=now_ms(),
    )


def to_openai_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 转成 OpenAI Chat Completions 的 messages。"""
    out: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
            text = "".join(b.text for b in msg.content if isinstance(b, TextContent))
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            payload: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in tool_calls
                ]
            out.append(payload)
        elif isinstance(msg, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "content": "\n".join(
                        p.text for p in msg.content if isinstance(p, TextContent) and isinstance(p.text, str)
                    ),
                }
            )
    return out


def to_openai_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转成 OpenAI tools。"""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def to_anthropic_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 转成 Anthropic Messages API payload。"""
    out: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": part.mime_type, "data": part.data},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCall):
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )
            out.append({"role": "assistant", "content": parts})
        elif isinstance(msg, ToolResultMessage):
            text = "\n".join(p.text for p in msg.content if isinstance(p, TextContent))
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": [{"type": "text", "text": text}],
                            "is_error": msg.is_error,
                        }
                    ],
                }
            )
    return out


def to_anthropic_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转成 Anthropic tools。"""
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]

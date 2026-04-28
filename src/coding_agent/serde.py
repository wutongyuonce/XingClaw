from __future__ import annotations

"""
消息序列化/反序列化工具。

目标：
1) 把 ai 层 dataclass 消息安全写入 jsonl；
2) 下次启动时恢复为同等结构，继续参与 Agent 推理。
"""

from typing import Any

from ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _user_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    return {"type": "image", "data": block.data, "mime_type": block.mime_type}


def _assistant_block_to_dict(block: TextContent | ThinkingContent | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    if isinstance(block, ThinkingContent):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}


def _tool_result_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    return _user_block_to_dict(block)


def message_to_dict(message: Message) -> dict[str, Any]:
    """
    将 Message 转成可持久化 dict。
    """

    if isinstance(message, UserMessage):
        content: str | list[dict[str, Any]]
        if isinstance(message.content, str):
            content = message.content
        else:
            content = [_user_block_to_dict(b) for b in message.content]
        return {"role": "user", "content": content, "timestamp": message.timestamp}

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_assistant_block_to_dict(b) for b in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": {
                "input": message.usage.input,
                "output": message.usage.output,
                "cache_read": message.usage.cache_read,
                "cache_write": message.usage.cache_write,
                "total_tokens": message.usage.total_tokens,
                "cost": {
                    "input": message.usage.cost.input,
                    "output": message.usage.cost.output,
                    "cache_read": message.usage.cost.cache_read,
                    "cache_write": message.usage.cost.cache_write,
                    "total": message.usage.cost.total,
                },
            },
            "stop_reason": message.stop_reason,
            "response_id": message.response_id,
            "error_message": message.error_message,
            "timestamp": message.timestamp,
        }

    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": [_tool_result_block_to_dict(b) for b in message.content],
            "is_error": message.is_error,
            "details": message.details,
            "timestamp": message.timestamp,
        }

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _user_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    if data.get("type") == "image":
        return ImageContent(data=data.get("data", ""), mime_type=data.get("mime_type", "image/png"))
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _assistant_block_from_dict(data: dict[str, Any]) -> TextContent | ThinkingContent | ToolCall:
    t = data.get("type")
    if t == "thinking":
        return ThinkingContent(
            thinking=data.get("thinking", ""),
            thinking_signature=data.get("thinking_signature"),
            redacted=bool(data.get("redacted", False)),
        )
    if t == "toolCall":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}) if isinstance(data.get("arguments"), dict) else {},
        )
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _tool_result_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    return _user_block_from_dict(data)


def message_from_dict(data: dict[str, Any]) -> Message:
    """
    将持久化 dict 恢复成 Message。
    """

    role = data.get("role")
    if role == "user":
        raw_content = data.get("content", "")
        if isinstance(raw_content, str):
            content: str | list[TextContent | ImageContent] = raw_content
        else:
            content = [_user_block_from_dict(i) for i in raw_content if isinstance(i, dict)]
        return UserMessage(content=content, timestamp=int(data.get("timestamp", 0)))

    if role == "assistant":
        return AssistantMessage(
            content=[_assistant_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            stop_reason=data.get("stop_reason", "stop"),
            response_id=data.get("response_id"),
            error_message=data.get("error_message"),
            timestamp=int(data.get("timestamp", 0)),
        )

    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=[_tool_result_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            is_error=bool(data.get("is_error", False)),
            details=data.get("details"),
            timestamp=int(data.get("timestamp", 0)),
        )

    raise ValueError(f"Unknown role: {role!r}")

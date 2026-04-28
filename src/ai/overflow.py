from __future__ import annotations

"""
上下文溢出检测。

根据模型 context_window 估算当前 token 总量，
判定是否超出上下文窗口，便于 Agent 侧在调用前主动压缩。
"""

from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

CHARS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 1000
TOOL_SCHEMA_TOKEN_ESTIMATE = 200


def estimate_message_tokens(msg: Message) -> int:
    total = 0
    if isinstance(msg, UserMessage):
        if isinstance(msg.content, str):
            total += len(msg.content)
        else:
            for block in msg.content:
                if isinstance(block, TextContent):
                    total += len(block.text)
                elif isinstance(block, ImageContent):
                    total += IMAGE_TOKEN_ESTIMATE * CHARS_PER_TOKEN
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += len(block.text)
            elif isinstance(block, ThinkingContent):
                total += len(block.thinking)
            elif isinstance(block, ToolCall):
                total += len(str(block.arguments)) + len(block.name) + 20
    elif isinstance(msg, ToolResultMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += len(block.text)
            elif isinstance(block, ImageContent):
                total += IMAGE_TOKEN_ESTIMATE * CHARS_PER_TOKEN
    return max(1, total // CHARS_PER_TOKEN)


def estimate_context_tokens(
    messages: list[Message],
    system_prompt: str = "",
    tools: list | None = None,
) -> int:
    total = len(system_prompt) // CHARS_PER_TOKEN
    for msg in messages:
        total += estimate_message_tokens(msg)
    if tools:
        total += len(tools) * TOOL_SCHEMA_TOKEN_ESTIMATE
    return total


def is_context_overflow(
    model: Model,
    context: Context,
    *,
    safety_margin: float = 0.95,
) -> bool:
    """
    检查 context 是否超出模型 context_window。

    safety_margin 为安全系数（默认 0.95，即保留 5% 余量给输出）。
    """
    limit = int(model.context_window * safety_margin)
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    return estimated > limit


def overflow_ratio(model: Model, context: Context) -> float:
    """返回当前 token 占 context_window 的比例。"""
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )
    if model.context_window <= 0:
        return 0.0
    return estimated / model.context_window

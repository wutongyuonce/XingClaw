from __future__ import annotations

"""
convertToLlm：把 AgentMessage 列表转换为 LLM 可直接消费的 Message 列表。

对标 pi-coding-agent 的 convertToLlm 逻辑：
1) 过滤掉非标准消息类型；
2) 裁剪过长的 ToolResult 内容，防止上下文膨胀；
3) 处理 thinking 块：根据目标模型能力决定保留 / 转文本 / 移除；
4) 清理空消息。
"""

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
from agent_core.types import AgentMessage

TOOL_RESULT_MAX_CHARS = 30_000
TOOL_RESULT_TRUNCATION_NOTICE = "\n...<content truncated>..."


def convert_to_llm(
    messages: list[AgentMessage],
    *,
    strip_thinking: bool = False,
    thinking_to_text: bool = False,
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS,
) -> list[Message]:
    """
    将 AgentMessage 列表转换为 LLM 可消费的 Message 列表。

    Parameters
    ----------
    strip_thinking : bool
        若为 True，完全移除 thinking 块（适用于不支持 thinking 的模型）。
    thinking_to_text : bool
        若为 True，把 thinking 块转为 TextContent（跨 Provider 切换时）。
    tool_result_max_chars : int
        ToolResult 文本最大字符数，超出则截断。
    """
    result: list[Message] = []
    for msg in messages:
        converted = _convert_single(
            msg,
            strip_thinking=strip_thinking,
            thinking_to_text=thinking_to_text,
            tool_result_max_chars=tool_result_max_chars,
        )
        if converted is not None:
            result.append(converted)
    return _ensure_valid_sequence(result)


def _convert_single(
    msg: AgentMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
    tool_result_max_chars: int,
) -> Message | None:
    if isinstance(msg, UserMessage):
        return msg

    if isinstance(msg, AssistantMessage):
        return _process_assistant(msg, strip_thinking=strip_thinking, thinking_to_text=thinking_to_text)

    if isinstance(msg, ToolResultMessage):
        return _process_tool_result(msg, max_chars=tool_result_max_chars)

    return None


def _process_assistant(
    msg: AssistantMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
) -> AssistantMessage:
    if not strip_thinking and not thinking_to_text:
        return msg

    new_content = []
    for block in msg.content:
        if isinstance(block, ThinkingContent):
            if strip_thinking:
                continue
            if thinking_to_text and block.thinking:
                new_content.append(TextContent(text=f"[thinking]\n{block.thinking}\n[/thinking]"))
                continue
        new_content.append(block)

    if not new_content:
        new_content = [TextContent(text="(no content)")]

    return AssistantMessage(
        role=msg.role,
        content=new_content,
        api=msg.api,
        provider=msg.provider,
        model=msg.model,
        usage=msg.usage,
        stop_reason=msg.stop_reason,
        response_id=msg.response_id,
        error_message=msg.error_message,
        timestamp=msg.timestamp,
    )


def _process_tool_result(msg: ToolResultMessage, *, max_chars: int) -> ToolResultMessage:
    total_chars = sum(len(b.text) for b in msg.content if isinstance(b, TextContent))
    if total_chars <= max_chars:
        return msg

    new_content = []
    remaining = max_chars
    for block in msg.content:
        if isinstance(block, TextContent):
            if remaining <= 0:
                continue
            if len(block.text) > remaining:
                new_content.append(TextContent(text=block.text[:remaining] + TOOL_RESULT_TRUNCATION_NOTICE))
                remaining = 0
            else:
                new_content.append(block)
                remaining -= len(block.text)
        elif isinstance(block, ImageContent):
            new_content.append(block)

    return ToolResultMessage(
        role=msg.role,
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
        content=new_content,
        is_error=msg.is_error,
        details=msg.details,
        timestamp=msg.timestamp,
    )


def _ensure_valid_sequence(messages: list[Message]) -> list[Message]:
    """确保消息序列对 LLM 有效：不以 assistant 开头，不连续出现同 role。"""
    if not messages:
        return messages
    result: list[Message] = []
    for msg in messages:
        if not result:
            result.append(msg)
            continue
        prev = result[-1]
        if isinstance(prev, AssistantMessage) and isinstance(msg, AssistantMessage):
            continue
        result.append(msg)
    return result

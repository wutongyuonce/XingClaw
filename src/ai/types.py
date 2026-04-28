from __future__ import annotations

"""
本模块定义 ai 包的核心数据结构。

设计原则：
1. 对外暴露稳定的数据模型，避免调用方直接依赖 provider 私有字段。
2. 所有 provider 的输入/输出都先归一到这些结构，再交给上层 Agent 使用。
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

# 协议标识：用于把请求分发到对应 provider。
Api = str
# 供应商标识：用于鉴权和模型分组。
Provider = str
# 一次回答结束原因。
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
# 简化推理等级（留给 stream_simple 接口使用）。
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


@dataclass
class Cost:
    """成本统计，单位由上层自行约定（通常是美元）。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    """token 使用统计。"""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass
class TextContent:
    """普通文本块。"""

    type: Literal["text"] = "text"
    text: str = ""
    text_signature: Optional[str] = None


@dataclass
class ThinkingContent:
    """模型的思考块（如果 provider 支持）。"""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: Optional[str] = None
    redacted: bool = False


@dataclass
class ImageContent:
    """图片块，使用 base64 数据承载。"""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"


@dataclass
class ToolCall:
    """模型发起的工具调用。"""

    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


AssistantBlock = Union[TextContent, ThinkingContent, ToolCall]
UserBlock = Union[TextContent, ImageContent]
ToolResultBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    """用户消息。"""

    role: Literal["user"] = "user"
    content: Union[str, list[UserBlock]] = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    """助手消息（流式完成后的标准形态）。"""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantBlock] = field(default_factory=list)
    api: Api = ""
    provider: Provider = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    response_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: int = 0


@dataclass
class ToolResultMessage:
    """工具执行结果消息。"""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultBlock] = field(default_factory=list)
    is_error: bool = False
    details: Any = None
    timestamp: int = 0


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class Tool:
    """可被模型调用的工具定义。"""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Context:
    """一次请求上下文：系统提示词 + 消息历史 + 可用工具。"""

    messages: list[Message]
    system_prompt: Optional[str] = None
    tools: Optional[list[Tool]] = None


@dataclass
class StreamOptions:
    """流式调用的通用参数。"""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    headers: Optional[dict[str, str]] = None
    timeout_seconds: Optional[float] = None
    session_id: Optional[str] = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    """简化接口参数：额外支持 reasoning 等级。"""

    reasoning: Optional[ThinkingLevel] = None


@dataclass
class Model:
    """
    模型配置。

    注意：
    - api 决定请求走哪种协议实现；
    - provider 决定默认鉴权读取方式；
    - base_url 允许指向官方服务或自建兼容网关。
    """

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    context_window: int
    max_tokens: int
    cost: Cost = field(default_factory=Cost)
    headers: Optional[dict[str, str]] = None
    compat: Optional[dict[str, Any]] = None

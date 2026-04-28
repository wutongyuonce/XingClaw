from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass
class IMIncomingMessage:
    """统一后的 IM 入站消息。"""

    platform: str
    channel_id: str
    user_id: str
    text: str
    thread_id: str | None = None
    message_id: str | None = None
    created_at: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class IMOutgoingText:
    """统一的文本回包。"""

    channel_id: str
    text: str
    thread_id: str | None = None
    reply_to_message_id: str | None = None


@dataclass
class IMOutgoingCard:
    """飞书交互卡片消息。"""

    channel_id: str
    title: str
    markdown_content: str
    thread_id: str | None = None
    reply_to_message_id: str | None = None


@dataclass
class IMUserInfo:
    """缓存的用户信息。"""

    user_id: str
    name: str = ""
    avatar_url: str = ""
    department: str = ""


@dataclass
class IMChannelInfo:
    """缓存的频道/群信息。"""

    channel_id: str
    name: str = ""
    description: str = ""
    owner_id: str = ""
    member_count: int = 0


@dataclass
class IMWebhookResult:
    """
    webhook 解析结果：
    - ack: 立即返回给平台的响应体
    - messages: 可投递给 Agent 的消息列表
    """

    ack: dict[str, Any]
    messages: list[IMIncomingMessage]


class IMAdapter(Protocol):
    def handle_webhook(self, headers: Mapping[str, str], body: bytes) -> IMWebhookResult:
        """解析平台 webhook 请求，返回标准化消息。"""

    def send_text(self, message: IMOutgoingText) -> str | None:
        """发送文本消息到平台，返回 message_id（供后续更新用）。"""

    def update_text(self, message_id: str, text: str) -> None:
        """更新已发送消息的文本内容（流式打字效果）。"""

    def send_card(self, message: IMOutgoingCard) -> str | None:
        """发送交互卡片消息。"""

    def get_user_info(self, user_id: str) -> IMUserInfo | None:
        """查询用户信息（带缓存）。"""

    def get_chat_info(self, chat_id: str) -> IMChannelInfo | None:
        """查询频道/群信息（带缓存）。"""

"""
XingClaw im
===========

IM 接入层（当前实现飞书）：
- webhook 解析
- 会话路由（频道级长期 session）
- 流式消息更新
- 调用 coding_agent
- 回发文本 / 卡片消息
- MEMORY.md 记忆系统
"""

from .memory import load_channel_memory, load_global_memory, load_merged_memory, save_channel_memory, save_global_memory
from .service import IMService, IMServiceConfig
from .session_router import SessionRouter
from .events import IMEventWatcher, IMEventWatcherOptions
from .types import (
    IMAdapter,
    IMChannelInfo,
    IMIncomingMessage,
    IMOutgoingCard,
    IMOutgoingText,
    IMUserInfo,
    IMWebhookResult,
)

__all__ = [
    "IMAdapter",
    "IMChannelInfo",
    "IMIncomingMessage",
    "IMOutgoingCard",
    "IMOutgoingText",
    "IMUserInfo",
    "IMWebhookResult",
    "SessionRouter",
    "IMService",
    "IMServiceConfig",
    "IMEventWatcher",
    "IMEventWatcherOptions",
    "load_global_memory",
    "load_channel_memory",
    "load_merged_memory",
    "save_global_memory",
    "save_channel_memory",
]

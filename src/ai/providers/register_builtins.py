from __future__ import annotations

"""
内置 provider 注册入口。
"""

from ..api_registry import ApiProvider, clear_api_providers, register_api_provider
from .anthropic import stream_anthropic, stream_simple_anthropic
from .openai_compatible import stream_openai_compatible, stream_simple_openai_compatible


def register_builtin_api_providers() -> None:
    """注册第一阶段支持的两个协议。"""
    register_api_provider(
        ApiProvider(
            api="anthropic-messages",
            stream=stream_anthropic,
            stream_simple=stream_simple_anthropic,
        )
    )
    register_api_provider(
        ApiProvider(
            api="openai-standard",
            stream=stream_openai_compatible,
            stream_simple=stream_simple_openai_compatible,
        )
    )


def reset_api_providers() -> None:
    """重置并重新注册内置 provider。"""
    clear_api_providers()
    register_builtin_api_providers()


# 模块加载即注册，保证 stream() 可直接使用。
register_builtin_api_providers()

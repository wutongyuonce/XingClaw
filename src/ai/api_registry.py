from __future__ import annotations

"""
api -> provider 实现的注册中心。

这样可以做到：
1) stream() 时按 model.api 动态分发；
2) 后续扩展新 provider 时只需注册，不改调用方代码。
"""

from dataclasses import dataclass
from typing import Callable

from .event_stream import AssistantMessageEventStream
from .types import Context, Model, SimpleStreamOptions, StreamOptions

StreamFn = Callable[[Model, Context, StreamOptions | None], AssistantMessageEventStream]
SimpleStreamFn = Callable[[Model, Context, SimpleStreamOptions | None], AssistantMessageEventStream]


@dataclass
class ApiProvider:
    api: str
    stream: StreamFn
    stream_simple: SimpleStreamFn


_REGISTRY: dict[str, ApiProvider] = {}


def register_api_provider(provider: ApiProvider) -> None:
    """注册或覆盖某个 api 的 provider。"""
    _REGISTRY[provider.api] = provider


def get_api_provider(api: str) -> ApiProvider | None:
    """按 api 获取 provider；不存在返回 None。"""
    return _REGISTRY.get(api)


def clear_api_providers() -> None:
    """清空注册中心（通常用于测试或重置）。"""
    _REGISTRY.clear()

from __future__ import annotations

"""
统一调用入口：
- stream / complete
- stream_simple / complete_simple
"""

from .api_registry import get_api_provider
from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions


def _resolve_provider(api: str):
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    """返回统一事件流。"""
    provider = _resolve_provider(model.api)
    return provider.stream(model, context, options)


async def complete(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
    """返回一次完整回答（内部基于 stream.result()）。"""
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    reasoning: str | None = None,
) -> AssistantMessageEventStream:
    """
    简化版流式接口。

    reasoning 提供快捷写法：stream_simple(..., reasoning="low")
    """
    provider = _resolve_provider(model.api)
    effective_options = options or SimpleStreamOptions()
    if reasoning is not None:
        effective_options.reasoning = reasoning
    return provider.stream_simple(model, context, effective_options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    reasoning: str | None = None,
) -> AssistantMessage:
    """简化版完整回答接口。"""
    s = stream_simple(model, context, options, reasoning=reasoning)
    return await s.result()

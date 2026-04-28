from __future__ import annotations

"""
统一事件流容器。

调用方可以：
1) async for 逐个消费事件；
2) await result() 获取最终 AssistantMessage。
"""

import asyncio
from typing import Any, AsyncIterator, Optional

from .types import AssistantMessage


_SENTINEL = object()


class AssistantMessageEventStream:
    def __init__(self) -> None:
        # 事件队列：用于迭代消费。
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        # 最终结果 Future：用于一次性获取完整消息。
        self._result: "asyncio.Future[AssistantMessage]" = asyncio.get_event_loop().create_future()
        self._closed = False

    def push(self, event: dict[str, Any]) -> None:
        """推送一个事件（text_delta/toolcall_delta/...）。"""
        if self._closed:
            return
        self._queue.put_nowait(event)

    def end(self, message: AssistantMessage) -> None:
        """正常结束：写入最终消息并关闭流。"""
        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.set_result(message)
        self._queue.put_nowait(_SENTINEL)

    def fail(self, error: Exception, fallback: Optional[AssistantMessage] = None) -> None:
        """
        异常结束。

        fallback 存在时，result() 仍返回 fallback；
        否则 result() 抛出异常。
        """
        if self._closed:
            return
        self._closed = True
        if fallback is not None:
            if not self._result.done():
                self._result.set_result(fallback)
        else:
            if not self._result.done():
                self._result.set_exception(error)
        self._queue.put_nowait(_SENTINEL)

    async def result(self) -> AssistantMessage:
        """等待并返回最终 AssistantMessage。"""
        return await self._result

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item

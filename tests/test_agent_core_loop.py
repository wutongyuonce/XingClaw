from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Callable

# 允许直接从源码目录导入（不依赖是否已 pip install -e .）
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, UserMessage
from agent_core.agent_loop import run_agent_loop, run_agent_loop_continue
from agent_core.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)


class _FakeEventStream:
    def __init__(self, events: list[dict[str, Any]], final_message: AssistantMessage) -> None:
        self._events = events
        self._final = final_message
        self._idx = 0

    def __aiter__(self) -> "_FakeEventStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        item = self._events[self._idx]
        self._idx += 1
        return item

    async def result(self) -> AssistantMessage:
        return self._final


def _build_stream_fn(final_messages: list[AssistantMessage]) -> Callable[..., _FakeEventStream]:
    idx = {"value": 0}

    def _stream_fn(model: Any, context: Any, options: Any) -> _FakeEventStream:
        _ = model, context, options
        current = final_messages[idx["value"]]
        idx["value"] += 1
        return _FakeEventStream(events=[{"type": "done"}], final_message=current)

    return _stream_fn


def _make_config(**kwargs: Any) -> AgentLoopConfig:
    model = get_model("anthropic", "claude-sonnet-4-5")
    defaults: dict[str, Any] = {
        "model": model,
        "convert_to_llm": lambda messages: messages,
        "session_id": "test-session",
    }
    defaults.update(kwargs)
    return AgentLoopConfig(**defaults)


class AgentCoreLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_schema_contains_common_fields(self) -> None:
        events: list[AgentEvent] = []
        config = _make_config()
        context = AgentContext(system_prompt="test", messages=[], tools=[])
        prompt = UserMessage(content="hello")
        final = AssistantMessage(content=[TextContent(text="hi")], stop_reason="stop")

        new_messages = await run_agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([final]),
        )

        self.assertEqual(len(new_messages), 2)
        self.assertTrue(events)
        for e in events:
            self.assertIn("type", e)
            self.assertIn("runId", e)
            self.assertIn("turnId", e)
            self.assertIn("eventId", e)
            self.assertIn("timestamp", e)
            self.assertIn("sessionId", e)
            self.assertEqual(e["sessionId"], "test-session")

    async def test_parallel_tool_execution(self) -> None:
        events: list[AgentEvent] = []
        tool_calls = [
            ToolCall(id="tc1", name="slow_tool", arguments={}),
            ToolCall(id="tc2", name="slow_tool", arguments={}),
        ]

        async def slow_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            await asyncio.sleep(0.06)
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="slow_tool",
            label="Slow Tool",
            description="sleep",
            parameters={"type": "object", "properties": {}},
            execute=slow_tool,
        )

        first = AssistantMessage(content=tool_calls, stop_reason="toolUse")
        second = AssistantMessage(content=[TextContent(text="done")], stop_reason="stop")

        config = _make_config(tool_execution="parallel")
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])
        prompt = UserMessage(content="run tools")

        started = time.perf_counter()
        new_messages = await run_agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([first, second]),
        )
        elapsed = time.perf_counter() - started

        # 并行执行时，两个 60ms 工具总耗时应明显小于串行 120ms。
        self.assertLess(elapsed, 0.11)
        tool_result_count = len([m for m in new_messages if getattr(m, "role", "") == "toolResult"])
        self.assertEqual(tool_result_count, 2)

        started_events = [e for e in events if e["type"] == "tool_execution_start"]
        self.assertEqual(len(started_events), 2)

    async def test_before_hook_can_block_tool(self) -> None:
        events: list[AgentEvent] = []
        execute_count = {"value": 0}

        async def blocked_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            execute_count["value"] += 1
            return AgentToolResult(content=[TextContent(text="should-not-run")])

        tool = AgentTool(
            name="dangerous_tool",
            label="Danger",
            description="blocked tool",
            parameters={"type": "object", "properties": {}},
            execute=blocked_tool,
        )

        async def before_hook(ctx, signal=None) -> BeforeToolCallResult:
            _ = ctx, signal
            return BeforeToolCallResult(block=True, reason="blocked-by-test")

        first = AssistantMessage(
            content=[ToolCall(id="tc_block", name="dangerous_tool", arguments={})],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="fallback")], stop_reason="stop")

        config = _make_config(tool_execution="sequential", before_tool_call=before_hook)
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        new_messages = await run_agent_loop(
            prompts=[UserMessage(content="try tool")],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(execute_count["value"], 0)
        blocked_results = [
            m
            for m in new_messages
            if getattr(m, "role", "") == "toolResult"
            and any(isinstance(c, TextContent) and "blocked-by-test" in c.text for c in m.content)
        ]
        self.assertTrue(blocked_results)
        end_events = [e for e in events if e["type"] == "tool_execution_end" and e.get("isError")]
        self.assertTrue(end_events)

    async def test_continue_loop(self) -> None:
        events: list[AgentEvent] = []
        context = AgentContext(
            system_prompt="test",
            messages=[UserMessage(content="continue please")],
            tools=[],
        )
        config = _make_config()
        final = AssistantMessage(content=[TextContent(text="continued")], stop_reason="stop")

        new_messages = await run_agent_loop_continue(
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([final]),
        )
        self.assertEqual(len(new_messages), 1)
        self.assertEqual(getattr(new_messages[0], "role", ""), "assistant")

        bad_context = AgentContext(
            system_prompt="test",
            messages=[AssistantMessage(content=[TextContent(text="already assistant")], stop_reason="stop")],
            tools=[],
        )
        with self.assertRaises(ValueError):
            await run_agent_loop_continue(
                context=bad_context,
                config=config,
                emit=events.append,
                stream_fn=_build_stream_fn([final]),
            )


if __name__ == "__main__":
    unittest.main()

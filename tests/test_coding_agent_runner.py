from __future__ import annotations

import sys
import io
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.runner import RunOptions, run, run_interactive, run_rpc


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "s1"
        self.messages = []
        self._listeners = []
        self.extension_commands = {
            "ext_ping": type(
                "Cmd",
                (),
                {"name": "ext_ping", "description": "ext ping", "source": "extension", "handler": staticmethod(lambda ctx: "pong")},
            )()
        }

    async def prompt(self, text: str, *, images=None):
        _ = text, images
        for listener in list(self._listeners):
            listener({"type": "message_end", "message": {"role": "assistant"}})
        return []

    async def continue_run(self):
        for listener in list(self._listeners):
            listener({"type": "turn_end"})
        return []

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def list_entry_ids(self):
        return ["e1", "e2"]

    def list_entries(self):
        return [
            {"id": "e1", "parent_id": None, "depth": 0, "is_leaf": False},
            {"id": "e2", "parent_id": "e1", "depth": 1, "is_leaf": True},
        ]

    def get_leaf_id(self):
        return "e2"

    def get_entry_path(self, entry_id: str):
        if entry_id == "e2":
            return ["e1", "e2"]
        return [entry_id]

    def get_session_tree(self):
        return [{"id": "e1", "children": [{"id": "e2", "children": []}]}]

    def fork_from_entry(self, entry_id: str):
        _ = entry_id
        forked = _FakeSession()
        forked.session_id = "forked_s"
        forked.close = lambda: None
        return forked

    def switch_to_entry(self, entry_id: str):
        _ = entry_id
        return None

    def close(self):
        return None


class CodingAgentRunnerDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_dispatch_print_mode_requires_prompt(self) -> None:
        session = _FakeSession()
        with self.assertRaises(ValueError):
            await run(RunOptions(mode="print", session=session, prompt=None))

    async def test_run_dispatch_rpc_mode(self) -> None:
        session = _FakeSession()
        with patch("coding_agent.runner.run_rpc", new_callable=AsyncMock) as rpc_mock:
            await run(RunOptions(mode="rpc", session=session))
        rpc_mock.assert_awaited_once()

    async def test_run_rpc_protocol(self) -> None:
        session = _FakeSession()
        outputs: list[str] = []
        stdin_data = "\n".join(
            [
                json.dumps({"id": "1", "type": "state"}),
                json.dumps({"id": "2", "type": "list_entries"}),
                json.dumps({"id": "3", "type": "show_tree"}),
                json.dumps({"id": "4", "type": "entry_path", "entry_id": "e2"}),
                json.dumps({"id": "5", "type": "fork_entry", "entry_id": "e1"}),
                json.dumps({"id": "6", "type": "switch_entry", "entry_id": "e2"}),
                json.dumps({"id": "7", "type": "get_commands"}),
                json.dumps({"id": "8", "type": "prompt", "text": "hello"}),
                json.dumps({"id": "9", "type": "continue"}),
                json.dumps({"id": "10", "type": "shutdown"}),
            ]
        )
        with patch("sys.stdin", io.StringIO(stdin_data)):
            await run_rpc(session, output=outputs.append)

        parsed = [json.loads(line) for line in outputs]
        self.assertEqual(parsed[0]["type"], "rpc_ready")
        commands = [item.get("command") for item in parsed if item.get("type") == "response"]
        self.assertIn("state", commands)
        self.assertIn("list_entries", commands)
        self.assertIn("show_tree", commands)
        self.assertIn("entry_path", commands)
        self.assertIn("fork_entry", commands)
        self.assertIn("switch_entry", commands)
        self.assertIn("get_commands", commands)
        self.assertIn("prompt", commands)
        self.assertIn("continue", commands)
        self.assertIn("shutdown", commands)
        responses = [item for item in parsed if item.get("type") == "response"]
        self.assertTrue(all(item.get("status") == "ok" for item in responses))
        self.assertTrue(any(item.get("type") == "event" for item in parsed))
        get_cmd_resp = next(
            (item for item in responses if item.get("command") == "get_commands"),
            None,
        )
        self.assertIsNotNone(get_cmd_resp)
        cmd_names = [x.get("name") for x in get_cmd_resp.get("data", {}).get("commands", [])]
        self.assertIn("ext_ping", cmd_names)

    async def test_run_interactive_session_commands(self) -> None:
        session = _FakeSession()
        outputs: list[str] = []
        inputs = iter(["/session", "/tree", "/switch e2", "exit"])
        await run_interactive(
            session,
            input_fn=lambda _: next(inputs),
            output=outputs.append,
            show_tool_events=False,
        )
        joined = "\n".join(outputs)
        self.assertIn("session_id=s1", joined)
        self.assertIn("- e1", joined)
        self.assertIn("switched leaf", joined)
        self.assertIn("/ext_ping", joined)


if __name__ == "__main__":
    unittest.main()

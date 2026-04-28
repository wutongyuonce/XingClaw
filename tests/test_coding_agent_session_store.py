from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from coding_agent.agent_session import AgentSession
from coding_agent.factory import create_agent_session
from coding_agent.serde import message_from_dict, message_to_dict
from coding_agent.session_store import SessionStore
from coding_agent.types import AgentSessionOptions, CreateAgentSessionOptions


class CodingAgentStoreTests(unittest.TestCase):
    def test_message_roundtrip(self) -> None:
        messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=1),
            AssistantMessage(
                content=[TextContent(text="ok"), ToolCall(id="tc1", name="tool_a", arguments={"x": 1})],
                stop_reason="toolUse",
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="tc1",
                tool_name="tool_a",
                content=[TextContent(text="done")],
                is_error=False,
                timestamp=3,
            ),
        ]

        rebuilt = [message_from_dict(message_to_dict(m)) for m in messages]
        self.assertEqual(len(rebuilt), 3)
        self.assertEqual(getattr(rebuilt[0], "role", ""), "user")
        self.assertEqual(getattr(rebuilt[1], "role", ""), "assistant")
        self.assertEqual(getattr(rebuilt[2], "role", ""), "toolResult")

    def test_session_store_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")

            store.append_context_message(UserMessage(content="hello"))
            store.append_event({"type": "agent_start", "runId": "r1"})

            loaded = store.load_context_messages()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(getattr(loaded[0], "role", ""), "user")
            loaded_from_session = store.load_session_messages()
            self.assertEqual(len(loaded_from_session), 1)

            meta = json.loads((Path(tmp_dir) / ".xingclaw" / "sessions" / "s1" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["session_id"], "s1")

    def test_factory_resolve_from_provider_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="anthropic",
                    model_id="claude-sonnet-4-5",
                    system_prompt="test",
                )
            )
            self.assertEqual(session.agent.state.model.id, "claude-sonnet-4-5")
            session.close()

    def test_factory_restore_from_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    system_prompt="restored-system",
                )
            )
            sid = first.session_id
            first.close()

            restored = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    session_id=sid,
                )
            )
            self.assertEqual(restored.agent.state.model.id, "gpt-4o-mini")
            self.assertEqual(restored.agent.state.system_prompt, "restored-system")
            restored.close()

    def test_context_compaction_rewrites_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    max_context_messages=4,
                    retain_recent_messages=2,
                )
            )

            session.agent.set_messages(
                [
                    UserMessage(content="u1"),
                    AssistantMessage(content=[TextContent(text="a1")]),
                    UserMessage(content="u2"),
                    AssistantMessage(content=[TextContent(text="a2")]),
                    UserMessage(content="u3"),
                    AssistantMessage(content=[TextContent(text="a3")]),
                ]
            )
            session._compact_context_if_needed()  # 测试私有策略入口

            compacted = session.agent.state.messages
            self.assertEqual(len(compacted), 3)
            self.assertEqual(getattr(compacted[0], "role", ""), "user")

            summary_text = ""
            first = compacted[0]
            if isinstance(first, UserMessage) and isinstance(first.content, list):
                summary_text = "".join(b.text for b in first.content if isinstance(b, TextContent))
            self.assertIn("[Context Summary]", summary_text)

            reloaded = session.store.load_context_messages()
            self.assertEqual(len(reloaded), 3)
            session.close()

    def test_session_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_context_message(UserMessage(content="u1"))
            store.append_context_message(AssistantMessage(content=[TextContent(text="a1")]))

            entry_ids = store.list_entry_ids()
            self.assertGreaterEqual(len(entry_ids), 2)
            forked = store.fork_to("s2", from_entry_id=entry_ids[0])
            loaded = forked.load_session_messages()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(getattr(loaded[0], "role", ""), "user")

    def test_session_tree_and_switch_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_context_message(UserMessage(content="u1"))
            store.append_context_message(AssistantMessage(content=[TextContent(text="a1")]))
            store.append_context_message(UserMessage(content="u2"))

            tree = store.get_session_tree()
            self.assertEqual(len(tree), 1)
            self.assertEqual(tree[0]["role"], "user")
            self.assertEqual(len(tree[0]["children"]), 1)

            ids = store.list_entry_ids()
            self.assertGreaterEqual(len(ids), 3)
            path = store.get_entry_path(ids[2])
            self.assertEqual(path, [ids[0], ids[1], ids[2]])
            store.set_leaf(ids[1])
            self.assertEqual(store.get_leaf_id(), ids[1])
            branch = store.load_session_messages()
            self.assertEqual(len(branch), 2)
            entries = store.list_entries()
            self.assertEqual(len(entries), 3)
            leaf_entries = [item for item in entries if item.get("is_leaf")]
            self.assertEqual(len(leaf_entries), 1)
            self.assertEqual(leaf_entries[0]["id"], ids[1])


if __name__ == "__main__":
    unittest.main()

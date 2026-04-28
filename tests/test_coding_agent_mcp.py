from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.factory import create_agent_session
from coding_agent.mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from coding_agent.types import CreateAgentSessionOptions


class _FakeMCPClient:
    async def call_tool(self, server: str, tool: str, arguments: dict):
        return f"{server}.{tool}:{arguments.get('q', '')}"


class CodingAgentMCPTests(unittest.TestCase):
    def test_parse_mcp_configs_and_proxy_execute(self) -> None:
        cfg = parse_mcp_tool_configs(
            [
                {
                    "name": "demo",
                    "tools": [
                        {
                            "name": "mcp_search",
                            "tool": "search",
                            "description": "search via mcp",
                            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                        }
                    ],
                }
            ]
        )
        tools = create_mcp_proxy_tools(cfg, client=_FakeMCPClient())
        self.assertEqual(len(tools), 1)
        result = asyncio.run(tools[0].execute("tc1", {"q": "abc"}))
        self.assertIn("demo.search:abc", result.content[0].text if result.content else "")

    def test_factory_registers_mcp_proxy_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    mcp_client=_FakeMCPClient(),
                    mcp_servers=[
                        {
                            "name": "demo",
                            "tools": [
                                {
                                    "name": "mcp_echo",
                                    "tool": "echo",
                                    "description": "echo from mcp",
                                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                                }
                            ],
                        }
                    ],
                )
            )
            names = {t.name for t in session.agent.state.tools}
            self.assertIn("mcp_echo", names)
            session.close()


if __name__ == "__main__":
    unittest.main()

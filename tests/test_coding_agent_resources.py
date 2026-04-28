from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.builtin_tools import create_builtin_tools
from coding_agent.factory import create_agent_session
from coding_agent.resources import WorkspaceResourceLoader
from coding_agent.types import CreateAgentSessionOptions
from ai.types import AssistantMessage, ToolCall
from agent_core import AfterToolCallContext, AgentContext, AgentToolResult, BeforeToolCallContext


class CodingAgentResourceTests(unittest.TestCase):
    def test_workspace_loader_reads_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / ".xingclaw"
            root.mkdir(parents=True, exist_ok=True)
            (root / "prompt.md").write_text("system from prompt", encoding="utf-8")
            (root / "settings.json").write_text(
                json.dumps(
                    {
                        "provider": "openai-standard",
                        "model_id": "gpt-4o-mini",
                        "thinking_level": "minimal",
                        "tool_execution": "sequential",
                    }
                ),
                encoding="utf-8",
            )
            (root / "tools.json").write_text(json.dumps({"enabled": ["list_dir", "read_file"]}), encoding="utf-8")

            resources = WorkspaceResourceLoader(tmp_dir).load()
            self.assertEqual(resources.prompt, "system from prompt")
            self.assertEqual(resources.settings.provider, "openai-standard")
            self.assertEqual(resources.enabled_tools, ["list_dir", "read_file"])

    def test_builtin_tools_create_and_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = create_builtin_tools(
                tmp_dir,
                enabled_names=["write", "read", "edit", "grep", "find", "ls", "bash"],
            )
            names = sorted([t.name for t in tools])
            self.assertIn("write", names)
            self.assertIn("read", names)
            self.assertIn("edit", names)
            self.assertIn("grep", names)
            self.assertIn("find", names)
            self.assertIn("ls", names)
            self.assertIn("bash", names)

            write_tool = next(t for t in tools if t.name == "write")
            read_tool = next(t for t in tools if t.name == "read")
            edit_tool = next(t for t in tools if t.name == "edit")
            grep_tool = next(t for t in tools if t.name == "grep")
            find_tool = next(t for t in tools if t.name == "find")
            ls_tool = next(t for t in tools if t.name == "ls")
            bash_tool = next(t for t in tools if t.name == "bash")

            asyncio.run(write_tool.execute("tc1", {"path": "a.txt", "content": "hello"}))
            result = asyncio.run(read_tool.execute("tc2", {"path": "a.txt"}))
            text = result.content[0].text if result.content else ""
            self.assertIn("hello", text)

            asyncio.run(edit_tool.execute("tc3", {"path": "a.txt", "old_text": "hello", "new_text": "world"}))
            edited = asyncio.run(read_tool.execute("tc4", {"path": "a.txt"}))
            edited_text = edited.content[0].text if edited.content else ""
            self.assertIn("world", edited_text)

            grep_result = asyncio.run(grep_tool.execute("tc5", {"pattern": "world", "path": "."}))
            self.assertIn("a.txt", grep_result.content[0].text if grep_result.content else "")

            find_result = asyncio.run(find_tool.execute("tc6", {"path": ".", "pattern": "**/*.txt"}))
            self.assertIn("a.txt", find_result.content[0].text if find_result.content else "")

            ls_result = asyncio.run(ls_tool.execute("tc7", {"path": "."}))
            self.assertIn("a.txt", ls_result.content[0].text if ls_result.content else "")

            bash_result = asyncio.run(bash_tool.execute("tc8", {"command": "echo hello-from-bash"}))
            self.assertIn("hello-from-bash", bash_result.content[0].text if bash_result.content else "")

    def test_factory_uses_workspace_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / ".xingclaw"
            root.mkdir(parents=True, exist_ok=True)
            (root / "prompt.md").write_text("workspace prompt", encoding="utf-8")
            (root / "settings.json").write_text(
                json.dumps({"provider": "openai-standard", "model_id": "gpt-4o-mini", "thinking_level": "minimal"}),
                encoding="utf-8",
            )
            (root / "tools.json").write_text(json.dumps({"enabled": ["list_dir"]}), encoding="utf-8")

            session = create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_dir))
            self.assertEqual(session.agent.state.model.id, "gpt-4o-mini")
            self.assertEqual(session.agent.state.system_prompt, "workspace prompt")
            tool_names = [t.name for t in session.agent.state.tools]
            self.assertIn("list_dir", tool_names)
            self.assertNotIn("write_file", tool_names)
            session.close()

    def test_factory_uses_default_chinese_system_prompt_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    load_workspace_resources=False,
                )
            )
            prompt = session.agent.state.system_prompt
            self.assertIn("你是一个专业、可靠的编程助手", prompt)
            self.assertIn("工具使用规范", prompt)
            self.assertIn("当前日期", prompt)
            self.assertIn("当前工作目录", prompt)
            session.close()

    def test_factory_read_only_mode_filters_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / ".xingclaw"
            root.mkdir(parents=True, exist_ok=True)
            (root / "settings.json").write_text(
                json.dumps({"provider": "openai-standard", "model_id": "gpt-4o-mini", "read_only_mode": True}),
                encoding="utf-8",
            )
            session = create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_dir))
            names = {t.name for t in session.agent.state.tools}
            self.assertIn("read", names)
            self.assertIn("grep", names)
            self.assertIn("find", names)
            self.assertIn("ls", names)
            self.assertNotIn("write", names)
            self.assertNotIn("edit", names)
            self.assertNotIn("bash", names)
            session.close()

    def test_bash_dangerous_command_blocked_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = create_builtin_tools(tmp_dir, enabled_names=["bash"])
            bash_tool = next(t for t in tools if t.name == "bash")
            result = asyncio.run(bash_tool.execute("tc1", {"command": "rm -rf temp-dir"}))
            text = result.content[0].text if result.content else ""
            self.assertIn("Blocked dangerous command", text)

    def test_dynamic_prompt_from_workspace_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / ".xingclaw"
            root.mkdir(parents=True, exist_ok=True)
            (root / "settings.json").write_text(
                json.dumps(
                    {
                        "provider": "openai-standard",
                        "model_id": "gpt-4o-mini",
                        "prompt_guidelines": ["遇到失败时先给出可执行修复步骤"],
                        "append_system_prompt": "你必须始终用中文回答。",
                        "tool_snippets": {"read": "读取代码并显示关键上下文。"},
                    }
                ),
                encoding="utf-8",
            )
            session = create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_dir))
            prompt = session.agent.state.system_prompt
            self.assertIn("遇到失败时先给出可执行修复步骤", prompt)
            self.assertIn("你必须始终用中文回答。", prompt)
            self.assertIn("read: 读取代码并显示关键上下文", prompt)
            session.close()

    def test_skill_markdown_is_injected_into_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / ".xingclaw" / "skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "review.md").write_text(
                "# 代码审查技能\n\n优先识别风险和回归点，再给修复建议。",
                encoding="utf-8",
            )
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                )
            )
            prompt = session.agent.state.system_prompt
            self.assertIn("技能约束（代码审查技能）", prompt)
            self.assertIn("## Skill: 代码审查技能", prompt)
            self.assertIn("优先识别风险和回归点", prompt)
            self.assertTrue(any(name.startswith("skill:") for name in session.extension_commands.keys()))
            session.close()

    def test_skill_frontmatter_registers_command_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / ".xingclaw" / "skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "s1.md").write_text(
                "---\nname: 审查技能\ncommand: skill:review\ndescription: 执行审查\n---\n请按清单审查。",
                encoding="utf-8",
            )
            (skill_dir / "s2.md").write_text(
                "---\nname: 冲突技能\ncommand: skill:review\n---\n覆盖命令。",
                encoding="utf-8",
            )
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    prompt_debug_sources=True,
                )
            )
            self.assertIn("skill:review", session.extension_commands)
            prompt = session.agent.state.system_prompt
            self.assertIn("Prompt Sources", prompt)
            self.assertIn("skill command conflict", prompt)
            session.close()

    def test_extension_registers_tool_and_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ext_dir = Path(tmp_dir) / ".xingclaw" / "extensions"
            ext_dir.mkdir(parents=True, exist_ok=True)
            (ext_dir / "sample_ext.py").write_text(
                "\n".join(
                    [
                        "from agent_core import AgentTool, AgentToolResult, BeforeToolCallResult, AfterToolCallResult",
                        "from ai.types import TextContent",
                        "",
                        "async def ping_exec(tool_call_id, params, signal=None, on_update=None):",
                        "    _ = tool_call_id, params, signal, on_update",
                        "    return AgentToolResult(content=[TextContent(text='pong')], details={'from': 'ext'})",
                        "",
                        "def before_hook(ctx, signal=None):",
                        "    _ = signal",
                        "    if ctx.tool_call.name == 'write':",
                        "        return BeforeToolCallResult(block=True, reason='blocked by extension')",
                        "    return None",
                        "",
                        "def after_hook(ctx, signal=None):",
                        "    _ = signal",
                        "    return AfterToolCallResult(details={'after': 'ok'})",
                        "",
                        "def before_prompt(ctx):",
                        "    _ = ctx",
                        "    return None",
                        "",
                        "def after_prompt(ctx):",
                        "    _ = ctx",
                        "    return None",
                        "",
                        "def register(api):",
                        "    api.register_tool(AgentTool(",
                        "        name='ping_ext',",
                        "        label='Ping Ext',",
                        "        description='extension tool',",
                        "        parameters={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},",
                        "        execute=ping_exec,",
                        "    ))",
                        "    api.on_before_tool_call(before_hook)",
                        "    api.on_after_tool_call(after_hook)",
                        "    api.on_before_prompt(before_prompt)",
                        "    api.on_after_prompt(after_prompt)",
                        "    api.register_command('ext_ping', lambda ctx: f'ext-ping:{len(ctx.args)}', 'ext ping cmd')",
                        "    api.add_prompt_guideline('扩展要求：优先返回可执行步骤')",
                        "    api.append_system_prompt('扩展提示：输出必须中文。')",
                    ]
                ),
                encoding="utf-8",
            )
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                )
            )
            tool_names = {t.name for t in session.agent.state.tools}
            self.assertIn("ping_ext", tool_names)
            prompt = session.agent.state.system_prompt
            self.assertIn("扩展要求：优先返回可执行步骤", prompt)
            self.assertIn("扩展提示：输出必须中文。", prompt)
            self.assertIn("ext_ping", session.extension_commands)
            self.assertEqual(len(session.before_prompt_hooks), 1)
            self.assertEqual(len(session.after_prompt_hooks), 1)

            before = session.agent._options.before_tool_call
            self.assertIsNotNone(before)
            blocked = asyncio.run(
                before(  # type: ignore[misc]
                    BeforeToolCallContext(
                        assistant_message=AssistantMessage(),
                        tool_call=ToolCall(id="tc1", name="write", arguments={}),
                        args={},
                        context=AgentContext(system_prompt="", messages=[], tools=[]),
                    ),
                    None,
                )
            )
            self.assertTrue(blocked.block if blocked else False)

            after = session.agent._options.after_tool_call
            self.assertIsNotNone(after)
            after_result = asyncio.run(
                after(  # type: ignore[misc]
                    AfterToolCallContext(
                        assistant_message=AssistantMessage(),
                        tool_call=ToolCall(id="tc2", name="read", arguments={}),
                        args={},
                        result=AgentToolResult(content=[]),
                        is_error=False,
                        context=AgentContext(system_prompt="", messages=[], tools=[]),
                    ),
                    None,
                )
            )
            self.assertEqual(after_result.details, {"after": "ok"} if after_result else None)
            session.close()

    def test_bash_block_and_allow_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = create_builtin_tools(
                tmp_dir,
                enabled_names=["bash"],
                bash_block_patterns=[r"secret"],
                bash_allow_patterns=[r"^echo secret$"],
            )
            bash_tool = next(t for t in tools if t.name == "bash")

            blocked = asyncio.run(bash_tool.execute("tc1", {"command": "echo my-secret"}))
            blocked_text = blocked.content[0].text if blocked.content else ""
            self.assertIn("Blocked by bash block patterns", blocked_text)

            allowed = asyncio.run(bash_tool.execute("tc2", {"command": "echo secret"}))
            allowed_text = allowed.content[0].text if allowed.content else ""
            self.assertIn("echo secret", allowed_text)

    def test_edit_occurrence_index_and_expected_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = create_builtin_tools(tmp_dir, enabled_names=["write", "read", "edit"])
            write_tool = next(t for t in tools if t.name == "write")
            read_tool = next(t for t in tools if t.name == "read")
            edit_tool = next(t for t in tools if t.name == "edit")

            asyncio.run(write_tool.execute("tc1", {"path": "m.txt", "content": "x x x"}))

            failed = asyncio.run(edit_tool.execute("tc2", {"path": "m.txt", "old_text": "x", "new_text": "y"}))
            failed_text = failed.content[0].text if failed.content else ""
            self.assertIn("Multiple matches found", failed_text)

            replaced = asyncio.run(
                edit_tool.execute(
                    "tc3",
                    {"path": "m.txt", "old_text": "x", "new_text": "y", "occurrence_index": 2, "expected_occurrences": 3},
                )
            )
            self.assertIn("replacements=1", replaced.content[0].text if replaced.content else "")

            final_text = asyncio.run(read_tool.execute("tc4", {"path": "m.txt"})).content[0].text
            self.assertEqual(final_text, "x y x")


if __name__ == "__main__":
    unittest.main()

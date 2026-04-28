from __future__ import annotations

"""
coding_agent 内置工具集合。

当前提供（对标常见 coding-agent 能力）：
- read / read_file
- write / write_file
- edit
- bash
- grep
- find
- ls / list_dir
"""

import asyncio
import re
from pathlib import Path
from typing import Any

from ai.types import TextContent
from agent_core import AgentTool, AgentToolResult

READ_ONLY_TOOL_NAMES = {"read", "read_file", "grep", "find", "ls", "list_dir"}
MUTATING_TOOL_NAMES = {"write", "write_file", "edit", "bash"}


def _resolve_workspace_path(workspace_dir: Path, path_text: str) -> Path:
    target = (workspace_dir / path_text).resolve()
    workspace = workspace_dir.resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Path escapes workspace boundary") from exc
    return target


def _is_dangerous_bash_command(command: str) -> bool:
    text = command.lower()
    patterns = [
        "rm -rf",
        "rm -r ",
        "rm -fr",
        "del /f",
        "rmdir /s",
        "format ",
        "mkfs",
        "shutdown",
        "reboot",
        "remove-item -recurse",
    ]
    return any(p in text for p in patterns)


def _matches_any_pattern(text: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


def _replace_nth(text: str, old: str, new: str, nth: int) -> str:
    if nth <= 0:
        raise ValueError("nth must be >= 1")
    start = 0
    match_count = 0
    while True:
        idx = text.find(old, start)
        if idx < 0:
            raise ValueError("nth occurrence not found")
        match_count += 1
        if match_count == nth:
            return text[:idx] + new + text[idx + len(old) :]
        start = idx + len(old)


def create_builtin_tools(
    workspace_dir: str | Path,
    enabled_names: list[str] | None = None,
    *,
    block_dangerous_bash: bool = True,
    bash_allow_patterns: list[str] | None = None,
    bash_block_patterns: list[str] | None = None,
    edit_require_unique_match: bool = True,
) -> list[AgentTool]:
    workspace = Path(workspace_dir)
    enabled = set(enabled_names) if enabled_names else None

    def _allow(name: str) -> bool:
        if enabled is None:
            return True
        return name in enabled

    tools: list[AgentTool] = []

    async def ls_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", "."))
        max_entries = int(params.get("max_entries", 100))
        target = _resolve_workspace_path(workspace, path_text)
        if not target.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {path_text}")], details={})
        if not target.is_dir():
            return AgentToolResult(content=[TextContent(text=f"Not a directory: {path_text}")], details={})

        items = sorted(target.iterdir(), key=lambda p: p.name)[:max_entries]
        lines = []
        for item in items:
            suffix = "/" if item.is_dir() else ""
            size = "-" if item.is_dir() else str(item.stat().st_size)
            lines.append(f"{item.name}{suffix}\t{size}")
        return AgentToolResult(content=[TextContent(text="\n".join(lines) if lines else "(empty)")], details={})

    async def read_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        max_chars = int(params.get("max_chars", 4000))
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})
        target = _resolve_workspace_path(workspace, path_text)
        if not target.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {path_text}")], details={})
        if not target.is_file():
            return AgentToolResult(content=[TextContent(text=f"Not a file: {path_text}")], details={})

        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...<truncated>..."
        return AgentToolResult(content=[TextContent(text=text)], details={})

    async def write_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        content = str(params.get("content", ""))
        overwrite = bool(params.get("overwrite", True))
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})

        target = _resolve_workspace_path(workspace, path_text)
        if target.exists() and not overwrite:
            return AgentToolResult(content=[TextContent(text=f"File exists: {path_text}")], details={})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return AgentToolResult(content=[TextContent(text=f"Wrote file: {path_text}")], details={})

    async def edit_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        path_text = str(params.get("path", ""))
        old_text = str(params.get("old_text", ""))
        new_text = str(params.get("new_text", ""))
        replace_all = bool(params.get("replace_all", False))
        occurrence_index_raw = params.get("occurrence_index")
        expected_occurrences_raw = params.get("expected_occurrences")
        if not path_text:
            return AgentToolResult(content=[TextContent(text="Missing path")], details={})
        if old_text == "":
            return AgentToolResult(content=[TextContent(text="old_text cannot be empty")], details={})
        occurrence_index = None if occurrence_index_raw is None else int(occurrence_index_raw)
        expected_occurrences = None if expected_occurrences_raw is None else int(expected_occurrences_raw)
        if occurrence_index is not None and occurrence_index <= 0:
            return AgentToolResult(content=[TextContent(text="occurrence_index must be >= 1")], details={})
        if expected_occurrences is not None and expected_occurrences < 0:
            return AgentToolResult(content=[TextContent(text="expected_occurrences must be >= 0")], details={})
        target = _resolve_workspace_path(workspace, path_text)
        if not target.exists() or not target.is_file():
            return AgentToolResult(content=[TextContent(text=f"Path not found or not file: {path_text}")], details={})

        original = target.read_text(encoding="utf-8", errors="replace")
        count = original.count(old_text)
        if expected_occurrences is not None and count != expected_occurrences:
            return AgentToolResult(
                content=[TextContent(text=f"Expected {expected_occurrences} matches, but found {count}")],
                details={"matches": count, "expected_occurrences": expected_occurrences},
            )
        if count == 0:
            return AgentToolResult(content=[TextContent(text="No match found")], details={"replacements": 0})
        if not replace_all and count > 1 and occurrence_index is None and edit_require_unique_match:
            return AgentToolResult(
                content=[TextContent(text="Multiple matches found; set replace_all=true or provide more unique old_text")],
                details={"matches": count},
            )
        if replace_all:
            updated = original.replace(old_text, new_text)
            replaced = count
        else:
            if occurrence_index is not None:
                if occurrence_index > count:
                    return AgentToolResult(
                        content=[TextContent(text=f"occurrence_index={occurrence_index} is out of range (matches={count})")],
                        details={"matches": count, "occurrence_index": occurrence_index},
                    )
                updated = _replace_nth(original, old_text, new_text, occurrence_index)
            else:
                updated = original.replace(old_text, new_text, 1)
            replaced = 1
        target.write_text(updated, encoding="utf-8")
        return AgentToolResult(
            content=[TextContent(text=f"Edited file: {path_text} (replacements={replaced})")],
            details={"replacements": replaced},
        )

    async def grep_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        pattern = str(params.get("pattern", ""))
        start_path = str(params.get("path", "."))
        glob_pattern = str(params.get("glob", "**/*"))
        max_matches = int(params.get("max_matches", 200))
        case_sensitive = bool(params.get("case_sensitive", True))
        if not pattern:
            return AgentToolResult(content=[TextContent(text="Missing pattern")], details={})

        root = _resolve_workspace_path(workspace, start_path)
        if not root.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {start_path}")], details={})

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return AgentToolResult(content=[TextContent(text=f"Invalid regex: {exc}")], details={})

        matches: list[str] = []
        files = [p for p in root.glob(glob_pattern) if p.is_file()]
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = file_path.relative_to(workspace).as_posix()
                    matches.append(f"{rel}:{idx}:{line[:220]}")
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break

        return AgentToolResult(
            content=[TextContent(text="\n".join(matches) if matches else "(no matches)")],
            details={"matches": len(matches)},
        )

    async def find_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal, on_update
        start_path = str(params.get("path", "."))
        pattern = str(params.get("pattern", "**/*"))
        max_results = int(params.get("max_results", 200))
        root = _resolve_workspace_path(workspace, start_path)
        if not root.exists():
            return AgentToolResult(content=[TextContent(text=f"Path not found: {start_path}")], details={})

        results = []
        for path in root.glob(pattern):
            rel = path.relative_to(workspace).as_posix()
            results.append(rel + ("/" if path.is_dir() else ""))
            if len(results) >= max_results:
                break
        return AgentToolResult(
            content=[TextContent(text="\n".join(results) if results else "(no files)")],
            details={"count": len(results)},
        )

    async def bash_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        _ = tool_call_id, signal
        command = str(params.get("command", "")).strip()
        timeout_seconds = int(params.get("timeout_seconds", 30))
        cwd_text = str(params.get("cwd", "."))
        allow_dangerous = bool(params.get("allow_dangerous", False))
        if not command:
            return AgentToolResult(content=[TextContent(text="Missing command")], details={})
        is_allowlisted = _matches_any_pattern(command, bash_allow_patterns)
        is_blocklisted = _matches_any_pattern(command, bash_block_patterns)
        if is_blocklisted and not is_allowlisted:
            return AgentToolResult(
                content=[TextContent(text="Blocked by bash block patterns.")],
                details={"blocked": True, "reason": "block_pattern"},
            )
        if (
            block_dangerous_bash
            and _is_dangerous_bash_command(command)
            and not allow_dangerous
            and not is_allowlisted
        ):
            return AgentToolResult(
                content=[
                    TextContent(
                        text="Blocked dangerous command. Set allow_dangerous=true only when you fully understand the risk."
                    )
                ],
                details={"blocked": True, "reason": "dangerous_command"},
            )
        cwd = _resolve_workspace_path(workspace, cwd_text)
        if not cwd.exists() or not cwd.is_dir():
            return AgentToolResult(content=[TextContent(text=f"Invalid cwd: {cwd_text}")], details={})

        if on_update:
            on_update(AgentToolResult(content=[TextContent(text=f"Running command: {command}")], details={"phase": "start"}))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            out_text = stdout.decode("utf-8", errors="replace")
            err_text = stderr.decode("utf-8", errors="replace")
            merged = f"$ {command}\n{out_text}"
            if err_text:
                merged += ("\n[stderr]\n" + err_text)
            return AgentToolResult(
                content=[TextContent(text=merged.strip() or "(no output)")],
                details={"exit_code": proc.returncode},
            )
        except asyncio.TimeoutError:
            return AgentToolResult(content=[TextContent(text=f"Command timed out after {timeout_seconds}s")], details={"timeout": True})

    if _allow("ls") or _allow("list_dir"):
        tools.append(
            AgentTool(
                name="ls",
                label="List Directory",
                description="列出目录内容，返回文件名和大小。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的目录路径"},
                        "max_entries": {"type": "number", "description": "最多返回条目数，默认 100"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                execute=ls_tool,
            )
        )
        # 兼容旧名称
        tools.append(
            AgentTool(
                name="list_dir",
                label="List Directory (compat)",
                description="兼容别名：等价于 ls。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的目录路径"},
                        "max_entries": {"type": "number", "description": "最多返回条目数，默认 100"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                execute=ls_tool,
            )
        )

    if _allow("read") or _allow("read_file"):
        tools.append(
            AgentTool(
                name="read",
                label="Read File",
                description="读取文本文件内容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "max_chars": {"type": "number", "description": "最大返回字符数，默认 4000"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=read_tool,
            )
        )
        # 兼容旧名称
        tools.append(
            AgentTool(
                name="read_file",
                label="Read File (compat)",
                description="兼容别名：等价于 read。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "max_chars": {"type": "number", "description": "最大返回字符数，默认 4000"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=read_tool,
            )
        )

    if _allow("write") or _allow("write_file"):
        tools.append(
            AgentTool(
                name="write",
                label="Write File",
                description="写入文本文件。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "content": {"type": "string", "description": "写入内容"},
                        "overwrite": {"type": "boolean", "description": "是否覆盖已存在文件，默认 true"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                execute=write_tool,
            )
        )
        # 兼容旧名称
        tools.append(
            AgentTool(
                name="write_file",
                label="Write File (compat)",
                description="兼容别名：等价于 write。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "content": {"type": "string", "description": "写入内容"},
                        "overwrite": {"type": "boolean", "description": "是否覆盖已存在文件，默认 true"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                execute=write_tool,
            )
        )

    if _allow("edit"):
        tools.append(
            AgentTool(
                name="edit",
                label="Edit File",
                description="按 old_text -> new_text 替换文件内容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                        "old_text": {"type": "string", "description": "待替换原文"},
                        "new_text": {"type": "string", "description": "替换后的新文本"},
                        "replace_all": {"type": "boolean", "description": "是否替换全部匹配，默认 false"},
                        "occurrence_index": {"type": "number", "description": "替换第几次匹配（1-based）"},
                        "expected_occurrences": {"type": "number", "description": "期望 old_text 出现次数，不匹配则拒绝修改"},
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
                execute=edit_tool,
            )
        )

    if _allow("grep"):
        tools.append(
            AgentTool(
                name="grep",
                label="Search Content",
                description="在文件内容里按正则搜索。",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "正则表达式"},
                        "path": {"type": "string", "description": "起始目录，默认 ."},
                        "glob": {"type": "string", "description": "glob 过滤，默认 **/*"},
                        "max_matches": {"type": "number", "description": "最大匹配条数，默认 200"},
                        "case_sensitive": {"type": "boolean", "description": "是否大小写敏感，默认 true"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                execute=grep_tool,
            )
        )

    if _allow("find"):
        tools.append(
            AgentTool(
                name="find",
                label="Find Files",
                description="按 glob 查找文件/目录路径。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "起始目录，默认 ."},
                        "pattern": {"type": "string", "description": "glob 表达式，默认 **/*"},
                        "max_results": {"type": "number", "description": "最大返回条数，默认 200"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                execute=find_tool,
            )
        )

    if _allow("bash"):
        tools.append(
            AgentTool(
                name="bash",
                label="Run Command",
                description="执行 shell 命令。",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的命令"},
                        "cwd": {"type": "string", "description": "命令执行目录，默认 ."},
                        "timeout_seconds": {"type": "number", "description": "超时时间（秒），默认 30"},
                        "allow_dangerous": {"type": "boolean", "description": "是否允许高风险命令（默认 false）"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                execute=bash_tool,
            )
        )

    return tools

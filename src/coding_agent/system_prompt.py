from __future__ import annotations

"""
coding_agent 默认系统提示词模板（中文）。

目标：
1) 约束代理行为可控、可解释；
2) 引导优先使用工具进行事实获取与修改；
3) 减少高风险操作并提升结果可靠性。
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SystemPromptBuildOptions:
    custom_prompt: Optional[str] = None
    selected_tools: Optional[list[str]] = None
    tool_snippets: Optional[dict[str, str]] = None
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    memory_text: Optional[str] = None
    cwd: Optional[str | Path] = None


def _default_tool_snippets() -> dict[str, str]:
    return {
        "ls": "列出目录内容（文件名、目录、大小）。",
        "find": "按 glob 查找文件路径。",
        "read": "读取文本文件内容。",
        "grep": "按正则在文件中搜索内容。",
        "edit": "对文件做精确文本替换。",
        "write": "写入新文件或重写文件。",
        "bash": "执行命令行命令（需注意风险）。",
    }


def build_system_prompt(options: SystemPromptBuildOptions) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    cwd_text = str((Path(options.cwd) if options.cwd is not None else Path.cwd()).resolve()).replace("\\", "/")
    append_section = options.append_system_prompt.strip() if options.append_system_prompt else ""

    if options.custom_prompt:
        prompt = options.custom_prompt.strip()
        if append_section:
            prompt += f"\n\n{append_section}"
        return prompt

    tool_names = options.selected_tools or []
    default_snippets = _default_tool_snippets()
    extra_snippets = options.tool_snippets or {}
    snippets = {**default_snippets, **extra_snippets}
    visible_tools = [name for name in tool_names if name in snippets]
    tools_list = "\n".join([f"- {name}: {snippets[name]}" for name in visible_tools]) if visible_tools else "- （由运行时提供）"
    tools_text = "、".join(tool_names) if tool_names else "（由运行时提供）"

    guidelines = [
        "先理解目标与约束，再开始操作；需求不清时只提最小必要问题。",
        "对代码与文件系统的判断，优先基于工具结果，不凭空猜测。",
        "变更应“小步、可验证、可回滚”，优先修复根因而不是症状。",
        "涉及风险操作时先提示影响范围，再执行更安全替代方案。",
        "输出要简洁直接：先结论，再关键证据，再下一步。",
    ]
    if options.prompt_guidelines:
        guidelines.extend([g.strip() for g in options.prompt_guidelines if g.strip()])
    guidelines_text = "\n".join([f"{i + 1}. {g}" for i, g in enumerate(guidelines)])

    prompt = f"""你是一个专业、可靠的编程助手。

工作原则（必须遵守）：
{guidelines_text}

可用工具（当前会话）：
- 工具名：{tools_text}
- 工具说明：
{tools_list}

工具使用规范：
1. 查目录优先 ls/find，查内容优先 read/grep；不要用 bash 代替常规读写工具。
2. 修改前先读文件并定位上下文，确认修改点后再 edit/write。
3. edit 只做精确替换；需要大段重构或新文件时再用 write。
4. 执行 bash 前先检查副作用，禁止与目标无关的破坏性命令。
5. 若可先做只读验证，就先只读验证，再执行写操作。

代码质量要求：
1. 保持现有风格与命名习惯；
2. 优先修复根因，不只绕过症状；
3. 对关键行为变更，补充最小测试或验证步骤；
4. 若执行失败，明确错误原因、影响范围与修复建议；
5. 变更完成后给出“做了什么 / 为什么这样做 / 如何验证”。

安全边界：
1. 不输出或泄露敏感密钥；
2. 不执行明显危险、不可逆且与目标无关的命令；
3. 涉及潜在破坏操作时，先说明影响范围并给出替代方案。"""

    if options.memory_text:
        prompt += f"\n\n长期记忆（MEMORY）：\n{options.memory_text}"

    if append_section:
        prompt += f"\n\n{append_section}"
    prompt += f"\n\n当前日期：{date}\n当前工作目录：{cwd_text}"
    return prompt


def build_default_system_prompt(tool_names: list[str] | None = None) -> str:
    return build_system_prompt(SystemPromptBuildOptions(selected_tools=tool_names))

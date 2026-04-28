from __future__ import annotations

"""
MEMORY.md 记忆系统。

支持两级记忆：
1) 全局：workspace/.xingclaw/MEMORY.md
2) 频道级：workspace/.xingclaw/im/<channel_id>/MEMORY.md

加载后拼入 system prompt，让 Agent 具备长期记忆。
"""

import logging
from pathlib import Path

logger = logging.getLogger("xingclaw.im.memory")


def load_global_memory(workspace_dir: str | Path) -> str:
    """加载全局 MEMORY.md。"""
    path = Path(workspace_dir) / ".xingclaw" / "MEMORY.md"
    return _read_memory_file(path)


def load_channel_memory(workspace_dir: str | Path, channel_id: str) -> str:
    """加载频道级 MEMORY.md。"""
    path = Path(workspace_dir) / ".xingclaw" / "im" / channel_id / "MEMORY.md"
    return _read_memory_file(path)


def load_merged_memory(workspace_dir: str | Path, channel_id: str | None = None) -> str:
    """
    加载并合并全局 + 频道级记忆。

    返回格式化后的记忆文本，可直接注入 system prompt。
    """
    sections: list[str] = []
    global_mem = load_global_memory(workspace_dir)
    if global_mem:
        sections.append(f"## Global Memory\n{global_mem}")

    if channel_id:
        channel_mem = load_channel_memory(workspace_dir, channel_id)
        if channel_mem:
            sections.append(f"## Channel Memory ({channel_id})\n{channel_mem}")

    if not sections:
        return ""
    return "\n\n".join(sections)


def save_channel_memory(workspace_dir: str | Path, channel_id: str, content: str) -> None:
    """保存频道级 MEMORY.md。"""
    path = Path(workspace_dir) / ".xingclaw" / "im" / channel_id / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("channel memory saved channel_id=%s chars=%d", channel_id, len(content))


def save_global_memory(workspace_dir: str | Path, content: str) -> None:
    """保存全局 MEMORY.md。"""
    path = Path(workspace_dir) / ".xingclaw" / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("global memory saved chars=%d", len(content))


def _read_memory_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            logger.debug("loaded memory file=%s chars=%d", path, len(text))
        return text
    except Exception as exc:
        logger.warning("failed to read memory file=%s: %s", path, exc)
        return ""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .agent_session import AgentSession
from .extensions.types import RegisteredCommand

CommandSource = Literal["builtin", "extension", "skill", "prompt"]


@dataclass
class RuntimeCommand:
    name: str
    description: str
    source: CommandSource


def builtin_commands() -> list[RuntimeCommand]:
    return [
        RuntimeCommand(name="help", description="显示可用命令", source="builtin"),
        RuntimeCommand(name="session", description="查看当前会话与叶子节点", source="builtin"),
        RuntimeCommand(name="tree", description="查看当前会话树", source="builtin"),
        RuntimeCommand(name="fork", description="从指定节点分叉新会话", source="builtin"),
        RuntimeCommand(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        RuntimeCommand(name="switch", description="切换到指定叶子节点", source="builtin"),
        RuntimeCommand(name="clear", description="IM 中等价于 /new", source="builtin"),
    ]


def list_runtime_commands(session: AgentSession) -> list[RuntimeCommand]:
    items: dict[str, RuntimeCommand] = {cmd.name: cmd for cmd in builtin_commands()}
    for cmd in session.extension_commands.values():
        items[cmd.name] = RuntimeCommand(
            name=cmd.name,
            description=cmd.description or "扩展命令",
            source=cmd.source,
        )
    return sorted(items.values(), key=lambda x: (x.source, x.name))


def format_commands_for_help(session: AgentSession) -> str:
    lines: list[str] = ["可用命令："]
    for item in list_runtime_commands(session):
        lines.append(f"- `/{item.name}` [{item.source}] {item.description}")
    return "\n".join(lines)


def resolve_registered_command(session: AgentSession, name: str) -> RegisteredCommand | None:
    return session.extension_commands.get(name.strip().lstrip("/"))

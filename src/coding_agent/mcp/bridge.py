from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai.types import TextContent
from agent_core import AgentTool, AgentToolResult


class MCPClient(Protocol):
    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """
        调用 MCP 服务器工具并返回结果对象。
        """


@dataclass
class MCPToolConfig:
    name: str
    description: str
    parameters: dict[str, Any]
    server: str
    tool: str


def parse_mcp_tool_configs(raw_servers: list[dict[str, Any]] | None) -> list[MCPToolConfig]:
    if not raw_servers:
        return []
    result: list[MCPToolConfig] = []
    for server in raw_servers:
        if not isinstance(server, dict):
            continue
        server_name = server.get("name")
        tools = server.get("tools")
        if not isinstance(server_name, str) or not isinstance(tools, list):
            continue
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            tool = item.get("tool") or name
            description = item.get("description") or f"MCP tool proxy: {server_name}.{tool}"
            params = item.get("parameters")
            if not isinstance(name, str) or not isinstance(tool, str):
                continue
            if not isinstance(description, str):
                description = str(description)
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": [], "additionalProperties": True}
            result.append(
                MCPToolConfig(
                    name=name,
                    description=description,
                    parameters=params,
                    server=server_name,
                    tool=tool,
                )
            )
    return result


def create_mcp_proxy_tools(configs: list[MCPToolConfig], client: MCPClient | None) -> list[AgentTool]:
    tools: list[AgentTool] = []
    for cfg in configs:
        async def _execute(tool_call_id, params, signal=None, on_update=None, *, _cfg=cfg):  # type: ignore[no-untyped-def]
            _ = tool_call_id, signal, on_update
            args = params if isinstance(params, dict) else {}
            if client is None:
                return AgentToolResult(
                    content=[TextContent(text=f"MCP bridge unavailable for `{_cfg.name}`")],
                    is_error=True,
                )
            try:
                result = await client.call_tool(_cfg.server, _cfg.tool, args)
            except Exception as exc:  # pragma: no cover - adapter-specific
                return AgentToolResult(
                    content=[TextContent(text=f"MCP call failed `{_cfg.server}.{_cfg.tool}`: {exc}")],
                    is_error=True,
                )
            return AgentToolResult(
                content=[TextContent(text=_normalize_mcp_result(result))],
                details={"server": _cfg.server, "tool": _cfg.tool},
            )

        tools.append(
            AgentTool(
                name=cfg.name,
                label=f"MCP/{cfg.server}",
                description=cfg.description,
                parameters=cfg.parameters,
                execute=_execute,
            )
        )
    return tools


def _normalize_mcp_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return str(value)
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)

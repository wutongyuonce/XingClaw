"""
XingClaw coding_agent
=====================

应用层会话管理：
- AgentSession
- 会话持久化（context/events jsonl）
- 会话工厂 create_agent_session
"""

from .agent_session import AgentSession
from .builtin_tools import create_builtin_tools
from .cli import build_parser, main
from .command_registry import format_commands_for_help, list_runtime_commands
from .convert_to_llm import convert_to_llm
from .extensions import discover_extension_paths, discover_skill_paths, load_extensions, load_skills
from .factory import create_agent_session
from .mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from .runner import RunOptions, run, run_interactive, run_print, run_rpc
from .resources import WorkspaceResourceLoader, WorkspaceResources, WorkspaceSettings
from .system_prompt import SystemPromptBuildOptions, build_default_system_prompt, build_system_prompt
from .types import AgentSessionOptions, CreateAgentSessionOptions

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "CreateAgentSessionOptions",
    "convert_to_llm",
    "create_agent_session",
    "create_builtin_tools",
    "WorkspaceResourceLoader",
    "WorkspaceResources",
    "WorkspaceSettings",
    "build_default_system_prompt",
    "build_system_prompt",
    "SystemPromptBuildOptions",
    "discover_extension_paths",
    "discover_skill_paths",
    "load_extensions",
    "load_skills",
    "format_commands_for_help",
    "list_runtime_commands",
    "parse_mcp_tool_configs",
    "create_mcp_proxy_tools",
    "build_parser",
    "main",
    "RunOptions",
    "run",
    "run_print",
    "run_interactive",
    "run_rpc",
]

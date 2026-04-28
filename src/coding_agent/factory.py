from __future__ import annotations

"""
coding_agent 工厂入口。
"""

from pathlib import Path
import inspect
from typing import Any, Awaitable, Callable

from ai.models import get_model
from agent_core import AfterToolCallContext, AfterToolCallResult, BeforeToolCallContext, BeforeToolCallResult

from .agent_session import AgentSession
from .builtin_tools import READ_ONLY_TOOL_NAMES, create_builtin_tools
from .convert_to_llm import convert_to_llm
from .extensions import load_extensions, load_skills
from .mcp import create_mcp_proxy_tools, parse_mcp_tool_configs
from .resources import WorkspaceResourceLoader
from .session_store import SessionStore
from .system_prompt import SystemPromptBuildOptions, build_system_prompt
from .types import AgentSessionOptions, CreateAgentSessionOptions


def _canonical_tool_names(tools) -> list[str]:
    aliases = {"list_dir": "ls", "read_file": "read", "write_file": "write"}
    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        name = aliases.get(tool.name, tool.name)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _compose_before_tool_call(
    base: Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    | None,
    hooks: list[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ],
):
    chain: list[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: BeforeToolCallContext, signal: Any | None):
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if result and result.block:
                return result
        return None

    return _runner


def _compose_after_tool_call(
    base: Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    | None,
    hooks: list[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ],
):
    chain: list[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = []
    if base:
        chain.append(base)
    chain.extend(hooks)
    if not chain:
        return None

    async def _runner(ctx: AfterToolCallContext, signal: Any | None):
        final = AfterToolCallResult()
        for hook in chain:
            result = hook(ctx, signal)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if not result:
                continue
            if result.content is not None:
                ctx.result.content = result.content
                final.content = result.content
            if result.details is not None:
                ctx.result.details = result.details
                final.details = result.details
            if result.is_error is not None:
                ctx.is_error = result.is_error
                final.is_error = result.is_error
        if final.content is None and final.details is None and final.is_error is None:
            return None
        return final

    return _runner


def _compose_lifecycle_hooks(
    base_hooks: list[Callable[[Any], None | Awaitable[None]]] | None,
    loaded_hooks: list[Callable[[Any], None | Awaitable[None]]],
) -> list[Callable[[Any], None | Awaitable[None]]]:
    chain: list[Callable[[Any], None | Awaitable[None]]] = []
    chain.extend(base_hooks or [])
    chain.extend(loaded_hooks)
    return chain


def create_agent_session(options: AgentSessionOptions | CreateAgentSessionOptions) -> AgentSession:
    """
    创建 AgentSession，支持两种输入形式：
    - AgentSessionOptions: 低层参数，model 必填；
    - CreateAgentSessionOptions: 更友好，支持 provider/model_id 或 session 恢复。
    """

    if isinstance(options, AgentSessionOptions):
        return AgentSession(options)

    workspace = Path(options.workspace_dir)
    resources = WorkspaceResourceLoader(workspace).load() if options.load_workspace_resources else None
    restored_meta = None
    if options.session_id:
        restored_meta = SessionStore(workspace_dir=workspace, session_id=options.session_id).read_meta()

    model = options.model
    if model is None and options.provider and options.model_id:
        model = get_model(options.provider, options.model_id)
    if model is None and resources and resources.settings.provider and resources.settings.model_id:
        model = get_model(resources.settings.provider, resources.settings.model_id)
    if model is None and restored_meta:
        p = restored_meta.get("provider")
        mid = restored_meta.get("model_id")
        if isinstance(p, str) and isinstance(mid, str):
            model = get_model(p, mid)
    if model is None:
        raise ValueError("Unable to resolve model: provide model/provider+model_id or a valid existing session_id")

    system_prompt = options.system_prompt
    if not system_prompt and resources and resources.prompt:
        system_prompt = resources.prompt
    if not system_prompt and resources and resources.settings.system_prompt:
        system_prompt = resources.settings.system_prompt
    if not system_prompt and restored_meta and isinstance(restored_meta.get("system_prompt"), str):
        system_prompt = restored_meta["system_prompt"]

    thinking_level = options.thinking_level
    if thinking_level == "off" and resources and resources.settings.thinking_level:
        thinking_level = resources.settings.thinking_level

    tool_execution = options.tool_execution
    if tool_execution == "parallel" and resources and resources.settings.tool_execution:
        tool_execution = resources.settings.tool_execution

    max_context_messages = options.max_context_messages
    if max_context_messages is None and resources and resources.settings.max_context_messages is not None:
        max_context_messages = resources.settings.max_context_messages

    retain_recent_messages = options.retain_recent_messages
    if (
        retain_recent_messages == 24
        and resources
        and resources.settings.retain_recent_messages is not None
    ):
        retain_recent_messages = resources.settings.retain_recent_messages

    max_context_tokens = options.max_context_tokens
    if max_context_tokens is None and resources and resources.settings.max_context_tokens is not None:
        max_context_tokens = resources.settings.max_context_tokens

    retry_enabled = options.retry_enabled
    if resources and resources.settings.retry_enabled is not None:
        retry_enabled = resources.settings.retry_enabled

    max_retries = options.max_retries
    if resources and resources.settings.max_retries is not None and options.max_retries == 2:
        max_retries = resources.settings.max_retries

    retry_base_delay_ms = options.retry_base_delay_ms
    if resources and resources.settings.retry_base_delay_ms is not None and options.retry_base_delay_ms == 1200:
        retry_base_delay_ms = resources.settings.retry_base_delay_ms

    read_only_mode = options.read_only_mode
    if resources and resources.settings.read_only_mode is not None:
        read_only_mode = resources.settings.read_only_mode

    block_dangerous_bash = options.block_dangerous_bash
    if resources and resources.settings.block_dangerous_bash is not None:
        block_dangerous_bash = resources.settings.block_dangerous_bash

    bash_allow_patterns = options.bash_allow_patterns
    if bash_allow_patterns is None and resources and resources.settings.bash_allow_patterns is not None:
        bash_allow_patterns = resources.settings.bash_allow_patterns

    bash_block_patterns = options.bash_block_patterns
    if bash_block_patterns is None and resources and resources.settings.bash_block_patterns is not None:
        bash_block_patterns = resources.settings.bash_block_patterns

    edit_require_unique_match = options.edit_require_unique_match
    if resources and resources.settings.edit_require_unique_match is not None:
        edit_require_unique_match = resources.settings.edit_require_unique_match

    extension_paths = options.extension_paths
    if extension_paths is None and resources and resources.settings.extension_paths is not None:
        extension_paths = resources.settings.extension_paths
    loaded_extensions = load_extensions(workspace, configured_paths=extension_paths)

    skill_paths = options.skill_paths
    if skill_paths is None and resources and resources.settings.skill_paths is not None:
        skill_paths = resources.settings.skill_paths
    loaded_skills = load_skills(workspace, configured_paths=skill_paths)

    mcp_servers = options.mcp_servers
    if mcp_servers is None and resources and resources.settings.mcp_servers is not None:
        mcp_servers = resources.settings.mcp_servers
    mcp_tools = create_mcp_proxy_tools(parse_mcp_tool_configs(mcp_servers), client=options.mcp_client)

    builtin_enabled = options.enabled_builtin_tools
    if builtin_enabled is None and resources:
        builtin_enabled = resources.enabled_tools
    builtin_tools = create_builtin_tools(
        workspace,
        enabled_names=builtin_enabled,
        block_dangerous_bash=block_dangerous_bash,
        bash_allow_patterns=bash_allow_patterns,
        bash_block_patterns=bash_block_patterns,
        edit_require_unique_match=edit_require_unique_match,
    )

    # 同名时优先使用业务层自定义工具覆盖内置工具。
    tool_map = {tool.name: tool for tool in builtin_tools}
    for tool in options.tools:
        tool_map[tool.name] = tool
    for tool in loaded_extensions.tools:
        tool_map[tool.name] = tool
    for tool in mcp_tools:
        tool_map[tool.name] = tool
    merged_tools = list(tool_map.values())

    if read_only_mode:
        merged_tools = [tool for tool in merged_tools if tool.name in READ_ONLY_TOOL_NAMES]

    prompt_guidelines = options.prompt_guidelines
    if prompt_guidelines is None and resources and resources.settings.prompt_guidelines is not None:
        prompt_guidelines = resources.settings.prompt_guidelines
    prompt_guidelines = [
        *(prompt_guidelines or []),
        *loaded_extensions.prompt_guidelines,
        *loaded_skills.prompt_guidelines,
    ]
    prompt_guidelines.extend([f"[skill-diagnostic] {d}" for d in loaded_skills.diagnostics])

    append_system_prompt = options.append_system_prompt
    if append_system_prompt is None and resources and resources.settings.append_system_prompt is not None:
        append_system_prompt = resources.settings.append_system_prompt
    merged_append_sections = [*loaded_extensions.append_prompts, *loaded_skills.append_prompts]
    if merged_append_sections:
        ext_append = "\n\n".join(merged_append_sections)
        append_system_prompt = f"{append_system_prompt}\n\n{ext_append}".strip() if append_system_prompt else ext_append

    prompt_debug_sources = options.prompt_debug_sources
    if not prompt_debug_sources and resources and resources.settings.prompt_debug_sources:
        prompt_debug_sources = True
    if prompt_debug_sources:
        debug_lines: list[str] = ["## Prompt Sources", "### extensions"]
        debug_lines.extend([f"- {p}" for p in loaded_extensions.loaded_paths] or ["- (none)"])
        debug_lines.append("### skills")
        debug_lines.extend([f"- {p}" for p in loaded_skills.loaded_paths] or ["- (none)"])
        if loaded_extensions.errors or loaded_skills.errors:
            debug_lines.append("### errors")
            debug_lines.extend([f"- {e}" for e in [*loaded_extensions.errors, *loaded_skills.errors]])
        if loaded_skills.diagnostics:
            debug_lines.append("### diagnostics")
            debug_lines.extend([f"- {d}" for d in loaded_skills.diagnostics])
        append_system_prompt = (
            f"{append_system_prompt}\n\n" + "\n".join(debug_lines)
            if append_system_prompt
            else "\n".join(debug_lines)
        )

    tool_snippets = options.tool_snippets
    if tool_snippets is None and resources and resources.settings.tool_snippets is not None:
        tool_snippets = resources.settings.tool_snippets

    from im.memory import load_global_memory
    memory_text = load_global_memory(workspace)

    system_prompt = build_system_prompt(
        SystemPromptBuildOptions(
            custom_prompt=system_prompt or None,
            selected_tools=_canonical_tool_names(merged_tools),
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
            append_system_prompt=append_system_prompt,
            memory_text=memory_text,
            cwd=workspace,
        )
    )

    before_tool_call = _compose_before_tool_call(options.before_tool_call, loaded_extensions.before_tool_hooks)
    after_tool_call = _compose_after_tool_call(options.after_tool_call, loaded_extensions.after_tool_hooks)
    before_prompt_hooks = _compose_lifecycle_hooks(options.before_prompt_hooks, loaded_extensions.before_prompt_hooks)
    after_prompt_hooks = _compose_lifecycle_hooks(options.after_prompt_hooks, loaded_extensions.after_prompt_hooks)

    concrete = AgentSessionOptions(
        model=model,
        workspace_dir=workspace,
        system_prompt=system_prompt,
        tools=merged_tools,
        session_id=options.session_id,
        messages=options.messages,
        thinking_level=thinking_level,
        tool_execution=tool_execution,
        convert_to_llm=convert_to_llm,
        max_context_messages=max_context_messages,
        max_context_tokens=max_context_tokens,
        retain_recent_messages=retain_recent_messages,
        summary_builder=options.summary_builder,
        retry_enabled=retry_enabled,
        max_retries=max_retries,
        retry_base_delay_ms=retry_base_delay_ms,
        read_only_mode=read_only_mode,
        block_dangerous_bash=block_dangerous_bash,
        bash_allow_patterns=bash_allow_patterns,
        bash_block_patterns=bash_block_patterns,
        edit_require_unique_match=edit_require_unique_match,
        extension_paths=extension_paths,
        skill_paths=skill_paths,
        prompt_debug_sources=prompt_debug_sources,
        mcp_servers=mcp_servers,
        mcp_client=options.mcp_client,
        extension_commands={**options.extension_commands, **loaded_skills.commands, **loaded_extensions.commands},
        before_prompt_hooks=before_prompt_hooks,
        after_prompt_hooks=after_prompt_hooks,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )
    return AgentSession(concrete)

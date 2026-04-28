from __future__ import annotations

"""
运行模式入口。

当前支持：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
"""

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import inspect
import json
import sys
from typing import Any, Callable

from ai.types import AssistantMessage, TextContent
from agent_core import AgentEvent

from .agent_session import AgentSession
from .command_registry import format_commands_for_help, list_runtime_commands, resolve_registered_command
from .extensions.types import ExtensionCommandContext
from .types import InputFn, OutputFn, RunMode


@dataclass
class RunOptions:
    mode: RunMode
    session: AgentSession
    prompt: str | None = None
    output: OutputFn = print
    input_fn: InputFn = input
    show_tool_events: bool = True
    exit_commands: tuple[str, ...] = field(default_factory=lambda: ("exit", "quit", ":q"))


def _extract_assistant_text(message: AssistantMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()


async def run_print(
    session: AgentSession,
    prompt: str,
    *,
    output: OutputFn = print,
    show_tool_events: bool = True,
) -> AssistantMessage | None:
    """
    单次问答模式：
    - 监听流式 delta
    - 打印工具执行开始/结束
    - 返回最后一条 assistant 消息
    """

    deltas: list[str] = []

    def on_event(event: AgentEvent) -> None:
        t = event["type"]
        if show_tool_events and t in {"tool_execution_start", "tool_execution_end"}:
            output(f"[tool-event] {t}: {event.get('toolName', '')}")
            return

        if t == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = str(assistant_event.get("delta", ""))
                deltas.append(delta)

    unsubscribe = session.subscribe(on_event)
    try:
        await session.prompt(prompt)
    finally:
        unsubscribe()

    final_assistant = next((m for m in reversed(session.messages) if isinstance(m, AssistantMessage)), None)

    if deltas:
        output("".join(deltas).strip())
    elif final_assistant is not None:
        output(_extract_assistant_text(final_assistant) or "(empty)")

    if final_assistant is not None:
        output(f"[assistant.stop_reason] {final_assistant.stop_reason}")
        output(f"[assistant.error_message] {final_assistant.error_message}")
    return final_assistant


async def run_interactive(
    session: AgentSession,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    show_tool_events: bool = True,
    exit_commands: tuple[str, ...] = ("exit", "quit", ":q"),
) -> None:
    """
    交互模式：
    持续读取输入并执行 prompt，直到命中退出命令。
    """

    output("Entering interactive mode. Type 'exit' or '/exit' to quit.")
    output(format_commands_for_help(session))
    current_session = session
    while True:
        text = input_fn("you> ").strip()
        bare = text.lstrip("/")
        if bare in exit_commands:
            output("Bye.")
            return
        if not text:
            continue
        if text.startswith("/"):
            handled, switched = await _handle_interactive_command(current_session, text, output=output)
            if switched is not None:
                current_session.close()
                current_session = switched
            if handled:
                continue
        await run_print(current_session, text, output=output, show_tool_events=show_tool_events)


def _create_fresh_session(old: AgentSession) -> AgentSession:
    """创建全新空白 session，保留模型/工具/设置但不带历史消息。"""
    from .session_store import new_session_id
    from .types import AgentSessionOptions

    return AgentSession(
        AgentSessionOptions(
            model=old.agent.state.model,
            workspace_dir=old.workspace_dir,
            system_prompt=old.agent.state.system_prompt,
            tools=list(old.agent.state.tools),
            session_id=new_session_id(),
            messages=[],
            thinking_level=old.agent.state.thinking_level,
            tool_execution=old.tool_execution,
            max_context_messages=old.max_context_messages,
            max_context_tokens=old.max_context_tokens,
            retain_recent_messages=old.retain_recent_messages,
            summary_builder=old.summary_builder,
            retry_enabled=old.retry_enabled,
            max_retries=old.max_retries,
            retry_base_delay_ms=old.retry_base_delay_ms,
            mcp_servers=old.mcp_servers,
            mcp_client=old.mcp_client,
            extension_commands=old.extension_commands,
            before_prompt_hooks=old.before_prompt_hooks,
            after_prompt_hooks=old.after_prompt_hooks,
            before_tool_call=old.before_tool_call,
            after_tool_call=old.after_tool_call,
        )
    )


async def _handle_interactive_command(
    session: AgentSession, text: str, *, output: OutputFn = print
) -> tuple[bool, AgentSession | None]:
    cmd, _, rest = text.partition(" ")
    arg = rest.strip()
    if cmd == "/help":
        output(format_commands_for_help(session))
        return True, None
    if cmd == "/session":
        output(f"session_id={session.session_id} leaf_id={session.get_leaf_id()}")
        return True, None
    if cmd == "/tree":
        entries = session.list_entries()
        if not entries:
            output("(empty)")
            return True, None
        for item in entries:
            depth = int(item.get("depth", 0))
            prefix = "  " * max(depth, 0)
            leaf_mark = " *" if item.get("is_leaf") else ""
            output(f"{prefix}- {item.get('id')}{leaf_mark}")
        return True, None
    if cmd == "/clear":
        fresh = _create_fresh_session(session)
        output(f"context cleared → new session_id={fresh.session_id}")
        return True, fresh
    if cmd in {"/new", "/fork"}:
        from_entry = arg or session.get_leaf_id() or ""
        if not from_entry:
            output("cannot resolve source entry")
            return True, None
        forked = session.fork_from_entry(from_entry)
        output(f"forked to session_id={forked.session_id}")
        return True, forked
    if cmd == "/switch":
        if not arg:
            output("usage: /switch <entry_id>")
            return True, None
        session.switch_to_entry(arg)
        output(f"switched leaf -> {session.get_leaf_id()}")
        return True, None
    reg = resolve_registered_command(session, cmd)
    if reg:
        value = reg.handler(
            ExtensionCommandContext(
                name=reg.name,
                args=[p for p in arg.split(" ") if p],
                raw_text=text,
                session=session,
                message=None,
            )
        )
        if inspect.isawaitable(value):
            value = await value
        if value:
            output(str(value))
        return True, None
    return False, None


async def run(options: RunOptions) -> AssistantMessage | None:
    """
    统一运行入口。
    """

    if options.mode == "print":
        if not options.prompt:
            raise ValueError("print mode requires prompt")
        return await run_print(
            options.session,
            options.prompt,
            output=options.output,
            show_tool_events=options.show_tool_events,
        )

    if options.mode == "rpc":
        await run_rpc(options.session, output=options.output)
        return None

    await run_interactive(
        options.session,
        input_fn=options.input_fn,
        output=options.output,
        show_tool_events=options.show_tool_events,
        exit_commands=options.exit_commands,
    )
    return None


async def run_rpc(
    session: AgentSession,
    *,
    output: OutputFn = print,
) -> None:
    """
    极简 RPC 模式（jsonl）：
    - {"type":"prompt","text":"..."}
    - {"type":"continue"}
    - {"type":"state"}
    - {"type":"shutdown"}
    """

    def _json_default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, set):
            return list(value)
        return str(value)

    def _emit(obj: dict[str, Any]) -> None:
        output(json.dumps(obj, ensure_ascii=False, default=_json_default))

    def _emit_error(*, req_id: Any, command: Any, code: str, message: str) -> None:
        _emit(
            {
                "type": "response",
                "id": req_id,
                "command": command,
                "status": "error",
                "error": {"code": code, "message": message},
            }
        )

    def _emit_ok(*, req_id: Any, command: str, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "ok",
        }
        if data is not None:
            payload["data"] = data
        _emit(payload)

    unsubscribe = session.subscribe(
        lambda event: _emit(
            {
                "type": "event",
                "event": event,
            }
        )
    )

    _emit({"type": "rpc_ready", "session_id": session.session_id, "protocol_version": "1.2"})
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as exc:
                _emit_error(req_id=None, command=None, code="invalid_json", message=f"Invalid JSON: {exc}")
                continue
            if not isinstance(req, dict):
                _emit_error(req_id=None, command=None, code="invalid_request", message="Request must be object")
                continue

            cmd = req.get("type")
            req_id = req.get("id")

            try:
                if cmd == "prompt":
                    text = str(req.get("text", ""))
                    await session.prompt(text)
                    _emit_ok(req_id=req_id, command="prompt")
                elif cmd == "continue":
                    await session.continue_run()
                    _emit_ok(req_id=req_id, command="continue")
                elif cmd == "state":
                    _emit_ok(
                        req_id=req_id,
                        command="state",
                        data={
                            "session_id": session.session_id,
                            "message_count": len(session.messages),
                            "entry_ids": session.list_entry_ids(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )
                elif cmd == "list_entries":
                    _emit_ok(
                        req_id=req_id,
                        command="list_entries",
                        data={
                            "session_id": session.session_id,
                            "entry_ids": session.list_entry_ids(),
                            "entries": session.list_entries(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )
                elif cmd == "show_tree":
                    _emit_ok(
                        req_id=req_id,
                        command="show_tree",
                        data={
                            "session_id": session.session_id,
                            "tree": session.get_session_tree(),
                            "leaf_id": session.get_leaf_id(),
                        },
                    )
                elif cmd == "entry_path":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("entry_path requires entry_id")
                    _emit_ok(
                        req_id=req_id,
                        command="entry_path",
                        data={"session_id": session.session_id, "entry_id": entry_id, "path": session.get_entry_path(entry_id)},
                    )
                elif cmd == "fork_entry":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("fork_entry requires entry_id")
                    forked = session.fork_from_entry(entry_id)
                    try:
                        _emit_ok(
                            req_id=req_id,
                            command="fork_entry",
                            data={
                                "from_session_id": session.session_id,
                                "from_entry_id": entry_id,
                                "new_session_id": forked.session_id,
                            },
                        )
                    finally:
                        forked.close()
                elif cmd == "switch_entry":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("switch_entry requires entry_id")
                    session.switch_to_entry(entry_id)
                    _emit_ok(
                        req_id=req_id,
                        command="switch_entry",
                        data={
                            "session_id": session.session_id,
                            "entry_id": entry_id,
                            "path": session.get_entry_path(entry_id),
                        },
                    )
                elif cmd == "get_commands":
                    _emit_ok(
                        req_id=req_id,
                        command="get_commands",
                        data={
                            "session_id": session.session_id,
                            "commands": [
                                {"name": c.name, "description": c.description, "source": c.source}
                                for c in list_runtime_commands(session)
                            ],
                        },
                    )
                elif cmd == "shutdown":
                    _emit_ok(req_id=req_id, command="shutdown")
                    return
                else:
                    _emit_error(req_id=req_id, command=cmd, code="unknown_command", message="Unknown command")
            except Exception as exc:
                _emit_error(req_id=req_id, command=cmd, code="execution_error", message=str(exc))
    finally:
        unsubscribe()

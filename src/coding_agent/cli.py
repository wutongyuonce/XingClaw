from __future__ import annotations

"""
coding_agent 命令行入口。

示例：
python -m coding_agent --mode print --prompt "你好"
python -m coding_agent --mode interactive --provider anthropic --model-id glm-4.7
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from .factory import create_agent_session
from .runner import RunOptions, run
from .types import CreateAgentSessionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XingClaw coding-agent CLI")
    parser.add_argument("--mode", choices=["print", "interactive", "rpc"], default="interactive")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--session-id", default=None, help="Existing session id to resume")
    parser.add_argument("--list-entries", action="store_true", help="Print session entry ids and exit")
    parser.add_argument("--show-tree", action="store_true", help="Print session tree as JSON and exit")
    parser.add_argument("--fork-entry", default=None, help="Fork from entry id and print new session id")
    parser.add_argument("--switch-entry", default=None, help="Switch current session leaf to entry id")
    parser.add_argument("--provider", default=None, help="Model provider, e.g. anthropic/openai-standard")
    parser.add_argument("--model-id", default=None, help="Model id")
    parser.add_argument("--system-prompt", default="", help="System prompt")
    parser.add_argument("--thinking-level", default="off", help="Thinking level: off/minimal/low/medium/high/xhigh")
    parser.add_argument("--tool-execution", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--max-context-messages", type=int, default=None, help="Compaction message threshold")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="Compaction token threshold (approx)")
    parser.add_argument("--retain-recent-messages", type=int, default=24, help="Keep recent messages when compacting")
    parser.add_argument("--no-retry", action="store_true", help="Disable automatic retry on transient errors")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retry count")
    parser.add_argument("--retry-base-delay-ms", type=int, default=1200, help="Retry base delay in milliseconds")
    parser.add_argument("--read-only", action="store_true", help="Enable read-only mode (disable write/edit/bash)")
    parser.add_argument("--allow-dangerous-bash", action="store_true", help="Disable dangerous bash blocking")
    parser.add_argument(
        "--bash-allow-pattern",
        action="append",
        default=None,
        help="Regex pattern to allow bash command (can be repeated)",
    )
    parser.add_argument(
        "--bash-block-pattern",
        action="append",
        default=None,
        help="Regex pattern to block bash command (can be repeated)",
    )
    parser.add_argument(
        "--relaxed-edit",
        action="store_true",
        help="Disable strict unique-match requirement for edit tool",
    )
    parser.add_argument("--prompt", default=None, help="Prompt text (required in print mode)")
    parser.add_argument("--no-tool-events", action="store_true", help="Hide tool events in output")
    parser.add_argument(
        "--disable-workspace-resources",
        action="store_true",
        help="Disable reading .xingclaw/{settings,prompt,tools}",
    )
    return parser


async def _run_from_args(args: argparse.Namespace) -> int:
    options = CreateAgentSessionOptions(
        workspace_dir=Path(args.workspace),
        provider=args.provider,
        model_id=args.model_id,
        system_prompt=args.system_prompt,
        session_id=args.session_id,
        thinking_level=args.thinking_level,
        tool_execution=args.tool_execution,
        max_context_messages=args.max_context_messages,
        max_context_tokens=args.max_context_tokens,
        retain_recent_messages=args.retain_recent_messages,
        retry_enabled=not bool(args.no_retry),
        max_retries=args.max_retries,
        retry_base_delay_ms=args.retry_base_delay_ms,
        read_only_mode=bool(args.read_only),
        block_dangerous_bash=not bool(args.allow_dangerous_bash),
        bash_allow_patterns=args.bash_allow_pattern,
        bash_block_patterns=args.bash_block_pattern,
        edit_require_unique_match=not bool(args.relaxed_edit),
        load_workspace_resources=not bool(args.disable_workspace_resources),
    )
    session = create_agent_session(options)
    try:
        if args.switch_entry:
            session.switch_to_entry(str(args.switch_entry))
            print(json.dumps({"type": "switch_entry", "session_id": session.session_id, "entry_id": args.switch_entry}))
            return 0

        if args.fork_entry:
            forked = session.fork_from_entry(str(args.fork_entry))
            try:
                print(
                    json.dumps(
                        {
                            "type": "forked",
                            "from_session_id": session.session_id,
                            "from_entry_id": args.fork_entry,
                            "new_session_id": forked.session_id,
                        },
                        ensure_ascii=False,
                    )
                )
            finally:
                forked.close()
            return 0

        if args.list_entries:
            print(json.dumps({"session_id": session.session_id, "entry_ids": session.list_entry_ids()}, ensure_ascii=False))
            return 0

        if args.show_tree:
            print(json.dumps({"session_id": session.session_id, "tree": session.get_session_tree()}, ensure_ascii=False))
            return 0

        await run(
            RunOptions(
                mode=args.mode,
                session=session,
                prompt=args.prompt,
                show_tool_events=not bool(args.no_tool_events),
            )
        )
    finally:
        session.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(_run_from_args(args))
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

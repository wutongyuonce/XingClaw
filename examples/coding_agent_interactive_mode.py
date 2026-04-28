"""
interactive 模式示例：启动一个简易对话 REPL。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from coding_agent import CreateAgentSessionOptions, RunOptions, create_agent_session, run


async def main() -> None:
    session = create_agent_session(
        CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            provider="anthropic",
            model_id="glm-4.7",
            system_prompt="你是一个简洁可靠的助手。",
            thinking_level="minimal",
        )
    )
    try:
        await run(
            RunOptions(
                mode="interactive",
                session=session,
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())

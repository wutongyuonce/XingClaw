"""
coding_agent 最小示例：
1) 创建 AgentSession
2) 发起一次对话
3) 打印会话目录与历史条数
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai import AssistantMessage, TextContent
from coding_agent import CreateAgentSessionOptions, create_agent_session


async def main() -> None:
    session = create_agent_session(
        CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            provider="anthropic",
            model_id="glm-4.7",
            system_prompt="你是一个简洁可靠的编程助手。",
            thinking_level="minimal",
        )
    )

    await session.prompt("请用一句话介绍你自己。")

    final_assistant = next((m for m in reversed(session.messages) if isinstance(m, AssistantMessage)), None)
    if final_assistant is not None:
        text = "".join(b.text for b in final_assistant.content if isinstance(b, TextContent)).strip()
        print("[assistant.stop_reason]", final_assistant.stop_reason)
        print("[assistant.error_message]", final_assistant.error_message)
        print("[assistant.text]", text if text else "(empty)")

    print("[session.id]", session.session_id)
    print("[session.dir]", Path.cwd() / ".xingclaw" / "sessions" / session.session_id)
    print("[session.message_count]", len(session.messages))

    session.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
会话恢复示例：
1) 第一次创建会话并提问；
2) 用同一个 session_id 重建会话对象；
3) 继续提问并验证历史仍在。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from coding_agent import CreateAgentSessionOptions, create_agent_session


async def main() -> None:
    first = create_agent_session(
        CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            provider="anthropic",
            model_id="glm-4.7",
            system_prompt="你是一个会话型助手。",
            thinking_level="minimal",
        )
    )
    await first.prompt("请记住：我的最喜欢语言是 Python。")
    sid = first.session_id
    print("[first.session_id]", sid)
    print("[first.message_count]", len(first.messages))
    first.close()

    second = create_agent_session(
        CreateAgentSessionOptions(
            workspace_dir=Path.cwd(),
            session_id=sid,  # 不再重复传 model，工厂会从 meta 中恢复
            thinking_level="minimal",
        )
    )
    print("[second.message_count.before]", len(second.messages))
    await second.prompt("我最喜欢的语言是什么？")
    print("[second.message_count.after]", len(second.messages))
    second.close()


if __name__ == "__main__":
    asyncio.run(main())

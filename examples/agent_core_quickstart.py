"""
agent_core 最小示例：
1) 创建 Agent
2) 注册一个简单工具
3) 发起一次对话并打印事件
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ai import AssistantMessage, TextContent, get_model
from agent_core import Agent, AgentOptions, AgentTool, AgentToolResult


async def get_time_tool(tool_call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
    _ = tool_call_id, signal
    timezone = params.get("timezone", "local")
    if on_update:
        on_update(AgentToolResult(content=[TextContent(text="正在查询时间...")], details={"stage": "start"}))
    now = datetime.now().isoformat(timespec="seconds")
    return AgentToolResult(content=[TextContent(text=f"当前时间({timezone}): {now}")], details={"timezone": timezone})


async def main() -> None:
    model = get_model("anthropic", "glm-4.7")
    tool = AgentTool(
        name="get_time",
        label="Get Time",
        description="获取当前时间",
        parameters={
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "时区描述，如 Asia/Shanghai"}},
            "required": [],
            "additionalProperties": False,
        },
        execute=get_time_tool,
    )

    agent = Agent(
        AgentOptions(
            model=model,
            system_prompt="你是一个简洁的助手。需要时间时请调用 get_time 工具。",
            tools=[tool],
            thinking_level="minimal",
        )
    )

    def on_event(event: dict) -> None:
        event_type = event.get("type")
        if event_type in {"tool_execution_start", "tool_execution_end"}:
            print(f"[tool-event] {event_type}: {event.get('toolName')}")
            return

        if event_type == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                print(assistant_event.get("delta", ""), end="", flush=True)
            return

        if event_type == "message_end":
            message = event.get("message")
            if getattr(message, "role", "") == "assistant":
                print()

    agent.subscribe(on_event)

    await agent.prompt("请告诉我现在时间，并说明你使用了哪个工具。")

    # 无论是否出现 text_delta，都打印最终 assistant 结果，方便排查问题。
    final_assistant = next(
        (m for m in reversed(agent.state.messages) if isinstance(m, AssistantMessage)),
        None,
    )
    if final_assistant is not None:
        text_blocks = [b.text for b in final_assistant.content if isinstance(b, TextContent)]
        final_text = "".join(text_blocks).strip()
        print(f"[assistant.stop_reason] {final_assistant.stop_reason}")
        print(f"[assistant.error_message] {final_assistant.error_message}")
        print(f"[assistant.text] {final_text if final_text else '(empty)'}")

    print("---- 对话结束 ----")
    for m in agent.state.messages:
        role = getattr(m, "role", "unknown")
        print(f"- {role}")


if __name__ == "__main__":
    asyncio.run(main())

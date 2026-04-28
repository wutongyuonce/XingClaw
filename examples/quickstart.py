import asyncio
import time

# 示例脚本：演示最小对话流程
# 1) 选模型
# 2) 准备上下文
# 3) 逐步消费 text_delta
# 4) 拿最终消息
from ai import Context, TextContent, Tool, UserMessage, get_model, stream_simple


async def main() -> None:
    # 选择 anthropic provider 的默认示例模型
    model = get_model("anthropic", "glm-4.7")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                content=[TextContent(text="Say hello from XingClaw in one sentence.")],
                timestamp=int(time.time() * 1000),
            )
        ],
        tools=[
            Tool(
                name="get_time",
                description="Get current UTC time",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            )
        ],
    )

    # 简化接口：直接传 reasoning 等级
    s = stream_simple(model, context, reasoning="low")
    async for event in s:
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)

    msg = await s.result()
    print(f"\nstop_reason={msg.stop_reason}")
    if msg.stop_reason in {"error", "aborted"}:
        print(f"error_message={msg.error_message}")


if __name__ == "__main__":
    asyncio.run(main())

"""
XingClaw agent_core
===================

最小可运行的 Agent 编排内核，包含：
- Agent 类
- 运行循环
- 工具执行协议
- 事件回调机制
"""

from .agent import Agent, AgentOptions
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentEventBase,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    ErrorEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)

__all__ = [
    "Agent",
    "AgentOptions",
    "run_agent_loop",
    "run_agent_loop_continue",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEvent",
    "AgentEventBase",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "ErrorEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ToolExecutionMode",
]

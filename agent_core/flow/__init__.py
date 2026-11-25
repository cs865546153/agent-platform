"""通用业务控制层：计划、装配、执行、观察、决策与确认。

与 ``agent_core.harness`` 的分工：

- ``flow`` 决定业务流程与状态：下一步做什么、为什么、何时停；
- ``harness`` 决定执行策略：步数、上下文、工具、预算与流式钩子。

领域只实现 ``FlowNodes``；通用循环、版本事实、确认校验、幂等与轨迹复用本模块。
"""
from .confirm import ConfirmExecutor, payload_hash, validate_pending_action
from .context import ContextAssembler, ContextPack
from .contracts import (
    ConfirmCheck,
    ConfirmCode,
    Decision,
    DecisionKind,
    Execution,
    FlowStatus,
    Observation,
    ObservationKind,
    PendingAction,
    PendingStatus,
    Plan,
    StopReason,
)
from .engine import FlowEngine, FlowNodes
from .guards import IdempotencyStore, InMemoryIdempotencyStore, LoopGuard
from .langgraph import build_flow_graph
from .state import FlowState, VersionedFacts, ensure_state, new_state
from .trace import FlowTrace, InMemoryTraceSink, TraceEvent, TraceSink

__all__ = [
    "ConfirmCheck",
    "ConfirmCode",
    "ConfirmExecutor",
    "ContextAssembler",
    "ContextPack",
    "Decision",
    "DecisionKind",
    "Execution",
    "FlowEngine",
    "FlowNodes",
    "FlowState",
    "FlowStatus",
    "FlowTrace",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "InMemoryTraceSink",
    "LoopGuard",
    "Observation",
    "ObservationKind",
    "PendingAction",
    "PendingStatus",
    "Plan",
    "StopReason",
    "TraceEvent",
    "TraceSink",
    "VersionedFacts",
    "build_flow_graph",
    "ensure_state",
    "new_state",
    "payload_hash",
    "validate_pending_action",
]

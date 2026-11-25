"""控制层轨迹：一次运行为一条 Trace，事件只记录决策依据与结构化结果。

轨迹不替代应用日志，也不记录原始 PII；领域在写入前自行脱敏。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Protocol
import uuid


@dataclass(frozen=True)
class TraceEvent:
    event: str
    trace_id: str
    step: int = 0
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            'event': self.event,
            'trace_id': self.trace_id,
            'step': self.step,
            'reason': self.reason,
            'data': self.data,
            'timestamp': self.timestamp,
        }


class TraceSink(Protocol):
    """轨迹出口；可以是 JSONL、内存、LangFuse 或审计库。"""

    def emit(self, event: TraceEvent) -> None:
        ...


class InMemoryTraceSink:
    """测试与本地调试用轨迹收集器。"""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


class FlowTrace:
    """一次业务运行的轨迹；``record`` 同时写入本地列表与可选出口。"""

    def __init__(self, trace_id: str | None = None, sink: TraceSink | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.sink = sink
        self.events: list[TraceEvent] = []

    def record(self, event: str, *, step: int = 0, reason: str | None = None,
               **data: Any) -> TraceEvent:
        item = TraceEvent(
            event=event,
            trace_id=self.trace_id,
            step=step,
            reason=reason,
            data=data,
        )
        self.events.append(item)
        if self.sink is not None:
            self.sink.emit(item)
        return item

    def as_dict(self) -> dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'events': [event.as_dict() for event in self.events],
        }


__all__ = ["FlowTrace", "InMemoryTraceSink", "TraceEvent", "TraceSink"]

"""业务控制层契约：计划、执行、观察、决策、确认与终止原因。

本模块只描述领域无关的控制面数据形态，不懂任何具体业务。领域图 / 循环按这里的
协议产出结构化结果，执行面策略（``agent_core.harness``）仍只负责资源与裁剪。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class FlowStatus(str, Enum):
    """业务控制层的生命周期状态。"""

    PLANNING = 'planning'
    ASSEMBLING = 'assembling'
    EXECUTING = 'executing'
    OBSERVING = 'observing'
    WAITING_USER = 'waiting_user'
    CONFIRMING = 'confirming'
    ESCALATED = 'escalated'
    DONE = 'done'
    FAILED = 'failed'


class ObservationKind(str, Enum):
    """观察结果的确定性分类；领域只解释业务含义，不改变分类集合。"""

    SUCCESS = 'success'
    RETRYABLE_ERROR = 'retryable_error'
    TERMINAL_ERROR = 'terminal_error'
    STALE = 'stale'
    CONFLICT = 'conflict'
    OUT_OF_SCOPE = 'out_of_scope'
    MISSING_INFO = 'missing_info'
    NO_PROGRESS = 'no_progress'


class DecisionKind(str, Enum):
    """观察之后允许的下一跳。"""

    CONTINUE = 'continue'
    WAIT_USER = 'wait_user'
    CONFIRM = 'confirm'
    ESCALATE = 'escalate'
    DONE = 'done'
    FAILED = 'failed'


_NO_PROGRESS_KINDS = frozenset({
    ObservationKind.RETRYABLE_ERROR,
    ObservationKind.TERMINAL_ERROR,
    ObservationKind.STALE,
    ObservationKind.CONFLICT,
    ObservationKind.OUT_OF_SCOPE,
    ObservationKind.MISSING_INFO,
    ObservationKind.NO_PROGRESS,
})


class StopReason(str, Enum):
    """执行面策略触发的终止原因；控制层据此决定转人工或失败。"""

    MAX_STEPS = 'max_steps'
    MAX_TOOL_CALLS = 'max_tool_calls'
    DEADLINE = 'deadline'
    NO_PROGRESS = 'no_progress'


class PendingStatus(str, Enum):
    """待确认动作状态。"""

    PENDING = 'pending'
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    EXECUTED = 'executed'


class ConfirmCode(str, Enum):
    """确认前确定性校验的结果码。"""

    OK = 'ok'
    NOT_PENDING = 'not_pending'
    VERSION_MISMATCH = 'version_mismatch'
    EXPIRED = 'expired'
    PAYLOAD_MISMATCH = 'payload_mismatch'
    IDEMPOTENT_REPLAY = 'idempotent_replay'


@dataclass(frozen=True)
class Plan:
    """一次计划的结构化结果；``missing`` 非空时控制层等待用户补充。"""

    actions: tuple[dict[str, Any], ...] = ()
    missing: tuple[str, ...] = ()
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {
            'actions': list(self.actions),
            'missing': list(self.missing),
            'reason': self.reason,
            'data': self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Plan':
        return cls(
            actions=tuple(dict(item) for item in (data.get('actions') or ())),
            missing=tuple(str(item) for item in (data.get('missing') or ())),
            reason=data.get('reason'),
            data=dict(data.get('data') or {}),
        )


@dataclass(frozen=True)
class Execution:
    """一次执行的结构化结果；``tool_calls`` 供执行面策略累计。"""

    outputs: tuple[Any, ...] = ()
    errors: tuple[str, ...] = ()
    tool_calls: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            'outputs': list(self.outputs),
            'errors': list(self.errors),
            'tool_calls': self.tool_calls,
            'data': self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Execution':
        return cls(
            outputs=tuple(data.get('outputs') or ()),
            errors=tuple(str(item) for item in (data.get('errors') or ())),
            tool_calls=int(data.get('tool_calls') or 0),
            data=dict(data.get('data') or {}),
        )


@dataclass(frozen=True)
class Observation:
    """对执行结果的规范化观察；控制层据此决定继续、等待、确认或转人工。"""

    kind: ObservationKind
    reason: str = ''
    changed_fields: tuple[str, ...] = ()
    invalidated: tuple[str, ...] = ()
    next_hint: str | None = None
    progressed: bool | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def made_progress(self) -> bool:
        """显式值优先；未指定时按观察类型保守判断。"""
        if self.progressed is not None:
            return self.progressed
        return self.kind not in _NO_PROGRESS_KINDS

    def as_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind.value,
            'reason': self.reason,
            'changed_fields': list(self.changed_fields),
            'invalidated': list(self.invalidated),
            'next_hint': self.next_hint,
            'progressed': self.made_progress,
            'data': self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Observation':
        progressed = data.get('progressed')
        return cls(
            kind=ObservationKind(data.get('kind', ObservationKind.NO_PROGRESS.value)),
            reason=str(data.get('reason') or ''),
            changed_fields=tuple(data.get('changed_fields') or ()),
            invalidated=tuple(data.get('invalidated') or ()),
            next_hint=data.get('next_hint'),
            progressed=progressed if isinstance(progressed, bool) else None,
            data=dict(data.get('data') or {}),
        )


@dataclass(frozen=True)
class PendingAction:
    """等待用户确认的写操作；确认前必须按版本、TTL、载荷与幂等键校验。

    该对象是可序列化的纯数据，不持有工具、模型或数据库连接。业务控制层负责
    生成与校验；执行面策略层不参与确认语义。
    """

    action_type: str
    target_id: str
    target_version: int
    payload_hash: str
    expires_at: float
    idempotency_key: str
    created_at: float = field(default_factory=time.time)
    status: PendingStatus = PendingStatus.PENDING

    def is_expired(self, *, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def as_dict(self) -> dict[str, Any]:
        return {
            'action_type': self.action_type,
            'target_id': self.target_id,
            'target_version': self.target_version,
            'payload_hash': self.payload_hash,
            'expires_at': self.expires_at,
            'idempotency_key': self.idempotency_key,
            'created_at': self.created_at,
            'status': self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PendingAction':
        return cls(
            action_type=str(data['action_type']),
            target_id=str(data['target_id']),
            target_version=int(data['target_version']),
            payload_hash=str(data['payload_hash']),
            expires_at=float(data['expires_at']),
            idempotency_key=str(data['idempotency_key']),
            created_at=float(data.get('created_at') or time.time()),
            status=PendingStatus(data.get('status', PendingStatus.PENDING.value)),
        )


@dataclass(frozen=True)
class Decision:
    """控制层决策；``CONFIRM`` / ``ESCALATE`` 必须带原因或待确认动作。"""

    kind: DecisionKind
    reason: str = ''
    action: PendingAction | None = None
    missing: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind.value,
            'reason': self.reason,
            'action': self.action.as_dict() if self.action is not None else None,
            'missing': list(self.missing),
            'data': self.data,
        }


@dataclass(frozen=True)
class ConfirmCheck:
    """确认前校验结果；``ok=False`` 时不得执行写操作。"""

    ok: bool
    code: ConfirmCode = ConfirmCode.OK
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {'ok': self.ok, 'code': self.code.value, 'reason': self.reason}


__all__ = [
    "ConfirmCheck",
    "ConfirmCode",
    "Decision",
    "DecisionKind",
    "Execution",
    "FlowStatus",
    "Observation",
    "ObservationKind",
    "PendingAction",
    "PendingStatus",
    "Plan",
    "StopReason",
]

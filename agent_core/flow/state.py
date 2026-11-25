"""业务控制层状态：事实版本、待决事项与通用 FlowState。

状态只保存可序列化的业务事实与决策结果；聊天消息、模型对象、连接不进入这里。
领域可以扩展自己的 TypedDict，但应复用这里的版本与待确认字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, TypedDict
import uuid


class FlowState(TypedDict, total=False):
    """通用控制层状态；领域可在此基础上增加自己的字段。"""

    thread_id: str
    trace_id: str
    status: str
    step_count: int
    tool_calls: int
    no_progress_count: int
    started_at: float
    facts: dict[str, Any]
    versions: dict[str, int]
    invalidated: dict[str, str]
    plan: dict[str, Any]
    context: dict[str, Any]
    result: dict[str, Any]
    observation: dict[str, Any]
    decision: dict[str, Any]
    confirmation: dict[str, Any] | None
    pending_action: dict[str, Any] | None
    stop_reason: str | None
    error: str | None
    trace_events: list[dict[str, Any]]


def new_state(thread_id: str, **overrides: Any) -> FlowState:
    """构造带默认值的控制层状态；``overrides`` 由领域补充业务字段。"""
    state: FlowState = {
        'thread_id': thread_id,
        'trace_id': uuid.uuid4().hex,
        'status': 'planning',
        'step_count': 0,
        'tool_calls': 0,
        'no_progress_count': 0,
        'started_at': time.time(),
        'facts': {},
        'versions': {},
        'invalidated': {},
        'plan': {},
        'context': {},
        'result': {},
        'observation': {},
        'decision': {},
        'confirmation': None,
        'pending_action': None,
        'stop_reason': None,
        'error': None,
        'trace_events': [],
    }
    state.update(overrides)
    return state


def ensure_state(state: FlowState) -> FlowState:
    """补齐调用方省略的通用字段，不覆盖已有值。"""
    defaults = new_state(str(state.get('thread_id') or 'default'))
    for key, value in defaults.items():
        state.setdefault(key, value)  # type: ignore[literal-required]
    return state


@dataclass
class VersionedFacts:
    """带版本号与失效标记的事实集合。

    - ``values`` 保存当前值；``versions`` 保存当前版本号；
    - ``invalidated`` 只标记失效原因，不物理删除事实；
    - 版本号单调递增，确认与写操作必须绑定版本。
    """

    values: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    invalidated: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: FlowState) -> 'VersionedFacts':
        return cls(
            values=dict(state.get('facts') or {}),
            versions=dict(state.get('versions') or {}),
            invalidated=dict(state.get('invalidated') or {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'VersionedFacts':
        return cls(
            values=dict(data.get('values') or {}),
            versions={str(key): int(value) for key, value in (data.get('versions') or {}).items()},
            invalidated={str(key): str(value) for key, value in (data.get('invalidated') or {}).items()},
        )

    def apply_to(self, state: FlowState) -> None:
        state['facts'] = dict(self.values)
        state['versions'] = dict(self.versions)
        state['invalidated'] = dict(self.invalidated)

    def set(self, key: str, value: Any) -> int:
        """写入当前值并递增版本；重新写入会清除该事实的失效标记。"""
        version = self.versions.get(key, 0) + 1
        self.values[key] = value
        self.versions[key] = version
        self.invalidated.pop(key, None)
        return version

    def current(self, key: str, default: Any = None) -> Any:
        """读取当前有效值；已失效事实一律返回默认值。"""
        if key in self.invalidated:
            return default
        return self.values.get(key, default)

    def version(self, key: str) -> int:
        return int(self.versions.get(key, 0))

    def is_current(self, key: str, version: int) -> bool:
        return key not in self.invalidated and self.version(key) == int(version)

    def invalidate(self, key: str, reason: str) -> bool:
        """标记事实失效；返回该事实此前是否存在。"""
        if key not in self.versions and key not in self.values:
            return False
        self.invalidated[key] = reason
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            'values': dict(self.values),
            'versions': dict(self.versions),
            'invalidated': dict(self.invalidated),
        }


__all__ = ["FlowState", "VersionedFacts", "ensure_state", "new_state"]

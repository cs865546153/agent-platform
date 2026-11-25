"""控制层上下文装配：常驻事实、版本与消息裁剪。

执行面 ``ContextPolicy`` 只负责裁到预算内；本模块决定哪些业务事实必须常驻，
并在装配时只选择当前有效版本。领域可在此基础上增加自己的压缩逻辑。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_core.harness import ContextPolicy

from .state import FlowState, VersionedFacts


@dataclass(frozen=True)
class ContextPack:
    """一轮执行使用的上下文包；``messages`` 不进入持久化状态，避免体积膨胀。"""

    messages: tuple[Any, ...] = ()
    facts: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    trimmed: bool = False

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def as_dict(self) -> dict[str, Any]:
        return {
            'facts': dict(self.facts),
            'versions': dict(self.versions),
            'trimmed': self.trimmed,
            'message_count': self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ContextPack':
        """从持久化状态还原；消息不入状态，需要时由领域在 execute 中重建。"""
        return cls(
            messages=(),
            facts=dict(data.get('facts') or {}),
            versions={str(key): int(value) for key, value in (data.get('versions') or {}).items()},
            trimmed=bool(data.get('trimmed')),
        )


_MISSING = object()


@dataclass(frozen=True)
class ContextAssembler:
    """按预算裁剪消息，并装配常驻事实的当前版本。"""

    policy: ContextPolicy = field(default_factory=ContextPolicy)
    pinned_facts: tuple[str, ...] = ()

    def assemble(self, messages: Iterable[Any], facts: VersionedFacts) -> ContextPack:
        original = list(messages)
        selected = list(self.policy.apply(original))
        pinned: dict[str, Any] = {}
        versions: dict[str, int] = {}
        for key in self.pinned_facts:
            value = facts.current(key, _MISSING)
            if value is _MISSING:
                continue
            pinned[key] = value
            versions[key] = facts.version(key)
        return ContextPack(
            messages=tuple(selected),
            facts=pinned,
            versions=versions,
            trimmed=len(selected) < len(original),
        )

    def assemble_state(self, messages: Iterable[Any], state: FlowState) -> ContextPack:
        return self.assemble(messages, VersionedFacts.from_state(state))


__all__ = ["ContextAssembler", "ContextPack"]

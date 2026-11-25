"""澄清闸门：内核确定性地判断"该不该先问用户"，不依赖模型自由发挥。

与 :mod:`agent_core.confirmation` 并列，二者都只读取领域清单的声明：

- **意图级**：路由结果 ``needs_clarification`` 为真时，用 ``intent_probabilities``
  取候选意图，再由清单里的 ``description`` 拼出"你要 A 还是 B"的问题；
- **槽位级**：意图声明了 ``required_slots`` 时，缺哪个问哪个，用户可读文案来自
  ``slot_labels``，轮次最多 ``max_rounds``（超过后不再追问，交给业务能力处理）。

内核不写死任何业务字段名或追问话术；问题文本全部由清单与路由概率派生。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_INTENT = 'intent'
SOURCE_SLOT = 'slot'
SOURCE_NONE = 'none'

# 写操作澄清必须同时告诉用户"模型不会直接执行"，避免把追问误解为已受理。
WRITE_NOTE = '该操作不会由模型直接执行。'
SLOT_PROMPT = '请补充：{slots}。'
INTENT_PROMPT = '你是要{options}？'


@dataclass(frozen=True)
class ClarificationDecision:
    """一次澄清裁决：是否需要问、问什么、还缺哪些要素。"""

    required: bool
    source: str = SOURCE_NONE
    kind: str = 'none'
    prompt: str | None = None
    missing_slots: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    round: int = 0
    max_rounds: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            'required': self.required,
            'source': self.source,
            'kind': self.kind,
            'prompt': self.prompt,
            'missing_slots': list(self.missing_slots),
            'candidates': list(self.candidates),
            'round': self.round,
            'max_rounds': self.max_rounds,
        }


class ClarifyGate:
    """从领域清单与路由概率派生澄清裁决；不调用模型。"""

    def __init__(self, registry: Any = None, *, max_rounds: int = 3) -> None:
        self.registry = registry
        self.max_rounds = max(0, int(max_rounds))

    @classmethod
    def from_registry(cls, registry: Any, *, max_rounds: int = 3) -> "ClarifyGate":
        return cls(registry, max_rounds=max_rounds)

    def decide(self, domain: str | None, intent: str | None, state: dict[str, Any], *,
               write_operation: bool = False) -> ClarificationDecision:
        """确定性裁决：槽位缺失优先于意图歧义，轮次用尽后停止追问。"""
        current_round = _round_of(state)
        if current_round >= self.max_rounds:
            return ClarificationDecision(required=False, round=current_round,
                                         max_rounds=self.max_rounds)

        manifest = self._manifest(domain)
        spec = manifest.intent(intent) if manifest is not None and intent else None
        slots = state.get('slots') or {}

        if spec is not None:
            missing = tuple(
                name for name in spec.required_slots
                if not str(slots.get(name) or '').strip()
            )
            if missing:
                labels = [spec.slot_labels.get(name) or name for name in missing]
                prompt = SLOT_PROMPT.format(slots='、'.join(labels))
                return ClarificationDecision(
                    required=True, source=SOURCE_SLOT, kind='slot',
                    prompt=_decorate(prompt, write_operation),
                    missing_slots=missing, round=current_round + 1,
                    max_rounds=self.max_rounds,
                )

        if state.get('needs_clarification') and manifest is not None:
            candidates = self._candidates(manifest, domain, state.get('intent_probabilities') or {})
            if len(candidates) >= 2:
                options = []
                for name in candidates:
                    candidate = manifest.intent(name)
                    if candidate is not None:
                        options.append(candidate.description)
                if len(options) >= 2:
                    prompt = INTENT_PROMPT.format(options='，还是要'.join(options))
                    return ClarificationDecision(
                        required=True, source=SOURCE_INTENT, kind='intent',
                        prompt=_decorate(prompt, write_operation),
                        candidates=tuple(candidates), round=current_round + 1,
                        max_rounds=self.max_rounds,
                    )

        return ClarificationDecision(required=False, round=current_round,
                                     max_rounds=self.max_rounds)

    def _manifest(self, domain: str | None) -> Any:
        if self.registry is None or not domain:
            return None
        try:
            return self.registry.get(domain)
        except (KeyError, AttributeError):
            return None

    def _candidates(self, manifest: Any, domain: str | None,
                    probabilities: dict[str, Any]) -> list[str]:
        """按概率从高到低取清单内前两个意图；键形如 ``domain:intent``。"""
        prefix = f'{domain}:'
        ranked: list[str] = []
        for raw, _score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            name = raw[len(prefix):] if domain and raw.startswith(prefix) else raw
            if name in ranked:
                continue
            if manifest.intent(name) is None:
                continue
            ranked.append(name)
            if len(ranked) >= 2:
                break
        return ranked


def _decorate(prompt: str, write_operation: bool) -> str:
    return f'{WRITE_NOTE}{prompt}' if write_operation else prompt


def _round_of(state: dict[str, Any]) -> int:
    """澄清轮次：优先通用 ``round``，旧前端只回传 ``material_round`` 时兼容。"""
    raw = state.get('round')
    if raw is None:
        raw = state.get('material_round')
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "SOURCE_INTENT",
    "SOURCE_NONE",
    "SOURCE_SLOT",
    "ClarificationDecision",
    "ClarifyGate",
]

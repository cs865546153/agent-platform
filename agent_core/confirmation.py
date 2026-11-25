"""写操作确认闸门：内核的确定性裁决层。

对话链路（SSE）无状态、无 checkpointer、无 HITL。企业级设计要求「改变业务状态的
动作」不能由模型分类结果直接触发，这条约束以前是散落在领域 handler 里的隐含约定
（各自写一句"已拦截"）。本模块把它收敛成一条内核规则：

    声明了 ``confirmation`` 的意图，或 ``risk=high`` / ``mode=case`` 的意图
      -> 一律视为写操作，内核裁决 ``required=True``
      -> 绝不允许落到 ``answer``（自由问答）；落到 answer 时强制改派 ``guard``
      -> ``guard`` 输出结构化引导：目标独立 API、是否需人工审批、裁决来源

裁决是纯确定性代码，不调用 LLM；领域名、意图名、API 路径全部来自插件
``manifest.yaml``，内核不内置任何业务字符串。正常配置下（写操作意图都有
非 answer 的能力）落点原样保留，因此不会改变既有 SSE 事件序列。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agent_core.capabilities import ANSWER_CAPABILITY, GUARD_CAPABILITY
from agent_core.intent.contracts import ConfirmationSpec, IntentSpec

# 裁决来源：清单显式声明 / 由 risk-mode 推导 / 非写操作。
SOURCE_DECLARED = "declared"
SOURCE_RISK = "risk"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class ConfirmationDecision:
    """一次写操作裁决；``as_dict()`` 是写进 trace 与 state 的唯一形态。"""

    required: bool
    source: str = SOURCE_NONE
    api: str | None = None
    requires_human_review: bool = True
    summary: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "source": self.source,
            "api": self.api,
            "requires_human_review": self.requires_human_review if self.required else False,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ConfirmationRule:
    """从领域清单聚合出的一条写操作契约（含其绑定的落点能力）。"""

    domain: str
    intent: str
    spec: ConfirmationSpec
    capability: str


class ConfirmGate:
    """启动期从 ``DomainRegistry`` 聚合、运行期只读的确认闸门。"""

    def __init__(self, registry: Any = None) -> None:
        self.registry = registry
        self._rules: dict[tuple[str, str], ConfirmationRule] = {}
        self._specs: dict[tuple[str, str], IntentSpec] = {}
        if registry is None:
            return
        for manifest in registry.all():
            for spec in manifest.intents:
                key = (manifest.name, spec.name)
                self._specs[key] = spec
                if spec.confirmation is not None:
                    self._rules[key] = ConfirmationRule(
                        domain=manifest.name, intent=spec.name,
                        spec=spec.confirmation, capability=spec.resolved_capability,
                    )

    @classmethod
    def from_registry(cls, registry: Any) -> "ConfirmGate":
        return cls(registry)

    @property
    def rules(self) -> list[ConfirmationRule]:
        return list(self._rules.values())

    def decide(self, domain: str | None, intent: str | None) -> ConfirmationDecision:
        """确定性裁决：显式声明优先，其次由 risk/mode 推导，绝不问模型。"""
        key = (domain or "", intent or "")
        declared = self._rules.get(key)
        if declared is not None:
            return ConfirmationDecision(
                required=True, source=SOURCE_DECLARED, api=declared.spec.api,
                requires_human_review=declared.spec.requires_human_review,
                summary=declared.spec.summary,
            )
        spec = self._specs.get(key)
        if spec is not None and (spec.risk == "high" or spec.mode == "case"):
            # 没写 confirmation 契约的高风险意图也要拦截；内核不猜 API 路径。
            return ConfirmationDecision(required=True, source=SOURCE_RISK)
        return ConfirmationDecision(required=False, source=SOURCE_NONE, api=None,
                                    requires_human_review=False)

    def enforce(self, capability: str, decision: ConfirmationDecision) -> str:
        """写操作意图绝不落到自由问答；其余落点原样保留，不改派领域能力。"""
        if decision.required and capability == ANSWER_CAPABILITY:
            return GUARD_CAPABILITY
        return capability

    def audit_capabilities(self) -> list[str]:
        """装配自检 A：声明了写操作契约却落到自由问答（会被 enforce 改派）。"""
        return [
            f"{rule.domain}/{rule.intent} 声明了写操作确认契约，但落点为自由问答，"
            "闸门将强制改派 guard；请在清单里绑定拦截/预审能力"
            for rule in self._rules.values() if rule.capability == ANSWER_CAPABILITY
        ]

    def audit_routes(self, routes: Iterable[str]) -> list[str]:
        """装配自检 B：契约里的独立 API 必须真实注册，否则是死引导。"""
        known = set(routes)
        return [
            f"{rule.domain}/{rule.intent} 的确认契约指向未注册接口 {rule.spec.api}"
            for rule in self._rules.values() if rule.spec.api not in known
        ]


__all__ = [
    "SOURCE_DECLARED",
    "SOURCE_NONE",
    "SOURCE_RISK",
    "ConfirmGate",
    "ConfirmationDecision",
    "ConfirmationRule",
]

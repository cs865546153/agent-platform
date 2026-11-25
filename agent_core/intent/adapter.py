"""对话图适配层：把工作台请求映射为 IntentRequest，并把 RouteDecision 收敛为槽位。

`agent_core/intent` 只依赖公共契约，不认识 FastAPI、工作台请求模型，
也不内置任何业务字段形态（保单号、案件号等）。领域专属的槽位兜底抽取
由装配层通过 ``slot_extractor`` 注入，本模块只负责协议转换。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import IntentClassifier, IntentRequest, RouteDecision, Turn

HISTORY_LIMIT = 12
ATTACHMENT_LIMIT = 6

# 槽位抽取器签名：接收路由决策、当前问题、最近对话，返回槽位字典。
SlotExtractor = Callable[[RouteDecision, str, list[Turn]], dict[str, str]]


@dataclass(frozen=True)
class AssistantRoute:
    """主程序视角的路由结果：意图、置信度、槽位与完整决策。

    ``slots`` 是唯一的业务字段出口：键名由领域清单/槽位抽取器决定，内核不解释。
    """

    intent: str
    confidence: float
    slots: dict[str, str]
    decision: RouteDecision


class AssistantIntentRouter:
    """把 HybridIntentClassifier 适配到无状态对话图；只做协议转换，不参与业务决策。"""

    def __init__(
        self,
        classifier: IntentClassifier,
        *,
        slot_extractor: SlotExtractor | None = None,
    ) -> None:
        self.classifier = classifier
        # 领域插件声明的槽位兜底抽取；不注入时内核不做任何业务字段抽取。
        self.slot_extractor = slot_extractor

    @property
    def engine_label(self) -> str:
        """给 trace 用的引擎描述，例如 router(primary=jev,fallback=llm)。"""
        return f"router({getattr(self.classifier, 'engine_label', type(self.classifier).__name__)})"

    @property
    def model_label(self) -> str:
        """主引擎实际使用的模型名；避免把 LLM 分类模型误写成 JEV。"""
        primary = getattr(self.classifier, "primary", self.classifier)
        model = getattr(primary, "model", None)
        name = getattr(model, "model_name", None)
        if name is None and isinstance(model, str):
            name = model
        if name is None:
            name = getattr(primary, "engine_name", None) or type(primary).__name__
        return str(name)

    async def classify(
        self,
        *,
        question: str,
        history: Iterable[Any] = (),
        attachments: Iterable[Any] = (),
        active_domain: str | None = None,
    ) -> AssistantRoute:
        """执行一次意图路由；输入已脱敏，输出直接可用于工作台 IntentResult。"""
        turns = [
            Turn(role=turn.role, content=turn.content[:4000])
            for turn in list(history)[-HISTORY_LIMIT:]
        ]
        file_names = [self._attachment_name(item) for item in list(attachments)[:ATTACHMENT_LIMIT]]
        request = IntentRequest(
            message=question,
            history=turns,
            active_domain=active_domain,
            attachments=file_names,
        )
        decision = await self.classifier.classify(request)
        slots = self.slot_extractor(decision, question, turns) if self.slot_extractor else {}
        return AssistantRoute(
            intent=decision.intent,
            confidence=decision.confidence,
            slots=slots,
            decision=decision,
        )

    @staticmethod
    def _attachment_name(item: Any) -> str:
        name = getattr(item, "file_name", None)
        return str(name or "attachment")[:255]

    async def aclose(self) -> None:
        """关闭底层引擎持有的 HTTP 客户端（JEV 使用 httpx.AsyncClient）。"""
        close = getattr(self.classifier, "aclose", None)
        if close is not None:
            await close()


__all__ = ["AssistantIntentRouter", "AssistantRoute", "SlotExtractor"]

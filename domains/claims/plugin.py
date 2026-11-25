"""理赔领域插件：装配服务与图，注册理赔专属路由。

业务实现仍复用 ``agents/claim_agent.py`` 等既有模块；插件只负责生命周期编排，
这样"内核抽取"与"领域插件化"可以渐进完成而不破坏既有测试。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from agent_core.plugins import AppContext
from domains.base import DomainPluginBase

# 保单号形态：字母前缀 + 数字（POL-2024-001 / POL2024001）。
# JEV 只输出领域与意图、不返回槽位，因此保单号需本地正则兜底抽取；
# 这条业务规则属于理赔域，不下沉到 agent_core。
POLICY_ID_PATTERN = re.compile(r"\b[A-Za-z]{2,6}[-_]?\d{2,6}[-_]?\d{2,6}\b")
POLICY_SLOT_KEYS = ("policy_id", "policyid", "保单号", "保单编号")
POLICY_QUERY_INTENTS = frozenset({"保单查询"})


def extract_claim_slots(decision: Any, question: str, turns: list[Any]) -> dict[str, str]:
    """理赔域槽位兜底抽取：优先用路由槽位，其次在保单查询意图下从原文识别保单号。"""
    slots: dict[str, str] = {}
    for key in POLICY_SLOT_KEYS:
        value = decision.slots.get(key)
        if value:
            slots["policy_id"] = str(value)[:64]
            return slots
    if decision.intent not in POLICY_QUERY_INTENTS:
        return slots
    recent_user_text = "\n".join(
        [question, *(turn.content for turn in reversed(turns) if turn.role == "user")]
    )
    match = POLICY_ID_PATTERN.search(recent_user_text)
    if match:
        slots["policy_id"] = match.group(0)[:64]
    return slots


class ClaimsPlugin(DomainPluginBase):
    name = "claims"
    tags = ["理赔域"]
    graph_keys = ("claims",)

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        from adapters.business import BusinessBackend, DemoBackend
        from agents.claim_agent import WorkflowServices
        from config import get_model
        from tools.demo_clauses import build_demo_clause_retriever
        from tools.policy_store import ensure_policy_store

        demo = ctx.settings.business_mode == "demo"
        services = overrides.get("services")
        if services is None:
            if demo:
                backend = DemoBackend(policy_store=ensure_policy_store())
                retriever = cast(Callable[..., Any], build_demo_clause_retriever())
                services = WorkflowServices(backend, clause_retriever=retriever)
            else:
                services = WorkflowServices(BusinessBackend())
        ctx.services[self.name] = services
        # 领域专属槽位规则注册到上下文，由内核装配层注入意图路由适配器。
        ctx.slot_extractors[self.name] = extract_claim_slots
        # 对话编排能力同样在装配期注册：处理器持有本领域的服务与回答模型。
        from domains.claims.assistant_handlers import register_claims_capabilities

        register_claims_capabilities(
            ctx, service=services,
            model=ctx.models.get('customer_service') or get_model('customer_service'),
        )

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        from agents.claim_agent import build_claim_graph

        ctx.graphs[self.name] = build_claim_graph(ctx.services[self.name], saver)

    def register_routes(self, ctx: AppContext) -> None:
        from domains.claims.routes import register_demo_routes, register_routes

        register_routes(ctx)
        register_demo_routes(ctx)


__all__ = ["ClaimsPlugin"]

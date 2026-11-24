"""理赔领域插件：装配服务与图，注册理赔专属路由。

业务实现仍复用 ``agents/claim_agent.py`` 等既有模块；插件只负责生命周期编排，
这样"内核抽取"与"领域插件化"可以渐进完成而不破坏既有测试。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from agent_core.plugins import AppContext
from domains.base import DomainPluginBase


class ClaimsPlugin(DomainPluginBase):
    name = "claims"
    tags = ["理赔域"]

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        from adapters.business import BusinessBackend, DemoBackend
        from agents.claim_agent import WorkflowServices
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

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        from agents.claim_agent import build_claim_graph

        ctx.graphs[self.name] = build_claim_graph(ctx.services[self.name], saver)

    def register_routes(self, ctx: AppContext) -> None:
        from domains.claims.routes import register_routes

        register_routes(ctx)


__all__ = ["ClaimsPlugin"]

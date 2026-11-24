"""投资领域插件：装配服务与图，注入投资 RAG。"""
from __future__ import annotations

from typing import Any

from agent_core.plugins import AppContext
from domains.base import DomainPluginBase


class InvestmentPlugin(DomainPluginBase):
    name = "investment"
    tags = ["投资域"]

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        from adapters.investment import DemoInvestmentBackend, InvestmentBackend
        from agents.investment_agent import InvestmentServices

        services = overrides.get("investment_services")
        if services is None:
            backend = DemoInvestmentBackend() if ctx.settings.business_mode == "demo" else InvestmentBackend()
            services = InvestmentServices(backend)
        ctx.services[self.name] = services

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        from agents.investment_agent import build_investment_graph

        ctx.graphs[self.name] = build_investment_graph(ctx.services[self.name], saver)

    def register_routes(self, ctx: AppContext) -> None:
        from domains.investment.routes import register_routes

        register_routes(ctx)


__all__ = ["InvestmentPlugin"]

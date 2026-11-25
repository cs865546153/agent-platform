"""投资领域插件：装配服务与图，注入投资 RAG。"""
from __future__ import annotations

import logging
from typing import Any

from agent_core.plugins import AppContext
from domains.base import DomainPluginBase

LOGGER = logging.getLogger(__name__)


class InvestmentPlugin(DomainPluginBase):
    name = "investment"
    tags = ["投资域"]
    graph_keys = ("investment",)

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        from adapters.investment import DemoInvestmentBackend, InvestmentBackend
        from agents.investment_agent import InvestmentServices
        from config import get_model

        services = overrides.get("investment_services")
        if services is None:
            backend = DemoInvestmentBackend() if ctx.settings.business_mode == "demo" else InvestmentBackend()
            services = InvestmentServices(backend)
        ctx.services[self.name] = services
        # 投资知识库是本领域资源，由插件自己读取配置项装配；内核不认识 investment_rag_*。
        # 注入键由本领域解释：显式传 False 表示关闭（测试 / 精简部署不加载向量库）。
        injected = overrides.get('investment_rag')
        if injected is False:
            rag = None
        elif injected is not None:
            rag = injected
        elif ctx.settings.investment_rag_enabled:
            try:
                from tools.investment_rag import get_investment_rag

                rag = get_investment_rag()
            except Exception:  # noqa: BLE001 - 知识库损坏不应阻断整个应用启动
                LOGGER.exception('投资知识库初始化失败，已关闭投资 RAG')
                rag = None
        else:
            rag = None
        ctx.rag[self.name] = rag
        # 对话编排能力在装配期注册：投资问答与交易拦截都由本领域实现。
        from domains.investment.assistant_handlers import register_investment_capabilities

        register_investment_capabilities(
            ctx, rag=rag,
            model=ctx.models.get('investment_advice') or get_model('investment_advice'),
        )

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        from agents.investment_agent import build_investment_graph

        ctx.graphs[self.name] = build_investment_graph(ctx.services[self.name], saver)

    def register_routes(self, ctx: AppContext) -> None:
        from domains.investment.routes import register_routes

        register_routes(ctx)


__all__ = ["InvestmentPlugin"]

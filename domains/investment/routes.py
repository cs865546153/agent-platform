"""投资领域专属路由：从内核 app.py 抽出，由 InvestmentPlugin 注册。

依赖通过 AppContext 注入，不直接引用 app.py，避免循环依赖。
"""
from __future__ import annotations

import asyncio
from typing import Any, cast

from fastapi import Depends, HTTPException
from langgraph.types import Command

from agent_core.plugins import AppContext
from agent_core.security import authenticate, authenticate_reviewer
from agents.investment_agent import (
    build_demo_trace,
    investment_advice_from_state,
    investment_initial_state,
    investment_state_to_response,
    run_demo_investment,
)
from agents.middleware_audit import mask_pii
from domains.base import graph_config, interrupt_kind
from models.investment import (
    DemoInvestmentRunResponse,
    InvestmentConfirmation,
    InvestmentRAGQuery,
    InvestmentRAGResponse,
    InvestmentRequest,
    InvestmentResponse,
    InvestmentReviewInput,
)


def _graph_config(request_id: str) -> dict[str, Any]:
    return graph_config(f'investment:{request_id}', recursion_limit=30)


def _request_from_state(state: dict[str, Any]) -> InvestmentRequest:
    return InvestmentRequest(
        request_id=state['request_id'],
        question=state.get('question') or '投资请求',
        intent=state.get('intent') or '投资咨询',
        risk_profile=state.get('risk_profile'),
        portfolio_id=state.get('portfolio_id'),
        product_id=state.get('product_id'),
        amount=state.get('amount'),
        confirmed=bool(state.get('confirmed')),
    )


def register_routes(ctx: AppContext) -> None:
    application = ctx.application
    get_graph = lambda: ctx.state.investment_graph  # noqa: E731 - 延迟取图，避免 lifespan 前访问

    @application.post('/api/investment/process', response_model=InvestmentResponse,
                      dependencies=[Depends(authenticate)], tags=['投资域'])
    async def process_investment(request: InvestmentRequest) -> InvestmentResponse:
        """投资咨询直接形成建议；交易指令在二次确认前暂停，不执行下单。"""
        async with ctx.locks.setdefault(f'investment:{request.request_id}', asyncio.Lock()):
            config = _graph_config(request.request_id)
            graph = get_graph()
            snapshot = await graph.aget_state(config)
            if snapshot.values:
                return investment_state_to_response(
                    snapshot.values, request, interrupted=interrupt_kind(snapshot)
                )
            state = await graph.ainvoke(investment_initial_state(request), config=config)
            return investment_state_to_response(state, request, interrupted=interrupt_kind(state))

    @application.post('/api/investment/{request_id}/confirm', response_model=InvestmentResponse,
                      dependencies=[Depends(authenticate)], tags=['投资域'])
    async def confirm_investment(request_id: str, body: InvestmentConfirmation) -> InvestmentResponse:
        """二次确认交易指令；确认后进入人工审批，仍未下单。"""
        async with ctx.locks.setdefault(f'investment:{request_id}', asyncio.Lock()):
            config = _graph_config(request_id)
            graph = get_graph()
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise HTTPException(404, '未找到投资请求')
            request = _request_from_state(snapshot.values)
            if interrupt_kind(snapshot) != 'trade_confirmation':
                raise HTTPException(409, '当前请求不处于交易二次确认阶段')
            state = await graph.ainvoke(Command(resume=body.model_dump()), config=config)
            return investment_state_to_response(state, request, interrupted=interrupt_kind(state))

    @application.post('/api/investment/{request_id}/review', response_model=InvestmentResponse,
                      dependencies=[Depends(authenticate), Depends(authenticate_reviewer)], tags=['投资域'])
    async def review_investment(request_id: str, body: InvestmentReviewInput) -> InvestmentResponse:
        """人工审批交易指令；只有 approve 且适配器实际执行时才标记 executed。"""
        if body.decision == 'reject' and not body.reason:
            raise HTTPException(422, '拒绝交易必须记录理由')
        async with ctx.locks.setdefault(f'investment:{request_id}', asyncio.Lock()):
            config = _graph_config(request_id)
            graph = get_graph()
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise HTTPException(404, '未找到投资请求')
            request = _request_from_state(snapshot.values)
            if interrupt_kind(snapshot) != 'investment_review':
                raise HTTPException(409, '当前请求不处于人工审批阶段')
            state = await graph.ainvoke(Command(resume=mask_pii(body.model_dump())), config=config)
            return investment_state_to_response(state, request, interrupted=interrupt_kind(state))

    @application.post('/api/investment/rag/search', response_model=InvestmentRAGResponse,
                      dependencies=[Depends(authenticate)], tags=['投资域'])
    async def search_investment_rag(body: InvestmentRAGQuery) -> InvestmentRAGResponse:
        """只读检索投资知识库；返回引用，不生成交易建议。"""
        rag = ctx.rag.get('investment')
        if rag is None:
            raise HTTPException(503, '投资知识库未启用')
        result = await asyncio.to_thread(rag, body.question, body.top_k)
        return InvestmentRAGResponse(**result)

    @application.post('/api/demo/investment/process', response_model=DemoInvestmentRunResponse, tags=['本地演示'])
    async def process_demo_investment() -> DemoInvestmentRunResponse:
        """使用固定合成组合运行投资域图，不连接券商、不执行真实下单。"""
        if ctx.settings.app_env == 'production':
            raise HTTPException(404, '演示接口在生产环境不可用')
        request, state, result = await run_demo_investment()
        from agents.investment_agent import InvestmentState

        typed_state = cast(InvestmentState, state)
        return DemoInvestmentRunResponse(
            request=request,
            result=result,
            advice=investment_advice_from_state(typed_state),
            trace=build_demo_trace(typed_state, request),
        )


__all__ = ["register_routes"]

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
    awaiting_trade_confirmation,
    build_demo_trace,
    investment_advice_from_state,
    investment_initial_state,
    investment_state_to_response,
    pending_trade_gate,
    run_demo_investment,
)
from agents.middleware_audit import mask_pii
from domains.base import gate_resume, graph_config, resume_gate
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


def _interrupted(snapshot: Any) -> str | None:
    """快照所停留的 HITL 阶段。

    中断阶段有两个来源，缺一不可：
    - ``tasks[].interrupts``：同进程内的中断载荷，但**进程重启后不再重建**；
    - ``next`` 里的待执行节点：重启后只剩这个，按 ``intake`` / ``review`` 重建。

    注意不能图省事直接读 ``ainvoke`` 的返回值——该版本的返回状态不带
    ``__interrupt__``，读它会让"等待二次确认"永远显示成"无中断"。
    """
    return resume_gate(snapshot, pending_trade_gate)


async def _gate_after(graph: Any, config: dict[str, Any]) -> str | None:
    """跑完一步后重新读快照，取真正的停驻阶段。"""
    return _interrupted(await graph.aget_state(config))


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
    def get_graph() -> Any:
        """延迟取图：lifespan 装配完成后 ctx.graphs 才有值。"""
        return ctx.graphs['investment']

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
                    snapshot.values, request, interrupted=_interrupted(snapshot)
                )
            state = await graph.ainvoke(investment_initial_state(request), config=config)
            return investment_state_to_response(state, request,
                                                interrupted=await _gate_after(graph, config))

    @application.post('/api/investment/{request_id}/confirm', response_model=InvestmentResponse,
                      dependencies=[Depends(authenticate)], tags=['投资域'])
    async def confirm_investment(request_id: str, body: InvestmentConfirmation) -> InvestmentResponse:
        """二次确认交易指令；确认后进入人工审批，仍未下单。"""
        async with ctx.locks.setdefault(f'investment:{request_id}', asyncio.Lock()):
            config = _graph_config(request_id)
            graph = get_graph()
            snapshot = await gate_resume(graph, config, kind='trade_confirmation',
                                         not_found='未找到投资请求',
                                         conflict='当前请求不处于交易二次确认阶段',
                                         node='intake',
                                         still_waiting=awaiting_trade_confirmation)
            request = _request_from_state(snapshot.values)
            state = await graph.ainvoke(Command(resume=body.model_dump()), config=config)
            return investment_state_to_response(state, request,
                                                interrupted=await _gate_after(graph, config))

    @application.post('/api/investment/{request_id}/review', response_model=InvestmentResponse,
                      dependencies=[Depends(authenticate), Depends(authenticate_reviewer)], tags=['投资域'])
    async def review_investment(request_id: str, body: InvestmentReviewInput) -> InvestmentResponse:
        """人工审批交易指令；只有 approve 且适配器实际执行时才标记 executed。"""
        if body.decision == 'reject' and not body.reason:
            raise HTTPException(422, '拒绝交易必须记录理由')
        async with ctx.locks.setdefault(f'investment:{request_id}', asyncio.Lock()):
            config = _graph_config(request_id)
            graph = get_graph()
            snapshot = await gate_resume(graph, config, kind='investment_review',
                                         not_found='未找到投资请求',
                                         conflict='当前请求不处于人工审批阶段',
                                         node='review')
            request = _request_from_state(snapshot.values)
            state = await graph.ainvoke(Command(resume=mask_pii(body.model_dump())), config=config)
            return investment_state_to_response(state, request,
                                                interrupted=await _gate_after(graph, config))

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

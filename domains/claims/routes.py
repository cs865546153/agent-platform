"""理赔领域专属路由：从内核 app.py 抽出，由 ClaimsPlugin 注册。

包含核心理赔工作流（process/status/supplement/review）及状态查询辅助函数。
监控打点直接依赖 monitoring 模块（横切能力，领域无关）。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Depends, HTTPException
from langgraph.types import Command
from typing import Literal

from agent_core.plugins import AppContext
from agent_core.security import authenticate, authenticate_reviewer
from agents.middleware_audit import mask_pii
from agents.observability import tracing_config
from domains.base import graph_config
from models.schemas import ClaimRequest, ClaimResponse
from monitoring import CONFIDENCE, ERRORS, ITERATIONS, LATENCY, REQUESTS, TOKENS
from pydantic import BaseModel, ConfigDict, Field


class Supplement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    policy_id: str = Field(min_length=1, max_length=64)


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: Literal['review', 'accept', 'reject']
    reviewer: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=1000)
    clause_basis: list[str] = Field(default_factory=list)


def register_routes(ctx: AppContext) -> None:
    application = ctx.application
    service = ctx.services['claims']
    cfg = ctx.settings

    async def status(claim_id: str) -> ClaimResponse:
        snapshot = await ctx.state.graph.aget_state(graph_config(claim_id))
        state = snapshot.values
        if not state:
            raise HTTPException(404, '未找到案件')
        pending = snapshot.next
        phase = ('awaiting_information' if 'intake' in pending
                 else 'awaiting_review' if 'review' in pending
                 else state.get('phase', 'pending'))
        return ClaimResponse(claim_id=claim_id,
            decision=('pending' if phase == 'awaiting_information'
                      else 'review' if phase == 'awaiting_review'
                      else state.get('decision', 'pending')),
            phase=phase, confidence=state.get('confidence'),
            demo=getattr(service.backend, 'demo', False),
            message=mask_pii(state.get('message') or
                             ('需要补充保单编号' if phase == 'awaiting_information'
                              else '等待人工审核')))

    async def invoke(claim_id: str, payload: Any) -> ClaimResponse:
        config, handler = tracing_config(claim_id, cfg)
        start = time.monotonic()
        try:
            ctx.limiter.check(claim_id)
            await ctx.state.graph.ainvoke(payload, config=config)
            result = await status(claim_id)
            REQUESTS.labels(result.decision).inc()
            if result.confidence is not None:
                CONFIDENCE.observe(result.confidence)
            snapshot = await ctx.state.graph.aget_state(graph_config(claim_id))
            ITERATIONS.observe(len(snapshot.values.get('expert_results', [])) + snapshot.values.get('rounds', 0))
            usage = config['callbacks'][0].snapshot()
            TOKENS.labels('input').inc(usage['prompt_tokens'])
            TOKENS.labels('output').inc(usage['completion_tokens'])
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            ERRORS.labels('validation').inc()
            raise HTTPException(422, '审核输入或模型结果不符合契约') from exc
        except Exception as exc:
            ERRORS.labels(type(exc).__name__).inc()
            if str(exc) == 'rate_limit_exceeded':
                raise HTTPException(429, '该案件请求过于频繁') from exc
            raise HTTPException(503, '处理链路暂不可用，请查询案件状态或转人工') from exc
        finally:
            LATENCY.observe(time.monotonic() - start)
            if handler is not None:
                await asyncio.to_thread(handler.flush)

    @application.post('/api/claims/process', response_model=ClaimResponse,
                      dependencies=[Depends(authenticate)], tags=['理赔域'])
    async def process_claim(request: ClaimRequest) -> ClaimResponse:
        async with ctx.lock_for(request.claim_id):
            existing = await ctx.state.graph.aget_state(graph_config(request.claim_id))
            if existing.values:
                if existing.values['claim_data'] != mask_pii(request.model_dump(mode='json')):
                    raise HTTPException(409, '案件已存在；请使用补材料或审核接口')
                return await status(request.claim_id)
            return await invoke(request.claim_id, {
                'claim_id': request.claim_id,
                'claim_data': mask_pii(request.model_dump(mode='json')),
                'rounds': 0, 'expert_results': [],
            })

    @application.get('/api/claims/{claim_id}', response_model=ClaimResponse,
                     dependencies=[Depends(authenticate)], tags=['理赔域'])
    async def get_claim(claim_id: str) -> ClaimResponse:
        return await status(claim_id)

    @application.post('/api/claims/{claim_id}/supplement', response_model=ClaimResponse,
                      dependencies=[Depends(authenticate)], tags=['理赔域'])
    async def supplement(claim_id: str, body: Supplement) -> ClaimResponse:
        async with ctx.lock_for(claim_id):
            current = await status(claim_id)
            if current.phase != 'awaiting_information':
                raise HTTPException(409, '当前案件不处于补材料阶段')
            return await invoke(claim_id, Command(resume=body.model_dump()))

    @application.post('/api/claims/{claim_id}/review', response_model=ClaimResponse,
                      dependencies=[Depends(authenticate), Depends(authenticate_reviewer)], tags=['理赔域'])
    async def review(claim_id: str, body: ReviewInput) -> ClaimResponse:
        if body.decision == 'reject' and not body.clause_basis:
            raise HTTPException(422, '拒赔必须提供核实后的条款依据')
        async with ctx.lock_for(claim_id):
            current = await status(claim_id)
            if current.phase != 'awaiting_review':
                raise HTTPException(409, '当前案件不处于人工审核阶段')
            return await invoke(claim_id, Command(resume=mask_pii(body.model_dump())))


__all__ = ["register_routes"]

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
from domains.base import gate_resume, graph_config
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
        snapshot = await ctx.graphs['claims'].aget_state(graph_config(claim_id))
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
            await ctx.graphs['claims'].ainvoke(payload, config=config)
            result = await status(claim_id)
            REQUESTS.labels(result.decision).inc()
            if result.confidence is not None:
                CONFIDENCE.observe(result.confidence)
            snapshot = await ctx.graphs['claims'].aget_state(graph_config(claim_id))
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
            existing = await ctx.graphs['claims'].aget_state(graph_config(request.claim_id))
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
            await gate_resume(ctx.graphs['claims'], graph_config(claim_id),
                              kind='information', not_found='未找到案件',
                              conflict='当前案件不处于补材料阶段', node='intake')
            return await invoke(claim_id, Command(resume=body.model_dump()))

    @application.post('/api/claims/{claim_id}/review', response_model=ClaimResponse,
                      dependencies=[Depends(authenticate), Depends(authenticate_reviewer)], tags=['理赔域'])
    async def review(claim_id: str, body: ReviewInput) -> ClaimResponse:
        if body.decision == 'reject' and not body.clause_basis:
            raise HTTPException(422, '拒赔必须提供核实后的条款依据')
        async with ctx.lock_for(claim_id):
            await gate_resume(ctx.graphs['claims'], graph_config(claim_id),
                              kind='review', not_found='未找到案件',
                              conflict='当前案件不处于人工审核阶段', node='review')
            return await invoke(claim_id, Command(resume=mask_pii(body.model_dump())))


__all__ = ["register_routes", "register_demo_routes"]


def register_demo_routes(ctx: AppContext) -> None:
    """注册理赔域本地演示路由：固定合成数据 + 真实状态机，不接真实业务、不付款。

    演示属于示例领域而非内核能力，因此随 ClaimsPlugin 一起启用；
    自定义 registry 不装 claims 插件时这两个路由自然不存在。
    """
    from agents.claim_agent import build_claim_graph
    from claims_agent import DemoServices
    from demos.mock_claim import build_mock_claim
    from models.schemas import ClaimResponse
    from models.workbench import DemoClaimRunResponse

    application = ctx.application
    cfg = ctx.settings
    service = ctx.services['claims']

    @application.post('/api/demo/claims/process', response_model=DemoClaimRunResponse, tags=['本地演示'])
    async def process_demo_claim() -> DemoClaimRunResponse:
        """使用固定合成数据运行真实状态机，不连接真实业务或执行付款。"""
        if cfg.app_env == 'production':
            raise HTTPException(404, '演示接口在生产环境不可用')

        claim, documents = build_mock_claim()
        claim_data = claim.model_dump(mode='json')
        graph = build_claim_graph(DemoServices())
        state = await graph.ainvoke(
            {'claim_id': claim.claim_id, 'claim_data': claim_data, 'rounds': 0, 'expert_results': []},
            config={'configurable': {'thread_id': claim.claim_id}, 'recursion_limit': 30, 'max_concurrency': 3},
        )
        opinions = {item['expert']: item for item in state['expert_results']}
        trace: list[dict[str, Any]] = [
            {'node': 'claim_intake', 'status': 'done', 'summary': '报案字段完整，可提交正式工作流',
             'input': claim_data, 'output': {'phase': 'ready', 'missing_structured_fields': []}},
            {'node': 'claim_documents', 'status': 'done', 'summary': '已核查 1 份合成单证',
             'input': {'documents': [item.model_dump() for item in documents]},
             'output': {'recognized_count': 1, 'average_extraction_confidence': 0.97, 'warning_count': 0}},
            {'node': 'claim_followup', 'status': 'skipped', 'summary': '基础材料齐全，无需回流追问',
             'input': {'round': 0}, 'output': {'complete': True}},
            {'node': 'claim_policy', 'status': 'done', 'summary': '演示保单与保障责任核验完成',
             'input': {'policy_id': claim.policy_id},
             'output': {'policy': state['policy_info'], 'verification': state['verified']}},
            *(
                {'node': f'claim_{role}', 'status': 'done',
                 'summary': f"建议 {opinions[role]['recommendation']} · 置信度 {opinions[role]['confidence']:.0%}",
                 'input': {'role': role, 'claim_id': claim.claim_id}, 'output': opinions[role]}
                for role in ('damage', 'risk', 'liability')
            ),
            {'node': 'claim_confidence', 'status': 'done',
             'summary': f"三专家最低置信度 {state['confidence']:.0%}",
             'input': {'method': 'min(expert_confidence)',
                       'expert_scores': {role: opinions[role]['confidence'] for role in opinions}},
             'output': {'confidence': state['confidence'], 'risk_score': state['risk_score']}},
            {'node': 'claim_decision', 'status': 'done', 'summary': '规则校验通过，形成自动受理建议',
             'input': {'amount': str(claim.amount),
                       'coverage_verified': state['verified'].get('coverage_verified'),
                       'confidence': state['confidence'], 'risk_score': state['risk_score']},
             'output': {'decision': state['decision'], 'phase': state['phase']}},
        ]
        result = ClaimResponse(
            claim_id=claim.claim_id, decision=state['decision'], phase=state['phase'],
            confidence=state['confidence'], demo=True, message=state['message'],
        )
        return DemoClaimRunResponse(claim=claim, documents=documents, result=result, trace=trace)

    @application.post('/api/demo/claims/process-live', response_model=DemoClaimRunResponse, tags=['本地演示'])
    async def process_demo_claim_live() -> DemoClaimRunResponse:
        """用内存条款 + 真实模型跑完整状态机，演示判责 RAG 检索全过程。"""
        if cfg.app_env == 'production':
            raise HTTPException(404, '演示接口在生产环境不可用')

        claim, documents = build_mock_claim()
        claim_data = claim.model_dump(mode='json')
        graph = build_claim_graph(service)
        state = await graph.ainvoke(
            {'claim_id': claim.claim_id, 'claim_data': claim_data, 'rounds': 0, 'expert_results': []},
            config={'configurable': {'thread_id': claim.claim_id}, 'recursion_limit': 30, 'max_concurrency': 3},
        )
        opinions = {item.get('expert'): item for item in state.get('expert_results', [])}

        def summary(role: str) -> str:
            opinion = opinions.get(role)
            if not opinion:
                return '专家结果缺失'
            text = f"建议 {opinion.get('recommendation')} · 置信度 {opinion.get('confidence', 0):.0%}"
            if opinion.get('clause_ids'):
                text += f" · 引用条款 {', '.join(opinion['clause_ids'])}"
            return text

        trace: list[dict[str, Any]] = [
            {'node': 'claim_intake', 'status': 'done', 'summary': '报案字段完整，可提交正式工作流',
             'input': claim_data, 'output': {'phase': 'ready', 'missing_structured_fields': []}},
            {'node': 'claim_documents', 'status': 'done', 'summary': '已核查 1 份合成单证',
             'input': {'documents': [item.model_dump() for item in documents]},
             'output': {'recognized_count': 1, 'average_extraction_confidence': 0.97, 'warning_count': 0}},
            {'node': 'claim_followup', 'status': 'skipped', 'summary': '基础材料齐全，无需回流追问',
             'input': {'round': 0}, 'output': {'complete': True}},
            {'node': 'claim_policy', 'status': 'done', 'summary': '保单与保障责任核验完成',
             'input': {'policy_id': claim.policy_id},
             'output': {'policy': state.get('policy_info'), 'verification': state.get('verified')}},
            *(
                {'node': f'claim_{role}', 'status': 'done', 'summary': summary(role),
                 'input': {'role': role, 'claim_id': claim.claim_id}, 'output': opinions.get(role)}
                for role in ('damage', 'risk', 'liability')
            ),
            {'node': 'claim_confidence', 'status': 'done',
             'summary': f"三专家最低置信度 {state.get('confidence', 0):.0%}",
             'input': {'method': 'min(expert_confidence)',
                       'expert_scores': {role: opinions[role].get('confidence') for role in opinions}},
             'output': {'confidence': state.get('confidence'), 'risk_score': state.get('risk_score')}},
            {'node': 'claim_decision', 'status': 'done', 'summary': '规则校验完成，形成审核建议',
             'input': {'amount': str(claim.amount),
                       'coverage_verified': state.get('verified', {}).get('coverage_verified'),
                       'confidence': state.get('confidence'), 'risk_score': state.get('risk_score')},
             'output': {'decision': state.get('decision'), 'phase': state.get('phase')}},
        ]
        result = ClaimResponse(
            claim_id=claim.claim_id, decision=state.get('decision', 'review'), phase=state.get('phase', 'complete'),
            confidence=state.get('confidence'), demo=True, message=state.get('message', ''),
        )
        return DemoClaimRunResponse(claim=claim, documents=documents, result=result, trace=trace)

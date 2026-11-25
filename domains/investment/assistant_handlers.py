"""投资域对对话助手的编排贡献：检索问答能力 + 交易拦截能力。

内核只按领域清单里的能力名分派，不认识"投资"：

- ``investment_answer``：投资咨询 / 组合诊断 / 风险评估 / 产品查询，先检索投资
  Wiki（多路召回 + rerank），再交给投资模型流式回答；检索失败只降级不中断。
- ``trade_gate``：买入 / 赎回等会改变持仓的指令，在对话链路只做拦截提示，
  真正的下单要经 ``/api/investment/*`` 的适当性校验 + 二次确认 + 人工审批。

trace 节点名（``investment_rag`` / ``investment``）与事件顺序是工作台前端的既有
契约，改动需同步 static/app.js。
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_core.capabilities import AssistantCapabilities
from agents.middleware_audit import mask_pii
from monitoring import ERRORS
from prompts.investment_prompt import INVESTMENT_ADVISOR_PROMPT, INVESTMENT_RAG_USAGE_PROMPT

TRADE_GATE_MESSAGE = (
    '交易指令不会由模型直接执行。请提供产品代码、交易金额和风险承受能力；'
    '系统会先做适当性校验，再要求你二次确认，并转授权人员审批。'
    '在审批完成前不会下单。'
)


def build_investment_handlers(*, rag: Any, model: Any) -> dict[str, Any]:
    """装配投资域的对话编排能力；返回 ``{能力名: 处理器}``。

    ``rag`` 是投资知识库调用（可为 ``None``，表示未启用），
    ``model`` 是投资回答模型，两者都由 InvestmentPlugin 在 ``build_services`` 注入。
    """
    from agents.assistant_agent import conversation_messages
    from tools.investment_rag import format_investment_context

    async def investment_answer(state: dict[str, Any]) -> dict[str, Any]:
        """先检索投资 Wiki，再把引用资料交给投资模型；检索异常只降级。"""
        events: list[dict[str, Any]] = []
        knowledge_context = ''
        if rag is None:
            events.append({'trace': {'node': 'investment_rag', 'status': 'skipped',
                                     'summary': '投资知识库未启用',
                                     'output': {'enabled': False}}})
        else:
            top_k = int(getattr(rag, 'top_k', 4) or 4)
            events.append({'trace': {'node': 'investment_rag', 'status': 'running',
                                     'summary': '检索投资知识库（多路召回 + rerank）',
                                     'input': {'question': mask_pii(state['question']),
                                               'top_k': top_k}}})
            try:
                retrieved = await asyncio.to_thread(rag, state['question'], top_k)
                documents = retrieved.get('documents', []) if isinstance(retrieved, dict) else []
                knowledge_context = format_investment_context(documents)
                vector_ready = bool(retrieved.get('vector_ready')) if isinstance(retrieved, dict) else False
                degraded = bool(retrieved.get('degraded')) if isinstance(retrieved, dict) else not vector_ready
                vector_error = retrieved.get('vector_error') if isinstance(retrieved, dict) else None
                if not documents:
                    rag_status = 'skipped'
                    rag_summary = '投资知识库未命中，按通用投资原则回答'
                elif vector_ready:
                    rag_status = 'done'
                    rag_summary = f'投资知识库命中 {len(documents)} 条（BM25 + 向量 + 元数据 + rerank）'
                else:
                    rag_status = 'compat'
                    rag_summary = f'投资知识库命中 {len(documents)} 条（向量未连接，BM25/元数据 + rerank）'
                events.append({'trace': {
                    'node': 'investment_rag',
                    'status': rag_status,
                    'summary': rag_summary,
                    'output': {
                        'citations': [
                            {key: document.get(key) for key in (
                                'citation_id', 'doc_id', 'title', 'chapter', 'source', 'source_url',
                            )}
                            for document in documents
                        ],
                        'vector_ready': vector_ready,
                        'degraded': degraded,
                        'vector_error': vector_error,
                        'corpus_size': retrieved.get('corpus_size') if isinstance(retrieved, dict) else None,
                    },
                }})
            except Exception as exc:  # noqa: BLE001 - RAG 失败不能阻断投资问答
                ERRORS.labels('investment_rag').inc()
                events.append({'trace': {'node': 'investment_rag', 'status': 'error',
                                         'summary': f'投资知识库检索失败：{type(exc).__name__}',
                                         'output': {'error': type(exc).__name__, 'detail': str(exc)}}})
                knowledge_context = format_investment_context([])

        return {
            'llm': model,
            'messages': conversation_messages(
                state['question'], state['history'],
                INVESTMENT_ADVISOR_PROMPT
                + INVESTMENT_RAG_USAGE_PROMPT
                + f'用户风险承受能力：{state.get("risk_profile") or "未提供"}。'
                + f'已识别单证摘要：{state["context"]}\n\n'
                + knowledge_context,
            ),
            'needs_llm_answer': True,
            'events': events,
        }

    async def trade_gate(state: dict[str, Any]) -> dict[str, Any]:
        """交易指令只做安全拦截和要素提示，不调用下单工具。"""
        return {
            'events': [
                {'trace': {'node': 'investment', 'status': 'running',
                           'summary': '交易指令进入高风险案件模式',
                           'input': {'intent': state['intent'], 'question': state['question']}}},
                {'trace': {'node': 'investment', 'status': 'done',
                           'summary': '已拦截直接下单，等待二次确认与人工审批',
                           'output': {'executed': False, 'requires_confirmation': True,
                                      'requires_human_review': True}}},
                {'content': TRADE_GATE_MESSAGE},
            ],
            'needs_llm_answer': False,
        }

    return {'investment_answer': investment_answer, 'trade_gate': trade_gate}


def register_investment_capabilities(ctx: Any, *, rag: Any, model: Any) -> AssistantCapabilities:
    """把投资域的对话编排能力写进应用上下文；内核按能力名分派。"""
    capabilities = ctx.capabilities
    for capability, handler in build_investment_handlers(rag=rag, model=model).items():
        capabilities.register(capability, handler)
    return capabilities


__all__ = [
    "TRADE_GATE_MESSAGE",
    "build_investment_handlers",
    "register_investment_capabilities",
]

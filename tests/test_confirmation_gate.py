"""写操作确认闸门契约：对话链路绝不执行改变业务状态的动作。

企业级设计里 LLM 只提议意图，"是否需要确认 / 去哪个独立 API / 是否需人工审批"
必须由确定性代码裁决。本文件覆盖六条保证：

1. 契约形状：``confirmation`` 由领域清单声明，拼错字段直接报错（extra=forbid）；
2. 确定性裁决：显式声明优先，其次由 risk=high / mode=case 推导，绝不问模型；
3. 强制不落自由问答：写操作落到 ``answer`` 时改派 ``guard``，领域能力原样保留；
4. 结构化引导：``guard`` 从裁决输出独立 API 与人工审批，内核不内置业务路径；
5. 装配与可见性：闸门挂在 app.state、契约指向的 API 真实注册、裁决并入 SSE trace；
6. 注入不丢失：显式传入的自定义闸门贯穿建图，新增领域零内核改动即可生效。

正常配置下写操作意图都绑定了非 answer 的领域能力，因此闸门不新增事件、不改派
既有落点，SSE 事件序列保持不变。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from agent_core.capabilities import ANSWER_CAPABILITY, GUARD_CAPABILITY, AssistantCapabilities
from agent_core.confirmation import (
    SOURCE_DECLARED,
    SOURCE_NONE,
    SOURCE_RISK,
    ConfirmGate,
    ConfirmationDecision,
)
from agent_core.intent.contracts import ConfirmationSpec, DomainManifest, IntentSpec
from agent_core.intent.registry import DomainRegistry
from agents.assistant_agent import IntentRoutingPolicy, build_assistant_graph


class FakeModel:
    model_name = 'fake'

    def __init__(self, content: str = '{}') -> None:
        self.content = content

    async def ainvoke(self, messages: list[Any]) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.content)


class FakeRoute:
    """意图路由的替身：只回传意图名，不触网、不加载 JEV。"""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.confidence = 0.95
        self.slots: dict[str, str] = {}
        self.decision = None


class FakeRouter:
    model_label = 'fake'
    engine_label = 'fake'

    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def classify(self, **kwargs: Any) -> FakeRoute:
        return FakeRoute(self.intent)


def _state(intent: str = '交易指令', **extra: Any) -> dict[str, Any]:
    return {
        'question': '帮我赎回基金', 'claim_id': None, 'documents': [], 'history': [],
        'material_round': 0, 'context': '[]', 'started': time.monotonic(), 'events': [],
        **extra,
    }


def _manifest(*, with_confirmation: bool = True, capability: str = 'trade_gate') -> DomainManifest:
    intent: dict[str, Any] = {
        'name': '交易指令', 'description': '会改变持仓的指令',
        'risk': 'high', 'mode': 'case', 'capability': capability,
    }
    if with_confirmation:
        intent['confirmation'] = {
            'api': '/api/investment/process',
            'requires_human_review': False,
            'summary': '请通过投资业务接口提交。',
        }
    return DomainManifest(
        name='investment', description='投资领域',
        capabilities=[capability, 'answer', 'guard'],
        intents=[IntentSpec.model_validate(intent)],
    )


# --- 1. 契约形状 -----------------------------------------------------------

def test_confirmation_contract_is_declared_and_strict() -> None:
    """confirmation 随清单走，形状由契约约束，拼错字段必须当场报错。"""
    spec = IntentSpec.model_validate(_manifest().intents[0].model_dump())
    assert isinstance(spec.confirmation, ConfirmationSpec)
    assert spec.confirmation.api == '/api/investment/process'
    assert spec.confirmation.requires_human_review is False
    with pytest.raises(ValueError):
        IntentSpec.model_validate({
            'name': 'x', 'description': 'x',
            'confirmation': {'api': '/x', 'typo_field': 1},
        })


def test_confirmation_api_is_required_and_nonempty() -> None:
    with pytest.raises(ValueError):
        ConfirmationSpec.model_validate({'api': ''})
    with pytest.raises(ValueError):
        ConfirmationSpec.model_validate({})


# --- 2. 确定性裁决 ---------------------------------------------------------

def test_gate_decides_declared_risk_and_none() -> None:
    registry = DomainRegistry([
        DomainManifest(
            name='investment', description='投资领域',
            capabilities=['trade_gate', 'answer', 'guard'],
            intents=[
                IntentSpec(name='交易指令', description='写操作', risk='high', mode='case',
                           capability='trade_gate',
                           confirmation=ConfirmationSpec(api='/api/investment/process',
                                                         requires_human_review=False)),
                IntentSpec(name='高风险查询', description='未声明契约', risk='high', mode='conversation'),
                IntentSpec(name='一般咨询', description='只读', risk='low', mode='conversation'),
            ],
        ),
    ])
    gate = ConfirmGate.from_registry(registry)

    declared = gate.decide('investment', '交易指令')
    assert declared.required and declared.source == SOURCE_DECLARED
    assert declared.api == '/api/investment/process'
    assert declared.requires_human_review is False

    # 没写 confirmation 的高风险意图也要拦截；内核不猜 API 路径。
    by_risk = gate.decide('investment', '高风险查询')
    assert by_risk.required and by_risk.source == SOURCE_RISK and by_risk.api is None

    plain = gate.decide('investment', '一般咨询')
    assert not plain.required and plain.source == SOURCE_NONE and plain.api is None
    assert not gate.decide('unknown', '不存在').required


def test_new_domain_gate_needs_no_kernel_change() -> None:
    """新增领域只要声明 confirmation，闸门自动生效，内核不写死领域名/API。"""
    registry = DomainRegistry([
        DomainManifest(
            name='lending', description='全新领域',
            capabilities=['loan_apply', 'answer', 'guard'],
            intents=[IntentSpec(name='提交贷款申请', description='写操作', mode='case',
                                capability='loan_apply',
                                confirmation=ConfirmationSpec(api='/api/lending/apply'))],
        ),
    ])
    decision = ConfirmGate.from_registry(registry).decide('lending', '提交贷款申请')
    assert decision.required and decision.source == SOURCE_DECLARED
    assert decision.api == '/api/lending/apply'


# --- 3. 强制不落自由问答 ---------------------------------------------------

def test_gate_enforces_guard_only_for_free_answer() -> None:
    gate = ConfirmGate()
    required = ConfirmationDecision(required=True)
    assert gate.enforce(ANSWER_CAPABILITY, required) == GUARD_CAPABILITY
    # 领域已有的拦截/预审能力原样保留，不改派既有落点（SSE 序列不变）。
    assert gate.enforce('trade_gate', required) == 'trade_gate'
    assert gate.enforce(ANSWER_CAPABILITY, ConfirmationDecision(required=False)) == ANSWER_CAPABILITY


def test_writes_on_answer_are_rerouted_to_guard() -> None:
    """写操作意图误落到 answer 时，策略强制改派 guard 并带上裁决。"""
    registry = DomainRegistry([
        DomainManifest(
            name='demo', description='演示', capabilities=['answer', 'guard'],
            intents=[IntentSpec(name='写操作', description='写', mode='case',
                                capability='answer',
                                confirmation=ConfirmationSpec(api='/api/demo/write'))],
        ),
    ])
    gate = ConfirmGate.from_registry(registry)
    policy = IntentRoutingPolicy.from_registry(registry, confirm_gate=gate)
    target = policy.resolve('demo', '写操作', _state())
    assert target.capability == GUARD_CAPABILITY
    assert target.confirmation['source'] == SOURCE_DECLARED
    assert target.confirmation['api'] == '/api/demo/write'


# --- 4. 结构化引导 ---------------------------------------------------------

def test_guard_guidance_comes_from_decision_not_kernel_strings() -> None:
    registry = DomainRegistry([
        DomainManifest(
            name='demo', description='演示', capabilities=['answer', 'guard'],
            intents=[IntentSpec(name='写操作', description='写', mode='case',
                                capability='answer',
                                confirmation=ConfirmationSpec(api='/api/demo/write',
                                                              summary='走业务接口。'))],
        ),
    ])
    graph = build_assistant_graph(
        classifier=FakeModel(), general_model=FakeModel(),
        intent_router=FakeRouter('写操作'),
        domain_registry=registry,
    )
    state = asyncio.run(graph.ainvoke(_state()))
    assert state['capability'] == GUARD_CAPABILITY
    texts = ''.join(str(e.get('content', '')) for e in state['events'])
    assert '/api/demo/write' in texts
    done = [e['trace'] for e in state['events']
            if e.get('trace', {}).get('status') == 'done' and e.get('trace', {}).get('node') != 'classification']
    assert any(t['output'].get('confirmation', {}).get('api') == '/api/demo/write' for t in done)


# --- 5. 装配与 SSE 可见性 --------------------------------------------------

def test_domain_handler_trace_carries_confirmation_without_new_events() -> None:
    """裁决由内核并入领域 handler 已有的 done trace，不新增事件、不改序列。"""
    registry = DomainRegistry([_manifest()])

    async def trade_gate(state: dict[str, Any]) -> dict[str, Any]:
        return {
            'events': [
                {'trace': {'node': 'investment', 'status': 'running', 'summary': '进入拦截'}},
                {'trace': {'node': 'investment', 'status': 'done',
                           'output': {'executed': False, 'requires_confirmation': True}}},
                {'content': '交易需审批'},
            ],
            'needs_llm_answer': False,
        }

    capabilities = AssistantCapabilities()
    capabilities.register('trade_gate', trade_gate)
    gate = ConfirmGate.from_registry(registry)
    policy = IntentRoutingPolicy.from_registry(registry, capabilities, confirm_gate=gate)
    graph = build_assistant_graph(
        classifier=FakeModel(), general_model=FakeModel(),
        intent_router=FakeRouter('交易指令'), routing_policy=policy,
    )
    state = asyncio.run(graph.ainvoke(_state()))

    assert state['capability'] == 'trade_gate'  # 落点未被改派
    traces = [e['trace'] for e in state['events'] if 'trace' in e]
    # SSE 事件序列与未启用闸门时完全一致：闸门只加字段，不新增节点。
    assert [(t['node'], t['status']) for t in traces] == [
        ('classification', 'running'), ('classification', 'done'),
        ('route', 'done'),
        ('investment', 'running'), ('investment', 'done'),
    ]
    assert len(state['events']) == 7  # 5 个 trace + 1 个 intent + 1 个 content
    investment_done = next(t for t in traces if t['node'] == 'investment' and t['status'] == 'done')
    # 原有字段保持，额外并入裁决（结构化、可审计）。
    assert investment_done['output']['executed'] is False
    assert investment_done['output']['confirmation']['api'] == '/api/investment/process'
    assert investment_done['output']['confirmation']['required'] is True
    # 裁决只出现在 trace 内，不新增顶层事件。
    assert not any('confirmation' in e for e in state['events'] if 'trace' not in e)


def test_assembled_app_exposes_gate_and_has_no_dead_routes() -> None:
    """真实装配：闸门挂 state、契约 API 全注册、装配自检无告警。"""
    from fastapi.testclient import TestClient

    from app import create_app

    with TestClient(create_app(database=':memory:')) as client:
        application = cast(Any, client.app)
        state = application.state
        assert state.confirm_gate is not None
        assert state.confirmation_audit == []
        paths = {getattr(r, 'path', '') for r in application.routes}
        for rule in state.confirm_gate.rules:
            assert rule.spec.api in paths


def test_custom_gate_injection_is_preserved_through_graph() -> None:
    """显式注入的自定义闸门贯穿建图，不被默认派生覆盖。"""
    calls: list[tuple[str, str]] = []

    class SpyGate:
        def decide(self, domain: str | None, intent: str | None) -> ConfirmationDecision:
            calls.append((domain or '', intent or ''))
            return ConfirmationDecision(required=False, source=SOURCE_NONE)

        def enforce(self, capability: str, decision: ConfirmationDecision) -> str:
            return capability

    registry = DomainRegistry([_manifest()])
    capabilities = AssistantCapabilities()

    async def trade_gate(state: dict[str, Any]) -> dict[str, Any]:
        return {'events': [{'content': 'ok'}], 'needs_llm_answer': False}

    capabilities.register('trade_gate', trade_gate)
    policy = IntentRoutingPolicy.from_registry(
        registry, capabilities, confirm_gate=cast(ConfirmGate, SpyGate()),
    )
    graph = build_assistant_graph(
        classifier=FakeModel(), general_model=FakeModel(),
        intent_router=FakeRouter('交易指令'), routing_policy=policy,
    )
    asyncio.run(graph.ainvoke(_state()))
    assert ('investment', '交易指令') in calls


def test_audit_detects_bad_capability_binding_and_dead_api() -> None:
    """装配自检能发现：写操作落到自由问答、契约指向未注册接口。"""
    registry = DomainRegistry([
        DomainManifest(
            name='bad', description='坏配置', capabilities=['answer', 'guard'],
            intents=[IntentSpec(name='写操作', description='写', mode='case',
                                capability='answer',
                                confirmation=ConfirmationSpec(api='/api/not-registered'))],
        ),
    ])
    gate = ConfirmGate.from_registry(registry)
    assert any('自由问答' in issue for issue in gate.audit_capabilities())
    assert any('/api/not-registered' in issue for issue in gate.audit_routes({'/api/ok'}))

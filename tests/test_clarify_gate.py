"""澄清闸门契约：低置信 / 缺必填槽位时先追问，不调模型、不写死业务话术。

覆盖四条保证：

1. 槽位级：清单声明 ``required_slots``，缺哪个问哪个，文案来自 ``slot_labels``；
2. 意图级：``needs_clarification`` 为真时，用路由概率 + 清单 description 给候选；
3. 轮次上限：超过 ``max_rounds`` 不再追问，交回业务能力处理；
4. 优先级：澄清先于确认，写操作澄清必须声明"不会由模型直接执行"。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from agent_core.capabilities import CLARIFY_CAPABILITY, AssistantCapabilities
from agent_core.clarification import SOURCE_INTENT, SOURCE_SLOT, ClarifyGate
from agent_core.intent.contracts import (
    ConfirmationSpec,
    DomainManifest,
    IntentSpec,
    RouteDecision,
)
from agent_core.intent.registry import DomainRegistry
from agents.assistant_agent import IntentRoutingPolicy, build_assistant_graph


def _manifest(*, required_slots: list[str] | None = None,
              slot_labels: dict[str, str] | None = None,
              confirmation: ConfirmationSpec | None = None) -> DomainManifest:
    return DomainManifest(
        name='invest', description='投资领域',
        capabilities=['trade_gate', 'answer', 'guard'],
        intents=[
            IntentSpec(name='交易指令', description='买入或赎回，会改变持仓',
                       risk='high', mode='case', capability='trade_gate',
                       required_slots=required_slots or [],
                       slot_labels=slot_labels or {},
                       confirmation=confirmation),
            IntentSpec(name='投资咨询', description='询问配置思路与理财建议',
                       risk='low', mode='conversation', capability='trade_gate'),
        ],
    )


def _state(intent: str = '交易指令', **extra: Any) -> dict[str, Any]:
    return {
        'question': '帮我买入基金', 'claim_id': None, 'documents': [], 'history': [],
        'material_round': 0, 'round': 0, 'context': '[]',
        'started': time.monotonic(), 'events': [], **extra,
    }


class FakeModel:
    model_name = 'fake'

    def __init__(self, content: str = '{}') -> None:
        self.content = content
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> Any:
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=self.content)


class FakeRoute:
    def __init__(self, intent: str, *, slots: dict[str, str] | None = None,
                 decision: RouteDecision | None = None) -> None:
        self.intent = intent
        self.confidence = 0.55
        self.slots = slots or {}
        self.decision = decision


class FakeRouter:
    model_label = 'fake'
    engine_label = 'fake'

    def __init__(self, intent: str, *, slots: dict[str, str] | None = None,
                 decision: RouteDecision | None = None) -> None:
        self.route = FakeRoute(intent, slots=slots, decision=decision)

    async def classify(self, **kwargs: Any) -> FakeRoute:
        return self.route


def test_slot_level_clarification_asks_missing_slot_only() -> None:
    gate = ClarifyGate.from_registry(DomainRegistry([
        _manifest(required_slots=['product_id', 'amount'],
                  slot_labels={'product_id': '产品代码', 'amount': '交易金额'}),
    ]))
    decision = gate.decide('invest', '交易指令', _state(slots={'product_id': 'FUND-1'}))

    assert decision.required and decision.source == SOURCE_SLOT
    assert decision.missing_slots == ('amount',)
    assert decision.prompt == '请补充：交易金额。'
    assert decision.round == 1


def test_slot_labels_fall_back_to_slot_name() -> None:
    gate = ClarifyGate.from_registry(DomainRegistry([
        _manifest(required_slots=['product_id', 'amount']),
    ]))
    decision = gate.decide('invest', '交易指令', _state(slots={}))
    assert decision.missing_slots == ('product_id', 'amount')
    assert decision.prompt == '请补充：product_id、amount。'


def test_no_clarification_when_slots_are_filled() -> None:
    gate = ClarifyGate.from_registry(DomainRegistry([
        _manifest(required_slots=['amount'], slot_labels={'amount': '交易金额'}),
    ]))
    decision = gate.decide('invest', '交易指令', _state(slots={'amount': '10000'}))
    assert not decision.required
    assert decision.source == 'none'


def test_round_limit_stops_clarifying() -> None:
    gate = ClarifyGate.from_registry(DomainRegistry([
        _manifest(required_slots=['amount']),
    ]), max_rounds=2)
    assert gate.decide('invest', '交易指令', _state(round=1, slots={})).required
    assert not gate.decide('invest', '交易指令', _state(round=2, slots={})).required


def test_intent_level_clarification_uses_manifest_descriptions() -> None:
    registry = DomainRegistry([_manifest()])
    gate = ClarifyGate.from_registry(registry)
    decision = gate.decide('invest', '交易指令', _state(
        needs_clarification=True,
        intent_probabilities={'invest:交易指令': 0.55, 'invest:投资咨询': 0.44},
    ))

    assert decision.required and decision.source == SOURCE_INTENT
    assert decision.candidates == ('交易指令', '投资咨询')
    assert decision.prompt == '你是要买入或赎回，会改变持仓，还是要询问配置思路与理财建议？'
    assert '0.55' not in decision.prompt


def test_write_operation_clarification_states_no_direct_execution() -> None:
    gate = ClarifyGate.from_registry(DomainRegistry([
        _manifest(required_slots=['amount'], slot_labels={'amount': '交易金额'},
                  confirmation=ConfirmationSpec(api='/api/invest/process')),
    ]))
    decision = gate.decide('invest', '交易指令', _state(slots={}), write_operation=True)
    assert decision.prompt.startswith('该操作不会由模型直接执行。')
    assert '交易金额' in decision.prompt


def test_clarification_takes_priority_over_confirmation() -> None:
    registry = DomainRegistry([
        _manifest(required_slots=['amount'], slot_labels={'amount': '交易金额'},
                  confirmation=ConfirmationSpec(api='/api/invest/process')),
    ])

    async def trade_gate(state: dict[str, Any]) -> dict[str, Any]:
        return {'needs_llm_answer': False, 'events': [{'content': 'should not run'}]}

    capabilities = AssistantCapabilities({'trade_gate': trade_gate})
    policy = IntentRoutingPolicy.from_registry(
        registry, capabilities, clarify_gate=ClarifyGate.from_registry(registry),
    )

    missing = policy.resolve('invest', '交易指令', _state(slots={}, intent='交易指令'))
    assert missing.capability == CLARIFY_CAPABILITY
    assert missing.clarification['missing_slots'] == ['amount']
    assert missing.confirmation['required'] is True

    filled = policy.resolve('invest', '交易指令',
                            _state(slots={'amount': '10000'}, intent='交易指令'))
    assert filled.capability == 'trade_gate'
    assert not filled.clarification


def test_graph_clarify_node_asks_without_calling_model() -> None:
    registry = DomainRegistry([
        _manifest(required_slots=['amount'], slot_labels={'amount': '交易金额'}),
    ])
    general = FakeModel()
    router = FakeRouter('交易指令')
    graph = build_assistant_graph(
        classifier=FakeModel('{}'), general_model=general,
        domain_registry=registry, intent_router=router,
    )
    state = asyncio.run(graph.ainvoke(_state()))

    assert state['capability'] == CLARIFY_CAPABILITY
    assert state['needs_llm_answer'] is False
    assert general.calls == 0
    assert '请补充：交易金额。' in ''.join(
        str(event.get('content', '')) for event in state['events'])
    status = next(event['clarification_status'] for event in state['events']
                  if 'clarification_status' in event)
    assert status['round'] == 1
    assert status['missing_slots'] == ['amount']
    plan = next(event['trace'] for event in state['events']
                if event.get('trace', {}).get('node') == 'plan')
    assert plan['output']['selected']['capability'] == CLARIFY_CAPABILITY
    assert plan['output']['needs_clarification'] is True
    assert plan['output']['needs_confirmation'] is True
    assert plan['output']['missing_slots'] == ['amount']
    done = next(event['trace'] for event in state['events']
                if event.get('trace', {}).get('status') == 'done'
                and event['trace'].get('node') == 'invest')
    assert done['output']['executed'] is False


def test_real_trade_manifest_declares_required_slots() -> None:
    from domains.registry import default_domain_registry

    spec = default_domain_registry().get('investment').intent('交易指令')
    assert spec is not None
    assert set(spec.required_slots) == {'product_id', 'amount', 'risk_profile'}
    assert spec.slot_labels['amount'] == '交易金额'

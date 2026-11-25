"""对话编排能力注册表与动态建图契约。

内核按"能力名"分派，领域按"能力名"注册实现，两边都不出现对方的名字。这里覆盖
五条保证：

1. 重复注册显式报错，接管内置能力需显式 ``replace``；
2. 清单没声明 / 处理器缺失时，落点由 ``mode`` / ``risk`` 推导，不会静默自由问答；
3. 领域改写钩子优先于清单，但改写到未实现的能力上等于无效；
4. 对话图的节点集合由注册表决定——注册了新能力就能被路由到，不必改内核；
5. 清单之外的意图（模型幻觉、非契约回显）只降级为通用回答，不能让 SSE 中断。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from agent_core.capabilities import (
    ANSWER_CAPABILITY,
    GUARD_CAPABILITY,
    AssistantCapabilities,
    Redirect,
    declared_capabilities,
    missing_handlers,
)
from agent_core.intent.contracts import DomainManifest, fallback_capability
from agent_core.intent.registry import DomainRegistry
from agents.assistant_agent import UNROUTED_INTENT, build_assistant_graph


def _manifest(name: str, **intent: Any) -> DomainManifest:
    capabilities = [value for value in (intent.get('capability'),) if value]
    return DomainManifest(
        name=name, description=f'{name} 领域',
        capabilities=capabilities + ['answer', 'guard'],
        intents=[dict({'name': f'{name}-意图', 'description': '测试意图'}, **intent)],
    )


class FakeModel:
    model_name = 'fake'

    def __init__(self, content: str = '{}') -> None:
        self.content = content

    async def ainvoke(self, messages: list[Any]) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.content)


def _state(question: str = '你好', **extra: Any) -> dict[str, Any]:
    return {
        'question': question, 'claim_id': None, 'documents': [], 'history': [],
        'material_round': 0, 'context': '[]', 'started': time.monotonic(), 'events': [],
        **extra,
    }


async def _noop(state: dict[str, Any]) -> dict[str, Any]:
    return {'needs_llm_answer': False, 'events': [{'content': 'ok'}]}


def test_duplicate_capability_registration_is_rejected() -> None:
    """能力名即对话图节点名，两个领域撞名必须显式失败。"""
    capabilities = AssistantCapabilities()
    capabilities.register('finance_answer', _noop)
    with pytest.raises(ValueError, match='重复注册'):
        capabilities.register('finance_answer', _noop)
    capabilities.register('finance_answer', _noop, replace=True)

    # 领域可以接管内核内置能力：注册一次即可生效，with_kernel 补齐时用 setdefault。
    capabilities.register(ANSWER_CAPABILITY, _noop)
    armed = capabilities.with_kernel(answer=lambda state: None, guard=_noop)
    assert armed.get(ANSWER_CAPABILITY) is _noop
    assert GUARD_CAPABILITY in armed.names()
    with pytest.raises(ValueError, match='重复注册'):
        capabilities.register(ANSWER_CAPABILITY, _noop)


def test_dispatch_falls_back_to_mode_and_risk() -> None:
    """未声明能力 / 处理器缺失时，落点由 mode 与 risk 推导。"""
    assert fallback_capability('case', 'low') == GUARD_CAPABILITY
    assert fallback_capability('conversation', 'high') == GUARD_CAPABILITY
    assert fallback_capability('conversation', 'low') == ANSWER_CAPABILITY

    registry = DomainRegistry([
        _manifest('risky', capability='risky_gate', mode='case'),
        _manifest('plain'),
    ])
    idle = AssistantCapabilities()
    # 声明了 risky_gate 但没人实现：案件型意图宁可拦截，也不能退化成自由问答。
    assert idle.dispatch(registry, 'risky', 'risky-意图') == (GUARD_CAPABILITY, None)
    # 清单里没绑能力：按 mode/risk 推导。
    assert idle.dispatch(registry, 'plain', 'plain-意图') == (ANSWER_CAPABILITY, None)
    # 领域或意图未知时同样有确定落点。
    assert idle.dispatch(registry, None, None) == (ANSWER_CAPABILITY, None)
    assert idle.dispatch(registry, 'unknown-domain', 'plain-意图') == (ANSWER_CAPABILITY, None)

    armed = AssistantCapabilities()
    armed.register('risky_gate', _noop)
    assert armed.dispatch(registry, 'risky', 'risky-意图')[1] is _noop


def test_redirect_preferred_but_skipped_when_unimplemented() -> None:
    """改写钩子优先于清单；改写到没实现的能力上则视为无效。"""
    capabilities = AssistantCapabilities()
    capabilities.register_redirect('claims', lambda state: Redirect(
        domain='claims', capability='claims_flow', evidence={'forced': True}))
    assert capabilities.redirect({'question': 'x'}) is None

    capabilities.register('claims_flow', _noop)
    hit = capabilities.redirect({'question': 'x'})
    assert (hit.domain, hit.capability, hit.evidence) == ('claims', 'claims_flow', {'forced': True})

    capabilities.register_redirect('claims', lambda state: ('claims', 'claims_flow'))
    with pytest.raises(TypeError, match='Redirect'):
        capabilities.redirect({'question': 'x'})


def test_registration_helpers_report_gaps() -> None:
    """装配期缺口清单：声明了能力却没实现，且忽略内核内置能力。"""
    registry = DomainRegistry([
        _manifest('claims', capability='claims_flow', mode='case'),
        _manifest('general', capability='answer'),
    ])
    assert declared_capabilities(registry) == ['claims_flow']
    assert missing_handlers(registry, AssistantCapabilities()) == [('claims_flow', 'claims')]
    armed = AssistantCapabilities()
    armed.register('claims_flow', _noop)
    assert missing_handlers(registry, armed) == []


def test_graph_nodes_are_derived_from_registered_capabilities() -> None:
    """新能力只要注册即可成为落点，内核代码里没有出现该领域名。"""
    registry = DomainRegistry([_manifest('finance', capability='finance_quote')])
    classifier = FakeModel('{"intent":"finance-意图","confidence":0.9}')
    capabilities = AssistantCapabilities()
    visited: list[dict[str, Any]] = []

    async def finance_quote(state: dict[str, Any]) -> dict[str, Any]:
        visited.append(state)
        return {'needs_llm_answer': False, 'events': [{'content': '行情已查询'}]}

    capabilities.register('finance_quote', finance_quote)
    graph = build_assistant_graph(
        classifier=classifier, general_model=FakeModel('unused'),
        domain_registry=registry, capabilities=capabilities,
    )
    state = asyncio.run(graph.ainvoke(_state('帮我查一下行情')))

    assert state['domain'] == 'finance'
    assert state['capability'] == 'finance_quote'
    assert state['needs_llm_answer'] is False
    assert visited and visited[0]['question'] == '帮我查一下行情'
    route = next(event['trace'] for event in state['events']
                 if event.get('trace', {}).get('node') == 'route')
    assert route['output'] == {'selected': 'finance', 'capability': 'finance_quote',
                               'handler': True, 'declared': True}


def test_intent_outside_the_manifest_degrades_to_kernel_answer() -> None:
    """清单是意图合法性的唯一事实源：模型编造的意图只降级问答，不报错也不进业务链路。"""
    registry = DomainRegistry([_manifest('finance', capability='finance_quote', mode='case')])
    capabilities = AssistantCapabilities()
    called: list[str] = []

    async def finance_quote(state: dict[str, Any]) -> dict[str, Any]:
        called.append(state['intent'])
        return {'needs_llm_answer': False, 'events': [{'content': '不该被调用'}]}

    capabilities.register('finance_quote', finance_quote)
    graph = build_assistant_graph(
        classifier=FakeModel('{"intent":"随便编一个意图","confidence":0.95}'),
        general_model=FakeModel('unused'), domain_registry=registry, capabilities=capabilities,
    )
    state = asyncio.run(graph.ainvoke(_state('今天天气怎么样')))

    assert called == []
    assert state['capability'] == ANSWER_CAPABILITY
    assert state['needs_llm_answer'] is True
    route = next(event['trace'] for event in state['events']
                 if event.get('trace', {}).get('node') == 'route')
    assert route['output']['declared'] is False


def test_malformed_classifier_output_keeps_the_stream_alive() -> None:
    """分类器回显非契约 JSON 时不能让整条对话崩掉：按未识别处理并留给 trace 线索。"""
    graph = build_assistant_graph(
        classifier=FakeModel('我觉得用户是在问天气'), general_model=FakeModel('unused'),
        domain_registry=DomainRegistry([_manifest('finance')]),
    )
    state = asyncio.run(graph.ainvoke(_state('随便聊聊')))

    assert state['intent'] == UNROUTED_INTENT
    assert state['capability'] == ANSWER_CAPABILITY
    assert state['needs_llm_answer'] is True
    classification = [event['trace'] for event in state['events']
                      if event.get('trace', {}).get('node') == 'classification']
    assert classification[-1]['route_error'].startswith('classify:')
    # 兜底文本不能带出模型回显（可能含用户输入），只保留异常类型。
    assert '天气' not in classification[-1]['route_error']


def test_unimplemented_capability_lands_on_kernel_guard() -> None:
    """没实现的案件型能力落到内置 guard：拦截并说明需要确认，不编造业务结果。"""
    registry = DomainRegistry([_manifest('finance', capability='finance_trade', mode='case')])
    graph = build_assistant_graph(
        classifier=FakeModel('{"intent":"finance-意图","confidence":0.9}'),
        general_model=FakeModel('unused'), domain_registry=registry,
    )
    state = asyncio.run(graph.ainvoke(_state('帮我买入')))

    assert state['capability'] == GUARD_CAPABILITY
    assert state['needs_llm_answer'] is False
    assert '会改变业务状态' in ''.join(
        str(event.get('content', '')) for event in state['events'])


def test_failing_domain_capability_degrades_without_breaking_the_stream() -> None:
    """领域处理器抛异常时显式降级为转人工，而不是吞成空回答。"""
    registry = DomainRegistry([_manifest('finance', capability='finance_quote')])
    capabilities = AssistantCapabilities()

    async def broken(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError('后端不可用')

    capabilities.register('finance_quote', broken)
    graph = build_assistant_graph(
        classifier=FakeModel('{"intent":"finance-意图","confidence":0.9}'),
        general_model=FakeModel('unused'), domain_registry=registry, capabilities=capabilities,
    )
    state = asyncio.run(graph.ainvoke(_state('帮我查一下行情')))

    error = next(event['trace'] for event in state['events']
                 if event.get('trace', {}).get('status') == 'error')
    assert error['node'] == 'finance_quote'
    assert error['output']['error'] == 'RuntimeError'
    assert '转人工' in ''.join(str(event.get('content', '')) for event in state['events'])


def test_kernel_answer_is_used_without_any_domain_registry() -> None:
    """完全没有领域清单时仍能回答：这是"清单缺失不致命"的兜底。"""
    graph = build_assistant_graph(
        classifier=FakeModel('{"intent":"一般咨询","confidence":0.8}'),
        general_model=FakeModel('unused'),
    )
    state = asyncio.run(graph.ainvoke(_state('Python 列表推导式是什么？')))

    assert state['capability'] == ANSWER_CAPABILITY
    assert state['needs_llm_answer'] is True
    assert '通用智能助手' in state['messages'][0].content

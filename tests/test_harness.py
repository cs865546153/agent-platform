"""执行面 Harness 契约：上下文裁剪、循环/工具策略、注册表与领域覆盖。"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_core.harness import (
    ContextPolicy,
    Harness,
    HarnessHooks,
    HarnessRegistry,
    LoopPolicy,
    ToolPolicy,
    estimate_tokens,
)
from app import create_app


def _history_messages() -> list:
    return [SystemMessage(content='系统提示')] + [
        HumanMessage(content=f'问题{i}'.ljust(20, '补')) for i in range(6)
    ]


def test_context_policy_keeps_system_and_recent_messages() -> None:
    messages = _history_messages()
    trimmed = ContextPolicy(max_messages=2).apply(messages)
    assert trimmed[0].type == 'system'
    assert [message.content for message in trimmed[1:]] == [messages[-2].content, messages[-1].content]


def test_context_policy_trims_by_token_budget() -> None:
    messages = _history_messages()
    trimmed = ContextPolicy(max_tokens=40).apply(messages)
    assert trimmed[0].type == 'system'
    assert estimate_tokens(trimmed) <= 40
    assert trimmed[-1].content == messages[-1].content
    assert len(trimmed) < len(messages)


def test_context_policy_never_removes_last_non_system_message() -> None:
    messages = [SystemMessage(content='系统提示'), HumanMessage(content='唯一问题')]
    trimmed = ContextPolicy(max_tokens=1).apply(messages)
    assert [message.type for message in trimmed] == ['system', 'human']


def test_loop_and_tool_policies() -> None:
    loop = LoopPolicy(max_steps=3, max_tool_calls=2, deadline_s=5)
    assert loop.allow(steps=2, tool_calls=1, elapsed_s=1)
    assert not loop.allow(steps=3)
    assert not loop.allow(tool_calls=2)
    assert not loop.allow(elapsed_s=5)
    tools = ToolPolicy(allow=frozenset({'search'}), deny=frozenset({'delete'}), max_calls=1)
    assert tools.allows('search')
    assert not tools.allows('delete')
    assert not tools.allows('other')


def test_registry_prefers_domain_harness() -> None:
    default = Harness()
    claims = Harness()
    registry = HarnessRegistry(default=default)
    registry.register('claims', claims)
    assert registry.get('claims') is claims
    assert registry.get('investment') is default
    assert registry.get(None) is default
    with pytest.raises(ValueError):
        registry.register('claims', Harness())
    registry.register('claims', Harness(), replace=True)
    assert registry.domains() == ['claims']


def test_hooks_apply_to_model_round_trip() -> None:
    class FakeModel:
        model_name = 'fake'

        async def astream(self, messages):
            assert [message.content for message in messages] == ['最后一句']
            yield AIMessage(content='hello')
            yield AIMessage(content='')

    harness = Harness(hooks=HarnessHooks(
        before_model=lambda messages: messages[-1:],
        on_chunk=str.upper,
    ))

    async def collect() -> list[str]:
        return [chunk async for chunk in harness.stream_answer({
            'llm': FakeModel(),
            'messages': [SystemMessage(content='系统'), HumanMessage(content='最后一句')],
        })]

    assert asyncio.run(collect()) == ['HELLO']


class FakeGeneralModel:
    """只验证装配协议，不连接外部模型。"""

    model_name = 'fake'

    async def ainvoke(self, messages):
        return AIMessage(content=json.dumps({'intent': '一般咨询', 'confidence': 0.93},
                                            ensure_ascii=False))

    async def astream(self, messages):
        yield AIMessage(content='hello')


class UppercaseHarness(Harness):
    """演示领域只替换回答阶段，不触碰路由与事件序列。"""

    async def stream_answer(self, state):
        async for chunk in super().stream_answer(state):
            yield chunk.upper()


def test_domain_harness_override_is_used_by_assistant_stream() -> None:
    model = FakeGeneralModel()
    with TestClient(create_app(
        database=':memory:',
        document_text_model=model,
        document_vision_model=model,
        assistant_model=model,
        intent_model=model,
        general_model=model,
        harnesses={'general': UppercaseHarness()},
    )) as client:
        response = client.post('/api/assistant/stream', json={
            'question': 'Python 列表推导式是什么？',
            'documents': [],
            'history': [],
        })
        assert response.status_code == 200
        assert 'HELLO' in response.text
        assert 'hello' not in response.text

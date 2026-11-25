"""意图路由可插拔契约：领域与意图清单随插件走，内核不再写死。

覆盖三条保证：
1. DomainRegistry.from_plugins 聚合插件 manifest，并可回退 YAML 补齐无插件领域；
2. IntentRoutingPolicy 从领域清单派生图内分支，新领域意图不必改 assistant_agent.py；
3. create_app 可显式注入 intent_router，内核跳过构建且不接管其生命周期。
"""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from agent_core.intent.contracts import DomainManifest
from agent_core.intent.registry import DomainRegistry
from agents.assistant_agent import IntentRoutingPolicy
from app import create_app
from domains.example.plugin import ExamplePlugin
from domains.registry import default_registry


class FakeRouter:
    """最小 router 替身：验证装配路径，不触发真实模型。"""

    engine_label = 'router(fake)'
    model_label = 'fake'

    async def classify(self, *, question, history=(), attachments=(), active_domain=None):
        raise AssertionError('本测试不触发分类')

    async def aclose(self) -> None:
        self.closed = True


def test_domain_registry_aggregates_plugin_manifests() -> None:
    """插件声明优先；fallback 补齐未声明领域（如 general）。"""
    plugins = [ExamplePlugin()]
    registry = DomainRegistry.from_plugins(plugins)
    assert registry.names() == ['example']
    assert {i.name for i in registry.get('example').intents} == {'示例问候'}

    with_fallback = DomainRegistry.from_plugins(plugins, fallback=DomainRegistry.default())
    # YAML 已瘦身为回退清单，只补无插件的 general；claims/investment 由插件声明。
    assert set(with_fallback.names()) == {'example', 'general'}
    assert with_fallback.get('example').name == 'example'


def test_default_registry_manifest_matches_yaml() -> None:
    """默认插件的 manifest 是 claims/investment 事实源，YAML 回退只补 general。"""
    regs = default_registry().all()
    registry = DomainRegistry.from_plugins(regs, fallback=DomainRegistry.default())
    assert set(registry.names()) == {'claims', 'investment', 'general'}
    assert '保单查询' in {i.name for i in registry.get('claims').intents}


def test_from_plugins_rejects_conflicting_fallback() -> None:
    """同名领域两份清单分叉时显式报错，不再静默以插件为准。"""

    class Plugin:
        def intent_manifest(self) -> DomainManifest:
            return DomainManifest(
                name='claims', description='插件版描述',
                intents=[{'name': 'A', 'description': 'a'}],
            )

    consistent = DomainManifest(
        name='claims', description='插件版描述',
        intents=[{'name': 'A', 'description': 'a'}],
    )
    conflicting = consistent.model_copy(update={'description': 'YAML 版描述'})

    # 内容一致时允许共存（插件优先，不重复注册）。
    ok = DomainRegistry.from_plugins([Plugin()], fallback=DomainRegistry([consistent]))
    assert ok.get('claims').description == '插件版描述'

    with pytest.raises(ValueError, match='内容不一致'):
        DomainRegistry.from_plugins([Plugin()], fallback=DomainRegistry([conflicting]))


def test_routing_policy_derives_from_new_domain() -> None:
    """新领域声明的意图能被策略识别，不需改图代码里的硬编码集。"""
    from agent_core.capabilities import ANSWER_CAPABILITY, AssistantCapabilities, missing_handlers

    registry = DomainRegistry.from_plugins([ExamplePlugin()], fallback=DomainRegistry.default())
    policy = IntentRoutingPolicy.from_registry(registry)
    # 领域归属由清单反查，不再维护 ``GENERAL_INTENTS`` 之类的常量集合。
    assert policy.domain_for('一般咨询') == 'general'
    assert policy.domain_for('示例问候') == 'example'
    assert policy.domain_for('未知意图', fallback='claims') == 'claims'

    # 声明了能力但没实现：装配期能查出缺口，运行期按 mode/risk 回落到内核能力。
    assert missing_handlers(registry, AssistantCapabilities()) == [('example_answer', 'example')]
    target = policy.resolve('example', '示例问候', {'question': '你好'})
    assert (target.domain, target.capability, target.handler) == ('example', ANSWER_CAPABILITY, None)

    # 注册后即成为落点，且内核不认识"example"这个名字。
    async def example_answer(state: dict) -> dict:
        return {'needs_llm_answer': True}

    capabilities = AssistantCapabilities()
    capabilities.register('example_answer', example_answer)
    armed = IntentRoutingPolicy.from_registry(registry, capabilities)
    target = armed.resolve('example', '示例问候', {'question': '你好'})
    assert (target.domain, target.capability) == ('example', 'example_answer')
    assert target.handler is example_answer


def test_create_app_accepts_explicit_intent_router() -> None:
    """显式注入 router 时跳过内核构建，且内核不关闭外部注入的实例。"""
    router = FakeRouter()
    with TestClient(create_app(database=':memory:', intent_router=router)) as client:
        assert client.app.state.intent_router is router
        assert client.app.state.intent_routing_policy is not None
        assert not getattr(router, 'closed', False)
    # lifespan 结束后仍不关闭外部注入的 router（生命周期归调用方）。
    assert not getattr(router, 'closed', False)


def test_create_app_aggregates_manifest_for_custom_registry() -> None:
    """只装 example 插件时，其领域与意图声明仍进入内核装配的清单。"""
    from agent_core.plugins import PluginRegistry

    with TestClient(create_app(database=':memory:', registry=PluginRegistry([ExamplePlugin()]))) as client:
        names = client.app.state.intent_registry.names()
        # create_app 用 YAML 回退补齐无插件领域（general），
        # 这里证明的是 example 的意图确实进入清单，且清单元数据来自插件。
        assert 'example' in names
        assert '示例问候' in {i.name for i in client.app.state.intent_registry.get('example').intents}

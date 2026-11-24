"""模板可插拔性回归：新增领域无需改动内核即可注册路由与装配服务。

这是"企业级 Agent 模板"的核心保证：内核只认 DomainPlugin 协议，
业务域通过 registry 注入。本测试用一个零依赖的 echo 领域证明这一点。
"""
from fastapi.testclient import TestClient

from agent_core.plugins import PluginRegistry
from app import create_app
from domains.example import ExamplePlugin


def test_custom_registry_registers_domain_routes() -> None:
    """只传自定义 registry，example 域的路由即可用，且不触碰理赔/投资。"""
    registry = PluginRegistry([ExamplePlugin()])
    with TestClient(create_app(database=':memory:', registry=registry)) as client:
        response = client.post('/api/example/echo', json={'text': '模板可插拔'})
        assert response.status_code == 200
        assert response.json() == {'echo': '模板可插拔', 'length': 5}

        # 自定义 registry 未装配的领域路由不应存在。
        assert client.post('/api/investment/process', json={}).status_code == 404


def test_default_registry_exposes_both_example_domains() -> None:
    """默认注册表包含理赔与投资两个示例域。"""
    from domains.registry import default_registry

    assert set(default_registry().names()) == {'claims', 'investment'}


def test_plugin_registry_rejects_duplicate_names() -> None:
    import pytest

    with pytest.raises(ValueError, match='重复注册'):
        PluginRegistry([ExamplePlugin(), ExamplePlugin()])

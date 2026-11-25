"""apps/ 薄入口：按助手裁剪领域，共享同一套内核装配。"""
from fastapi.testclient import TestClient

from apps.claims_app import build_app as build_claims_app
from apps.investment_app import build_app as build_investment_app
from apps.unified_app import build_app as build_unified_app


def test_claims_app_only_registers_claims_domain() -> None:
    with TestClient(build_claims_app(database=':memory:')) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 200
        assert client.get('/api/investment/process').status_code == 404

    names = set(client.app.state.capabilities.names())
    assert 'claims_answer' in names
    assert 'investment_answer' not in names


def test_investment_app_only_registers_investment_domain() -> None:
    with TestClient(build_investment_app(database=':memory:')) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 200
        assert client.get('/api/claims/process').status_code == 404

    names = set(client.app.state.capabilities.names())
    assert 'investment_answer' in names
    assert 'claims_answer' not in names


def test_unified_app_registers_all_default_domains() -> None:
    with TestClient(build_unified_app(database=':memory:')) as client:
        assert client.get('/ready').status_code == 200

    names = set(client.app.state.capabilities.names())
    assert {'claims_answer', 'investment_answer'} <= names

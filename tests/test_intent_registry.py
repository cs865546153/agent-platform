"""领域注册表测试：YAML 回退、插件聚合、校验与解析。"""
from __future__ import annotations

import pytest

from agent_core.intent.contracts import RouteDecision
from agent_core.intent.registry import DomainRegistry
from domains.registry import default_domain_registry


def test_default_registry_is_fallback_only() -> None:
    """YAML 只保留无插件领域；claims/investment 的事实源在插件 manifest。"""
    registry = DomainRegistry.default()
    assert set(registry.names()) == {"general"}
    assert registry.get("general").intent("一般咨询") is not None


def test_full_registry_aggregates_plugin_manifests() -> None:
    """完整清单由插件 manifest + YAML 回退聚合，不再由 YAML 独载。"""
    registry = default_domain_registry()
    assert set(registry.names()) == {"claims", "investment", "general"}
    assert registry.get("claims").criteria_text().startswith("Insurance claims")
    assert registry.get("investment").intent("交易指令").risk == "high"
    assert registry.get("claims").intent("保单查询") is not None


def test_duplicate_domain_is_rejected() -> None:
    registry = default_domain_registry()
    with pytest.raises(ValueError):
        registry.register(registry.get("claims"))


def test_resolve_rejects_unknown_intent() -> None:
    registry = default_domain_registry()
    with pytest.raises(ValueError):
        registry.resolve(RouteDecision(domain="claims", intent="不存在的意图", confidence=0.9))
    manifest = registry.resolve(RouteDecision(domain="claims", intent="理赔报案", confidence=0.9))
    assert manifest.name == "claims"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        DomainRegistry.from_yaml("definitely-missing-domains.yaml")


def test_cross_domain_intent_name_is_rejected() -> None:
    """路由结果只带意图名，跨领域同名意图必须显式报错而不是静默歧义。"""
    from agent_core.intent.contracts import DomainManifest, IntentSpec

    registry = DomainRegistry([
        DomainManifest(name="alpha", description="甲", intents=[
            IntentSpec(name="查询", description="查东西")]),
    ])
    with pytest.raises(ValueError, match="重名无法判定归属"):
        registry.register(DomainManifest(name="beta", description="乙", intents=[
            IntentSpec(name="查询", description="查别的东西")]))
    assert registry.intent_owner("查询") == "alpha"


def test_default_manifests_have_unique_intent_names() -> None:
    """插件 manifest + 回退清单聚合后意图名全局唯一（校验不会让启动失败）。"""
    registry = default_domain_registry()
    owners: dict[str, str] = {}
    for domain in registry.all():
        for spec in domain.intents:
            assert spec.name not in owners, f"{spec.name} 同时属于 {owners.get(spec.name)} 与 {domain.name}"
            owners[spec.name] = domain.name


def test_intent_logger_namespace_is_shared() -> None:
    """JEV/LLM/Hybrid/Factory 的日志必须落在同一命名空间，否则按 logger 名排障会漏。"""
    import agent_core.intent.factory as factory
    import agent_core.intent.hybrid as hybrid
    import agent_core.intent.jev as jev
    import agent_core.intent.llm as llm
    from agent_core.intent.contracts import INTENT_LOGGER_NAME

    names = {module.LOGGER.name for module in (factory, hybrid, jev, llm) if hasattr(module, "LOGGER")}
    assert names and names == {INTENT_LOGGER_NAME}
    assert "claim" not in INTENT_LOGGER_NAME

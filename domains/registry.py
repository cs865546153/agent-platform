"""领域插件注册表：唯一的"哪些领域启用"事实源。

模板使用者新增领域时，在这里登记一次；内核（app.py）通过注册表装配，
不直接 import 任何具体业务模块。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_core.plugins import DomainPlugin, PluginRegistry

if TYPE_CHECKING:
    from agent_core.intent.registry import DomainRegistry


def default_registry() -> PluginRegistry:
    """构建默认插件集。可按部署需要裁剪或替换。"""
    from domains.claims.plugin import ClaimsPlugin
    from domains.investment.plugin import InvestmentPlugin

    plugins: list[DomainPlugin] = [ClaimsPlugin(), InvestmentPlugin()]
    return PluginRegistry(plugins)


def default_domain_registry() -> "DomainRegistry":
    """聚合完整领域/意图清单：插件 manifest 优先，YAML 回退补齐无插件领域。

    供离线评测与契约测试使用。运行时装配由 ``create_app`` 用
    ``DomainRegistry.from_plugins(registry.all(), fallback=DomainRegistry.default())``
    完成，两者语义一致：插件是事实源，YAML 只补 general。
    """
    from agent_core.intent.registry import DomainRegistry

    return DomainRegistry.from_plugins(
        default_registry().all(), fallback=DomainRegistry.default()
    )


__all__ = ["default_registry", "default_domain_registry"]

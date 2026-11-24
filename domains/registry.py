"""领域插件注册表：唯一的"哪些领域启用"事实源。

模板使用者新增领域时，在这里登记一次；内核（app.py）通过注册表装配，
不直接 import 任何具体业务模块。
"""
from __future__ import annotations

from agent_core.plugins import DomainPlugin, PluginRegistry


def default_registry() -> PluginRegistry:
    """构建默认插件集。可按部署需要裁剪或替换。"""
    from domains.claims.plugin import ClaimsPlugin
    from domains.investment.plugin import InvestmentPlugin

    plugins: list[DomainPlugin] = [ClaimsPlugin(), InvestmentPlugin()]
    return PluginRegistry(plugins)


__all__ = ["default_registry"]

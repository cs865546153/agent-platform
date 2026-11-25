"""理赔助手入口：只装配 claims 领域。"""
from typing import Any

from agent_core.plugins import PluginRegistry
from app_factory import create_app
from domains.claims.plugin import ClaimsPlugin


def build_app(**kwargs: Any):
    """构建只含理赔领域的 FastAPI 应用，``kwargs`` 透传 ``create_app``。"""
    return create_app(registry=PluginRegistry([ClaimsPlugin()]), **kwargs)


app = build_app()

__all__ = ['app', 'build_app']

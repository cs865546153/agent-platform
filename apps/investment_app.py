"""投资助手入口：只装配 investment 领域。"""
from typing import Any

from agent_core.plugins import PluginRegistry
from app_factory import create_app
from domains.investment.plugin import InvestmentPlugin


def build_app(**kwargs: Any):
    """构建只含投资领域的 FastAPI 应用，``kwargs`` 透传 ``create_app``。"""
    return create_app(registry=PluginRegistry([InvestmentPlugin()]), **kwargs)


app = build_app()

__all__ = ['app', 'build_app']

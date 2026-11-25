"""统一工作台入口：装配全部默认领域（等价于 ``app.py``）。"""
from typing import Any

from app_factory import create_app
from domains.registry import default_registry


def build_app(**kwargs: Any):
    """构建含全部默认领域的 FastAPI 应用，``kwargs`` 透传 ``create_app``。"""
    return create_app(registry=default_registry(), **kwargs)


app = build_app()

__all__ = ['app', 'build_app']

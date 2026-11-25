"""应用入口：组合默认领域，通用装配见 ``app_factory.py``。

本文件只负责“启用哪些领域”，不承载路由实现；按助手裁剪请使用 ``apps/*_app.py``。
"""
from app_factory import (
    HealthResponse,
    build_assistant_intent_router,
    composite_slot_extractor,
    create_app,
)
from domains.registry import default_registry

app = create_app(registry=default_registry())

__all__ = [
    'HealthResponse',
    'app',
    'build_assistant_intent_router',
    'composite_slot_extractor',
    'create_app',
]

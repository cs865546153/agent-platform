"""领域插件基类与共用辅助：减少样板，避免每个领域重写 LangGraph 中断解析。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from agent_core.plugins import AppContext


def graph_config(thread_id: str, *, recursion_limit: int | None = None) -> dict[str, Any]:
    """构造 LangGraph 调用配置；thread_id 即领域案件主键。"""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def interrupt_value(snapshot: Any) -> dict[str, Any] | None:
    """兼容 StateSnapshot 与 ainvoke 返回值中的 interrupt 表达。"""
    if isinstance(snapshot, dict):
        for item in snapshot.get("__interrupt__") or ():
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                return value
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return None


def interrupt_kind(snapshot: Any) -> str | None:
    value = interrupt_value(snapshot)
    return str(value.get("kind")) if value else None


class DomainPluginBase:
    """实现 ``DomainPlugin`` 协议的轻量基类（结构化子类型，无需显式继承）。

    子类覆盖 ``name`` 与 ``build_services`` / ``build_graphs`` /
    ``register_routes``；未覆盖的方法为空操作。
    """

    name: str = "unnamed"
    tags: list[str] = []
    _manifest_cache: Any = None

    @classmethod
    def _manifest_path(cls) -> Path:
        """插件目录下的 manifest.yaml；子类可覆盖以自定义位置。"""
        import inspect

        return Path(inspect.getfile(cls)).resolve().parent / "manifest.yaml"

    def intent_manifest(self) -> Any:
        """从插件同目录的 ``manifest.yaml`` 加载领域声明，缺失则返回 ``None``。

        结果按类缓存。子类也可直接覆盖本方法返回内存中的 ``DomainManifest``，
        适合不想落 YAML 的领域。
        """
        if self._manifest_cache is None:
            path = self._manifest_path()
            if not path.exists():
                return None
            # 延迟 import：避免 agent_core.intent <-> agent_core.plugins 循环依赖。
            from agent_core.intent.contracts import DomainManifest

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self._manifest_cache = DomainManifest.model_validate(raw)
        return self._manifest_cache

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:  # noqa: ANN401
        return None

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:  # noqa: ANN401
        return None

    def register_routes(self, ctx: AppContext) -> None:
        return None

    async def close(self) -> None:
        return None


# 用注册时的结构化校验替代 issubclass：Protocol 带数据成员时不支持运行时 issubclass。


__all__ = ["DomainPluginBase", "graph_config", "interrupt_value", "interrupt_kind"]

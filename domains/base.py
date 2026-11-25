"""领域插件基类与共用辅助：减少样板，避免每个领域重写 LangGraph 中断解析。"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import yaml  # type: ignore[import-untyped]
from fastapi import HTTPException

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


def resume_gate(snapshot: Any,
                derive: Callable[[dict[str, Any], tuple[str, ...]], str | None]) -> str | None:
    """当前停留的中断阶段，兼容进程重启。

    中断载荷（``kind``）只在同一进程内可靠：LangGraph 重启后不再重建
    ``tasks[].interrupts``，快照里只剩状态和 ``next`` 中的待执行节点。响应体如果
    直接依赖 ``interrupt_kind``，重启后就会把"等待二次确认"这样的关键状态告诉
    客户端为"没有中断"。这里让领域用 ``(状态, 待执行节点)`` 自行重建。
    """
    kind = interrupt_kind(snapshot)
    if kind is not None:
        return kind
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return None
    return derive(values, tuple(snapshot.next or ()))


async def gate_resume(graph: Any, config: dict[str, Any], *, kind: str,
                      not_found: str, conflict: str,
                      node: str | None = None,
                      still_waiting: Callable[[dict[str, Any]], bool] | None = None) -> Any:
    """恢复态门控：确认图正停在 ``kind`` 中断点，并返回快照。

    每个领域都有"补材料 / 二次确认 / 人工审批"这类 HITL 恢复入口，门控完全同形：
    会话不存在 404，没停在期望阶段 409。放在基类里，避免各领域各写一份判定。

    ``kind`` 只在**同一进程**内可靠：LangGraph 重启后不会重建 ``tasks[].interrupts``，
    快照里只剩 ``next`` 里的待执行节点名。所以领域可以额外给出 ``node``（中断所在的
    节点）作为重启后的判据；同一节点承载多种中断时，再用 ``still_waiting`` 从状态里
    重建"确实还停在这一种中断上"的条件，避免重启后把恢复值喂给错误的中断点。
    """
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(404, not_found)
    if interrupt_kind(snapshot) == kind:
        return snapshot
    if (node is not None and node in (snapshot.next or ())
            and (still_waiting is None or still_waiting(snapshot.values))):
        return snapshot
    raise HTTPException(409, conflict)


class DomainPluginBase:
    """实现 ``DomainPlugin`` 协议的轻量基类（结构化子类型，无需显式继承）。

    子类覆盖 ``name`` 与 ``build_services`` / ``build_graphs`` /
    ``register_routes``；未覆盖的方法为空操作。
    """

    name: str = "unnamed"
    tags: list[str] = []
    # 本插件在 build_graphs 中会写入 ctx.graphs 的 key。就绪探针据此核对
    # "声明的图都装配成功"，而不是只看"至少有一个图"。无图领域保持为空。
    graph_keys: tuple[str, ...] = ()
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


__all__ = ["DomainPluginBase", "graph_config", "gate_resume", "interrupt_value",
           "interrupt_kind", "resume_gate"]

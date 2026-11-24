"""领域插件契约：模板内核与业务领域之间唯一的扩展点。

新增一个业务领域只需实现 :class:`DomainPlugin` 并在 ``domains/registry.py`` 登记，
无需修改内核（app.py / agent_core / 接入层）。领域插件可以声明：

- ``build_services``：按配置装配领域服务（真实后端 / demo 后端）；
- ``build_graphs``：在 lifespan 中基于 checkpointer 构建 LangGraph；
- ``register_routes``：注册该领域专属的 HTTP 路由；
- ``intent_manifest``：声明该领域的意图清单，供意图路由器识别（可选）；
- ``close``：释放领域持有的资源（可选）。

内核只认识这里定义的协议，不认识理赔、投资或任何具体业务概念。
领域与意图的事实源随插件走；``configs/domains.yaml`` 仅作未声明插件的回退。
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI


@dataclass
class AppContext:
    """内核装配完成后传给领域插件的运行时上下文。

    插件通过它访问注入的模型、服务、RAG 和共享的锁，而不必 import app.py，
    从而避免内核与领域之间的循环依赖。
    """

    application: FastAPI
    settings: Any
    static_dir: Path
    limiter: Any
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)
    rag: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    intent_router: Any = None
    intent_registry: Any = None

    @property
    def state(self) -> Any:
        return self.application.state

    def lock_for(self, key: str) -> asyncio.Lock:
        """按业务主键取锁（进程内）；插件不应自行创建锁集合。"""
        return self.locks.setdefault(key, asyncio.Lock())


@runtime_checkable
class DomainPlugin(Protocol):
    """领域插件协议。所有方法都是同步的，异步资源在 lifespan 内构建。"""

    name: str

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        """装配领域服务；``overrides`` 是测试注入的替身。"""
        ...

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        """在 lifespan 中构建领域 LangGraph 并写入 ``ctx.graphs``。"""
        ...

    def register_routes(self, ctx: AppContext) -> None:
        """向 ``ctx.application`` 注册该领域的路由。"""
        ...

    def intent_manifest(self) -> Any:
        """返回该领域的 ``DomainManifest``；未声明则返回 ``None``。

        可选 hook：内核据此聚合领域清单供 JEV/LLM 意图路由使用，
        使新增领域无需修改全局 ``configs/domains.yaml``。
        """
        ...

    async def close(self) -> None:
        """释放领域资源（可选，默认无操作）。"""
        ...


class PluginRegistry:
    """启动时构建、运行时只读的领域插件注册表。"""

    def __init__(self, plugins: Iterable[DomainPlugin] | None = None) -> None:
        self._plugins: dict[str, DomainPlugin] = {}
        for plugin in plugins or ():
            self.register(plugin)

    def register(self, plugin: DomainPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"领域插件重复注册：{plugin.name}")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> DomainPlugin:
        try:
            return self._plugins[name]
        except KeyError:
            raise KeyError(f"未注册领域插件：{name}") from None

    def all(self) -> list[DomainPlugin]:
        return list(self._plugins.values())

    def names(self) -> list[str]:
        return list(self._plugins)


__all__ = ["AppContext", "DomainPlugin", "PluginRegistry"]

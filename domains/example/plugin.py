"""示例领域插件：无状态 echo 服务，演示插件契约的最小实现。

这个领域故意不接模型、不接向量库，便于模板使用者抄改。它展示了三件事：
1. 用 ``build_services`` 装配领域依赖；
2. 用 ``register_routes`` 注册领域路由（这里只有一个只读 echo）；
3. 通过 ``ctx.settings`` / ``ctx.application.state`` 访问内核能力。
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends

from agent_core.plugins import AppContext
from agent_core.security import authenticate
from domains.base import DomainPluginBase


class ExampleEchoService:
    """领域服务示例：真实项目里这里会是业务适配器或领域模型。"""

    @staticmethod
    def echo(text: str) -> dict[str, Any]:
        return {"echo": text, "length": len(text)}


class ExamplePlugin(DomainPluginBase):
    name = "example"
    tags: list = ["示例域"]

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        ctx.services[self.name] = overrides.get("example_service") or ExampleEchoService()

    def register_routes(self, ctx: AppContext) -> None:
        application = ctx.application

        @application.post("/api/example/echo", dependencies=[Depends(authenticate)], tags=self.tags)
        async def example_echo(payload: dict[str, Any]) -> dict[str, Any]:
            service: ExampleEchoService = ctx.services[self.name]
            return service.echo(str(payload.get("text", "")))


__all__ = ["ExamplePlugin", "ExampleEchoService"]

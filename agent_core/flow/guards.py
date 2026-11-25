"""控制层守卫：循环预算、无进展检测与幂等存储。

这些是业务控制层的通用骨架，不包含任何领域规则；业务节点只负责产出
``Observation`` / ``Decision``，守卫负责在越界时给出明确终止原因。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from agent_core.harness import LoopPolicy

from .contracts import StopReason


@dataclass
class LoopGuard:
    """累计步数/工具调用/无进展，并给出结构化终止原因。

    默认 ``max_steps=20``，避免业务图忘记配置时无限循环；领域可注入自己的
    ``LoopPolicy``。守卫只报告终止原因，不负责跳转或转人工。
    """

    policy: LoopPolicy = field(default_factory=lambda: LoopPolicy(max_steps=20))
    max_no_progress: int = 2
    steps: int = 0
    tool_calls: int = 0
    no_progress_count: int = 0
    started_at: float = field(default_factory=time.monotonic)
    clock: Callable[[], float] = time.monotonic

    def start_step(self) -> int:
        self.steps += 1
        return self.steps

    def record_tool_calls(self, count: int = 1) -> int:
        if count < 0:
            raise ValueError('工具调用次数不能为负数')
        self.tool_calls += count
        return self.tool_calls

    def record_progress(self, progressed: bool) -> int:
        if progressed:
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1
        return self.no_progress_count

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def check(self) -> StopReason | None:
        """返回第一个命中的终止原因；未命中则返回 ``None``。"""
        if self.policy.max_steps is not None and self.steps >= self.policy.max_steps:
            return StopReason.MAX_STEPS
        if self.policy.max_tool_calls is not None and self.tool_calls >= self.policy.max_tool_calls:
            return StopReason.MAX_TOOL_CALLS
        if self.policy.deadline_s is not None and self.elapsed_s >= self.policy.deadline_s:
            return StopReason.DEADLINE
        if self.max_no_progress > 0 and self.no_progress_count >= self.max_no_progress:
            return StopReason.NO_PROGRESS
        return None


class IdempotencyStore(Protocol):
    """写操作幂等存储；生产实现必须保证 ``run_once`` 的原子性。"""

    async def get(self, key: str) -> Any | None:
        ...

    async def put(self, key: str, value: Any) -> None:
        ...

    async def run_once(self, key: str,
                       factory: Callable[[], Awaitable[Any]]) -> tuple[Any, bool]:
        """返回 ``(结果, 是否重放)``；同一 key 只执行一次。"""
        ...


@dataclass
class InMemoryIdempotencyStore:
    """进程内幂等存储，仅用于单进程/测试；多副本应换 Redis 等原子存储。"""

    _values: dict[str, Any] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get(self, key: str) -> Any | None:
        return self._values.get(key)

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            self._values[key] = value

    async def run_once(self, key: str,
                       factory: Callable[[], Awaitable[Any]]) -> tuple[Any, bool]:
        async with self._lock:
            if key in self._values:
                return self._values[key], True
            value = await factory()
            self._values[key] = value
            return value, False


__all__ = ["IdempotencyStore", "InMemoryIdempotencyStore", "LoopGuard"]

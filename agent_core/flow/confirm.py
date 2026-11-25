"""确认闸门：待确认动作的版本、TTL、载荷与幂等校验。

确认属于业务控制层，不交给模型判断。执行面策略只负责超时与预算；本模块只做
确定性校验，并在执行写操作前调用幂等存储。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .contracts import ConfirmCheck, ConfirmCode, PendingAction, PendingStatus
from .guards import IdempotencyStore


def payload_hash(payload: Any) -> str:
    """对载荷做稳定哈希；用于确认时检测载荷是否被替换。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                           default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_pending_action(action: PendingAction, *, current_version: int, payload_hash: str,
                            now: float | None = None) -> ConfirmCheck:
    """确认前校验：状态、过期、版本、载荷；任一失败都不得执行写操作。"""
    if action.status is not PendingStatus.PENDING:
        return ConfirmCheck(False, ConfirmCode.NOT_PENDING,
                            f'待确认动作状态为 {action.status.value}，不能重复确认')
    if action.is_expired(now=now):
        return ConfirmCheck(False, ConfirmCode.EXPIRED, '待确认动作已过期')
    if int(action.target_version) != int(current_version):
        return ConfirmCheck(False, ConfirmCode.VERSION_MISMATCH,
                            f'目标版本 {action.target_version} 不是当前版本 {current_version}')
    if action.payload_hash != payload_hash:
        return ConfirmCheck(False, ConfirmCode.PAYLOAD_MISMATCH, '确认载荷与待确认时不一致')
    return ConfirmCheck(True)


@dataclass
class ConfirmExecutor:
    """在确认校验通过后执行一次写操作；相同幂等键只执行一次。"""

    store: IdempotencyStore

    async def execute(self, action: PendingAction, *,
                      current_version: int,
                      payload_hash: str,
                      execute: Callable[[PendingAction], Awaitable[Any]],
                      now: float | None = None) -> tuple[ConfirmCheck, Any | None]:
        check = validate_pending_action(
            action, current_version=current_version, payload_hash=payload_hash,
            now=time.time() if now is None else now,
        )
        if not check.ok:
            return check, None

        result, replayed = await self.store.run_once(
            action.idempotency_key, lambda: execute(action),
        )
        if replayed:
            return ConfirmCheck(True, ConfirmCode.IDEMPOTENT_REPLAY, '重复确认，返回首次执行结果'), result
        return check, result


__all__ = ["ConfirmExecutor", "payload_hash", "validate_pending_action"]

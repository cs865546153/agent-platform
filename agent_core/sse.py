"""SSE 事件编码：领域无关，只负责把 payload 转成标准 SSE 帧。"""
from __future__ import annotations

import json
from typing import Any


def sse_frame(payload: dict[str, Any]) -> str:
    """把一个事件字典编码为 ``data: ...\\n\\n``，保证中文不被转义。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_headers() -> dict[str, str]:
    """流式响应头：禁用缓存与 nginx 缓冲。"""
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


__all__ = ["sse_frame", "stream_headers"]

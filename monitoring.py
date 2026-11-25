"""Prometheus 指标：标签不使用案件号，避免高基数。

指标前缀可通过 ``METRIC_PREFIX`` 配置（默认 ``claims`` 兼容既有部署）；
领域插件用 ``register_counter`` / ``register_histogram`` 注册自己的指标，
通过 ``domain`` 参数隔离命名空间，内核不写死任何业务域名。
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()
METRIC_PREFIX = (os.getenv('METRIC_PREFIX', '') or 'claims').strip()


def metric_name(name: str, *, domain: str | None = None) -> str:
    """生成指标名；``domain`` 非空时覆盖全局前缀，供插件隔离命名空间。"""
    prefix = domain or METRIC_PREFIX
    return f'{prefix}_{name}' if prefix else name


def register_counter(
    name: str,
    documentation: str,
    labels: Iterable[str] = (),
    *,
    domain: str | None = None,
) -> Counter:
    """领域插件注册自己的 Counter；名称自动加前缀。"""
    return Counter(metric_name(name, domain=domain), documentation, list(labels), registry=REGISTRY)


def register_histogram(
    name: str,
    documentation: str,
    *,
    buckets: tuple[float, ...],
    labels: Iterable[str] = (),
    domain: str | None = None,
) -> Histogram:
    """领域插件注册自己的 Histogram；名称自动加前缀。"""
    return Histogram(
        metric_name(name, domain=domain), documentation,
        buckets=buckets, labelnames=list(labels), registry=REGISTRY,
    )


REQUESTS = register_counter('requests_total', '已完成HTTP请求数', ['decision'])
ERRORS = register_counter('errors_total', '处理异常数', ['kind'])
LATENCY = register_histogram('latency_seconds', '处理耗时',
                             buckets=(.1, .5, 1, 2, 5, 10, 30, 120))
CONFIDENCE = register_histogram('confidence', '已计算置信度分布',
                                buckets=(.1, .3, .5, .7, .9, 1))
TOKENS = register_counter('tokens_total', '服务端实际报告token用量', ['kind'])
ITERATIONS = register_histogram('iterations', '每案专家及补充轮次',
                                buckets=(1, 2, 3, 4, 6, 8, 12))
CACHE = register_counter('cache_total', '缓存访问结果', ['result'])


def metrics() -> bytes:
    """导出本进程累计指标，不推测未知的服务端token数量。"""
    return generate_latest(REGISTRY)

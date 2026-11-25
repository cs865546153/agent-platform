"""对话编排能力注册表：内核按"能力名"分派，领域插件按"能力名"注册实现。

内核认识的能力只有两个内置名字：

- ``answer``：用通用模型直接回答（兜底，任何领域都能回落到它）；
- ``guard``：高风险 / 案件型意图的拦截确认（不自动执行任何写操作）。

其余能力名一律由领域清单的 ``IntentSpec.capability`` 声明、由领域插件在
``build_services`` 里注册实现。对话图按清单动态选节点，因此新增领域只需
"声明意图 + 注册处理器"，不必改 ``agents/assistant_agent.py``。

声明了能力却没有注册处理器时，按 ``mode`` / ``risk`` 回落到内置能力，
宁可让人工确认，也不让未实现的分支悄悄变成自由问答。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Awaitable

from .intent.contracts import fallback_capability

# 处理器就是对话图的一个节点：读状态、返回状态补丁（events / llm / messages ...）。
AssistantHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Redirect:
    """领域对一次路由结果的改写：落到哪个领域的哪个能力，以及给用户看的依据。

    内核不判断改写是否合理，只按 ``(domain, capability)`` 分派；
    ``evidence`` 会并进 ``route`` trace，便于在工作台复核改写原因。
    """

    domain: str
    capability: str
    evidence: dict[str, Any] = field(default_factory=dict)


# 意图改写钩子：领域可以基于会话状态（如材料补充轮次）改写落点。
RedirectHook = Callable[[dict[str, Any]], "Redirect | None"]

ANSWER_CAPABILITY = 'answer'
GUARD_CAPABILITY = 'guard'
KERNEL_CAPABILITIES: tuple[str, ...] = (ANSWER_CAPABILITY, GUARD_CAPABILITY)


class AssistantCapabilities:
    """各领域对对话图的编排贡献；键为能力名（全局唯一）。"""

    def __init__(self, handlers: dict[str, AssistantHandler] | None = None,
                 redirects: dict[str, RedirectHook] | None = None) -> None:
        self._handlers = dict(handlers or {})
        self._redirects = dict(redirects or {})

    def register(self, capability: str, handler: AssistantHandler, *, replace: bool = False) -> None:
        """注册一个能力实现。

        能力名是对话图的节点名，两个领域注册同名能力会互相覆盖且难以排查，
        因此显式报错；确要接管内核内置能力（如替换默认 ``answer``）时传 ``replace``。
        """
        if capability in self._handlers and not replace:
            raise ValueError(
                f"对话能力重复注册：{capability}。能力名即对话图节点名，"
                "请为领域能力加业务前缀（如 claims_answer），或显式传 replace=True 接管"
            )
        self._handlers[capability] = handler

    def register_redirect(self, domain: str, hook: RedirectHook) -> None:
        """注册领域自己的意图改写钩子（按领域名保留一个，可覆盖）。"""
        self._redirects[domain] = hook

    def get(self, capability: str) -> AssistantHandler | None:
        return self._handlers.get(capability)

    def names(self) -> list[str]:
        return list(self._handlers)

    def with_kernel(self, *, answer: AssistantHandler, guard: AssistantHandler) -> "AssistantCapabilities":
        """补齐内核内置能力，返回新实例。

        注册表在应用装配期被各插件写入，而内置处理器持有本次请求的模型，
        因此这里复制而不是就地修改，避免跨请求共享可变状态。
        """
        merged = AssistantCapabilities(self._handlers, self._redirects)
        merged._handlers.setdefault(ANSWER_CAPABILITY, answer)
        merged._handlers.setdefault(GUARD_CAPABILITY, guard)
        return merged

    def redirect(self, state: dict[str, Any]) -> "Redirect | None":
        """依次询问领域改写钩子，第一个命中且已注册处理器的改写生效。"""
        for hook in self._redirects.values():
            hit = hook(state)
            if hit is None:
                continue
            if not isinstance(hit, Redirect):
                raise TypeError(f"意图改写钩子必须返回 Redirect，收到 {type(hit).__name__}")
            if hit.capability not in self._handlers:
                # 目标能力没实现，改写等于无效；继续走清单派生的落点，比静默失败安全。
                continue
            return hit
        return None

    def dispatch(self, registry: Any, domain: str | None,
                 intent: str | None) -> tuple[str, AssistantHandler | None]:
        """把 ``(domain, intent)`` 解析成 ``(能力名, 处理器)``。

        清单没声明、领域未知或处理器缺失时，回落到 ``mode`` / ``risk`` 推导的
        内置能力，保证任何路由结果都有确定的落点。
        """
        spec = _spec_of(registry, domain, intent)
        if spec is not None and spec.capability:
            handler = self._handlers.get(spec.capability)
            if handler is not None:
                return spec.capability, handler
        capability = (fallback_capability(spec.mode, spec.risk) if spec is not None
                      else ANSWER_CAPABILITY)
        return capability, self._handlers.get(capability)


def _spec_of(registry: Any, domain: str | None, intent: str | None) -> Any:
    if registry is None or not domain or not intent:
        return None
    try:
        manifest = registry.get(domain)
    except KeyError:
        return None
    return manifest.intent(intent)


def capability_of(registry: Any, domain: str | None, intent: str | None,
                  *, default: str = ANSWER_CAPABILITY) -> str:
    """只取能力名（不查处理器），供装配校验与日志使用。"""
    spec = _spec_of(registry, domain, intent)
    if spec is None:
        return default
    return spec.capability or fallback_capability(spec.mode, spec.risk)


def domain_of_intent(registry: Any, intent: str | None) -> str | None:
    """反查意图归属的领域；内置兜底分类只返回意图名时用它补齐领域。"""
    if registry is None or not intent:
        return None
    owner = getattr(registry, 'intent_owner', None)
    return owner(intent) if callable(owner) else None


def declared_capabilities(registry: Any) -> Iterable[str]:
    """领域清单声明的能力名（不含内核内置能力）。"""
    if registry is None:
        return ()
    return sorted({spec.capability for manifest in registry.all() for spec in manifest.intents
                   if spec.capability and spec.capability not in KERNEL_CAPABILITIES})


def missing_handlers(registry: Any, capabilities: AssistantCapabilities) -> list[tuple[str, str]]:
    """清单声明了能力但没人实现的 ``(capability, domain)`` 列表。

    装配期用它打 warning：这类缺口不会报错，只会静默回落到内置落点，
    是"新增领域忘记注册处理器"最常见的表现。内核能力（``answer`` / ``guard``）
    由 ``with_kernel`` 在每次建图时补齐，不参与装配期检查。
    """
    if registry is None:
        return []
    gaps: list[tuple[str, str]] = []
    for manifest in registry.all():
        for spec in manifest.intents:
            if not spec.capability or spec.capability in KERNEL_CAPABILITIES:
                continue
            if capabilities.get(spec.capability) is None:
                gaps.append((spec.capability, manifest.name))
    return gaps


__all__ = [
    "ANSWER_CAPABILITY",
    "GUARD_CAPABILITY",
    "KERNEL_CAPABILITIES",
    "AssistantCapabilities",
    "AssistantHandler",
    "Redirect",
    "RedirectHook",
    "capability_of",
    "declared_capabilities",
    "domain_of_intent",
    "missing_handlers",
]

"""Agent 执行面的可替换策略：循环、上下文、工具、预算与钩子。

路由面（接入 → 意图 → 领域）由 ``agent_core.intent`` / ``agent_core.capabilities``
负责；本模块只描述"选定领域之后怎么执行"。默认策略保持现有单次流式行为，
领域可以只替换其中一个维度，也可以整体覆盖 ``Harness.stream_answer``。
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import Any

MessageHook = Callable[[list[Any]], list[Any]]
ChunkHook = Callable[[str], str]
EventHook = Callable[[dict[str, Any]], dict[str, Any]]


def message_text(message: Any) -> str:
    """把 LangChain 消息或兼容对象规范化为纯文本，不依赖具体消息类。"""
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            str(item.get('text', '')) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def estimate_tokens(messages: Iterable[Any], *, chars_per_token: int = 4) -> int:
    """粗估 token 数；不引入 tokenizer 依赖，领域可换成精确计数。"""
    per_token = max(1, chars_per_token)
    return sum(max(1, len(message_text(message)) // per_token) + 4 for message in messages)


def _is_system(message: Any) -> bool:
    return getattr(message, 'type', None) == 'system' or getattr(message, 'role', None) == 'system'


@dataclass(frozen=True)
class LoopPolicy:
    """循环与终止条件；默认不设上限，沿用现有图/执行器行为。"""

    max_steps: int | None = None
    max_tool_calls: int | None = None
    deadline_s: float | None = None

    def allow(self, *, steps: int = 0, tool_calls: int = 0, elapsed_s: float = 0.0) -> bool:
        if self.max_steps is not None and steps >= self.max_steps:
            return False
        if self.max_tool_calls is not None and tool_calls >= self.max_tool_calls:
            return False
        if self.deadline_s is not None and elapsed_s >= self.deadline_s:
            return False
        return True


@dataclass(frozen=True)
class ContextPolicy:
    """上下文裁剪：保留系统消息，按消息条数或 token 预算丢弃最旧的历史。

    只做确定性裁剪，不调用模型、不修改入参对象；需要摘要压缩时由领域
    在 ``HarnessHooks.before_model`` 里替换成自己的实现。
    """

    max_messages: int | None = None
    max_tokens: int | None = None
    chars_per_token: int = 4
    preserve_system: bool = True

    def apply(self, messages: Iterable[Any]) -> list[Any]:
        items = list(messages)
        if self.max_messages is not None:
            limit = max(0, self.max_messages)
            if self.preserve_system:
                system = [message for message in items if _is_system(message)]
                rest = [message for message in items if not _is_system(message)]
                items = system + (rest[-limit:] if limit else [])
            else:
                items = items[-limit:] if limit else []
        if self.max_tokens is not None:
            while estimate_tokens(items, chars_per_token=self.chars_per_token) > self.max_tokens:
                removable = [
                    index for index, message in enumerate(items)
                    if not (self.preserve_system and _is_system(message))
                ]
                # 至少保留一条非系统消息，避免裁剪成只剩系统提示。
                if len(removable) <= 1:
                    break
                del items[removable[0]]
        return items


@dataclass(frozen=True)
class ToolPolicy:
    """工具白名单/黑名单与调用次数上限；不改变工具实现本身。"""

    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()
    max_calls: int | None = None

    def allows(self, name: str) -> bool:
        if name in self.deny:
            return False
        return self.allow is None or name in self.allow


@dataclass(frozen=True)
class BudgetPolicy:
    """单次执行的 token / 成本 / 时间预算；由运行时按需读取。"""

    max_tokens: int | None = None
    max_cost: float | None = None
    max_seconds: float | None = None

    @property
    def timeout_s(self) -> float | None:
        return self.max_seconds


@dataclass
class HarnessHooks:
    """执行前后的扩展点；默认全部为空，不改变现有行为。"""

    before_model: MessageHook | None = None
    on_chunk: ChunkHook | None = None
    on_event: EventHook | None = None


class Harness:
    """一个领域的执行策略。

    默认实现只做三件事：按 ``context`` 裁剪消息、调用模型流式输出、按 ``hooks``
    改写内容。领域可以只替换策略，也可以继承并重写 ``stream_answer`` /
    ``run_graph`` 来接管完整循环。
    """

    def __init__(
        self,
        *,
        loop: LoopPolicy | None = None,
        context: ContextPolicy | None = None,
        tools: ToolPolicy | None = None,
        budget: BudgetPolicy | None = None,
        hooks: HarnessHooks | None = None,
    ) -> None:
        self.loop = loop or LoopPolicy()
        self.context = context or ContextPolicy()
        self.tools = tools or ToolPolicy()
        self.budget = budget or BudgetPolicy()
        self.hooks = hooks or HarnessHooks()

    def prepare_messages(self, messages: Iterable[Any]) -> list[Any]:
        items = list(messages)
        if self.hooks.before_model is not None:
            items = list(self.hooks.before_model(items))
        return self.context.apply(items)

    def transform_chunk(self, text: str) -> str:
        if self.hooks.on_chunk is None:
            return text
        return self.hooks.on_chunk(text)

    def transform_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.hooks.on_event is None:
            return event
        return self.hooks.on_event(event)

    async def run_graph(self, graph: Any, state: dict[str, Any]) -> Any:
        """运行图并应用时间预算；默认无超时，保持现有调用语义。"""
        timeout = self.budget.timeout_s or self.loop.deadline_s
        if timeout is None:
            return await graph.ainvoke(state)
        return await asyncio.wait_for(graph.ainvoke(state), timeout)

    async def stream_answer(self, state: dict[str, Any]) -> AsyncIterator[str]:
        """默认回答执行器：裁剪消息后逐块产出模型文本。"""
        llm = state.get('llm')
        if llm is None:
            return
        messages = self.prepare_messages(state.get('messages') or [])
        async for chunk in llm.astream(messages):
            text = message_text(chunk)
            if not text:
                continue
            text = self.transform_chunk(text)
            if text:
                yield text


class HarnessRegistry:
    """按领域名查找执行策略；``'*'`` 为默认策略，领域注册优先。"""

    DEFAULT_KEY = '*'

    def __init__(self, default: Harness | None = None) -> None:
        self._default = default or Harness()
        self._harnesses: dict[str, Harness] = {}

    def register(self, domain: str, harness: Harness, *, replace: bool = False) -> None:
        if domain in self._harnesses and not replace:
            raise ValueError(f'执行策略重复注册：{domain}')
        self._harnesses[domain] = harness

    def get(self, domain: str | None = None) -> Harness:
        if domain and domain in self._harnesses:
            return self._harnesses[domain]
        return self._harnesses.get(self.DEFAULT_KEY, self._default)

    def set_default(self, harness: Harness) -> None:
        self._default = harness

    @property
    def default(self) -> Harness:
        return self._default

    def domains(self) -> list[str]:
        return sorted(self._harnesses)


__all__ = [
    "BudgetPolicy",
    "ChunkHook",
    "ContextPolicy",
    "EventHook",
    "Harness",
    "HarnessHooks",
    "HarnessRegistry",
    "LoopPolicy",
    "MessageHook",
    "ToolPolicy",
    "estimate_tokens",
    "message_text",
]

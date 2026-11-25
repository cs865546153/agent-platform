"""LangGraph 子图适配：把通用 FlowNodes 拆成可 checkpoint 的图节点。

用法：

```python
graph = build_flow_graph(RenewalNodes(...), checkpointer=saver)
await graph.ainvoke(new_state(thread_id, ...), config)
```

适配层只负责控制流：``plan → assemble → execute → observe → decide``。
确认场景拆成 ``request_confirm``（提交待确认动作）和 ``wait_confirm``（interrupt）
两个节点，避免在同一个节点里中断导致待确认动作未写入 checkpoint。
领域节点仍实现 ``FlowNodes``；执行面资源限制继续由 ``LoopPolicy`` 提供。
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_core.harness import LoopPolicy

from .context import ContextPack
from .contracts import (
    Decision,
    DecisionKind,
    Execution,
    FlowStatus,
    Observation,
    Plan,
    StopReason,
)
from .engine import FlowNodes
from .guards import LoopGuard
from .state import FlowState, ensure_state


def _ensure(raw: FlowState) -> FlowState:
    return ensure_state(dict(raw))


def _record(state: FlowState, event: str, **data: Any) -> None:
    state.setdefault('trace_events', []).append({
        'event': event,
        'step': int(state.get('step_count') or 0),
        'data': data,
        'timestamp': time.time(),
    })


def _guard(state: FlowState, loop: LoopPolicy, max_no_progress: int) -> LoopGuard:
    now = time.time()
    return LoopGuard(
        policy=loop,
        max_no_progress=max_no_progress,
        steps=int(state.get('step_count') or 0),
        tool_calls=int(state.get('tool_calls') or 0),
        no_progress_count=int(state.get('no_progress_count') or 0),
        started_at=float(state.get('started_at') or now),
        clock=time.time,
    )


def _stop(state: FlowState, reason: StopReason) -> FlowState:
    state['status'] = FlowStatus.ESCALATED.value
    state['stop_reason'] = reason.value
    _record(state, 'escalated', reason=reason.value)
    return state


def _plan_node(nodes: FlowNodes, loop: LoopPolicy, max_no_progress: int):
    async def plan_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        guard = _guard(state, loop, max_no_progress)
        stop = guard.check()
        if stop is not None:
            state['step_count'] = guard.steps
            state['tool_calls'] = guard.tool_calls
            return _stop(state, stop)

        guard.start_step()
        state['step_count'] = guard.steps
        state['status'] = FlowStatus.PLANNING.value
        plan = await nodes.plan(state)
        state['plan'] = plan.as_dict()
        _record(state, 'plan_created', actions=len(plan.actions),
                missing=list(plan.missing), reason=plan.reason)
        if plan.missing:
            state['status'] = FlowStatus.WAITING_USER.value
            state['decision'] = Decision(
                DecisionKind.WAIT_USER, reason='missing_info', missing=plan.missing,
            ).as_dict()
            _record(state, 'waiting_user', missing=list(plan.missing))
        else:
            state['status'] = FlowStatus.ASSEMBLING.value
        return state

    return plan_node


def _assemble_node(nodes: FlowNodes):
    async def assemble_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        plan = Plan.from_dict(dict(state.get('plan') or {}))
        context = await nodes.assemble(state, plan)
        state['context'] = context.as_dict()
        state['status'] = FlowStatus.EXECUTING.value
        _record(state, 'context_assembled', message_count=context.message_count,
                facts=list(context.facts), versions=context.versions,
                trimmed=context.trimmed)
        return state

    return assemble_node


def _execute_node(nodes: FlowNodes):
    async def execute_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        context = ContextPack.from_dict(dict(state.get('context') or {}))
        execution = await nodes.execute(state, context)
        if execution.tool_calls < 0:
            raise ValueError('工具调用次数不能为负数')
        state['result'] = execution.as_dict()
        state['tool_calls'] = int(state.get('tool_calls') or 0) + execution.tool_calls
        state['status'] = FlowStatus.OBSERVING.value
        _record(state, 'tool_called', tool_calls=execution.tool_calls,
                errors=list(execution.errors))
        return state

    return execute_node


def _observe_node(nodes: FlowNodes):
    async def observe_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        execution = Execution.from_dict(dict(state.get('result') or {}))
        observation = await nodes.observe(state, execution)
        state['observation'] = observation.as_dict()
        if observation.made_progress:
            state['no_progress_count'] = 0
        else:
            state['no_progress_count'] = int(state.get('no_progress_count') or 0) + 1
        _record(state, 'observation_created', kind=observation.kind.value,
                reason=observation.reason, progressed=observation.made_progress)
        return state

    return observe_node


def _decide_node(nodes: FlowNodes):
    async def decide_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        observation = Observation.from_dict(dict(state.get('observation') or {}))
        decision = nodes.decide(state, observation)
        state['decision'] = decision.as_dict()
        _record(state, 'decision_made', kind=decision.kind.value,
                reason=decision.reason, missing=list(decision.missing))

        if decision.kind is DecisionKind.CONTINUE:
            state['status'] = FlowStatus.PLANNING.value
        elif decision.kind is DecisionKind.WAIT_USER:
            state['status'] = FlowStatus.WAITING_USER.value
        elif decision.kind is DecisionKind.CONFIRM:
            state['status'] = FlowStatus.CONFIRMING.value
        elif decision.kind is DecisionKind.ESCALATE:
            state['status'] = FlowStatus.ESCALATED.value
            state['stop_reason'] = decision.reason or 'decision'
        elif decision.kind is DecisionKind.DONE:
            state['status'] = FlowStatus.DONE.value
        elif decision.kind is DecisionKind.FAILED:
            state['status'] = FlowStatus.FAILED.value
        return state

    return decide_node


def _request_confirm_node(raw: dict[str, Any]) -> dict[str, Any]:
    state = _ensure(raw)
    decision = dict(state.get('decision') or {})
    action = decision.get('action')
    if not isinstance(action, dict):
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = 'confirm_without_action'
        _record(state, 'escalated', reason='confirm_without_action')
        return state

    state['pending_action'] = dict(action)
    state['status'] = FlowStatus.CONFIRMING.value
    _record(state, 'confirmation_requested', action=dict(action))
    return state


def _wait_confirm_node(raw: dict[str, Any]) -> dict[str, Any]:
    state = _ensure(raw)
    action = state.get('pending_action')
    if not isinstance(action, dict):
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = 'confirm_without_action'
        _record(state, 'escalated', reason='confirm_without_action')
        return state

    response = interrupt({
        'kind': 'flow_confirm',
        'thread_id': state.get('thread_id'),
        'pending_action': dict(action),
    })
    state['confirmation'] = response if isinstance(response, dict) else {'value': response}
    state['status'] = FlowStatus.PLANNING.value
    _record(state, 'confirmation_received',
            approved=bool(state['confirmation'].get('approved')))
    return state


def _escalate_node(nodes: FlowNodes):
    async def escalate_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        reason = str(state.get('stop_reason')
                     or dict(state.get('decision') or {}).get('reason')
                     or 'decision')
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = reason
        state = ensure_state(dict(await nodes.escalate(state, reason)))
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = reason
        _record(state, 'escalated', reason=reason)
        return state

    return escalate_node


def _finish_node(nodes: FlowNodes):
    async def finish_node(raw: dict[str, Any]) -> dict[str, Any]:
        state = _ensure(raw)
        kind = dict(state.get('decision') or {}).get('kind')
        status = FlowStatus.FAILED if kind == DecisionKind.FAILED.value else FlowStatus.DONE
        state = ensure_state(dict(await nodes.finish(state, status)))
        state['status'] = status.value
        _record(state, 'run_completed', status=status.value)
        return state

    return finish_node


def _after_plan(state: FlowState) -> str:
    if state.get('status') == FlowStatus.ESCALATED.value:
        return 'escalate'
    if state.get('status') == FlowStatus.WAITING_USER.value:
        return END
    return 'assemble'


def _after_decide(state: FlowState) -> str:
    kind = dict(state.get('decision') or {}).get('kind')
    if kind == DecisionKind.CONTINUE.value:
        return 'plan'
    if kind == DecisionKind.WAIT_USER.value:
        return END
    if kind == DecisionKind.CONFIRM.value:
        return 'request_confirm'
    if kind == DecisionKind.ESCALATE.value:
        return 'escalate'
    if kind in (DecisionKind.DONE.value, DecisionKind.FAILED.value):
        return 'finish'
    return 'escalate'


def _after_confirm_request(state: FlowState) -> str:
    if state.get('status') == FlowStatus.ESCALATED.value:
        return 'escalate'
    return 'wait_confirm'


def _after_wait_confirm(state: FlowState) -> str:
    if state.get('status') == FlowStatus.ESCALATED.value:
        return 'escalate'
    return 'plan'


def build_flow_graph(
    nodes: FlowNodes,
    *,
    loop: LoopPolicy | None = None,
    max_no_progress: int = 2,
    checkpointer: Any = None,
):
    """构建可复用的 FlowNodes 子图。

    ``checkpointer`` 必须传业务图使用的 saver 才能支持确认后恢复；不传时仍可
    执行无确认流程，但 ``interrupt`` 无法跨调用恢复。
    """
    graph = StateGraph(dict[str, Any])
    graph.add_node('plan', _plan_node(nodes, loop or LoopPolicy(max_steps=20), max_no_progress))
    graph.add_node('assemble', _assemble_node(nodes))
    graph.add_node('execute', _execute_node(nodes))
    graph.add_node('observe', _observe_node(nodes))
    graph.add_node('decide', _decide_node(nodes))
    graph.add_node('request_confirm', _request_confirm_node)
    graph.add_node('wait_confirm', _wait_confirm_node)
    graph.add_node('escalate', _escalate_node(nodes))
    graph.add_node('finish', _finish_node(nodes))

    graph.add_edge(START, 'plan')
    graph.add_conditional_edges('plan', _after_plan)
    graph.add_edge('assemble', 'execute')
    graph.add_edge('execute', 'observe')
    graph.add_edge('observe', 'decide')
    graph.add_conditional_edges('decide', _after_decide)
    graph.add_conditional_edges('request_confirm', _after_confirm_request)
    graph.add_conditional_edges('wait_confirm', _after_wait_confirm)
    graph.add_edge('escalate', END)
    graph.add_edge('finish', END)
    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_flow_graph"]

"""通用业务控制层循环：plan → assemble → execute → observe → decide。

引擎只负责编排与不变量：步数/无进展终止、状态流转、确认与转人工出口。
业务节点由领域实现，执行面资源限制仍由 ``LoopPolicy`` 提供。
"""
from __future__ import annotations

import time
from typing import Protocol

from agent_core.harness import LoopPolicy

from .context import ContextPack
from .contracts import (
    Decision,
    DecisionKind,
    Execution,
    FlowStatus,
    Observation,
    Plan,
)
from .guards import LoopGuard
from .state import FlowState, ensure_state
from .trace import FlowTrace


class FlowNodes(Protocol):
    """领域业务节点；节点可以读取/更新状态，但控制流由 ``FlowEngine`` 决定。"""

    async def plan(self, state: FlowState) -> Plan:
        ...

    async def assemble(self, state: FlowState, plan: Plan) -> ContextPack:
        ...

    async def execute(self, state: FlowState, context: ContextPack) -> Execution:
        ...

    async def observe(self, state: FlowState, execution: Execution) -> Observation:
        ...

    def decide(self, state: FlowState, observation: Observation) -> Decision:
        ...

    async def escalate(self, state: FlowState, reason: str) -> FlowState:
        ...

    async def finish(self, state: FlowState, status: FlowStatus) -> FlowState:
        ...


class FlowEngine:
    """有界业务循环；默认最多 20 步，连续 2 轮无进展转人工。"""

    def __init__(self, nodes: FlowNodes, *, loop: LoopPolicy | None = None,
                 max_no_progress: int = 2, trace: FlowTrace | None = None) -> None:
        self.nodes = nodes
        self.loop = loop or LoopPolicy(max_steps=20)
        self.max_no_progress = max_no_progress
        self.trace = trace or FlowTrace()

    async def run(self, state: FlowState) -> FlowState:
        state = ensure_state(dict(state))
        state['trace_id'] = self.trace.trace_id
        state['started_at'] = float(state.get('started_at') or time.time())
        guard = LoopGuard(
            policy=self.loop, max_no_progress=self.max_no_progress,
            steps=int(state.get('step_count') or 0),
            tool_calls=int(state.get('tool_calls') or 0),
            no_progress_count=int(state.get('no_progress_count') or 0),
            started_at=state['started_at'], clock=time.time,
        )
        self.trace.record('run_started', data={'thread_id': state.get('thread_id')})
        try:
            while True:
                stop = guard.check()
                if stop is not None:
                    return await self._escalate(state, guard, stop.value)

                guard.start_step()
                state['step_count'] = guard.steps

                state['status'] = FlowStatus.PLANNING.value
                plan = await self.nodes.plan(state)
                state['plan'] = plan.as_dict()
                self.trace.record('plan_created', step=guard.steps, data={
                    'actions': len(plan.actions),
                    'missing': list(plan.missing),
                    'reason': plan.reason,
                })
                if plan.missing:
                    return self._wait_user(state, plan.missing)

                state['status'] = FlowStatus.ASSEMBLING.value
                context = await self.nodes.assemble(state, plan)
                state['context'] = context.as_dict()
                self.trace.record('context_assembled', step=guard.steps, data={
                    'message_count': context.message_count,
                    'facts': list(context.facts),
                    'versions': context.versions,
                    'trimmed': context.trimmed,
                })

                state['status'] = FlowStatus.EXECUTING.value
                execution = await self.nodes.execute(state, context)
                state['result'] = execution.as_dict()
                guard.record_tool_calls(execution.tool_calls)
                state['tool_calls'] = guard.tool_calls
                self.trace.record('tool_called', step=guard.steps, data={
                    'tool_calls': execution.tool_calls,
                    'errors': list(execution.errors),
                })

                state['status'] = FlowStatus.OBSERVING.value
                observation = await self.nodes.observe(state, execution)
                state['observation'] = observation.as_dict()
                guard.record_progress(observation.made_progress)
                state['no_progress_count'] = guard.no_progress_count
                self.trace.record('observation_created', step=guard.steps, reason=observation.reason,
                                  data={'kind': observation.kind.value,
                                        'progressed': observation.made_progress})

                decision = self.nodes.decide(state, observation)
                state['decision'] = decision.as_dict()
                self.trace.record('decision_made', step=guard.steps, reason=decision.reason,
                                  data={'kind': decision.kind.value,
                                        'missing': list(decision.missing)})

                if decision.kind is DecisionKind.CONTINUE:
                    continue
                if decision.kind is DecisionKind.WAIT_USER:
                    return self._wait_user(state, decision.missing)
                if decision.kind is DecisionKind.CONFIRM:
                    if decision.action is None:
                        return await self._escalate(state, guard, 'confirm_without_action')
                    state['pending_action'] = decision.action.as_dict()
                    state['status'] = FlowStatus.CONFIRMING.value
                    self.trace.record('confirmation_requested', step=guard.steps,
                                      data={'action': decision.action.as_dict()})
                    return state
                if decision.kind is DecisionKind.ESCALATE:
                    return await self._escalate(state, guard, decision.reason or 'decision')
                if decision.kind is DecisionKind.DONE:
                    return await self._finish(state, guard, FlowStatus.DONE)
                if decision.kind is DecisionKind.FAILED:
                    return await self._finish(state, guard, FlowStatus.FAILED)

                return await self._escalate(state, guard, f'unknown_decision:{decision.kind}')
        except Exception as exc:  # noqa: BLE001 - 记录失败后继续抛出，交给上层处理
            state['status'] = FlowStatus.FAILED.value
            state['error'] = f'{type(exc).__name__}: {exc}'
            self.trace.record('run_failed', step=state.get('step_count', 0),
                              reason=state['error'])
            raise
        finally:
            self.trace.record('run_finished', step=state.get('step_count', 0),
                              data={'status': state.get('status'),
                                    'stop_reason': state.get('stop_reason')})

    def _wait_user(self, state: FlowState, missing: tuple[str, ...]) -> FlowState:
        state['status'] = FlowStatus.WAITING_USER.value
        state['decision'] = Decision(
            DecisionKind.WAIT_USER, reason='missing_info', missing=missing,
        ).as_dict()
        self.trace.record('waiting_user', step=state.get('step_count', 0),
                          data={'missing': list(missing)})
        return state

    async def _escalate(self, state: FlowState, guard: LoopGuard, reason: str) -> FlowState:
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = reason
        state['step_count'] = guard.steps
        state['tool_calls'] = guard.tool_calls
        self.trace.record('escalated', step=guard.steps, reason=reason)
        state = await self.nodes.escalate(state, reason)
        state['status'] = FlowStatus.ESCALATED.value
        state['stop_reason'] = reason
        return state

    async def _finish(self, state: FlowState, guard: LoopGuard,
                      status: FlowStatus) -> FlowState:
        state['step_count'] = guard.steps
        state['tool_calls'] = guard.tool_calls
        state = await self.nodes.finish(state, status)
        state['status'] = status.value
        self.trace.record('run_completed', step=guard.steps, data={'status': status.value})
        return state


__all__ = ["FlowEngine", "FlowNodes"]

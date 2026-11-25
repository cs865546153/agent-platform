"""LangGraph 子图适配契约：逐步 checkpoint、确认中断与恢复。"""
import asyncio
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_core.flow import (
    ContextPack,
    Decision,
    DecisionKind,
    Execution,
    Observation,
    ObservationKind,
    PendingAction,
    Plan,
    StopReason,
    build_flow_graph,
    new_state,
)
from agent_core.harness import LoopPolicy


def _action() -> PendingAction:
    return PendingAction(
        action_type='confirm',
        target_id='quote-1',
        target_version=1,
        payload_hash='hash',
        expires_at=time.time() + 60,
        idempotency_key='idem-1',
    )


class DemoNodes:
    def __init__(self, *, done_after: int = 2) -> None:
        self.done_after = done_after
        self.executed = 0

    async def plan(self, state):
        return Plan(actions=({'tool': 'quote'},), reason='plan')

    async def assemble(self, state, plan):
        return ContextPack(facts={'budget': 100}, versions={'budget': 1})

    async def execute(self, state, context):
        self.executed += 1
        return Execution(outputs=('ok',), tool_calls=1)

    async def observe(self, state, execution):
        return Observation(ObservationKind.SUCCESS, reason='ok', progressed=True)

    def decide(self, state, observation):
        if state['step_count'] >= self.done_after:
            return Decision(DecisionKind.DONE, reason='complete')
        return Decision(DecisionKind.CONTINUE)

    async def escalate(self, state, reason: str):
        state['escalated'] = reason
        return state

    async def finish(self, state, status):
        state['finished'] = status.value
        return state


class ConfirmNodes(DemoNodes):
    def decide(self, state, observation):
        if state.get('confirmation'):
            return Decision(DecisionKind.DONE, reason='confirmed')
        return Decision(DecisionKind.CONFIRM, reason='need_confirm', action=_action())


class NoProgressNodes(DemoNodes):
    async def observe(self, state, execution):
        return Observation(ObservationKind.NO_PROGRESS, reason='same')

    def decide(self, state, observation):
        return Decision(DecisionKind.CONTINUE)


class MissingNodes(DemoNodes):
    async def plan(self, state):
        return Plan(missing=('budget',), reason='missing')


def test_flow_graph_runs_to_done_and_keeps_domain_fields() -> None:
    nodes = DemoNodes(done_after=2)
    graph = build_flow_graph(nodes, loop=LoopPolicy(max_steps=5))
    state = asyncio.run(graph.ainvoke(new_state('thread-lg-1', tenant='t1')))

    assert state['status'] == 'done'
    assert state['step_count'] == 2
    assert state['tool_calls'] == 2
    assert state['finished'] == 'done'
    assert state['tenant'] == 't1'
    events = [event['event'] for event in state['trace_events']]
    assert 'plan_created' in events
    assert 'context_assembled' in events
    assert 'tool_called' in events
    assert 'observation_created' in events
    assert 'decision_made' in events
    assert 'run_completed' in events


def test_flow_graph_waits_for_missing_info_without_executing() -> None:
    nodes = MissingNodes()
    graph = build_flow_graph(nodes, loop=LoopPolicy(max_steps=5))
    state = asyncio.run(graph.ainvoke(new_state('thread-lg-2')))

    assert state['status'] == 'waiting_user'
    assert state['decision']['missing'] == ['budget']
    assert nodes.executed == 0


def test_flow_graph_escalates_on_max_steps() -> None:
    nodes = DemoNodes(done_after=100)
    graph = build_flow_graph(nodes, loop=LoopPolicy(max_steps=1))
    state = asyncio.run(graph.ainvoke(new_state('thread-lg-3')))

    assert state['status'] == 'escalated'
    assert state['stop_reason'] == StopReason.MAX_STEPS.value
    assert state['escalated'] == StopReason.MAX_STEPS.value


def test_flow_graph_escalates_on_no_progress() -> None:
    nodes = NoProgressNodes()
    graph = build_flow_graph(
        nodes, loop=LoopPolicy(max_steps=10), max_no_progress=2,
    )
    state = asyncio.run(graph.ainvoke(new_state('thread-lg-4')))

    assert state['status'] == 'escalated'
    assert state['stop_reason'] == StopReason.NO_PROGRESS.value
    assert state['step_count'] == 2


def test_flow_graph_confirm_interrupts_and_resumes() -> None:
    nodes = ConfirmNodes()
    saver = MemorySaver()
    graph = build_flow_graph(nodes, loop=LoopPolicy(max_steps=5), checkpointer=saver)
    config = {'configurable': {'thread_id': 'thread-lg-5'}}

    asyncio.run(graph.ainvoke(new_state('thread-lg-5'), config))
    snapshot = graph.get_state(config)
    assert snapshot.next == ('wait_confirm',)
    assert snapshot.values['pending_action']['target_id'] == 'quote-1'
    interrupts = [item for task in snapshot.tasks for item in task.interrupts]
    assert interrupts[0].value['kind'] == 'flow_confirm'
    assert interrupts[0].value['pending_action']['target_id'] == 'quote-1'

    state = asyncio.run(graph.ainvoke(Command(resume={'approved': True}), config))
    assert state['status'] == 'done'
    assert state['confirmation'] == {'approved': True}
    assert nodes.executed == 2

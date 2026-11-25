"""业务控制层契约：版本事实、上下文、循环守卫、确认与通用引擎。"""
import asyncio
from dataclasses import replace
import time

from agent_core.flow import (
    ConfirmCode,
    ConfirmExecutor,
    ContextAssembler,
    ContextPack,
    Decision,
    DecisionKind,
    Execution,
    FlowEngine,
    FlowTrace,
    InMemoryIdempotencyStore,
    InMemoryTraceSink,
    LoopGuard,
    Observation,
    ObservationKind,
    PendingAction,
    PendingStatus,
    Plan,
    StopReason,
    VersionedFacts,
    new_state,
    payload_hash,
    validate_pending_action,
)
from agent_core.harness import ContextPolicy, LoopPolicy
from langchain_core.messages import HumanMessage, SystemMessage


def _action(**overrides) -> PendingAction:
    values = {
        'action_type': 'confirm',
        'target_id': 'quote-1',
        'target_version': 2,
        'payload_hash': 'hash',
        'expires_at': time.time() + 60,
        'idempotency_key': 'idem-1',
    }
    values.update(overrides)
    return PendingAction(**values)


def test_versioned_facts_version_invalidate_and_reactivate() -> None:
    facts = VersionedFacts()
    assert facts.set('budget', 100) == 1
    assert facts.set('budget', 120) == 2
    assert facts.current('budget') == 120
    assert facts.version('budget') == 2
    assert facts.is_current('budget', 2)
    assert not facts.is_current('budget', 1)

    assert facts.invalidate('budget', '用户改口')
    assert facts.current('budget') is None
    assert not facts.is_current('budget', 2)
    assert facts.invalidated['budget'] == '用户改口'

    assert facts.set('budget', 150) == 3
    assert 'budget' not in facts.invalidated
    assert facts.current('budget') == 150


def test_context_assembler_pins_current_facts_and_trims_messages() -> None:
    facts = VersionedFacts()
    facts.set('budget', 100)
    facts.set('coverage', 'full')
    facts.set('budget', 120)
    facts.invalidate('coverage', '明确删除')

    messages = [SystemMessage(content='系统提示')] + [
        HumanMessage(content=f'问题{i}') for i in range(3)
    ]
    pack = ContextAssembler(
        policy=ContextPolicy(max_messages=2),
        pinned_facts=('budget', 'coverage', 'not_exists'),
    ).assemble(messages, facts)

    assert pack.messages[0].type == 'system'
    assert len(pack.messages) == 3
    assert pack.facts == {'budget': 120}
    assert pack.versions == {'budget': 2}
    assert pack.trimmed is True
    assert pack.as_dict()['message_count'] == 3


def test_loop_guard_stops_on_steps_tools_deadline_and_no_progress() -> None:
    guard = LoopGuard(policy=LoopPolicy(max_steps=1))
    guard.start_step()
    assert guard.check() is StopReason.MAX_STEPS

    guard = LoopGuard(policy=LoopPolicy(max_tool_calls=1))
    guard.record_tool_calls()
    assert guard.check() is StopReason.MAX_TOOL_CALLS

    guard = LoopGuard(policy=LoopPolicy(deadline_s=5), started_at=0, clock=lambda: 5)
    assert guard.check() is StopReason.DEADLINE

    guard = LoopGuard(policy=LoopPolicy(max_steps=10), max_no_progress=2)
    guard.record_progress(False)
    assert guard.check() is None
    guard.record_progress(False)
    assert guard.check() is StopReason.NO_PROGRESS
    guard.record_progress(True)
    assert guard.check() is None


def test_pending_action_validation_and_serialization() -> None:
    action = _action()
    assert PendingAction.from_dict(action.as_dict()) == action
    assert validate_pending_action(
        action, current_version=2, payload_hash='hash', now=action.created_at,
    ).ok

    version = validate_pending_action(
        action, current_version=3, payload_hash='hash', now=action.created_at,
    )
    assert version.code is ConfirmCode.VERSION_MISMATCH

    expired = validate_pending_action(
        action, current_version=2, payload_hash='hash', now=action.expires_at + 1,
    )
    assert expired.code is ConfirmCode.EXPIRED

    payload = validate_pending_action(
        action, current_version=2, payload_hash='other', now=action.created_at,
    )
    assert payload.code is ConfirmCode.PAYLOAD_MISMATCH

    rejected = replace(action, status=PendingStatus.REJECTED)
    not_pending = validate_pending_action(
        rejected, current_version=2, payload_hash='hash', now=action.created_at,
    )
    assert not_pending.code is ConfirmCode.NOT_PENDING


def test_payload_hash_is_stable_across_key_order() -> None:
    assert payload_hash({'a': 1, 'b': 2}) == payload_hash({'b': 2, 'a': 1})
    assert payload_hash({'a': 1}) != payload_hash({'a': 2})


def test_confirm_executor_only_executes_once() -> None:
    action = _action(expires_at=time.time() + 60)
    calls: list[PendingAction] = []

    async def execute(item: PendingAction) -> dict[str, bool]:
        calls.append(item)
        return {'ok': True}

    async def scenario() -> tuple:
        executor = ConfirmExecutor(InMemoryIdempotencyStore())
        first = await executor.execute(
            action, current_version=2, payload_hash='hash', execute=execute,
        )
        second = await executor.execute(
            action, current_version=2, payload_hash='hash', execute=execute,
        )
        return first, second

    (first_check, first_result), (second_check, second_result) = asyncio.run(scenario())
    assert first_check.code is ConfirmCode.OK
    assert first_result == {'ok': True}
    assert second_check.code is ConfirmCode.IDEMPOTENT_REPLAY
    assert second_result == first_result
    assert len(calls) == 1


def test_trace_sink_receives_events() -> None:
    sink = InMemoryTraceSink()
    trace = FlowTrace(trace_id='trace-1', sink=sink)
    event = trace.record('decision_made', step=2, reason='ok', kind='continue')
    assert event.trace_id == 'trace-1'
    assert sink.events == [event]
    assert trace.as_dict()['trace_id'] == 'trace-1'


class DemoNodes:
    def __init__(self, *, done_after: int = 2) -> None:
        self.done_after = done_after
        self.executed = 0

    async def plan(self, state) -> Plan:
        return Plan(actions=({'tool': 'quote'},), reason='plan')

    async def assemble(self, state, plan) -> ContextPack:
        return ContextPack(
            messages=('m1',), facts={'budget': 100}, versions={'budget': 1},
        )

    async def execute(self, state, context) -> Execution:
        self.executed += 1
        return Execution(outputs=('ok',), tool_calls=1)

    async def observe(self, state, execution) -> Observation:
        return Observation(ObservationKind.SUCCESS, reason='ok', changed_fields=('budget',))

    def decide(self, state, observation) -> Decision:
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
    def decide(self, state, observation) -> Decision:
        return Decision(DecisionKind.CONFIRM, reason='need_confirm', action=_action())


class NoProgressNodes(DemoNodes):
    async def observe(self, state, execution) -> Observation:
        return Observation(ObservationKind.NO_PROGRESS, reason='same')

    def decide(self, state, observation) -> Decision:
        return Decision(DecisionKind.CONTINUE)


class MissingNodes(DemoNodes):
    async def plan(self, state) -> Plan:
        return Plan(missing=('budget',), reason='missing')


def test_flow_engine_runs_plan_to_done() -> None:
    nodes = DemoNodes(done_after=2)
    engine = FlowEngine(nodes, loop=LoopPolicy(max_steps=5))
    state = asyncio.run(engine.run(new_state('thread-1')))

    assert state['status'] == 'done'
    assert state['step_count'] == 2
    assert state['finished'] == 'done'
    assert state['result']['tool_calls'] == 1
    assert state['decision']['kind'] == 'done'
    events = [event.event for event in engine.trace.events]
    assert events[0] == 'run_started'
    assert 'plan_created' in events
    assert 'context_assembled' in events
    assert 'tool_called' in events
    assert 'observation_created' in events
    assert 'decision_made' in events
    assert 'run_completed' in events
    assert events[-1] == 'run_finished'


def test_flow_engine_stops_for_confirmation() -> None:
    engine = FlowEngine(ConfirmNodes(), loop=LoopPolicy(max_steps=5))
    state = asyncio.run(engine.run(new_state('thread-2')))

    assert state['status'] == 'confirming'
    assert state['pending_action']['target_id'] == 'quote-1'
    assert state['step_count'] == 1
    assert 'confirmation_requested' in [event.event for event in engine.trace.events]


def test_flow_engine_escalates_on_max_steps() -> None:
    nodes = DemoNodes(done_after=100)
    engine = FlowEngine(nodes, loop=LoopPolicy(max_steps=1))
    state = asyncio.run(engine.run(new_state('thread-3')))

    assert state['status'] == 'escalated'
    assert state['stop_reason'] == StopReason.MAX_STEPS.value
    assert state['escalated'] == StopReason.MAX_STEPS.value
    assert state['step_count'] == 1


def test_flow_engine_escalates_on_no_progress() -> None:
    nodes = NoProgressNodes()
    engine = FlowEngine(nodes, loop=LoopPolicy(max_steps=10), max_no_progress=2)
    state = asyncio.run(engine.run(new_state('thread-4')))

    assert state['status'] == 'escalated'
    assert state['stop_reason'] == StopReason.NO_PROGRESS.value
    assert state['step_count'] == 2


def test_flow_engine_waits_for_missing_info_without_executing() -> None:
    nodes = MissingNodes()
    engine = FlowEngine(nodes, loop=LoopPolicy(max_steps=5))
    state = asyncio.run(engine.run(new_state('thread-5')))

    assert state['status'] == 'waiting_user'
    assert state['decision']['missing'] == ['budget']
    assert nodes.executed == 0
    assert 'waiting_user' in [event.event for event in engine.trace.events]

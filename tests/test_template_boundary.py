"""模板边界门禁：内核不得再出现业务领域名与意图字符串。

可插拔化之后，这些约束是"新增领域不改内核"的前提，但源码一旦被写回业务词就会
悄悄失效，所以用源码扫描把它钉住：

- 内核不 import 任何具体领域包 / 领域图 / 领域适配器；
- 内核不出现意图字符串（意图只存在于领域清单里）；
- 内核不出现旧的分支词汇（``BranchKind`` 一类），落点只能是"能力名"；
- 工作台契约不枚举意图名（``IntentName`` 必须是受长度约束的 str）。

任何一条失败都意味着：把这段逻辑放回 ``domains/<领域>/``，而不是加进内核。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 真正的内核文件。domains/registry.py 是"启用哪些领域"的登记表，允许出现领域名。
KERNEL_FILES = [
    'app.py',
    'app_factory.py',
    'agents/assistant_agent.py',
    'agent_core/capabilities.py',
    'agent_core/plugins.py',
    'agent_core/security.py',
    'agent_core/sse.py',
    'domains/base.py',
    'models/workbench.py',
] + [f'agent_core/intent/{path.name}' for path in (ROOT / 'agent_core' / 'intent').glob('*.py')] \
  + [f'agent_core/harness/{path.name}' for path in (ROOT / 'agent_core' / 'harness').glob('*.py')]

# 意图字符串只属于领域清单（manifest.yaml / DomainManifest）。
INTENT_STRINGS = [
    '理赔报案', '材料审核', '进度查询', '条款咨询', '保单查询', '补充材料',
    '投资咨询', '组合诊断', '风险评估', '产品查询', '交易指令', '一般咨询',
]
# 已被"能力名分派"取代的旧分支词汇。
BRANCH_VOCABULARY = ['BranchKind', 'branch_kind', 'INVESTMENT_INTENTS', 'GENERAL_INTENTS', 'CLAIM_INTENTS']
# 内核不得依赖的具体领域/业务模块。
FORBIDDEN_IMPORTS = [
    'domains.claims', 'domains.investment', 'domains.example',
    'agents.claim_agent', 'agents.investment_agent', 'claims_agent', 'adapters.',
]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding='utf-8')


def test_kernel_does_not_import_business_modules() -> None:
    offenders: list[str] = []
    for relative in KERNEL_FILES:
        for imported in FORBIDDEN_IMPORTS:
            if re.search(rf"(^|\s)(from|import)\s+{re.escape(imported)}(\.|\s|$)", _source(relative), re.M):
                offenders.append(f'{relative} -> {imported}')
    assert not offenders, '内核开始依赖具体业务模块：' + '；'.join(offenders)


def test_kernel_has_no_intent_strings() -> None:
    offenders = [f'{relative}:{intent}'
                 for relative in KERNEL_FILES
                 for intent in INTENT_STRINGS
                 if intent in _source(relative)]
    assert not offenders, '内核写死了意图名，应改为领域清单声明：' + '；'.join(offenders)


def test_kernel_routes_by_capability_not_branch() -> None:
    offenders = [f'{relative}:{word}'
                 for relative in KERNEL_FILES
                 for word in BRANCH_VOCABULARY
                 if word in _source(relative)]
    assert not offenders, '内核出现旧分支词汇，落点只能是能力名：' + '；'.join(offenders)


def test_workbench_contract_does_not_enumerate_intents() -> None:
    """意图白名单交给领域清单，契约只约束形态。"""
    source = _source('models/workbench.py')
    assert re.search(r'IntentName:\s*TypeAlias\s*=\s*Annotated\[str', source), \
        'IntentName 不应再是 Literal 枚举'


def test_default_manifests_only_reuse_kernel_capabilities_by_declaration() -> None:
    """真实领域清单里，落点要么是本领域能力，要么是显式声明的内核能力。"""
    from agent_core.capabilities import KERNEL_CAPABILITIES
    from domains.registry import default_domain_registry

    for manifest in default_domain_registry().all():
        for spec in manifest.intents:
            if spec.capability in KERNEL_CAPABILITIES:
                # general 只做通用问答，可以显式绑定内核 answer；其余领域不得借用。
                assert manifest.name == 'general', f'{manifest.name} 借用了内核能力 {spec.capability}'
            else:
                assert spec.capability in manifest.capabilities


def test_assembled_app_closes_every_declared_capability() -> None:
    """装配后不得有"声明了能力却没人实现"的缺口——它会静默回落内核兜底。"""
    from fastapi.testclient import TestClient

    from agent_core.capabilities import missing_handlers
    from app import create_app

    with TestClient(create_app(database=':memory:')) as client:
        capabilities = client.app.state.capabilities
        registry = client.app.state.intent_registry
        assert missing_handlers(registry, capabilities) == []

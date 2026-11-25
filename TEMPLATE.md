# 企业级 Agent 模板说明

本仓库既是可运行的理赔助手，也是一套**企业级 Agent 工程模板**。核心承诺：
**新增一个业务领域，不需要修改内核（`app.py` / `agent_core/` / 接入层），只需实现插件并登记。**

个性化教学资料已归档至工作区根目录 `../teaching-archive/`（My_Study、培训实现追踪清单），
不属于模板运行时。

---

## 1. 分层与边界

```
app.py                     内核：FastAPI 装配 + 通用路由（health/ready/metrics/RAG/工作台/SSE）
  └─ create_app(registry=)  唯一扩展点：注入领域插件集；顶层不 import 任何领域类型
     └─ overrides / models  领域依赖注入点，键名由插件解释（内核签名无领域名）
agent_core/                领域无关内核
  ├─ plugins.py            DomainPlugin 协议（intent_manifest）、AppContext（slot_extractors/capabilities）、PluginRegistry
  ├─ capabilities.py       对话编排能力注册表：内核按"能力名"分派，只内置 answer / guard / clarify
  ├─ confirmation.py       写操作确认闸门：按清单确定性裁决，写操作绝不落到自由问答
  ├─ clarification.py      澄清闸门：required_slots / 低置信候选，确定性追问且不调用模型
  ├─ security.py           统一认证唯一事实源（API_KEY 优先，CLAIMS_API_KEY 兼容旧部署）
  ├─ sse.py                SSE 帧编码（已接入 app.py，无内联副本）
  └─ intent/               JEV/LLM/Hybrid/Shadow（清单从插件 manifest 聚合，YAML 仅回退 general）
domains/                   领域插件（可增删）
  ├─ registry.py           默认领域登记表 + default_domain_registry()（评测/测试聚合完整清单）
  ├─ base.py               插件基类 + LangGraph 中断/配置/HITL 恢复门控共用辅助
  ├─ claims/               示例域：理赔状态机、三专家、RAG、HITL、演示路由、对话能力与保单号槽位规则
  ├─ investment/           示例域：投资咨询、交易二次确认、人工审批、对话能力与投资 RAG 装配
  └─ example/              最小示例：60 行接入一个新领域（可直接抄改）
configs/domains.yaml       运行时回退清单：只保留无插件领域（general）
```

**横切能力**（审计哈希链、接入限流、PII 脱敏、模型缓存/降级、Prometheus/LangFuse）
在 `agents/http_middleware.py`、`agents/audit_store.py`、`agents/middleware_audit.py` 等，
对所有领域一律生效，领域插件无需重复实现。

---

## 2. 内核提供的能力（开箱即用）

| 能力 | 位置 | 领域是否需关心 |
| --- | --- | --- |
| RequestID / 认证 / 限流 / 审计 / 500 兜底 | `agents/http_middleware.py` + `agent_core/security.py`（同一密钥解析） | 否 |
| append-only + 哈希链审计 | `agents/audit_store.py` | 否 |
| 意图分类 JEV/LLM/Hybrid/Shadow | `agent_core/intent/` | 在插件 manifest.yaml 声明意图即可 |
| 模型工厂 / 任务档位 / 缓存 / 降级 | `config.py`（`LLMFactory`、`register_task_profile()`）、`agents/middleware_audit.py` | 否；领域可注册自己的档位 |
| 领域槽位兜底抽取 | `AppContext.slot_extractors`，由插件在 `build_services` 注册 | 内核不内置业务字段形态 |
| 对话编排能力分派 | `agent_core/capabilities.py` + `agents/assistant_agent.py` 动态建图 | 在 `build_services` 里 `ctx.capabilities.register(名字, handler)` |
| 写操作确认闸门 | `agent_core/confirmation.py`（`ConfirmGate` 确定性裁决） | 写操作意图在 manifest 声明 `confirmation`（指向独立 API）；内核自动拦截并引导，不执行动作 |
| 澄清闸门 | `agent_core/clarification.py`（`ClarifyGate` 确定性裁决） | 需要要素的意图在 manifest 声明 `required_slots` / `slot_labels`；缺槽位或低置信由内核 `clarify` 追问 |
| 多路召回 + RRF + BGE-Rerank + 引用校验 | `tools/knowledge_rag.py`、`tools/rerank.py` | 可选复用 |
| HITL（中断 + 恢复 + 人工审批） | LangGraph `interrupt` + `domains/base.py:gate_resume/resume_gate` | 图内声明中断，恢复门控用基类 |
| SSE 流式回答 | `app.py` assistant 路由 + `agent_core/sse.py` | 否 |
| Prometheus / LangFuse / 日志脱敏 | `monitoring.py`（`METRIC_PREFIX` 可配、插件可 `register_counter`）、`agents/observability.py` | 否 |
| 上线门禁 / 影子对拍 | `deploy/regression_gate.py`、`deploy/shadow_mode.py` | 否 |

---

## 3. 新增一个领域（五步）

以 `domains/example/` 为骨架：

### 步骤 1：实现插件

```python
# domains/my_domain/plugin.py
from typing import Any
from agent_core.plugins import AppContext
from domains.base import DomainPluginBase

class MyDomainPlugin(DomainPluginBase):
    name = 'my_domain'
    tags = ['我的业务域']

    def build_services(self, ctx: AppContext, overrides: dict[str, Any]) -> None:
        # 装配领域依赖；overrides 是测试注入点
        ctx.services[self.name] = overrides.get('my_service') or MyService()

    def build_graphs(self, ctx: AppContext, saver: Any) -> None:
        # 可选：构建 LangGraph，checkpointer 由内核在 lifespan 中提供
        ctx.graphs[self.name] = build_my_graph(ctx.services[self.name], saver)

    def register_routes(self, ctx: AppContext) -> None:
        # 注册领域专属路由，依赖全部从 ctx 取，不要 import app.py
        from domains.my_domain.routes import register_routes
        register_routes(ctx)
```

`DomainPluginBase.intent_manifest()` 会自动加载插件目录下的 `manifest.yaml`，
领域与意图声明从此随插件走，不必改全局 YAML（见步骤 2）。

需要领域专属槽位（如保单号）或模型档位时，在 `build_services` 里注册，
内核只负责把抽取器注入意图路由适配器，不认识具体字段：

```python
from config import TaskProfile, register_task_profile

ctx.slot_extractors[self.name] = my_slot_extractor  # (decision, question, turns) -> dict[str, str]
register_task_profile('my_task', TaskProfile('plus', 0.3, 1200))
```

### 步骤 2：声明领域意图与能力绑定（随插件走）

```yaml
# domains/my_domain/manifest.yaml —— 格式与 DomainManifest 一致
name: my_domain
description: 我的业务领域
description_en: "My business domain"   # JEV 英文优先，务必提供
version: 0.1.0
# 能力名是本领域对话落点的声明集合；意图绑定的能力必须在这里出现，否则清单校验报错。
capabilities: [my_domain_answer, my_domain_submit]
intents:
  - name: 我的意图
    description: 用户会怎么表达
    description_en: "How users express it"
    examples: ["一个示例问题"]
    risk: low
    mode: conversation
    capability: my_domain_answer     # 对话图落点；留空则按 mode/risk 回落 answer / guard
  - name: 我的写操作
    description: 会改变业务状态的请求
    description_en: "Requests that change business state"
    risk: high
    mode: case
    capability: my_domain_submit
    # 写操作契约：对话链路只拦截引导，真正执行交给独立 API 的校验/二次确认/人工审批。
    # ConfirmGate 据此做确定性裁决（不问模型"要不要确认"）；省略时 high/case 也会被拦截，
    # 但引导信息里没有目标接口。api 指向的路由必须真实注册，装配自检会校验死链。
    confirmation:
      api: /api/my_domain/submit
      requires_human_review: true
      summary: 请通过业务接口提交，由系统完成校验与授权审批。
```

内核启动时从已启用插件的 `intent_manifest()` 聚合领域清单；
`configs/domains.yaml` 已瘦身为**只含 general 的回退清单**（完整三域清单请用
`domains.registry.default_domain_registry()` 从插件聚合）。若插件声明与回退清单
同名却内容不一致，启动即抛 `ValueError`，避免两份事实源悄悄分叉。
图内分支与分类提示词都从聚合清单派生；领域专属槽位抽取由插件注入（见步骤 1）。

`capability` 是对话图里的节点名。省略时按 `mode` / `risk` 推导：
`case` 或 `high` → `guard`（拦截并说明需要确认），其余 → `answer`（通用模型直接回答）。
宁可回落到内置落点，也不会让未实现的分支静默变成自由问答。

**写操作必须声明 `confirmation`。** 内核的 `ConfirmGate` 从聚合清单读取契约，
对声明了 `confirmation`、或 `risk=high` / `mode=case` 的意图一律裁决为"需要确认"：
绝不允许落到 `answer`（误落时强制改派 `guard`），并把目标独立 API、是否需人工审批、
裁决来源并入对话图已有的 done trace（只加字段、不新增事件，SSE 序列不变）。
裁决是纯确定性代码、不调用 LLM；领域 handler 无需感知闸门，新增领域自动生效。
装配期还会自检：契约落到自由问答、或 `api` 指向未注册路由，都会打 warning
（`application.state.confirmation_audit` 可查）。契约由 `tests/test_confirmation_gate.py` 守护。

### 步骤 3：注册对话能力（让新领域在 `/api/assistant/stream` 可达）

```python
# domains/my_domain/assistant_handlers.py
from agents.assistant_agent import conversation_messages, model_inputs


async def my_domain_answer(state: dict) -> dict:
    """state 里可读 intent / slots / documents / history / question；返回状态补丁。"""
    messages = conversation_messages(state['question'], state['history'], MY_PROMPT)
    return {'events': [{'trace': {'node': 'my_domain', 'status': 'done',
                                  'input': model_inputs(MODEL, messages)}}],
            'llm': MODEL, 'messages': messages, 'needs_llm_answer': True}


def register_my_capabilities(ctx, *, model) -> None:
    ctx.capabilities.register('my_domain_answer', my_domain_answer)
    # 需要基于会话状态改写落点时（如补材料续答继承上一轮意图）：
    # ctx.capabilities.register_redirect('my_domain', lambda state: Redirect(...))
```

在插件的 `build_services` 里调用 `register_my_capabilities(...)` 即可。约束：

- 能力名全局唯一，撞名直接 `ValueError`（能力名即图节点名）；
- 声明了能力但没注册实现不会报错，只会在装配时打 warning，
  可用 `agent_core.capabilities.missing_handlers()` 自检（`tests/test_template_boundary.py` 已钉住）；
- 处理器抛异常时内核显式降级为"转人工"，SSE 不中断。
- 需要追问要素时，在 manifest 的 `IntentSpec` 声明 `required_slots`（snake_case 字段名）
  与 `slot_labels`；缺失时内核先走 `clarify`，澄清优先级高于 `ConfirmGate`。
- 有业务图的插件在 `DomainPluginBase` 声明 `graph_keys`（如 `("claims",)`），
  `/ready` 会据此检查装配是否完整；跳过该字段的纯能力插件不受影响。

### 步骤 4：登记领域

```python
# domains/registry.py —— 唯一的启用开关
from domains.my_domain.plugin import MyDomainPlugin

def default_registry() -> PluginRegistry:
    return PluginRegistry([ClaimsPlugin(), InvestmentPlugin(), MyDomainPlugin()])
```

`configs/domains.yaml` 不再是新增领域的必改项；它只保留无插件领域的声明。

### 步骤 5：测试

```bash
pytest -q tests/test_domain_plugin_template.py   # 领域可插拔契约
pytest -q tests/test_intent_plugin_template.py   # 意图路由可插拔契约
pytest -q tests/test_capabilities.py             # 能力分派与动态建图契约
pytest -q tests/test_confirmation_gate.py        # 写操作确认闸门契约
pytest -q tests/test_template_boundary.py        # 内核边界门禁（源码扫描）
pytest -q                                        # 全量
```

测试里可用 `create_app(registry=PluginRegistry([MyDomainPlugin()]))` 只装自己的域，
与新领域无关的路由不会注册；也可用 `intent_router=FakeRouter()` 显式注入意图路由，
内核会跳过构建且不接管其生命周期（见两个契约测试文件）。

---

## 4. 模板的使用边界（如实说明）

- **演示路由已迁出内核**：`/api/demo/claims/*` 由 `ClaimsPlugin` 注册，不装 claims 插件就不存在；
  `app.py` 只保留通用路由与 assistant 对话编排，原创业务可删除后者，不影响领域插件机制。
- **对话编排已按能力分派**：`agents/assistant_agent.py` 不再有业务节点或意图字符串，
  图节点由"已注册的能力名"动态生成，落点来自领域清单的 `capability`。
  新增领域按步骤 2/3 声明并注册后，在 `/api/assistant/stream` 自动可达，无需改内核。
  边界由 `tests/test_template_boundary.py` 用源码扫描钉住（内核不 import 业务模块、
  不出现意图字符串、不残留旧分支词汇）。
- **工作台请求模型仍带示例形态**：`AssistantRequest.claim_id` / `material_round` 与
  `models/workbench.py` 的理赔演示响应属于示例前端契约，内核只透传不解释；
  通用澄清轮次走新增的 `round`（与 `material_round` 兼容，上限 3），原创业务可替换为
  自己的案件主键，或改用 `slots` 承载。
- **HITL 恢复跨进程可用**：LangGraph 重启后不再重建中断载荷，快照只剩待执行节点，
  因此恢复门控与响应阶段都用 `domains/base.py` 的 `gate_resume` / `resume_gate`
  按 `(状态, 待执行节点)` 重建；领域新增中断点时要一并给出 `node` 判据。
- **确认闸门不是完整审批流**：`ConfirmGate` 在对话链路里做确定性拦截与结构化引导，
  不承担二次确认状态机与人工审批存储；真正执行写操作仍走独立 API（checkpointer +
  HITL 中断/恢复 + 审核员二次认证）。对话链路无状态、无检查点，永不落业务写入。
- **教学产物已迁出**：`My_Study/`、培训追踪清单在 `../teaching-archive/`，不在模板运行时内。
- **真实联调未完成**：本地 LLM、Milvus、Redis、LangFuse、核心业务系统需按部署环境接入；
  模板不虚构准确率、延迟或成本。详见 `docs/` 各专题与 `deploy/runbook.md`。
- **进程内限流/锁**：多副本部署需替换为 Redis；`AppContext.locks` 是进程内实现。
- **关于 Harness**：材料中的 `deep_agents.Harness/Fuse/Memory/AgenticRAG` 未被官方 SDK 证实，
  本项目主链路使用 LangGraph 显式状态机；`isolated_deep/` 里用官方
  `create_deep_agent`/subagents 做了隔离验证的等价替代，**主 app 未接入**。

---

## 5. 验证基线（2026-11-25）

- `244 passed, 1 skipped`（跳过项为需真实 Milvus 的集成测试）；写操作确认闸门 11 条
  契约 + 澄清闸门 9 条契约（槽位 / 意图候选 / 轮次上限 / 优先级 / 图内追问 / 清单声明）
- Ruff 本次改动文件全部通过
- 可插拔契约由 `tests/test_domain_plugin_template.py`、`tests/test_intent_plugin_template.py`、
  `tests/test_capabilities.py`、`tests/test_confirmation_gate.py`、`tests/test_clarify_gate.py`、
  `tests/test_template_boundary.py` 守护
- 真实装配冒烟：`/health`、`/ready`、`/openapi.json` 均 200，19 条路由全部注册；
  `/ready` 额外校验插件 `graph_keys` 声明的 `claims` / `investment` 两张图；
  默认插件集注册 `claim_workflow / claims_answer / investment_answer / policy_query / trade_gate`
  五个领域对话能力，`clarify` 为内核内置能力，`missing_handlers()` 为空

> 启动脚本注意：`start_mac.sh` 开了 `set -u`，在 echo 中文提示里 `$var` 后紧跟全角括号时
> Bash 会把 UTF-8 首字节并进变量名（如 `$app_log（` 误判为 `app_log\xef`），导致启动即报
> `unbound variable`。2026-11-24 已改为 `${var}` 花括号边界修复；新增脚本时同类写法请照此处理。

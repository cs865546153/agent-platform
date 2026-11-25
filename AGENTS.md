# AGENTS.md

本文件是项目级的 Agent 工作说明。根目录使用 `AGENTS.md`（复数、全大写）是通用约定；
个人学习笔记已归档至工作区根目录 `../teaching-archive/`，不作为执行规范。

最后全量扫描：2026-11-25（本轮复跑全量测试与关键路径核对；上次全量 2026-10-26）。

## 项目定位

理赔智能助手 + 投资知识库 RAG + 通用知识库 RAG，并带一套可插拔的意图路由
（JEV / LLM / Hybrid / Shadow）与可视化工作台。

核心链路：

```text
FastAPI
  → 接入层（RequestID / 认证 / 限流 / 审计 / 500 兜底）
  → 路由层（纯 URL 匹配，无模型）
  → 对话助手图 classify（intent_router：JEV/LLM/Hybrid，默认 shadow；失败回落内置分类）
  → 理赔状态机 / 投资图 / 通用问答
  → 多路召回 → RRF → BGE-Reranker → Top-K → 引用校验
  → LLM 流式回答（SSE）
```

## 环境

| 用途 | 环境 / 版本 |
| --- | --- |
| 主应用 | `.venv-py315`，Python 3.15.1 |
| Embedding / Reranker 服务 | `.venv-embed`，Python 3.12.15 |
| 工作台前端 | `static/`，原生 HTML/CSS/JS，无构建步骤 |
| 本地 LLM | OpenAI 兼容接口，默认 `http://127.0.0.1:8000/v1` |
| 本地向量库 | Milvus Lite，`.data/milvus.db`（macOS/Linux） |
| 远程/容器向量库 | `MILVUS_URI=http://127.0.0.1:19530` |
| 应用端口 | `8001` |
| Embed / Rerank 端口 | `8010` |

模型提供方由 `MODEL_PROVIDER` 选择：`local`（默认）、`deepseek`、`bailian`。
扫描时本机 `.env` 使用 `MODEL_PROVIDER=deepseek`（非密钥，随环境调整）；
不要把 `.env` 中的密钥写进文档、日志或提交。

个人学习目录已迁至工作区根目录 `../teaching-archive/My_Study/`，不属于本项目实现；不要把它当作规范或测试对象。

## 目录

```text
app.py                  应用入口：组合默认领域（create_app + default_registry）
app_factory.py          通用 FastAPI 装配、lifespan、SSE、工作台接口
apps/                   薄应用入口：claims / investment / unified，按需裁剪领域
settings.py             Pydantic 环境配置、URL/密钥校验
config.py               模型工厂、任务档位路由、LangChain 模型缓存
logging_setup.py        日志、请求 ID、轮转、脱敏
monitoring.py           Prometheus 指标（METRIC_PREFIX 可配）
health_check.py         健康与状态机验证
agents/                 接入层、理赔/投资/对话编排、审计、可观测性
  assistant_agent.py    /api/assistant/stream 的无状态对话图（按能力名动态建图，无业务词）
  claim_agent.py        理赔 LangGraph 状态机、Send 三专家、HITL
  investment_agent.py   投资域图、交易二次确认与人工审批
  http_middleware.py    请求 ID / 认证 / 限流 / 审计 / 异常兜底
  audit_store.py        append-only + 哈希链审计
  middleware_audit.py   PII 脱敏、模型层限流/缓存/重试/审计
  memory.py             会话窗口、摘要、实体冲突、Redis（可选）
  confidence.py         置信度评估与阈值规则
  risk_agent.py         理赔风险识别专家
  observability.py      LangFuse / 链路观测出口（可选）
tools/                  业务工具、RAG、Milvus、rerank、文档解析
  knowledge_rag.py      通用知识库多路召回 + RRF + rerank + 引用
  investment_rag.py     投资 Wiki RAG
  rag_retrieval.py      理赔条款检索与引用校验
  rerank.py             BGE-Reranker HTTP 客户端与词项降级
  vector_store.py       Milvus 集合与可达性检查
  milvus_conn.py        Milvus 连接参数与 keepalive（统一入口）
  document_analysis.py  附件校验、文本抽取、多模态单证识别
  claim_tool.py         理赔工具集（查询 / 核验 / 回写）
  claim_write_producer.py outbox 回写队列（幂等键、租约、去重）
  policy_store.py       本地模拟保单（demo，`.data/policies.sqlite`）
  material_check.py     材料齐套校验
  demo_clauses.py       合成理赔条款（演示用）
adapters/               真实业务适配器（HTTP + 熔断 + 降级）与 demo 后端
  legacy_adapter.py     旧接口兼容适配器
agent_core/             领域无关内核
  plugins.py            DomainPlugin 协议、AppContext、PluginRegistry（模板扩展点）
  capabilities.py       对话编排能力注册表：按能力名分派，内置 answer / guard / clarify
  confirmation.py       写操作确认闸门：按清单确定性裁决，写操作绝不落到自由问答
  clarification.py      澄清闸门：按 required_slots / 低置信候选确定性追问，不调用 LLM
  security.py           统一认证唯一事实源（API_KEY 优先，CLAIMS_API_KEY 兼容）、生产密钥校验
  sse.py                SSE 帧编码（app.py 已接入）
  intent/               意图路由公共层：JEV / LLM / Hybrid / Registry / adapter
  harness/              执行面策略：Harness + 循环/上下文/工具/预算 + 按领域注册表
domains/                领域插件（可增删，新增领域不改内核）
  registry.py           默认领域登记表（唯一"启用哪些领域"的事实源）
  base.py               插件基类 + LangGraph 中断/配置/HITL 恢复门控共用辅助
  claims/               示例域：理赔服务装配、图、核心路由、对话能力与演示路由
  investment/           示例域：投资服务装配、图、核心路由与对话能力
  example/              最小示例：演示新领域如何接入
TEMPLATE.md             模板定位、内核/领域边界、新增领域五步指南
configs/domains.yaml    领域/意图回退清单（含 description_en）；主事实源已随插件 manifest.yaml
models/                 领域无关模型（Pydantic / 工作台）+ 本地 BGE 权重目录（权重 gitignore）
data/investment_wiki/   投资 Wiki 源文件（约 55 chunks）
data/knowledge_base/    通用知识库源文件（当前约 3 chunks）
data/intent_golden.jsonl 意图评测集（39 条合成样本）
scripts/                启动、建索引、评测、验证、embed 服务、VPS 部署、create_domain 脚手架
templates/domain/       新领域骨架（manifest / plugin / handlers / routes / 冒烟测试）
docs/                   架构、RAG、结构化输出、平台工作流说明
demos/                  链路图、状态机图、模块演示
deploy/                 回归门禁、shadow 对拍、监控告警、runbook
isolated_deep/          官方 Deep Agents 独立依赖与运行时（不在主环境）
static/                 工作台前端
tests/                  pytest 测试
vendor/                 Python 3.15 下使用的纯 Python PyYAML wheel
.opencode/skills/       本机 Agent 技能（vps-trojan-proxy 等），与业务运行时代码无关
invest.md               投资域链路草稿；claims-ai-agent-architecture.html 为架构可视化
.workbuddy/             个人工作记忆（非规范，不作为实现依据）
.data/                  运行期数据，已 gitignore
```

## 启动与日志

推荐：

```bash
bash scripts/start_mac.sh
```

脚本会优先使用 `.venv-py315`，默认 `8001`，开发热重载开启；可用
`bash scripts/start_mac.sh 8002` 指定端口，`--no-reload` 关闭热重载；
`bash scripts/start_mac.sh --app=apps.claims_app:app` 只装理赔领域。

按助手裁剪入口（不启动不需要的领域）：

```bash
.venv-py315/bin/uvicorn apps.claims_app:app --host 127.0.0.1 --port 8001
.venv-py315/bin/uvicorn apps.investment_app:app --host 127.0.0.1 --port 8002
.venv-py315/bin/uvicorn apps.unified_app:app --host 127.0.0.1 --port 8003
```

`app.py` 仍等价于 `apps.unified_app`，是工作台默认入口。

Embedding / Reranker 服务：

```bash
.venv-embed/bin/uvicorn scripts.embed_server:app \
  --host 127.0.0.1 --port 8010
```

服务端点：`GET /health`、`POST /v1/embeddings`、`POST /v1/rerank`。

日志位置：

```text
.data/logs/app.log        # 应用日志，含 rag.retrieval
.data/logs/uvicorn.log    # uvicorn 启动/访问/异常日志
.data/audit.sqlite        # 审计库，含 details.retrieval
```

日志配置：

```dotenv
LOG_LEVEL=INFO
LOG_FILE=.data/logs/app.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
# LOG_JSON=true
```

约定：

- `LOG_FILE` 父目录自动创建。
- 使用 `RotatingFileHandler`，默认 10MB × 5。
- `LOG_MAX_BYTES=0` 关闭轮转。
- 日志文件不可写时退回 stderr，不阻断启动。
- `.env` 修改后必须重启 uvicorn。
- 不要把日志只写到容器文件系统；容器环境优先 stdout/stderr + `LOG_JSON=true`。

查看：

```bash
tail -f .data/logs/app.log
tail -f .data/logs/uvicorn.log
grep 'rag.retrieval' .data/logs/app.log
```

重启前先确认端口和进程：

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

## API 一览

| 方法 | 路径 | 认证 | 说明 |
| --- | --- | --- | --- |
| GET | `/health` | 无 | 存活探针 |
| GET | `/ready` | 无 | 就绪探针（插件 `graph_keys` 声明的图 + 审计库），未就绪返回 503 |
| GET | `/metrics` | `X-API-Key` | Prometheus 指标 |
| GET | `/` | 无 | 工作台页面 |
| POST | `/api/assistant/stream` | `X-API-Key` | 对话助手，SSE 流式 |
| POST | `/api/documents/analyze` | `X-API-Key` | 单证识别，1–6 个附件，内存处理 |
| POST | `/api/claims/process` | `X-API-Key` | 发起理赔案件 |
| GET | `/api/claims/{claim_id}` | `X-API-Key` | 查询案件状态 |
| POST | `/api/claims/{claim_id}/supplement` | `X-API-Key` | 补材料并恢复图 |
| POST | `/api/claims/{claim_id}/review` | `X-API-Key` + `X-Reviewer-Key` | 人工审核；拒赔必须给条款依据 |
| POST | `/api/investment/process` | `X-API-Key` | 投资咨询 / 交易指令 |
| POST | `/api/investment/{request_id}/confirm` | `X-API-Key` | 交易二次确认 |
| POST | `/api/investment/{request_id}/review` | `X-API-Key` + `X-Reviewer-Key` | 人工审批，未接真实下单 |
| POST | `/api/investment/rag/search` | `X-API-Key` | 投资知识库只读检索 |
| GET | `/api/rag/domains` | `X-API-Key` | 列出知识库 domain |
| POST | `/api/rag/search` | `X-API-Key` | 通用知识库检索 |
| POST | `/api/rag/answer` | `X-API-Key` | 检索 + LLM 回答 + 引用校验 |
| POST | `/api/demo/claims/process` | 无 | 本地演示，合成数据 |
| POST | `/api/demo/claims/process-live` | 无 | 演示 + 真实模型链路 |
| POST | `/api/demo/investment/process` | 无 | 投资演示 |

- 认证由 `EnterpriseHttpMiddleware` 与路由依赖共同负责，请求头为 `X-API-Key`；
  审核接口另需 `X-Reviewer-Key`。
- `APP_ENV=production` 启动时强制要求 `API_KEY`（兼容旧名 `CLAIMS_API_KEY`）与
  `REVIEWER_API_KEY`，长度至少 16 位且不能是常见弱口令；解析统一走
  `agent_core/security.resolve_api_key()`，中间件与路由依赖共用，不再有两套实现。
- 限流是进程内实现，多副本需要换 Redis。

## 编排链路

### 1. 对话助手：`POST /api/assistant/stream`

`agents/assistant_agent.py`，方案 A：无状态多节点图，图只做编排，LLM 流式回答由
`app.py` 外层执行器负责，以保留 token 级 SSE。

```text
START
  → classify（意图路由 → 领域清单解析出「能力名」）
  → plan（SSE trace：所选能力 / 依据 / 澄清 / 确认，非图节点）
  → 条件路由（节点集合 = 已注册能力名 + 内核内置 answer / guard / clarify）
      ├─ answer（内核）        → END（general 模型流式回答）
      ├─ guard（内核）         → END（案件型 / 高风险意图未实现时拦截，不自动执行）
      ├─ clarify（内核）       → END（缺必填槽位或低置信候选时确定性追问，不走 LLM）
      ├─ claims_answer / policy_query / claim_workflow（理赔插件注册）
      │     └─ claim_workflow：缺材料 → follow_up（最多 3 轮，超限转人工）
      │                          材料齐 → 三专家 + RAG 判责
      └─ investment_answer / trade_gate（投资插件注册）
            └─ trade_gate：只拦截，不下单
```

- 内核里没有业务领域名，也没有意图字符串：落点来自领域清单的
  `IntentSpec.capability`，处理器来自 `ctx.capabilities`（`agent_core/capabilities.py`）。
  清单未声明能力或没人实现时，按 `mode` / `risk` 回落：`case` 或 `high` → `guard`，其余 → `answer`。
- 分类模型输出的 `intent` 不再受内核枚举约束（`models/workbench.py` 的 `IntentName` 是
  受长度约束的 str）；清单外的意图只降级为通用问答并在 `route` trace 记 `declared=false`，
  分类结果非契约 JSON 时降级为 `未识别`，两者都不中断 SSE。
- 业务编号等字段统一进 `IntentResult.slots`（额外键自动收拢），内核不枚举字段名。
- 意图分类：优先走 `IntentRouter`（`application.state.intent_router`），默认 `shadow`
  （LLM 主路由 + JEV 影子）；路由异常回落 `get_model('classification')` 内置提示分类，
  两者最终都收敛为 `IntentResult`。
- 历史、`material_round` 与通用澄清轮次 `round` 由前端回传；该图没有 checkpointer。
- `needs_clarification` 不再只是 trace 标记：`ClarifyGate` 优先级高于 `ConfirmGate`，
  缺 `required_slots` 或低置信候选时落 `clarify`，追问文案来自 manifest 的
  `slot_labels` / `description`，超过 3 轮停止追问；澄清阶段不调用模型、不执行写操作。
- 通用问答和理赔问答当前**不走知识库 RAG**；只有投资意图会调用 `investment_rag`。
- 交易指令在对话链路只做拦截和提示，不会下单；正式流程走 `/api/investment/*` 的
  二次确认 + 人工审批。
- 修改意图分类或路由时，优先用真实 `/api/assistant/stream` 验证 SSE 事件顺序。
- 边界门禁：`tests/test_template_boundary.py` 用源码扫描钉住"内核不 import 业务模块、
  不出现意图字符串、不残留旧分支词汇、装配后无未实现能力"，改内核前先跑它。

### 2. 理赔状态机：`/api/claims/*`

`agents/claim_agent.py` + `AsyncSqliteSaver`：

```text
intake → policy（保单/证据核验）→ experts（Send 并行三专家）→ aggregate → route_decision
                                                              ├─ auto
                                                              ├─ review（HITL）
                                                              └─ investigate
```

- 专家：`damage` / `risk` / `liability`，结果用 `operator.add` 汇聚。
- 材料不齐最多追问 3 轮，超限转人工。
- 低置信度、缺证据、高风险只允许转人工/调查，不能仅凭分数拒赔。
- 拒赔必须提供核实后的 `clause_basis`。
- 写接口按 `claim_id` 加进程内锁；同一案号内容不一致返回 409。

### 3. 投资图：`/api/investment/*`

`agents/investment_agent.py`：

- 咨询/诊断/评估/产品查询 → 教育性建议，不构成交易指令。
- 交易指令 → 适当性校验 → 二次确认（interrupt）→ 人工审批（interrupt）。
- 只有审批通过且适配器真实执行时才标记 `executed`；当前没有接券商，不会真实下单。

## 意图路由：JEV / LLM / Hybrid / Shadow

**结论：JEV 路由层已接入主对话链路。代码默认 `shadow`（LLM 主路由 + JEV 影子）；
当前仓库 `.env` 已配置 `JEV_API_KEY` 并设 `INTENT_ENGINE=hybrid`（JEV 主 + LLM 兜底）。**

** 多agent 协作， 可以尝试使用 身份带入+jev意图识别 **

代码与配置：

- `app.py`：`build_assistant_intent_router(cfg, intent_model, domain_registry)` 装配时构建一次，
  领域清单从插件 `intent_manifest()` 聚合；显式传 `create_app(intent_router=...)` 时跳过构建。
  存入 `application.state.intent_router`，仅内核自建时 `aclose()`；初始化失败只记 warning。
- `agent_core/intent/adapter.py`：`AssistantIntentRouter` 把工作台请求转成
  `IntentRequest`，并把 `RouteDecision` 转成 `IntentResult`（含保单号兜底抽取）。
- `agent_core/intent/jev.py`：`JevIntentClassifier`，调用 `ateve-jev-v1` 兼容接口；
  `POST {JEV_BASE_URL}`，Bearer 认证，一次请求同时返回 `domain` 与 `intent`；
  对 `429` / `5xx`（含网关 `529 model busy`）按 `JEV_MAX_RETRIES` 做指数退避重试。
- `agent_core/intent/llm.py`：LLM 路由器，中文场景默认主路由，也是 JEV 的兜底。
- `agent_core/intent/hybrid.py`：主路由 + 兜底 + 影子组合；影子并行执行，只记录不生效。
  主路由异常记 `intent.primary_failed`，低置信记 `intent.primary_low_confidence`，
  最终结果再由 `app.py` 记 `intent.route`。
- `agent_core/intent/factory.py`：`shadow / hybrid / jev / llm` 四种预设。
- `agent_core/intent/registry.py`：`DomainRegistry.from_plugins()` 从插件 `manifest.yaml`
  聚合领域清单，`configs/domains.yaml` 仅回退补齐无插件领域（如 general）。
  `agents/assistant_agent.py` 的 `IntentRoutingPolicy` 据此把 `(领域, 意图)` 解析成
  **能力名**（`IntentSpec.capability`，未声明时按 `mode`/`risk` 回落），
  新增领域不要改图代码里的意图字符串。JEV 使用 `description_en`，新增意图务必补英文描述。
- `agent_core/capabilities.py`：`AssistantCapabilities` 注册表（能力名全局唯一，
  即对话图节点名）、`Redirect` 意图改写钩子、`with_kernel()` 补内置 `answer`/`guard`/`clarify`、
  `missing_handlers()` 供装配期自检（`app.py` 会打 warning）。
- `agent_core/confirmation.py`：`ConfirmGate` 写操作确认闸门。从聚合清单读取
  `IntentSpec.confirmation`，对声明契约或 `risk=high`/`mode=case` 的意图确定性裁决
  "需确认"，`enforce()` 把误落 `answer` 的写操作改派 `guard`；`audit_capabilities()` /
  `audit_routes()` 供装配自检（落点错误 / `api` 死链）。裁决并入对话图已有 done trace，
  只加字段不新增事件。**新增写操作意图必须在插件 manifest 声明 `confirmation`**，
  内核不写死业务路径。契约由 `tests/test_confirmation_gate.py` 守护。
- `agent_core/clarification.py`：`ClarifyGate` 澄清闸门。从 `IntentSpec.required_slots` /
  `slot_labels` 与路由层 `needs_clarification` / `intent_probabilities` 确定性裁决；
  槽位级优先（缺什么问什么）、意图级其次（清单 `description` 生成候选项），
  优先级高于确认闸门，文案不写死在内核。**新增需追问要素的意图必须在 manifest 声明
  `required_slots`**；契约由 `tests/test_clarify_gate.py` 守护。
- `agent_core/harness/`：执行面策略。`Harness` 默认只做"裁剪消息 → 模型流式 → 内容钩子"，
  并提供 `LoopPolicy`（步数/工具次数/截止时间）、`ContextPolicy`（条数/token 预算，
  保留 system）、`ToolPolicy`（白名单/黑名单/调用上限）、`BudgetPolicy`（token/成本/秒）、
  `HarnessHooks`（before_model / on_chunk / on_event）。`HarnessRegistry` 按域名查找，
  未注册用默认策略。接入点：`create_app(harnesses={'claims': ...}, assistant_harness=...)`，
  或插件在 `build_services` 写 `ctx.harnesses`；对话处理器可从 `state['harnesses']` 取。
  领域只换一个维度时改 policy，要接管整套循环则继承 `Harness` 重写 `stream_answer` /
  `run_graph`。默认策略不改变现有 SSE 事件序列；契约由 `tests/test_harness.py` 守护。
- `settings.py` / `.env.example`：`INTENT_ROUTER_ENABLED`、`INTENT_*`、`JEV_*`，
  重试相关为 `JEV_MAX_RETRIES`（默认 2）和 `JEV_RETRY_BACKOFF_S`（默认 0.25）。
- 测试：`tests/test_intent_jev.py`、`tests/test_intent_hybrid.py`、
  `tests/test_intent_registry.py`、`tests/test_intent_router_integration.py`、
  `tests/test_capabilities.py`、`tests/test_template_boundary.py`。
- 评测：`scripts/eval_intent.py` + `data/intent_golden.jsonl`。

预设语义：

| `INTENT_ENGINE` | primary | fallback | shadow |
| --- | --- | --- | --- |
| `shadow`（代码默认） | `llm` | 无 | `jev` |
| `hybrid` | `jev` | `llm` | 无 |
| `jev` | `jev` | 无 | 无 |
| `llm` | `llm` | 无 | 无 |

- `INTENT_PRIMARY / FALLBACK / SHADOW` 可覆盖预设；空值用预设，`none`/`off` 关闭该角色。
- 默认阈值：置信度 `0.8`、margin `0.2`；低于阈值或 `needs_clarification` 时触发兜底，
  兜底后仍不确定则由 `ClarifyGate` 生成候选项进入 `clarify` 节点。
- JEV 是英文优先模型；中文输入可开 `INTENT_TRANSLATE_TO_ENGLISH=true` 走 LLM 翻译。
  未开翻译时会记录 `cjk_without_translation` 警告。
- 发送前先 `mask_pii` 再翻译/请求，PII 不离开本进程。
- 缺少 `JEV_API_KEY` 时，factory 会跳过 JEV 影子并记录 warning，不会让服务启动失败；
  但把 JEV 设为主路由且无 key 会直接报错。
- `JEV_API_KEY` 只从环境变量/.env 读取，不写仓库、不写报告、不打印日志。

**当前实际状态（重要）**：

- `/api/assistant/stream` 的 `classify` 节点优先调用 `application.state.intent_router`；
  路由异常（JEV 超时、LLM 返回非契约 JSON、领域/意图不在注册表）时回落
  `assistant_agent` 内置分类模型，SSE 不中断，trace 记录 `route_error`。
- 当前 `.env`：`INTENT_ROUTER_ENABLED=true`、`INTENT_ENGINE=hybrid`、`JEV_API_KEY` 已配置；
  代码默认仍是 `shadow`，新环境未配 key 时会自动跳过 JEV 影子。
- 2026-10-27 实测：39 条评测集 JEV 联合准确率 `0.897`、领域 `1.000`、P50/P95 `571/628ms`；
  真实 SSE 验证 `engine=jev` 命中交易指令、保单查询，低置信样例正确触发 `fallback_used=true`。
- 2026-10-27 VPS 实测：JEV 网关偶发 `529 model busy`（直连 20 次失败 2 次）；已加
  `429/5xx` 重试。`intent.route` 记录的是兜底后的最终结果，主路由尝试看
  `intent.primary_low_confidence` / `intent.primary_failed`，重试看 `intent.jev_retry`。
- `INTENT_ROUTER_ENABLED=false` 可整体关闭路由层；`INTENT_ENGINE=jev/hybrid` 但缺
  key 时 `build_assistant_intent_router()` 返回 `None`，主链路回落内置分类。
- SSE `classification` trace 新增：`engine`（`jev`/`llm`/`llm-prompt`）、
  `route`（domain/intent/confidence/margin/needs_clarification/fallback/warnings/shadow）、
  `route_error`。
- `classify` 后追加 `plan` trace：`selected`（domain/intent/capability）、`basis`
  （declared/fallback/redirect）、`missing_slots`、`needs_clarification`、
  `needs_confirmation`；澄清落 `clarify` 时另发 `clarification_status`
  （`status=asking`/`round`/`max_rounds`/`missing_slots`/`candidates`）。
- 启用 JEV：先配 `JEV_API_KEY`，选 `INTENT_ENGINE=hybrid` 或 `jev`，用评测集对比
  准确率、兜底率和延迟后再放量；影子模式会与主路由并行，延迟取两者较慢值。
- `needs_clarification` 现在会由 `ClarifyGate` 决定是否进入 `clarify` 节点；这里没有
  LangGraph `interrupt`/checkpointer，追问以 SSE 内容结束本轮，状态由前端 `round` 回传。

意图评测：

```bash
.venv-py315/bin/python -m scripts.eval_intent --engine jev --data data/intent_golden.jsonl
.venv-py315/bin/python -m scripts.eval_intent --engine shadow --limit 20
.venv-py315/bin/python -m scripts.eval_intent --engine hybrid --min-accuracy 0.9
.venv-py315/bin/python -m scripts.eval_intent --engine jev --translate
```

- 评测集当前 39 条合成样本（`data/intent_golden.jsonl`）。
- 报告默认写到 `.data/intent_eval.json`，包含准确率、混淆矩阵、兜底率、影子一致性和延迟。

## 模型与任务路由

- `config.py` 统一创建 `ChatOpenAI`；导入时不发网络请求。
- 三档通用模型：`flash`（分类/轻量）、`plus`（通用/客服）、`max`（强推理）；
  另有 `vision` 处理图片单证。
- 任务档位见 `TASK_PROFILES`：`classification` 用 flash，`customer_service` / `investment_advice`
  用 plus，`coverage` / `assessment` / `risk` / `decision` 用 max。
- 未知任务调用 `get_model()` 会显式报错，不自动降级判责。
- DeepSeek 默认关闭思考模式（`DEEPSEEK_THINKING=false`）；开启会占用 `max_tokens`。

## RAG 约定

### 通用 / 投资知识库

- 使用共享 `split_markdown_sections()` 切分 Markdown。
- 文件标题、免责声明合并到第一个 `##` 章节，不单独成块。
- `参考资料 / 参考来源 / 参考文献 / 延伸阅读 / 相关链接` 章节不进入索引。
- 纯风险提示/短免责声明块会被过滤。
- 通用知识库支持 `.md/.txt/.pdf/.docx/.html`。
- 通用知识库与投资 Wiki 共用 `KnowledgeBase`，三路召回：
  - `dense`：BGE-M3 embedding + Milvus，语义召回；
  - `bm25`：本地 `BM25Okapi`，中文按“单字 + 双字”切 token，负责专名/数字精确匹配；
  - `metadata`：`title / chapter / topic / source` 精确匹配 + token 重叠。
- 三路权重：`dense=0.55`、`bm25=0.30`、`metadata=0.15`，加权 RRF `k=60`，再交 BGE-Reranker 精排。
- 向量不可用：自动降级为 `bm25 + metadata` 两路，`vector_ready=false`，原因写进 `route_errors`。
- `RERANK_TOP_N` 是进入 BGE-Reranker 的候选池上限；实际候选数为
  `min(fused_count, RERANK_TOP_N)`，并至少保留 Top-K。
- BGE 不可用时自动降级为词项重排，`rerank_fallback=true`；失败后默认冷却 60 秒。
- 检索诊断字段：
  - `route_hits`
  - `route_errors`
  - `fused_count`
  - `rerank_candidates`
  - `rerank_top_n`
  - `reranked`
  - `rerank_fallback`
  - `rerank_error`
  - `returned`
- 每次检索同时：
  - 写入结构化 `rag.retrieval` 日志；
  - 追加到审计 `details.retrieval`；
  - 返回 `retrieval` 给 API。
- 引用校验只允许回答使用本轮检索到的 `citation_id`；非法引用必须被过滤或报错。

### 索引重建

```bash
python scripts/build_investment_rag.py
python scripts/build_knowledge_rag.py --domain general
# 也支持 --domain investment；--root / --collection 可覆盖默认值
```

注意：

- 修改切块规则会改变 `doc_id`。
- `upsert` 只覆盖同 ID 记录，不会删除旧 chunk。
- 切块规则变化后，先删除对应 collection 再重建：
  - `investment_wiki`
  - `knowledge_general`
- 否则 dense 召回仍可能命中旧标题块或参考资料块。

### 理赔 RAG

- 理赔条款使用独立集合 `claims_<insurance_type>`。
- 理赔条款走 `tools/rag_retrieval.py` 的 `HybridRetriever`，只有 `vector + bm25` 两路，
  默认权重 `0.7/0.3`、加权 RRF `k=60`，并按 `insurance_type / version` 过滤；
  不要和通用/投资知识库的三路召回混淆。
- 默认 `drop_old=False`，不要随意自动清空。
- 仓库没有内置真实条款语料；演示用 `tools/demo_clauses.py` 的合成条款。
- 修改理赔检索行为时必须跑相关测试，不要影响投资/通用 RAG。

### 当前验证基线（2026-10-26）

`POST /api/assistant/stream`，问题 `投资基础知识`：

```text
intent=投资咨询
corpus_size=55
route_hits={dense:16, bm25:16, metadata:16}
route_errors={}
fused_count=27
rerank_candidates=20
reranked=true
rerank_fallback=false
returned=4
```

- citations：`K1` 投资基础/1、`K2` 合规边界/5、`K3` 流动性/5、`K4` 核实平台和人员/3。
- 回答引用 `[K1]`～`[K4]` 全部属于本轮检索结果，无非法引用。
- 总耗时约 12.8s；分类约 1.0s，投资回答生成约 6.7s。
- 宽泛问题下 K2/K4 偏合规/防诈骗，但回答用于风险提示和平台核实，属于可接受结果。
- 该基线使用当时模型/网络环境；切到 DeepSeek 或换模型后延迟会变化，但检索诊断字段应保持稳定。

### 本地 / 远程 Milvus

- 本地 Lite：`VECTOR_STORE_URI=./.data/milvus.db`。
- 远程：删除 `VECTOR_STORE_URI` 或设为 HTTP 地址。
- 不要把 `.db` 路径写进 `MILVUS_URI`。
- 同一个 `.data/milvus.db` 只允许一个进程写；不要开多个 uvicorn worker。
- Milvus 连接参数集中在 `tools/milvus_conn.py`（含 keepalive 配置），不要各处手拼。

## 数据与持久化

```text
.data/checkpoints.sqlite  理赔/投资图检查点（AsyncSqliteSaver）
.data/audit.sqlite        接入层审计，只追加 + 哈希链，可 verify_chain
.data/policies.sqlite     本地模拟保单（PolicyStore，demo）
.data/outbox.sqlite       回写队列（组件已实现，主图未注入；入队 ≠ 付款）
.data/milvus.db           Milvus Lite 向量库
.data/logs/               应用与 uvicorn 日志
```

- `.data/` 全部 gitignore，不要提交。
- 审计哈希链可用 `scripts/audit_verify.py` 校验。
- 单机 SQLite + Milvus Lite 只适合单进程；多副本前换 Postgres/Redis + 远程 Milvus。
- Outbox 的消费者必须使用同一幂等键去重，才能覆盖“业务成功但本地未确认”的窗口。

## 测试与质量

```bash
# 全量
.venv-py315/bin/python -m pytest -q

# 指定模块
.venv-py315/bin/python -m pytest -q \
  tests/test_knowledge_rag.py \
  tests/test_investment_rag.py \
  tests/test_intent_jev.py \
  tests/test_intent_hybrid.py \
  tests/test_intent_registry.py \
  tests/test_intent_router_integration.py \
  tests/test_capabilities.py \
  tests/test_confirmation_gate.py \
  tests/test_clarify_gate.py \
  tests/test_harness.py \
  tests/test_create_domain.py \
  tests/test_app_shells.py \
  tests/test_template_boundary.py \
  tests/test_logging_setup.py

# Ruff
.venv-py315/bin/python -m ruff check <changed-files>

# HTTP 冒烟（独立检查点库，不碰业务数据）
bash scripts/verify.sh
```

当前基线（2026-11-25 执行面 harness 接入后实跑）：

- 全量测试：`258 passed, 1 skipped`（含领域可插拔、意图路由可插拔、能力分派、
  写操作确认闸门 11 条契约、澄清闸门 9 条契约、执行面 harness 7 条契约、
  create-domain 脚手架 4 条契约、薄应用入口 3 条契约、`/ready` 图声明校验、内核边界门禁、
  P0 认证一致性回归；跳过项为需真实 Milvus 的集成测试）。
- 模板相关契约：`tests/test_domain_plugin_template.py`、
  `tests/test_intent_plugin_template.py`、`tests/test_capabilities.py`、
  `tests/test_confirmation_gate.py`、`tests/test_clarify_gate.py`、
  `tests/test_harness.py`、`tests/test_create_domain.py`、`tests/test_app_shells.py`、
  `tests/test_template_boundary.py`（改内核路由前必须先跑）。
- 跳过项：`tests/test_milvus_integration.py`，需要真实 Milvus。
- 已知无害日志：测试结束时 `milvus_lite` 可能打印
  `ValueError: I/O operation on closed file`，不影响结果。
- Ruff：`line-length=110`，`target-version=py312`，只启用 `E4/E7/E9/F`。
- 测试里大量 Pydantic/LangChain 弃用 warning，属当前依赖版本预期。

## 部署

- `Dockerfile`：Python 3.14，非 root（uid 10001），单 worker，暴露 8001。
- `Dockerfile.embed`：Python 3.12，CPU torch，加载 BGE-M3 + Reranker。
- `docker-compose.prod.yml`：`app` + `embed`，app 只绑定 `127.0.0.1:8001`；
  `data/`、`static/`、`models/` 只读挂载，`.data/` 持久化。
- VPS 形态：nginx TLS + Basic Auth，`/ai/` 反代到 `127.0.0.1:8001`，并去掉前缀。
- 当前实际部署脚本：`scripts/deploy_vps.sh`（rsync + Compose），不是根目录 `deploy.sh`。
- `APP_ENV=production` 会强制校验 API key；当前线上仍是 `APP_ENV=development` +
  nginx Basic Auth，因为前端还不会发 `X-API-Key`。
- `.github/workflows/deploy.yml` 只在 `main` push 触发，且 deploy job 仍假设
  “服务器 Git 仓库 + `deploy.sh`”，与 rsync 流程未统一。
- 容器日志优先 stdout/stderr + `LOG_JSON=true`；不要把日志只留在容器层。

## 修改约定

- 不要主动 commit / push，除非用户明确要求。
- 不要删除或覆盖不认识的未提交改动；`README.md` 可能有用户未提交修改。
- 不要提交 `.env`、`.data/`、`models/bge-*`、虚拟环境、密钥和临时产物；
  `vendor/pyyaml-*.whl` 是 Python 3.15 安装所需的例外，需要保留。
- 日志和审计中的输入输出必须经过 `mask_pii`；不要把密钥、手机号、身份证、银行卡写入日志。
- 新增领域优先用脚手架：`.venv-py315/bin/python -m scripts.create_domain <name> --description "..."`，
  生成 `domains/<name>/` 与冒烟测试；再按需补 manifest / handler / 路由。
- 新增需追问要素的意图必须在插件 manifest 声明 `required_slots` / `slot_labels`；新增有业务图的
  插件必须声明 `graph_keys`；需要不同循环/上下文裁剪的领域在 `build_services` 注册
  `ctx.harnesses`；改 SSE 序列时同步前端静态节点与序列断言测试。
- 修改行为、配置、接口、RAG 或意图链路时，同步更新：
  - `README.md`
  - `Architecture.md`
  - `TEMPLATE.md`（内核/领域边界与新增领域步骤，改装配点时必须同步）
  - `docs/knowledge-rag.md`
  - `docs/investment-rag.md`
  - `docs/platform-architecture-workflow.md`
  - `demos/完整链路图.md`
  - `.env.example`
  - `AGENTS.md`
- 往 `app.py` / `agent_core/` / `models/workbench.py` 里加代码前，先跑
  `tests/test_template_boundary.py`：内核不 import 业务模块、不写意图字符串、
  不枚举业务字段名；需要领域信息时走 manifest + `ctx.capabilities`。
- `docs/platform-architecture-workflow.md` 中的 `platform/intent/` 旧路径已清理；
  实际代码在 `agent_core/intent/`，新增代码不要再使用 `platform` 作为包名
  （会遮蔽标准库 `platform`）。
- 保持现有代码风格：Python 类型标注、中文注释/文档、Ruff 通过。
- 涉及运行期行为时，除了单元测试，尽量用真实 `/health`、`/ready`、`/api/.../search` 验证。
- 重启用户正在使用的 uvicorn 前先确认端口和进程；不要直接 `kill -9`。

## 已知边界

- **JEV 已接管主路由（hybrid）**：路由层已接入 `/api/assistant/stream`，当前 `.env` 为
  `INTENT_ENGINE=hybrid`（JEV 主 + LLM 兜底）；39 条评测联合准确率 89.7%，
  主要残余混淆是「组合诊断 ↔ 投资咨询」，扩充语料前不要切 `jev` 单引擎。
- **JEV 中文边界仍不稳**：官方定位英文优先；39 条合成样本联合准确率 89.7%，
  错误集中在「组合诊断 ↔ 投资咨询」；当前用 `hybrid` + LLM 兜底，暂不切单引擎。
- **翻译开关当前不可用**：`LLMTranslator` 复用 `classification` 模型（`json_object`），
  DeepSeek 会因 prompt 不含 “json” 返回 400；开启 `INTENT_TRANSLATE_TO_ENGLISH=true`
  前需先改为 text 模式模型，并重新跑评测。
- **对话编排已能力化，但工作台请求模型仍带示例形态**：`agents/assistant_agent.py` 与
  `app.py` 里不再有领域名或意图字符串，落点全部来自领域清单的 `capability`；
  但 `AssistantRequest.claim_id` / `material_round` 仍是示例理赔工作台的字段
  （内核只透传，不解释），通用澄清轮次新增 `round`（兼容 `material_round`，上限 3），
  原创业务需要替换为自己的案件主键。
- **HITL 中断跨进程会降级**：LangGraph 重启后 `tasks[].interrupts` 不再重建，
  只有 `snapshot.next` 保留。恢复门控与响应阶段必须用 `domains/base.py` 的
  `gate_resume(node=...)` / `resume_gate(...)` 按待执行节点重建，只看 `interrupt_kind`
  会让重启后的 `/confirm`、`/review` 永远 409（投资域曾如此，已修并有回归测试）。
- **真实理赔闭环未完成**：`BusinessBackend` 只内置 `query_policy`，
  `verify_claim_evidence` 需注册适配器；`CORE_QUERY_URL` 为空时安全降级为转人工。
- **真实条款语料缺失**：支持 `claims_<险种>` 集合，但仓库没有真实条款，主要用演示条款。
- **RAG 评测缺失**：有意图评测，但没有覆盖投资/通用/理赔的 golden QA、
  Recall@K / MRR / nDCG / 引用准确率回归。
- **Reranker 懒加载**：`/health` 在 reranker 未加载时仍可能返回 ok，首次检索会多几秒。
- **单机限制**：SQLite + Milvus Lite 只适合单进程；多副本需换基础设施。
- **线上安全**：仍是 development + nginx Basic Auth，需换默认口令、补 nginx 限流、
  SSH 密钥、日志轮转、`.data` 备份和外部监控。
- **CI/CD 未统一**：工作流与 `scripts/deploy_vps.sh` 的 rsync 流程不一致，需要二选一。

## 后续规划（未完成，按优先级）

1. **`packages/` 拆包（P1）**
   - 目标：把 `app_factory.py`、`agents/`、`tools/`、`models/` 收拢到 `packages/`（或 `src/` 布局），
     `app.py` / `apps/*` 只保留组合根；先留兼容导入壳，分两步迁移，避免一次性打断现有测试。
   - 验收：全量测试与 `scripts/verify.sh` 通过；`import app`、`apps.claims_app`、
     `uvicorn app:app` 与 `uvicorn apps.*:app` 均可用；文档同步更新路径。
2. **领域级配置与数据隔离（P1）**
   - 现状：`investment_rag_*` 等示例领域配置仍在全局 `Settings`，`.data/` 也是全局路径；
     同时启动多个助手会共享检查点、审计、outbox 与向量库。
   - 目标：插件声明/读取自己的配置前缀（如 `INVESTMENT_*` / `CLAIMS_*`），应用入口可覆盖
     checkpoint / audit / outbox / vector 路径；未声明时回退全局默认，保持单助手兼容。
   - 验收：`apps.claims_app` 与 `apps.investment_app` 可用不同 `.data` 目录并行启动；
     新增配置隔离与路径隔离契约测试；`AGENTS.md`、`TEMPLATE.md`、`README.md` 同步。
3. **可选增强（P2）**
   - RAG golden QA 与 Recall@K / MRR / nDCG / 引用准确率回归；
   - 真实理赔适配器闭环（保单查询、单证核验、回写/通知）；
   - CI/CD 二选一（GitHub Actions 与 `scripts/deploy_vps.sh` 对齐）。

## 关键文档

| 文档 | 内容 |
| --- | --- |
| `README.md` | 安装、启动、接口、配置、VPS 部署 |
| `Architecture.md` | 分层架构、三条业务链路、RAG、意图路由与 SQLite 职责 |
| `docs/knowledge-rag.md` | 通用 RAG、多路召回、rerank、审计诊断 |
| `docs/investment-rag.md` | 投资 Wiki、索引构建、降级策略 |
| `docs/platform-architecture-workflow.md` | 平台架构、意图路由、演进方向 |
| `docs/structured-output.md` | 结构化输出选型与迁移 |
| `TEMPLATE.md` | 模板定位、内核/领域边界、新增领域指南 |
| `deploy/runbook.md` | 发布、回退与运维 |
| `demos/完整链路图.md` | 端到端链路 Mermaid |
| `demos/理赔Agent端到端状态机图.md` | 理赔状态机 Mermaid |

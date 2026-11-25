# 企业级 Agent 平台模板架构（Agent Platform Template）

> 内核 + 领域插件；理赔 / 投资 / 通用知识库 RAG 为随模板提供的示例领域。
> 使用与扩展方法见 [TEMPLATE.md](TEMPLATE.md)，本文描述系统实际如何组织。

> 2026-11-24 模板化重构：抽出领域无关内核（`app.py` / `agent_core/`），
> 业务领域降为 `domains/` 下可插拔插件；新增领域不改内核。教学资料已归档至
> 工作区 `../teaching-archive/`，不在运行时内。

## 一、全景分层架构

```
【前端层】
  智能工作台 UI（聊天 / 单证上传 / 链路可视化）
       |
       v   HTTP / SSE
【接入层】 EnterpriseHttpMiddleware（纯 ASGI，每个请求；领域无关）
  CORS -> RequestID -> 认证 -> 体积上限 -> 限流 -> 审计 -> 500 兜底
  认证：/api/* 校验 X-API-Key；Depends 与接入层中间件共用 agent_core/security.resolve_api_key()
        （API_KEY 优先，CLAIMS_API_KEY 兼容旧部署）；生产启动强制配置密钥
  限流：身份 + IP + 接口类别（进程内；多副本需 Redis）
  审计：脱敏入参 + RAG 检索诊断写入 audit.sqlite，只追加 + 哈希链
       |
       v
【内核装配层】 app_factory.py (FastAPI) —— 唯一扩展点 create_app(registry=)；app.py/apps/* 是组合根
  通用路由：/health · /ready · /metrics · /api/rag/* · /api/assistant/stream · /api/documents/analyze
  领域路由：由 domains/<域>/routes.py 注册，装配层不认识具体业务
  演示路由：/api/demo/claims/* 由 ClaimsPlugin 注册（不装 claims 插件即不存在）
       |
       |  lifespan 中按 registry 装配 AppContext
       |  （services / graphs / rag / slot_extractors / locks / settings）
       v
【领域插件层】 domains/ —— 可增删，新增领域不改内核
  PluginRegistry（domains/registry.py，启用哪些领域的唯一事实源）
    ├─ ClaimsPlugin     : /api/claims/*  + 理赔状态机（链路②）      [默认启用]
    ├─ InvestmentPlugin : /api/investment/* + 投资图、二次确认、人工审批 [默认启用]
    └─ ExamplePlugin    : /api/example/echo（最小接入骨架，按需注入，默认不启用）
  每个插件实现：build_services / build_graphs / register_routes（见下节）
       |
       v
【内核能力层】 agent_core/ —— 领域无关
  plugins.py  : DomainPlugin 协议、AppContext、PluginRegistry
  capabilities.py : 对话编排能力表（内核只内置 answer / guard / clarify，其余由领域注册）
  confirmation.py : 写操作确认闸门（按清单确定性裁决，写操作绝不落自由问答）
  clarification.py : 澄清闸门（required_slots / 低置信候选，确定性追问，不调用模型）
  harness/    : 执行面策略（Harness + 循环/上下文/工具/预算），领域可按域名覆盖
  flow/       : 业务控制层（plan/assemble/execute/observe/decide + 版本/确认/轨迹）
  security.py : 统一认证与生产密钥校验；sse.py : SSE 帧编码（app.py 已接入）
  intent/     : JEV / LLM / Hybrid / Shadow（清单从插件 manifest 聚合，YAML 仅回退 general）
       |
       v
【编排层】（领域图由各插件构建，checkpointer 由内核在 lifespan 注入）
  链路① 对话助手(LangGraph)  : classify(IntentRouter) -> plan trace -> 按能力名分派（answer / guard / clarify / 领域能力）
  链路② 理赔状态机(LangGraph) : intake -> policy -> experts(子图) -> auto/review/investigate
       └ experts 子图          : expert(Send 扇出三专家) -> aggregate
  链路③ 工具/规划(未接主app)  : AgentExecutor、Plan-and-Execute、RunnableWithMessageHistory
       |
       v
【服务 / 适配器层】（领域自行装配）
  WorkflowServices（policy / verify / expert + clause_retriever 条款检索）
  MiddlewareChain（三专家：脱敏 -> 按案限流 -> 缓存 -> 推理 -> 降级 -> 审计 -> HITL 标记）
  BusinessBackend -> CoreSystemAdapter（HTTP + 熔断 + 只读缓存）-> 核心系统
  DemoBackend     -> PolicyStore（SQLite: policies.sqlite）
       |
       v
【模型层】
  flash（分类/轻量/降级）· plus（通用/客服/主专家）· max（责任专家/强推理）· vision（单证视觉）
  JEV（远程意图分类，可选；ateve-jev-v1，英文优先）
       |
       v
【数据层】
  SQLite : checkpoints.sqlite（状态机检查点）
           audit.sqlite     （请求审计，只追加 + 哈希链）
           policies.sqlite  （本地保单）
           outbox.sqlite    （回写队列；组件已实现，主图未注入）
  Redis（会话历史，可选）· Milvus（条款向量，可选）· 核心系统 HTTP（真实保单）

【横切】 接入层：RequestID / 认证 / 限流 / 审计；意图路由：JEV / LLM / Hybrid / Shadow；
        模型层：脱敏 / 缓存 / 降级；观测：LangFuse / Prometheus / stdout JSON
```

## 一·补充、生产部署形态（单机）

```text
公网
  |-- 443 nginx
  |     |-- /                    -> trojan-web 面板
  |     `-- /ai/  (Basic Auth)   -> 127.0.0.1:8001
  |                                  app 容器（FastAPI / SSE）
  |                                  `-- http://embed:8010/v1
  |                                        embedding / reranker 容器
  |-- 444 trojan
  `-- 22  SSH
```

- 应用不监听公网，只绑定 `127.0.0.1:8001`；nginx 负责 TLS、Basic Auth、上传体积和 SSE 不缓冲。
- `/ai/` 反向代理时去掉前缀：`/ai/api/... -> /api/...`、`/ai/static/... -> /static/...`。前端资源与 API 使用相对路径，因此本地 `/` 和线上 `/ai/` 共用一份代码。
- BGE-M3 与 BGE-Reranker 放在独立 `embed` 容器，模型以只读 bind mount 挂载；应用和模型服务共享 Compose 网络。
- `.data/` 使用 bind mount 持久化 Milvus Lite、审计库、检查点和日志。不要多进程写同一个 Milvus Lite 文件。
- 当前前端不发送 `X-API-Key`，公网入口使用 `APP_ENV=development` + nginx Basic Auth；若要切到 `APP_ENV=production`，必须同时解决前端密钥注入。

## 一·架构核心、内核与领域插件（模板化）

模板的核心承诺：**新增一个业务领域不需要修改内核，只需实现插件并登记。**

**三个边界**

| 层 | 位置 | 职责 | 是否认识业务 |
|---|---|---|---|
| 内核装配 | `app_factory.py`（组合根 `app.py` / `apps/*`） | `create_app(registry=)` 装配全局、注册通用路由、提供 lifespan | 否（只认插件协议） |
| 内核能力 | `agent_core/` | plugins 契约、认证、SSE、意图路由框架、执行面 Harness 策略、业务控制层骨架 | 否 |
| 领域插件 | `domains/<域>/` | 服务装配、图构建、领域专属路由 | 是（各领域自持） |

**PluginRegistry 是唯一启用开关**：`domains/registry.py` 的 `default_registry()` 返回要启用的插件列表。
测试里可传 `create_app(registry=PluginRegistry([MyDomainPlugin()]))` 只装某个领域，
无关路由不会注册——`tests/test_domain_plugin_template.py` 守护这一契约。

**插件生命周期**（内核在 lifespan 中按序调用，依赖全部经 `AppContext` 传递）：

```text
create_app(registry)
  -> 各插件 build_services(ctx, overrides)   # 装配领域依赖；overrides 是测试注入点
  -> lifespan: AsyncSqliteSaver 建立 checkpointer
  -> 各插件 build_graphs(ctx, saver)          # 构建 LangGraph，saver 由内核注入
  -> 各插件 register_routes(ctx)              # 注册领域专属路由，不 import app.py
  -> 领域图留在 ctx.graphs，由插件路由自行取用；内核不再按领域名挂全局 state
```

`AppContext` 持有 `services` / `graphs` / `rag` / `slot_extractors` / `locks` /
`settings` / `application` / `intent_router` / `intent_registry`，是内核与插件之间
唯一的依赖通道；插件不直接 import `app.py`，内核也不 import 任何领域类型。

**意图路由同样可插拔**（领域与意图事实源随插件走，内核不再写死）：

```text
插件 manifest.yaml（intent_manifest hook）
  -> DomainRegistry.from_plugins(plugins, fallback=DomainRegistry.default())
       插件声明优先；YAML 只补 general；同名内容分叉时启动抛 ValueError
  -> build_assistant_intent_router(cfg, intent_model, domain_registry, slot_extractor)
       显式 intent_router 注入时跳过构建，内核不接管其生命周期
       slot_extractor 来自 AppContext.slot_extractors（领域插件注册的业务槽位规则）
   -> IntentRoutingPolicy.from_registry(intent_domain_registry, ctx.capabilities,
        confirm_gate, clarify_gate)
        意图 -> IntentSpec.capability（能力名）+ 领域处理器；内核只内置 answer / guard / clarify
   -> ConfirmGate.from_registry(...)（确定性裁决，不调用 LLM）
        声明 confirmation / risk=high / mode=case 一律判"需确认"；
        误落 answer 时强制改派 guard，其余领域能力原样保留
   -> ClarifyGate.from_registry(...)（确定性追问，优先级高于确认，不调用 LLM）
        required_slots 缺槽位 / needs_clarification 低置信 -> 内核 clarify 节点
   -> assistant 图按能力名动态建节点（节点集合 = 已注册能力名 + 三个内置落点）
        classify 追加 plan trace（所选能力 / 依据 / 澄清 / 确认），裁决并入已有 done trace
   -> HarnessRegistry 按 domain 选择执行面策略（循环/上下文/工具/预算）
        默认 Harness 不改变 SSE 事件序列；领域可只换 policy 或重写 stream_answer / run_graph
```

**业务控制层（可选，`agent_core/flow/`）**：面向续保、审批、多轮写操作这类长程任务，
提供 `FlowEngine` 的 `plan → assemble → execute → observe → decide` 有界循环。
领域实现 `FlowNodes`；`VersionedFacts` 维护事实版本/失效，`PendingAction` +
`ConfirmExecutor` 负责版本 / TTL / 载荷 / 幂等，`FlowTrace` 记录决策依据。
默认最多 20 步、连续 2 轮无进展转人工。**flow 决定业务下一步，harness 决定执行资源**；
需要 LangGraph checkpoint / HITL 时用 `build_flow_graph(nodes, checkpointer=saver)` 把
`FlowNodes` 拆成 `plan / assemble / execute / observe / decide` 子图；确认拆成
`request_confirm` + `wait_confirm`，确保待确认动作先入 checkpoint 再 `interrupt`。
契约由 `tests/test_flow.py`、`tests/test_flow_langgraph.py` 守护。

`configs/domains.yaml` 已瘦身为只含 general 的回退清单；完整三域清单由
`domains.registry.default_domain_registry()` 从插件聚合。新增领域只需在插件目录写
`manifest.yaml`，无需改全局 YAML，也无需在 `agent_core` 里写业务字段。
契约由 `tests/test_intent_plugin_template.py` 守护；写操作确认闸门由
`tests/test_confirmation_gate.py` 守护（含装配自检：契约落到自由问答、
或 `api` 指向未注册路由都会告警）。

**写操作确认闸门**：对话链路（SSE）无状态、无 checkpointer、无 HITL，
改变业务状态的动作绝不在这里执行。`ConfirmGate` 从聚合清单读取
`IntentSpec.confirmation`（`api` / `requires_human_review` / `summary`），
由确定性代码裁决"是否需要确认 / 去哪个独立 API / 是否需人工审批"，
LLM 只提议意图。`guard` 与该领域能力据此输出结构化引导，真正执行仍走独立 API
（checkpointer + HITL 中断恢复 + 审核员二次认证）。新增领域只要声明契约即自动生效。

**领域无关的横切能力**（对全部领域一律生效，插件无需重复实现）：审计哈希链
（`agents/audit_store.py`）、接入限流与 500 兜底（`agents/http_middleware.py`）、
PII 脱敏与模型缓存/降级（`agents/middleware_audit.py`）、观测
（`monitoring.py` / `agents/observability.py`）。

> 关于 Harness：培训材料中的 `deep_agents.Harness/Fuse/Memory/AgenticRAG` 未经官方 SDK 证实，
> 本模板主链路使用 LangGraph 显式状态机；`isolated_deep/` 用官方 `create_deep_agent`/subagents
> 做隔离验证的等价替代，**主 app 未接入**。详见 [TEMPLATE.md](TEMPLATE.md)。

## 二、入口路由：三层，只有意图层做模型判断

```
HTTP 路由（无模型，静态 URL 匹配）
  内核注册：/api/assistant/stream  /api/rag/*  /api/documents/analyze ...
  插件注册：/api/claims/*（ClaimsPlugin）  /api/investment/*（InvestmentPlugin）
        │
        ├─ /api/assistant/stream ──▶ IntentRouter（JEV / LLM / Hybrid / Shadow）
        └─ /api/claims/* ─────────▶ 直接进入理赔状态机（无意图路由）
```

1. **HTTP 层路由**（FastAPI 装饰器）—— 纯 URL 匹配，请求打到哪个 endpoint 是写死的，**没有大模型**。
   通用路由在 `app_factory.py`，领域路由在 `domains/<域>/routes.py` 由各插件注册（见上一节）。
2. **意图路由层** —— 进入 `/api/assistant/stream` 后，`IntentRouter` 判定 `domain + intent + confidence`：
   - `shadow`（默认）：LLM 主路由，JEV 影子并行只记录；
   - `hybrid`：JEV 主路由，低置信或异常时 LLM 兜底；
   - `jev` / `llm`：单引擎。
3. **图内业务分流** —— 依据 `intent` 进入投资、保单查询、材料门控等节点。

```
用户消息 + 附件 + 历史
  -> 意图路由（IntentRouter）   <-- 唯一的意图判断入口
       JEV: ateve-jev-v1（远程） / LLM: flash 档
       输出: RouteDecision(domain, intent, confidence, margin, warnings, shadow)
  -> 适配为工作台 IntentResult(intent, confidence, policy_id)
  -> 按 intent 分流
       |-- 一般咨询     -> general 模型
       |-- 投资意图     -> investment_rag -> investment 模型
       |-- 交易指令     -> 拦截，不执行下单
       |-- 保单查询     -> 查后端（PolicyStore / CoreSystemAdapter）
       |-- 其他理赔意图 -> customer_service 模型（材料门控在前）
```

对应代码：

```python
# app.py：启动时按配置构建一次，失败降级为内置分类
registry = DomainRegistry.from_plugins(plugins, fallback=DomainRegistry.default())  # 插件声明优先
router = build_assistant_intent_router(cfg, intent_model, registry)  # 可注入；显式 router 时跳过构建
# agents/assistant_agent.py：classify 节点优先走路由层
route = await intent_router.classify(question=..., history=..., active_domain=...)
intent = IntentResult(intent=route.intent, confidence=route.confidence, slots=route.slots)
# 业务编号统一进 slots；路由异常时回落内置分类模型，SSE 不中断，trace 记录 route_error
```

`INTENT_ROUTER_ENABLED=false` 可整体关闭路由层，回到内置 LLM 分类；路由引擎与阈值见
`INTENT_ENGINE` / `INTENT_*` / `JEV_*`。

## 三、链路① 对话助手（LangGraph 多节点，/api/assistant/stream）

方案 A：无状态多节点图（`agents/assistant_agent.py`）。图负责编排决策，LLM 流式回答由外层执行器负责。

内核不认识任何领域，也不做意图字符串比较：`classify` 把 `(领域, 意图)` 解析成
**能力名**（领域清单 `IntentSpec.capability`），对话图的节点集合 = 已注册能力名
（`ctx.capabilities`）+ 内核内置 `answer` / `guard` / `clarify`。因此新增领域的对话分支只要
注册能力就能被路由到。

```
START
  -> classify（IntentRouter：JEV/LLM/Hybrid，默认 shadow；解析出 capability + domain + slots）
  -> plan（SSE trace：所选能力 / 依据 / 澄清 / 确认裁决，非图节点）
  -> 条件路由 route_after_classify（按能力名选节点，落点未实现时回内置 answer）
       |-- answer（内核）              -> END（general 模型流式回答）
       |-- guard（内核）               -> END（案件型/高风险未实现时拦截，不执行）
       |-- clarify（内核）             -> END（缺必填槽位或低置信时确定性追问，不走 LLM）
       |-- policy_query（理赔插件）     -> END（反问保单号 / 输出保单摘要）
       |-- claim_workflow（理赔插件）   -> END（缺材料回流追问 ≤3 轮；材料齐三专家 + RAG 判责）
       |-- claims_answer（理赔插件）    -> END（理赔通用问答）
       |-- investment_answer（投资插件）-> END（投资 RAG + 流式回答）
       |-- trade_gate（投资插件）       -> END（只拦截，不下单）
```

**内置落点**

| 能力 | 归属 | 职责 |
|---|---|---|
| answer | 内核 | 通用模型直接回答；清单外意图、无处理器的对话都落到这里 |
| guard | 内核 | 案件型 / 高风险意图没有领域处理器时拦截，提示走业务接口确认 |
| clarify | 内核 | 缺 `required_slots` 或 `needs_clarification` 时确定性追问；文案来自 manifest，不调用模型 |

**领域注册的落点（示例）**

| 能力 | 领域 | 职责 |
|---|---|---|
| policy_query | claims | 查保单（PolicyStore / CoreSystemAdapter），缺保单号反问，降级/失败转人工 |
| claim_workflow | claims | 材料完整性门控 + 回流追问（≤3 轮，超限转人工）+ 正式核赔（保单核验 + 三专家并行 + RAG 判责） |
| claims_answer | claims | 理赔相关但无需案件上下文的问答 |
| investment_answer | investment | 投资 RAG 检索 + 教育性建议 |
| trade_gate | investment | 交易指令拦截，引导走 `/api/investment/*` 的确认与审批 |

代码位置：`domains/claims/assistant_handlers.py`、`domains/investment/assistant_handlers.py`，
在各自插件的 `build_services` 里注册；理赔的意图改写钩子（补材料续答继承上一轮意图）
由 `ctx.capabilities.register_redirect()` 注入，内核不判断改写是否合理。

**trace 观测**

`classification` 节点在 SSE trace 中额外输出：

- `engine`：`jev` / `llm` / `llm-prompt`（回落内置分类时）；
- `route`：`domain` / `intent` / `confidence` / `margin` / `needs_clarification` /
  `fallback_used` / `warnings` / `shadow`；
- `route_error`：路由异常摘要，出现时表示本轮已回落到内置分类；兜底分类自身
  不符合契约时追加 `classify:<异常名>`（不带模型回显），意图按 `未识别` 处理。

`plan` trace 输出执行计划：`selected`（`domain` / `intent` / `capability`）、`basis`
（`declared` / `fallback` / `redirect`）、`missing_slots`、`needs_clarification`、
`needs_confirmation`；`route` 节点输出落点与依据：`selected`（领域）、`capability`
（对话图节点名）、`handler`（该能力是否有领域实现）、`declared`（意图是否在领域清单里）。
进入 `clarify` 节点时另发 `clarification_status`：`status=asking`、`round`、`max_rounds`、
`missing_slots`、`candidates`。

工作台展示位置：「调用链路 → 平台内核」下

- **领域意图路由**：显示 `JEV 意图模型 / LLM 意图路由 / 内置分类模型`、是否兜底、
  影子一致性，展开「输入 / 输出参数」可见 `engine`、`domain`、`intent`、`confidence`、
  `fallback_reason`、`warnings`、`shadow`；
- **领域注册校验**：显示 `已校验 <domain> / <intent>`；
- **执行计划**：显示 `<intent> → <capability>`，并标注「先澄清 / 需确认 / 缺 <槽位>」；
- 聊天消息上的意图徽章同时显示引擎，例如 `交易指令 · 100% · JEV`。

**外层执行器**（`app_factory.py` 的 `assistant_stream`）

```
ainvoke 图 -> 重放 events（trace + intent + clarification_status + material_status + content）-> 流式 LLM 回答 -> done
```

> 链路① 的对话预审与正式核赔已分开：材料齐全才由理赔域的 `claim_workflow` 能力进入核赔
> （三专家 + RAG 判责），缺材料只回流追问；投资域的交易指令由 `trade_gate` 能力只拦截不下单。
> 这些落点都写在领域 manifest 的 `capability:` 里，内核只认 `answer` / `guard` / `clarify` 三个内置落点。
> 当前为方案 A（无状态，每次请求新建图）；方案 B（checkpointer + interrupt 断点续跑）见 README。

## 四、链路② 理赔状态机（LangGraph，/api/claims/*）

主图（`agents/claim_agent.py` 的 `build_claim_graph`），带 checkpointer：

```
START
  -> intake（报案字段核验）
       |-- 缺 policy_id
       |     |-- 已补 < 3 轮 -> interrupt 挂起 -> 等 /supplement -> 回到 intake
       |     |-- 已补 >= 3 轮 -> review 转人工
       |
       |-- 有 policy_id
            -> policy（query_policy 查保单 + verify 核验）
                 |-- 查询/核验失败 -> review 转人工
                 |-- 成功
                      -> experts（子图，见下）
                      -> route_decision（业务规则优先于模型分数）
                            |-- 高风险(risk>.7)       -> investigate 调查
                            |-- 大额 / 低置信 / 缺证据 -> review 转人工（interrupt 挂起等 /review）
                            |-- 通过                  -> auto 受理建议（不执行付款）
  -> END
```

**experts 子图**（`build_expert_graph`，作为主图的一个节点）

```
START
  -> dispatch（Send 扇出，并行）
       |-- expert(damage)     [plus/main]
       |-- expert(risk)       [plus/main]
       |-- expert(liability)  [max/pro]
  -> aggregate（置信度 = min(三专家)，风险分取 risk 专家）
  -> END
```

> 每个专家有降级兜底：主档失败回退 `flash/fast`（经 `MiddlewareChain` 的 fallback）。
> 判责条款检索（RAG）：`WorkflowServices` 注入 `clause_retriever` 后，专家先检索适用条款
> （`build_clause_retriever` -> `HybridRetriever`，向量 + BM25 加权 RRF），条款注入 prompt，
> 专家在 `clause_ids` 中引用；引用必须落在已检索条款内，未配置或检索失败降级为无条款。
>
> 通用知识库 RAG：`tools/knowledge_rag.py` 对 `data/knowledge_base/` 和投资 Wiki 做
> dense + BM25 + 元数据三路召回，加权 RRF 后用 BGE-Reranker 精排；`/api/rag/search`
> 返回片段与引用，`/api/rag/answer` 生成答案并校验 `citation_ids`。
> 当前 `models/bge-reranker-base` 已下载并启用，正常检索返回
> `rerank_fallback=false`；模型或服务不可用时自动降级为词项重排。

## 五、其他 LangGraph/LangChain 结构（`claim_agent.py`，未接主 app）

| 结构 | 说明 |
|---|---|
| `build_plan_execute` | Plan-and-Execute 图：planner -> execute 循环（≤8 步） |
| `create_model_plan_execute` | pro(max) 规划 + main(plus) 执行，只分析不执行 |
| `create_basic_agent` | `AgentExecutor` + `create_tool_calling_agent`（工具调用闭环） |
| `create_conversational_agent` | `RunnableWithMessageHistory`，按 session_id 隔离历史 |

> 八类业务工具经 `tools/claim_tool.py` 的 `build_claim_tools` 注册；`isolated_deep/runtime.py`（deep agents 隔离环境）单独使用该工具集，主 app 未接入。

## 六、大模型判断位置汇总

| 环节 | 模型档位 | 说明 |
|---|---|---|
| 意图分类（入口路由） | JEV `ateve-jev-v1` / flash | 输出 domain + intent + confidence；默认 shadow 由 LLM 主路由 |
| 单证识别 | vision / plus | 识别上传的发票/病历/事故单 |
| 损失/风险专家 | plus（main） | 降级回退 flash |
| 责任专家 | max（pro） | 降级回退 flash |
| 回答生成 | plus | 客服/通用回答 |
| 意图翻译（可选） | flash | `INTENT_TRANSLATE_TO_ENGLISH=true` 时把中文转英文再送 JEV |

## 六·补充、意图路由（JEV / LLM / Hybrid / Shadow）

接入位置：`agents/assistant_agent.py` 的 `classify` 节点。
适配层：`agent_core/intent/adapter.py`（`AssistantIntentRouter`）把工作台请求转成
`IntentRequest`，再把 `RouteDecision` 转成 `IntentResult`。

```text
INTENT_ROUTER_ENABLED=true
  -> IntentConfig.from_settings(Settings)
  -> build_intent_classifier()  # agent_core/intent/factory.py
       primary / fallback / shadow
  -> AssistantIntentRouter
  -> graph classify 节点
```

| `INTENT_ENGINE` | primary | fallback | shadow | 适用 |
|---|---|---|---|---|
| `shadow`（代码默认） | LLM | 无 | JEV | 生产稳定 + 影子对拍 |
| `hybrid`（当前 `.env`） | JEV | LLM | 无 | 中文评测达标后接管 |
| `jev` | JEV | 无 | 无 | 单引擎验证 |
| `llm` | LLM | 无 | 无 | 与旧链路等价 |

约定：

- 阈值：`confidence >= INTENT_CONFIDENCE_THRESHOLD(0.8)`、
  `margin >= INTENT_MARGIN_THRESHOLD(0.2)`；低于阈值触发兜底或标记 `needs_clarification`，
  后者由 `ClarifyGate` 结合清单 `description` 生成候选项并进入内核 `clarify` 节点。
- JEV 是英文优先模型；中文输入可开 `INTENT_TRANSLATE_TO_ENGLISH=true`，
  未开翻译会记录 `cjk_without_translation` 警告。
- 发送前先 `mask_pii`，JEV 只收到脱敏文本；`JEV_API_KEY` 不写入日志、报告和镜像。
- 缺 `JEV_API_KEY` 时影子自动跳过；若设成 `jev/hybrid` 主路由而缺 key，
  `build_assistant_intent_router()` 返回 `None`，主链路回落内置分类，服务照常启动。
- 保单号：JEV 不返回槽位，`AssistantIntentRouter` 对「保单查询」意图用本地正则从
  当前问题与用户历史中补齐；识别不到仍按原逻辑反问保单号。
- 关闭开关：`INTENT_ROUTER_ENABLED=false`。
- 观测：`intent.route` 日志（不含用户原文）+ SSE trace 的 `engine` / `plan` / `route` /
  `route_error` / `clarification_status`。
- 评测：`scripts/eval_intent.py` + `data/intent_golden.jsonl`。
- 2026-10-27 实测（JEV 单引擎，39 条）：联合准确率 `0.897`、领域 `1.000`、意图 `0.897`、
  P50/P95 `571/628ms`；主要错误为「组合诊断 ↔ 投资咨询」。

## 七、四处 SQLite 职责

```
checkpoints.sqlite  <- 链路② 状态机断点/恢复（AsyncSqliteSaver）
audit.sqlite        <- 接入层 请求审计（只追加 + 哈希链，可 verify_chain）
policies.sqlite     <- 链路① 保单查询的本地模拟数据（PolicyStore）
outbox.sqlite       <- 赔付/通知回写队列（Outbox 已实现，主图未注入，入队 ≠ 付款）
```

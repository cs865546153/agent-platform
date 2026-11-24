# 企业级 Agent 模板说明

本仓库既是可运行的理赔助手，也是一套**企业级 Agent 工程模板**。核心承诺：
**新增一个业务领域，不需要修改内核（`app.py` / `agent_core/` / 接入层），只需实现插件并登记。**

个性化教学资料已归档至工作区根目录 `../teaching-archive/`（My_Study、培训实现追踪清单），
不属于模板运行时。

---

## 1. 分层与边界

```
app.py                     内核：FastAPI 装配 + 通用路由 + 健康检查/RAG/工作台/SSE
  └─ create_app(registry=)  唯一扩展点：注入领域插件集，Kernel 不认识任何业务
agent_core/                领域无关内核
  ├─ plugins.py            DomainPlugin 协议（含 intent_manifest hook）、AppContext、PluginRegistry
  ├─ security.py           API Key / 审核员认证、生产密钥校验
  ├─ sse.py                SSE 帧编码
  └─ intent/               JEV/LLM/Hybrid/Shadow 意图路由（领域清单从插件 manifest 聚合）
domains/                   领域插件（可增删）
  ├─ registry.py           默认领域登记表（唯一"启用哪些领域"的事实源）
  ├─ base.py               插件基类 + LangGraph 中断/配置共用辅助
  ├─ claims/               示例域：理赔状态机、三专家、RAG、HITL
  ├─ investment/           示例域：投资咨询、交易二次确认、人工审批
  └─ example/              最小示例：60 行接入一个新领域（可直接抄改）
```

**横切能力**（审计哈希链、接入限流、PII 脱敏、模型缓存/降级、Prometheus/LangFuse）
在 `agents/http_middleware.py`、`agents/audit_store.py`、`agents/middleware_audit.py` 等，
对所有领域一律生效，领域插件无需重复实现。

---

## 2. 内核提供的能力（开箱即用）

| 能力 | 位置 | 领域是否需关心 |
| --- | --- | --- |
| RequestID / 认证 / 限流 / 审计 / 500 兜底 | `agents/http_middleware.py` | 否 |
| append-only + 哈希链审计 | `agents/audit_store.py` | 否 |
| 意图分类 JEV/LLM/Hybrid/Shadow | `agent_core/intent/` | 在插件 manifest.yaml 声明意图即可 |
| 模型工厂 / 任务档位 / 缓存 / 降级 | `config.py`、`agents/middleware_audit.py` | 否 |
| 多路召回 + RRF + BGE-Rerank + 引用校验 | `tools/knowledge_rag.py`、`tools/rerank.py` | 可选复用 |
| HITL（中断 + 恢复 + 人工审批） | LangGraph `interrupt` + `Command(resume=)` | 图内声明 |
| SSE 流式回答 | `app.py` assistant 路由 + `agent_core/sse.py` | 否 |
| Prometheus / LangFuse / 日志脱敏 | `monitoring.py`、`agents/observability.py` | 否 |
| 上线门禁 / 影子对拍 | `deploy/regression_gate.py`、`deploy/shadow_mode.py` | 否 |

---

## 3. 新增一个领域（四步）

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

### 步骤 2：声明领域意图（随插件走，可选但推荐）

```yaml
# domains/my_domain/manifest.yaml —— 与 configs/domains.yaml 单个领域条目同格式
name: my_domain
description: 我的业务领域
description_en: "My business domain"   # JEV 英文优先，务必提供
version: 0.1.0
capabilities: []
intents:
  - name: 我的意图
    description: 用户会怎么表达
    description_en: "How users express it"
    examples: ["一个示例问题"]
    risk: low
    mode: conversation
```

内核启动时从已启用插件的 `intent_manifest()` 聚合领域清单，
`configs/domains.yaml` 只用于补齐没有插件的领域（如 general）；
意图路由器的构建、保单号兜底抽取、图内分支策略都从这份聚合清单派生。

### 步骤 3：登记领域

```python
# domains/registry.py —— 唯一的启用开关
from domains.my_domain.plugin import MyDomainPlugin

def default_registry() -> PluginRegistry:
    return PluginRegistry([ClaimsPlugin(), InvestmentPlugin(), MyDomainPlugin()])
```

`configs/domains.yaml` 不再是新增领域的必改项；它只保留无插件领域的声明。

### 步骤 4：测试

```bash
pytest -q tests/test_domain_plugin_template.py   # 领域可插拔契约
pytest -q tests/test_intent_plugin_template.py   # 意图路由可插拔契约
pytest -q                                         # 全量
```

测试里可用 `create_app(registry=PluginRegistry([MyDomainPlugin()]))` 只装自己的域，
与新领域无关的路由不会注册；也可用 `intent_router=FakeRouter()` 显式注入意图路由，
内核会跳过构建且不接管其生命周期（见两个契约测试文件）。

---

## 4. 模板的使用边界（如实说明）

- **`app.py` 仍保留 demo 路由与对话编排**：它们是"示例应用"而非内核的一部分，
  原创业务可删除 `/api/demo/*` 与 assistant 图，不影响领域插件机制。
- **教学产物已迁出**：`My_Study/`、培训追踪清单在 `../teaching-archive/`，不在模板运行时内。
- **真实联调未完成**：本地 LLM、Milvus、Redis、LangFuse、核心业务系统需按部署环境接入；
  模板不虚构准确率、延迟或成本。详见 `docs/` 各专题与 `deploy/runbook.md`。
- **进程内限流/锁**：多副本部署需替换为 Redis；`AppContext.locks` 是进程内实现。
- **关于 Harness**：材料中的 `deep_agents.Harness/Fuse/Memory/AgenticRAG` 未被官方 SDK 证实，
  本项目主链路使用 LangGraph 显式状态机；`isolated_deep/` 里用官方
  `create_deep_agent`/subagents 做了隔离验证的等价替代，**主 app 未接入**。

---

## 5. 验证基线（2026-11-24）

- `196 passed, 1 skipped`（原 188 + 3 条领域可插拔 + 5 条意图路由可插拔契约测试；跳过项为需真实 Milvus 的集成测试）
- Ruff（F 规则）全项目通过；新增领域模块 Mypy 通过
- 可插拔契约由 `tests/test_domain_plugin_template.py` 守护
- 真实启动：`bash scripts/start_mac.sh` 后 `/health`、`/ready`、首页与 19 个路由均 200；
  `POST /api/demo/claims/process` 走完整状态机返回 `accept/confidence=0.95`。

> 启动脚本注意：`start_mac.sh` 开了 `set -u`，在 echo 中文提示里 `$var` 后紧跟全角括号时
> Bash 会把 UTF-8 首字节并进变量名（如 `$app_log（` 误判为 `app_log\xef`），导致启动即报
> `unbound variable`。2026-11-24 已改为 `${var}` 花括号边界修复；新增脚本时同类写法请照此处理。

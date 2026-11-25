"""FastAPI入口：真实状态机、持久检查点、补材料、审核恢复和可观测指标。"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core.plugins import AppContext, PluginRegistry
from agent_core.security import authenticate, validate_production_keys
from agent_core.sse import sse_frame, stream_headers
from domains.registry import default_registry
from agents.audit_store import AuditStore
from agents.http_middleware import EnterpriseHttpMiddleware
from agents.middleware_audit import configure_logging, mask_pii, RateLimitMiddleware
from config import LLMFactory, get_model
from models.knowledge import (
    KnowledgeAnswerResponse,
    KnowledgeRAGQuery,
    KnowledgeSearchResponse,
)
from models.workbench import AssistantRequest, DocumentBatchResponse
from monitoring import metrics, ERRORS
from settings import Settings
from tools.document_analysis import (
    MAX_DOCUMENT_BYTES,
    DocumentModelOutputError,
    DocumentValidationError,
    analyze_document,
)


logger = logging.getLogger(__name__)


def composite_slot_extractor(extractors: dict[str, Any]) -> Any:
    """把各领域注册的槽位抽取器组合成单个抽取器，按 ``RouteDecision.domain`` 分派。

    领域插件各自往 ``AppContext.slot_extractors`` 里放一个抽取器；内核只按路由
    领域选择，不认识具体字段形态。取"第一个"会让后注册的领域被静默丢弃。
    """
    if not extractors:
        return None

    def extract(decision: Any, question: str, turns: list[Any]) -> dict[str, str]:
        handler = extractors.get(getattr(decision, 'domain', None) or '')
        return dict(handler(decision, question, turns)) if callable(handler) else {}

    return extract


def build_assistant_intent_router(
    cfg: Settings,
    intent_model: Any = None,
    domain_registry: Any = None,
    slot_extractor: Any = None,
) -> Any:
    """按配置构建主链路意图路由；失败时返回 None，由内置 LLM 分类兜底。

    - 路由层只决定 domain/intent/confidence，业务分流仍由对话图负责。
    - ``domain_registry`` 是领域/意图清单；由 ``create_app`` 从已启用插件聚合后传入。
      独立调用本函数时应显式传完整清单（``domains.registry.default_domain_registry()``），
      内核不再从 ``configs/domains.yaml`` 读取完整意图树（该文件已是回退清单）。
    - ``slot_extractor`` 是领域插件声明的槽位兜底抽取器；内核不内置任何业务字段形态。
    - 缺少 JEV_API_KEY 时 factory 会跳过 JEV 影子；INTENT_ENGINE=jev 则会因缺 key
      初始化失败，这里降级为内置分类而不是让服务启动失败。
    """
    if not cfg.intent_router_enabled:
        logger.info('INTENT_ROUTER_ENABLED=false，回退内置 LLM 意图分类')
        return None

    from agent_core.intent.adapter import AssistantIntentRouter
    from agent_core.intent.contracts import RouteDecision
    from agent_core.intent.factory import IntentConfig, build_intent_classifier

    def log_decision(decision: RouteDecision) -> None:
        # 这是兜底后的最终决策；主路由尝试与异常原因见 agent_core.intent 的 intent.primary_* 日志。
        # 只记录意图元数据，不记录用户原文；影子字段用于 JEV/LLM 对拍。
        logger.info(
            'intent.route engine=%s domain=%s intent=%s confidence=%.3f margin=%s '
            'fallback=%s reason=%s clarification=%s shadow=%s warnings=%s',
            decision.engine, decision.domain, decision.intent, decision.confidence,
            decision.margin, decision.fallback_used, decision.fallback_reason,
            decision.needs_clarification, decision.shadow, decision.warnings,
        )

    try:
        config = IntentConfig.from_settings(cfg)
        classifier = build_intent_classifier(
            config,
            registry=domain_registry,
            llm_model=intent_model or get_model('classification'),
            mask=mask_pii,
            on_decision=log_decision,
        )
        router = AssistantIntentRouter(classifier, slot_extractor=slot_extractor)
        logger.info('意图路由已启用：%s（INTENT_ENGINE=%s）', router.engine_label, cfg.intent_engine)
        return router
    except Exception as exc:  # noqa: BLE001 - 路由初始化失败不能阻断服务启动
        logger.warning('意图路由初始化失败，回退内置 LLM 分类：%s: %s', type(exc).__name__, exc)
        return None


class HealthResponse(BaseModel):
    status: Literal['ok'] = 'ok'


def create_app(
    services: Any = None,
    *,
    database: str | None = None,
    document_text_model: Any = None,
    document_vision_model: Any = None,
    assistant_model: Any = None,
    intent_model: Any = None,
    general_model: Any = None,
    knowledge_rag: Any = None,
    knowledge_model: Any = None,
    overrides: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    intent_router: Any = None,
    intent_domain_registry: Any = None,
) -> FastAPI:
    """装配应用。

    ``registry`` 是模板的核心扩展点：传入自定义领域插件集即可增删业务域，
    内核（本文件）无需改动。默认装配 ``domains/registry.py`` 登记的示例领域。

    意图路由同样可插拔：``intent_router`` 显式传入时跳过内核构建（测试/自定义引擎）；
    否则 ``intent_domain_registry`` 决定领域清单，未传时从已启用插件的
    ``intent_manifest()`` 聚合（YAML 回退补齐无插件的领域，如 general）。

    ``overrides`` / ``models`` 是领域依赖的注入点，键名由插件自己解释
    （如 ``overrides={'investment_rag': ...}``、``models={'investment_advice': ...}``）：
    内核不 import 业务类型，签名与装配里也不出现领域名。模型键名一律用任务档位。
    """
    cfg = Settings()
    registry = registry or default_registry()
    overrides = {
        'services': services,
        'knowledge_rag': knowledge_rag,
        **(overrides or {}),
    }
    limiter = RateLimitMiddleware()
    locks: dict[str, asyncio.Lock] = {}
    # 模型替身按"任务档位"登记，领域插件按档位自取；内核不解释档位属于哪个领域。
    model_overrides = {
        'classification': intent_model,
        'customer_service': assistant_model,
        'general': general_model,
        **(models or {}),
    }
    # 内核上下文：服务/图/模型/RAG/能力表全部由领域插件填充，内核只持有容器。
    ctx = AppContext(
        application=None,  # type: ignore[arg-type]  # application 在下方创建后回填
        settings=cfg,
        static_dir=Path(__file__).resolve().parent / 'static',
        limiter=limiter,
        locks=locks,
        models={key: value for key, value in model_overrides.items() if value is not None},
    )
    for plugin in registry.all():
        plugin.build_services(ctx, overrides)

    # 知识库：通用库（/api/rag/*）由内核装配，领域自己的库由插件写入 ctx.rag。
    effective_knowledge_rag = knowledge_rag
    if effective_knowledge_rag is None and cfg.knowledge_rag_enabled:
        try:
            from tools.knowledge_rag import get_knowledge_rag
            effective_knowledge_rag = get_knowledge_rag()
        except Exception:  # noqa: BLE001 - 通用知识库损坏不应阻断既有业务
            logger.exception('通用知识库初始化失败，已关闭 /api/rag')
    ctx.rag['knowledge'] = effective_knowledge_rag

    # 意图路由可插拔：显式注入 > 自定义领域清单 > 从插件聚合（YAML 回退）。
    if intent_domain_registry is None:
        from agent_core.intent.registry import DomainRegistry

        intent_domain_registry = DomainRegistry.from_plugins(
            registry.all(), fallback=DomainRegistry.default()
        )
    ctx.intent_registry = intent_domain_registry
    # 对话图的分支策略 = 领域清单 + 能力注册表。两者都在 build_services 之后就位，
    # 因此新增领域只要声明意图并注册能力即可被路由，不必改内核。
    from agent_core.capabilities import missing_handlers
    from agents.assistant_agent import IntentRoutingPolicy

    for capability, domain in missing_handlers(intent_domain_registry, ctx.capabilities):
        logger.warning('领域 %s 声明了对话能力 %s 但没有实现，将回落到内核内置能力', domain, capability)
    routing_policy = IntentRoutingPolicy.from_registry(intent_domain_registry, ctx.capabilities)
    if intent_router is None:
        # 领域插件在 build_services 注册的槽位抽取器交给适配层；内核不认识具体槽位名。
        slot_extractor = composite_slot_extractor(ctx.slot_extractors)
        # models={'classification': ...} 与 intent_model=... 都是分类模型注入点，
        # 两者都要传给路由构建，否则离线测试里路由层会落到真实模型并触网。
        effective_intent_router = build_assistant_intent_router(
            cfg, intent_model or ctx.models.get('classification'),
            domain_registry=intent_domain_registry,
            slot_extractor=slot_extractor,
        )
        owns_intent_router = True
    else:
        effective_intent_router = intent_router
        owns_intent_router = False
    ctx.intent_router = effective_intent_router

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        # 先配置日志，后续启动校验与业务日志才不会被默认级别丢弃。
        configure_logging()
        validate_production_keys()
        if os.getenv('APP_ENV') == 'production' and Path('.env').exists():
            logger.warning('生产环境检测到 .env 文件：请改用环境变量或密钥管理（Vault/KMS）注入')
        path = database or os.getenv('CHECKPOINT_DB') or os.getenv('CLAIMS_CHECKPOINT_DB') or '.data/checkpoints.sqlite'
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        audit_path = (os.getenv('AUDIT_DB') or os.getenv('CLAIMS_AUDIT_DB')
                      or (':memory:' if path == ':memory:' else '.data/audit.sqlite'))
        audit = AuditStore(audit_path)
        application.state.audit = audit
        application.state.intent_router = effective_intent_router
        application.state.intent_registry = intent_domain_registry
        application.state.capabilities = ctx.capabilities
        # 图内分支策略从领域清单与能力注册表派生；未声明领域时只剩内核内置能力。
        application.state.intent_routing_policy = routing_policy
        try:
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                await saver.setup()
                for plugin in registry.all():
                    plugin.build_graphs(ctx, saver)
                # 领域图由插件通过 ctx.graphs 自行取用；内核不再按领域名挂全局 state。
                # 自定义 registry 未装配任何图时 ctx.graphs 为空，就绪探针会报未就绪。
                yield
        finally:
            # 只关闭内核自建的 router；外部注入的由调用方负责生命周期。
            if owns_intent_router and effective_intent_router is not None:
                try:
                    await effective_intent_router.aclose()
                except Exception:  # noqa: BLE001 - 关闭失败不影响进程退出
                    logger.warning('意图路由关闭失败', exc_info=True)
            audit.close()

    application = FastAPI(title='智能业务助手服务', version='0.3.0', lifespan=lifespan)
    # 注册顺序即嵌套顺序：后添加的在外层。CORS 必须最外层，错误响应才带 CORS 头。
    application.add_middleware(EnterpriseHttpMiddleware)
    application.add_middleware(CORSMiddleware,
        allow_origins=[value.strip() for value in os.getenv('CORS_ALLOW_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')],
        allow_credentials=False, allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'Authorization', 'X-API-Key', 'X-Reviewer-Key', 'X-Request-ID'])
    static_dir = Path(__file__).resolve().parent / 'static'
    application.mount('/static', StaticFiles(directory=static_dir), name='static')

    # 回填内核上下文并让领域插件注册各自路由（内核不再硬编码业务路由）。
    ctx.application = application
    for plugin in registry.all():
        plugin.register_routes(ctx)

    # 认证统一走 agent_core.security：Depends 版与接入层中间件共用同一密钥解析，
    # 避免只配 API_KEY 时 /metrics、/api/rag/*、/api/assistant/stream 认证失效。
    @application.get('/health', response_model=HealthResponse, tags=['健康检查'])
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get('/ready', tags=['健康检查'])
    async def ready(response: Response) -> dict[str, Any]:
        """就绪探针：依赖（图与审计库）未就绪时返回 503，与存活探针区分。"""
        audit = getattr(application.state, 'audit', None)
        checks = {
            'graph': bool(ctx.graphs),
            'audit': bool(audit is not None and await asyncio.to_thread(audit.ping)),
        }
        ok = all(checks.values())
        if not ok:
            response.status_code = 503
        return {'status': 'ready' if ok else 'not_ready', 'checks': checks}

    @application.get('/', include_in_schema=False)
    async def workbench() -> FileResponse:
        return FileResponse(static_dir / 'index.html')

    @application.get('/favicon.ico', include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @application.get('/metrics', dependencies=[Depends(authenticate)])
    async def export_metrics() -> Response:
        return Response(content=metrics(), media_type='text/plain; version=0.0.4')

    # 演示路由 /api/demo/claims/* 由 claims 插件注册
    # （domains/claims/routes.py:register_demo_routes）；内核不再承载示例业务逻辑。

    @application.get('/api/rag/domains', dependencies=[Depends(authenticate)], tags=['RAG'])
    async def list_rag_domains() -> dict[str, Any]:
        """列出已配置的知识库 domain；不触发向量库连接。"""
        if effective_knowledge_rag is None:
            raise HTTPException(503, '知识库 RAG 未启用')
        return {'domains': effective_knowledge_rag.domains()}

    @application.post('/api/rag/search', response_model=KnowledgeSearchResponse,
                      dependencies=[Depends(authenticate)], tags=['RAG'])
    async def search_knowledge_rag(body: KnowledgeRAGQuery) -> KnowledgeSearchResponse:
        """多路召回 + RRF + rerank，返回 Top-K 片段和引用。"""
        if effective_knowledge_rag is None:
            raise HTTPException(503, '知识库 RAG 未启用')
        from tools.knowledge_rag import KnowledgeBaseUnavailable
        try:
            result = await asyncio.to_thread(
                effective_knowledge_rag.retrieve,
                body.domain,
                body.question,
                body.top_k,
                filters=body.filters or None,
            )
        except KeyError as exc:
            raise HTTPException(404, f'未知知识库：{body.domain}') from exc
        except KnowledgeBaseUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return KnowledgeSearchResponse(**result)

    @application.post('/api/rag/answer', response_model=KnowledgeAnswerResponse,
                      dependencies=[Depends(authenticate)], tags=['RAG'])
    async def answer_knowledge_rag(body: KnowledgeRAGQuery) -> KnowledgeAnswerResponse:
        """检索后调用 LLM 生成答案，并校验 citation_ids 只能来自检索结果。"""
        if effective_knowledge_rag is None:
            raise HTTPException(503, '知识库 RAG 未启用')
        from tools.knowledge_rag import KnowledgeAnswerError, KnowledgeBaseUnavailable
        model = knowledge_model or LLMFactory.create(
            'main', temperature=0.2, max_tokens=1500, response_format='json_object', stop=()
        )
        try:
            result = await effective_knowledge_rag.answer(
                body.domain,
                body.question,
                model,
                body.top_k,
                filters=body.filters or None,
            )
        except KeyError as exc:
            raise HTTPException(404, f'未知知识库：{body.domain}') from exc
        except KnowledgeBaseUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except KnowledgeAnswerError as exc:
            raise HTTPException(502, str(exc)) from exc
        return KnowledgeAnswerResponse(**result)

    @application.post('/api/documents/analyze', response_model=DocumentBatchResponse,
                      dependencies=[Depends(authenticate)], tags=['工作台'])
    async def analyze_documents(files: list[UploadFile] = File(...)) -> DocumentBatchResponse:
        """在内存中识别上传附件，原文件不会写入服务器磁盘。"""
        if not 1 <= len(files) <= 6:
            raise HTTPException(422, '每次必须上传 1 至 6 个附件')
        payloads: list[tuple[str | None, bytes]] = []
        for upload in files:
            content = await upload.read(MAX_DOCUMENT_BYTES + 1)
            payloads.append((upload.filename, content))
            await upload.close()
        text_llm = document_text_model or get_model('extraction')
        vision_llm = document_vision_model or LLMFactory.create(
            'vision', temperature=0, max_tokens=2000, response_format='json_object', stop=()
        )
        try:
            results = await asyncio.gather(*(
                analyze_document(name, content, text_model=text_llm, vision_model=vision_llm)
                for name, content in payloads
            ))
            return DocumentBatchResponse(documents=results)
        except DocumentValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except DocumentModelOutputError as exc:
            ERRORS.labels('document_output').inc()
            raise HTTPException(502, '单证识别结果格式无效，请重试') from exc
        except Exception as exc:
            ERRORS.labels('document_model').inc()
            raise HTTPException(503, '单证识别模型暂不可用，请检查本地推理服务') from exc

    @application.post('/api/assistant/stream', dependencies=[Depends(authenticate)], tags=['工作台'])
    async def assistant_stream(body: AssistantRequest) -> StreamingResponse:
        """识别意图后，按领域清单把请求分派到该领域注册的对话能力（LangGraph 多节点编排）。"""
        # ``model_inputs`` 供下方的回答流 trace 复用：与分类节点记录"送进模型的完整输入"同一口径。
        from agents.assistant_agent import build_assistant_graph, chat_content, model_inputs

        context = json.dumps(
            mask_pii([document.model_dump() for document in body.documents]),
            ensure_ascii=False,
        )
        classifier = ctx.models.get('classification') or get_model('classification')
        # 内核内置 answer 能力用的通用模型；领域回答模型由各能力处理器自带。
        effective_general = ctx.models.get('general') or LLMFactory.create(
            'main', temperature=0.6, max_tokens=1500, response_format='text', stop=()
        )

        async def events():
            started = time.monotonic()
            stage = 'classification'

            try:
                graph = build_assistant_graph(
                    classifier=classifier,
                    general_model=effective_general,
                    routing_policy=getattr(application.state, 'intent_routing_policy', None)
                    or routing_policy,
                    intent_router=getattr(application.state, 'intent_router', None)
                    or effective_intent_router,
                )
                state = await graph.ainvoke({
                    'question': body.question,
                    'claim_id': body.claim_id,
                    'documents': body.documents,
                    'history': body.history,
                    'material_round': body.material_round,
                    'context': context,
                    'started': started,
                    'events': [],
                })
                stage = state.get('stage') or 'general'
                for event in state.get('events', []):
                    # trace / content 有固定形态，单独处理；其余（intent、领域自定义状态
                    # 事件）按原结构透传，统一走脱敏，内核不再枚举领域事件名。
                    if 'trace' in event:
                        yield sse_frame({'trace': mask_pii(event['trace'])})
                    elif 'content' in event:
                        yield sse_frame({'content': mask_pii(event['content'])})
                    else:
                        yield sse_frame(mask_pii(event))
                if state.get('needs_llm_answer', False):
                    llm = state['llm']
                    messages = state['messages']
                    answer_started = time.monotonic()
                    yield sse_frame({'trace': mask_pii({'node': stage, 'status': 'running',
                                                  'model': getattr(llm, 'model_name', '回答模型'),
                                                  'input': model_inputs(llm, messages)})})
                    answer_parts: list[str] = []
                    async for chunk in llm.astream(messages):
                        content = chat_content(chunk)
                        if content:
                            answer_parts.append(content)
                            yield sse_frame({'content': mask_pii(content)})
                    yield sse_frame({'trace': mask_pii({'node': stage, 'status': 'done',
                                                  'elapsed_ms': round((time.monotonic() - answer_started) * 1000),
                                                  'output': {'content': ''.join(answer_parts)}})})
                yield sse_frame({'done': True})
            except Exception as exc:
                logger.exception(
                    'assistant model call failed at stage=%s provider=%s',
                    stage,
                    cfg.model_provider,
                )
                yield sse_frame({'trace': mask_pii({'node': stage, 'status': 'error',
                                              'elapsed_ms': round((time.monotonic() - started) * 1000),
                                              'output': {'error': '调用失败，未返回有效结果',
                                                         'error_type': type(exc).__name__}})})
                ERRORS.labels('assistant_model').inc()
                yield sse_frame({'error': (
                    f'模型调用失败（阶段：{stage}，类型：{type(exc).__name__}），'
                    f'请检查 {cfg.model_provider} 模型服务配置、模型名称和网络连接'
                )})

        return StreamingResponse(
            events(),
            media_type='text/event-stream',
            headers=stream_headers(),
        )

    return application


app = create_app()

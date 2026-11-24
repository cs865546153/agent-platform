"""FastAPI入口：真实状态机、持久检查点、补材料、审核恢复和可观测指标。"""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core.plugins import AppContext, PluginRegistry
from agent_core.security import validate_production_keys
from domains.registry import default_registry
from agents.audit_store import AuditStore
from agents.http_middleware import EnterpriseHttpMiddleware
from agents.claim_agent import WorkflowServices, build_claim_graph
from agents.investment_agent import InvestmentServices
from agents.middleware_audit import configure_logging, mask_pii, RateLimitMiddleware
from config import ClaimLLMFactory, get_model
from models.knowledge import (
    KnowledgeAnswerResponse,
    KnowledgeRAGQuery,
    KnowledgeSearchResponse,
)
from models.schemas import ClaimResponse
from models.workbench import AssistantRequest, DemoClaimRunResponse, DocumentBatchResponse
from monitoring import metrics, ERRORS
from settings import Settings
from tools.document_analysis import (
    MAX_DOCUMENT_BYTES,
    DocumentModelOutputError,
    DocumentValidationError,
    analyze_document,
)


logger = logging.getLogger(__name__)


def build_assistant_intent_router(
    cfg: Settings,
    intent_model: Any = None,
    domain_registry: Any = None,
) -> Any:
    """按配置构建主链路意图路由；失败时返回 None，由内置 LLM 分类兜底。

    - 路由层只决定 domain/intent/confidence，业务分流仍由对话图负责。
    - ``domain_registry`` 是领域/意图清单；不传时从 ``configs/domains.yaml`` 加载。
      内核装配时由 ``create_app`` 从已启用的领域插件聚合后传入。
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
        # 这是兜底后的最终决策；主路由尝试与异常原因见 claims.intent 的 intent.primary_* 日志。
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
            on_decision=log_decision,
        )
        # 保单号兜底抽取的触发意图从领域清单派生，不再写死"保单查询"。
        policy_intents: set[str] = set()
        if domain_registry is not None:
            for domain in domain_registry.all():
                if 'policy_query' in (getattr(domain, 'capabilities', None) or ()):
                    # 只取语义上确实是保单查询的意图，避免把同领域其他意图也算进来。
                    policy_intents.update(
                        spec.name for spec in domain.intents if '保单' in spec.name
                    )
        router = AssistantIntentRouter(
            classifier,
            policy_intents=frozenset(policy_intents) if policy_intents else frozenset({'保单查询'}),
        )
        logger.info('意图路由已启用：%s（INTENT_ENGINE=%s）', router.engine_label, cfg.intent_engine)
        return router
    except Exception as exc:  # noqa: BLE001 - 路由初始化失败不能阻断服务启动
        logger.warning('意图路由初始化失败，回退内置 LLM 分类：%s: %s', type(exc).__name__, exc)
        return None


class HealthResponse(BaseModel):
    status: Literal['ok'] = 'ok'


def create_app(
    services: WorkflowServices | None = None,
    *,
    database: str | None = None,
    document_text_model: Any = None,
    document_vision_model: Any = None,
    assistant_model: Any = None,
    intent_model: Any = None,
    general_model: Any = None,
    investment_model: Any = None,
    investment_services: InvestmentServices | None = None,
    investment_rag: Any = None,
    knowledge_rag: Any = None,
    knowledge_model: Any = None,
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
    """
    cfg = Settings()
    registry = registry or default_registry()
    overrides: dict[str, Any] = {
        'services': services,
        'investment_services': investment_services,
    }
    limiter = RateLimitMiddleware()
    locks: dict[str, asyncio.Lock] = {}
    # 内核只持有 AppContext；具体服务/图由领域插件装配，路由从 ctx 按需取用。
    ctx = AppContext(
        application=None,  # type: ignore[arg-type]  # application 在下方创建后回填
        settings=cfg,
        static_dir=Path(__file__).resolve().parent / 'static',
        limiter=limiter,
        locks=locks,
    )
    for plugin in registry.all():
        plugin.build_services(ctx, overrides)
    service: WorkflowServices | None = ctx.services.get('claims')

    effective_investment_rag = investment_rag
    if effective_investment_rag is None and cfg.investment_rag_enabled:
        try:
            from tools.investment_rag import get_investment_rag
            effective_investment_rag = get_investment_rag()
        except Exception:  # noqa: BLE001 - 投资知识库损坏不应阻断理赔主链路
            logger.exception('投资知识库初始化失败，已关闭投资 RAG')
    effective_knowledge_rag = knowledge_rag
    if effective_knowledge_rag is None and cfg.knowledge_rag_enabled:
        try:
            from tools.knowledge_rag import get_knowledge_rag
            effective_knowledge_rag = get_knowledge_rag()
        except Exception:  # noqa: BLE001 - 通用知识库损坏不应阻断既有业务
            logger.exception('通用知识库初始化失败，已关闭 /api/rag')
    ctx.rag['investment'] = effective_investment_rag
    ctx.rag['knowledge'] = effective_knowledge_rag

    # 意图路由可插拔：显式注入 > 自定义领域清单 > 从插件聚合（YAML 回退）。
    if intent_domain_registry is None:
        from agent_core.intent.registry import DomainRegistry

        intent_domain_registry = DomainRegistry.from_plugins(
            registry.all(), fallback=DomainRegistry.default()
        )
    ctx.intent_registry = intent_domain_registry
    if intent_router is None:
        effective_intent_router = build_assistant_intent_router(
            cfg, intent_model, domain_registry=intent_domain_registry
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
        path = database or os.getenv('CLAIMS_CHECKPOINT_DB') or '.data/checkpoints.sqlite'
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        audit_path = (os.getenv('CLAIMS_AUDIT_DB')
                      or (':memory:' if path == ':memory:' else '.data/audit.sqlite'))
        audit = AuditStore(audit_path)
        application.state.audit = audit
        application.state.intent_router = effective_intent_router
        application.state.intent_registry = intent_domain_registry
        # 图内分支策略从领域清单派生；未声明领域时 IntentRoutingPolicy 用默认兼容值。
        from agents.assistant_agent import IntentRoutingPolicy
        application.state.intent_routing_policy = IntentRoutingPolicy.from_registry(
            intent_domain_registry
        )
        try:
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                await saver.setup()
                for plugin in registry.all():
                    plugin.build_graphs(ctx, saver)
                # 兼容层：既有 demo/workbench 路由通过 application.state.* 访问图。
                # 自定义 registry 未装配某领域时置 None，路由按需报错而非启动失败。
                application.state.graph = ctx.graphs.get('claims')
                application.state.investment_graph = ctx.graphs.get('investment')
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

    def authenticate(x_api_key: str = Header(default='')) -> None:
        expected = os.getenv('CLAIMS_API_KEY', '')
        if os.getenv('APP_ENV') == 'production' and not expected:
            raise HTTPException(503, '生产环境尚未配置API认证')
        if expected and not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(401, 'API认证失败')

    def authenticate_reviewer(x_reviewer_key: str = Header(default='')) -> None:
        expected = os.getenv('REVIEWER_API_KEY', '')
        if not expected:
            raise HTTPException(503, '尚未配置审核员认证')
        if not hmac.compare_digest(x_reviewer_key, expected):
            raise HTTPException(403, '需要授权审核员')

    @application.get('/health', response_model=HealthResponse, tags=['健康检查'])
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get('/ready', tags=['健康检查'])
    async def ready(response: Response) -> dict[str, Any]:
        """就绪探针：依赖（图与审计库）未就绪时返回 503，与存活探针区分。"""
        audit = getattr(application.state, 'audit', None)
        checks = {
            'graph': hasattr(application.state, 'graph'),
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

    @application.post('/api/demo/claims/process', response_model=DemoClaimRunResponse, tags=['本地演示'])
    async def process_demo_claim() -> DemoClaimRunResponse:
        """使用固定合成数据运行真实状态机，不连接真实业务或执行付款。"""
        if cfg.app_env == 'production':
            raise HTTPException(404, '演示接口在生产环境不可用')
        from claims_agent import DemoServices
        from demos.mock_claim import build_mock_claim

        claim, documents = build_mock_claim()
        claim_data = claim.model_dump(mode='json')
        graph = build_claim_graph(DemoServices())
        state = await graph.ainvoke(
            {'claim_id': claim.claim_id, 'claim_data': claim_data, 'rounds': 0, 'expert_results': []},
            config={'configurable': {'thread_id': claim.claim_id}, 'recursion_limit': 30, 'max_concurrency': 3},
        )
        opinions = {item['expert']: item for item in state['expert_results']}
        trace = [
            {'node': 'claim_intake', 'status': 'done', 'summary': '报案字段完整，可提交正式工作流',
             'input': claim_data, 'output': {'phase': 'ready', 'missing_structured_fields': []}},
            {'node': 'claim_documents', 'status': 'done', 'summary': '已核查 1 份合成单证',
             'input': {'documents': [item.model_dump() for item in documents]},
             'output': {'recognized_count': 1, 'average_extraction_confidence': 0.97, 'warning_count': 0}},
            {'node': 'claim_followup', 'status': 'skipped', 'summary': '基础材料齐全，无需回流追问',
             'input': {'round': 0}, 'output': {'complete': True}},
            {'node': 'claim_policy', 'status': 'done', 'summary': '演示保单与保障责任核验完成',
             'input': {'policy_id': claim.policy_id},
             'output': {'policy': state['policy_info'], 'verification': state['verified']}},
            *(
                {'node': f'claim_{role}', 'status': 'done',
                 'summary': f"建议 {opinions[role]['recommendation']} · 置信度 {opinions[role]['confidence']:.0%}",
                 'input': {'role': role, 'claim_id': claim.claim_id}, 'output': opinions[role]}
                for role in ('damage', 'risk', 'liability')
            ),
            {'node': 'claim_confidence', 'status': 'done',
             'summary': f"三专家最低置信度 {state['confidence']:.0%}",
             'input': {'method': 'min(expert_confidence)',
                       'expert_scores': {role: opinions[role]['confidence'] for role in opinions}},
             'output': {'confidence': state['confidence'], 'risk_score': state['risk_score']}},
            {'node': 'claim_decision', 'status': 'done', 'summary': '规则校验通过，形成自动受理建议',
             'input': {'amount': str(claim.amount),
                       'coverage_verified': state['verified'].get('coverage_verified'),
                       'confidence': state['confidence'], 'risk_score': state['risk_score']},
             'output': {'decision': state['decision'], 'phase': state['phase']}},
        ]
        result = ClaimResponse(
            claim_id=claim.claim_id, decision=state['decision'], phase=state['phase'],
            confidence=state['confidence'], demo=True, message=state['message'],
        )
        return DemoClaimRunResponse(
            claim=claim, documents=documents, result=result, trace=trace,
        )

    @application.post('/api/demo/claims/process-live', response_model=DemoClaimRunResponse, tags=['本地演示'])
    async def process_demo_claim_live() -> DemoClaimRunResponse:
        """用内存条款 + 真实模型跑完整状态机，演示判责 RAG 检索全过程。"""
        if cfg.app_env == 'production':
            raise HTTPException(404, '演示接口在生产环境不可用')
        from demos.mock_claim import build_mock_claim

        claim, documents = build_mock_claim()
        claim_data = claim.model_dump(mode='json')
        graph = build_claim_graph(service)
        state = await graph.ainvoke(
            {'claim_id': claim.claim_id, 'claim_data': claim_data, 'rounds': 0, 'expert_results': []},
            config={'configurable': {'thread_id': claim.claim_id}, 'recursion_limit': 30, 'max_concurrency': 3},
        )
        opinions = {item.get('expert'): item for item in state.get('expert_results', [])}

        def summary(role: str) -> str:
            opinion = opinions.get(role)
            if not opinion:
                return '专家结果缺失'
            text = f"建议 {opinion.get('recommendation')} · 置信度 {opinion.get('confidence', 0):.0%}"
            if opinion.get('clause_ids'):
                text += f" · 引用条款 {', '.join(opinion['clause_ids'])}"
            return text

        trace = [
            {'node': 'claim_intake', 'status': 'done', 'summary': '报案字段完整，可提交正式工作流',
             'input': claim_data, 'output': {'phase': 'ready', 'missing_structured_fields': []}},
            {'node': 'claim_documents', 'status': 'done', 'summary': '已核查 1 份合成单证',
             'input': {'documents': [item.model_dump() for item in documents]},
             'output': {'recognized_count': 1, 'average_extraction_confidence': 0.97, 'warning_count': 0}},
            {'node': 'claim_followup', 'status': 'skipped', 'summary': '基础材料齐全，无需回流追问',
             'input': {'round': 0}, 'output': {'complete': True}},
            {'node': 'claim_policy', 'status': 'done', 'summary': '保单与保障责任核验完成',
             'input': {'policy_id': claim.policy_id},
             'output': {'policy': state.get('policy_info'), 'verification': state.get('verified')}},
            *(
                {'node': f'claim_{role}', 'status': 'done', 'summary': summary(role),
                 'input': {'role': role, 'claim_id': claim.claim_id}, 'output': opinions.get(role)}
                for role in ('damage', 'risk', 'liability')
            ),
            {'node': 'claim_confidence', 'status': 'done',
             'summary': f"三专家最低置信度 {state.get('confidence', 0):.0%}",
             'input': {'method': 'min(expert_confidence)',
                       'expert_scores': {role: opinions[role].get('confidence') for role in opinions}},
             'output': {'confidence': state.get('confidence'), 'risk_score': state.get('risk_score')}},
            {'node': 'claim_decision', 'status': 'done', 'summary': '规则校验完成，形成审核建议',
             'input': {'amount': str(claim.amount),
                       'coverage_verified': state.get('verified', {}).get('coverage_verified'),
                       'confidence': state.get('confidence'), 'risk_score': state.get('risk_score')},
             'output': {'decision': state.get('decision'), 'phase': state.get('phase')}},
        ]
        result = ClaimResponse(
            claim_id=claim.claim_id, decision=state.get('decision', 'review'), phase=state.get('phase', 'complete'),
            confidence=state.get('confidence'), demo=True, message=state.get('message', ''),
        )
        return DemoClaimRunResponse(claim=claim, documents=documents, result=result, trace=trace)

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
        model = knowledge_model or ClaimLLMFactory.create(
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
        """在内存中识别理赔附件，原文件不会写入服务器磁盘。"""
        if not 1 <= len(files) <= 6:
            raise HTTPException(422, '每次必须上传 1 至 6 个附件')
        payloads: list[tuple[str | None, bytes]] = []
        for upload in files:
            content = await upload.read(MAX_DOCUMENT_BYTES + 1)
            payloads.append((upload.filename, content))
            await upload.close()
        text_llm = document_text_model or get_model('extraction')
        vision_llm = document_vision_model or ClaimLLMFactory.create(
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
        """识别意图后，将理赔、投资和一般问题路由到对应模型提示（LangGraph 多节点编排）。"""
        from agents.assistant_agent import build_assistant_graph, chat_content

        context = json.dumps(
            mask_pii([document.model_dump() for document in body.documents]),
            ensure_ascii=False,
        )
        classifier = intent_model or get_model('classification')
        effective_general = general_model or ClaimLLMFactory.create(
            'main', temperature=0.6, max_tokens=1500, response_format='text', stop=()
        )
        effective_assistant = assistant_model or get_model('customer_service')
        effective_investment = investment_model or ClaimLLMFactory.create(
            'main', temperature=0.4, max_tokens=1800, response_format='text', stop=()
        )

        def model_inputs(model: Any, messages: list[Any]) -> dict[str, Any]:
            return {
                'model': getattr(model, 'model_name', '测试模型'),
                'temperature': getattr(model, 'temperature', None),
                'max_tokens': getattr(model, 'max_tokens', None),
                'messages': [{'role': message.type, 'content': message.content} for message in messages],
            }

        async def events():
            started = time.monotonic()
            stage = 'classification'

            def sse(payload: dict[str, Any]) -> str:
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            try:
                graph = build_assistant_graph(
                    classifier=classifier,
                    general_model=effective_general,
                    assistant_model=effective_assistant,
                    investment_model=effective_investment,
                    service=service,
                    investment_rag=effective_investment_rag,
                    intent_router=getattr(application.state, 'intent_router', None),
                    routing_policy=getattr(application.state, 'intent_routing_policy', None),
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
                stage = state.get('stage', 'claims')
                for event in state.get('events', []):
                    if 'trace' in event:
                        yield sse({'trace': mask_pii(event['trace'])})
                    elif 'intent' in event:
                        yield sse(event)
                    elif 'material_status' in event:
                        yield sse(event)
                    elif 'content' in event:
                        yield sse({'content': mask_pii(event['content'])})
                if state.get('needs_llm_answer', False):
                    llm = state['llm']
                    messages = state['messages']
                    answer_started = time.monotonic()
                    yield sse({'trace': mask_pii({'node': stage, 'status': 'running',
                                                  'model': getattr(llm, 'model_name', '回答模型'),
                                                  'input': model_inputs(llm, messages)})})
                    answer_parts: list[str] = []
                    async for chunk in llm.astream(messages):
                        content = chat_content(chunk)
                        if content:
                            answer_parts.append(content)
                            yield sse({'content': mask_pii(content)})
                    yield sse({'trace': mask_pii({'node': stage, 'status': 'done',
                                                  'elapsed_ms': round((time.monotonic() - answer_started) * 1000),
                                                  'output': {'content': ''.join(answer_parts)}})})
                yield sse({'done': True})
            except Exception as exc:
                logger.exception(
                    'assistant model call failed at stage=%s provider=%s',
                    stage,
                    cfg.model_provider,
                )
                yield sse({'trace': mask_pii({'node': stage, 'status': 'error',
                                              'elapsed_ms': round((time.monotonic() - started) * 1000),
                                              'output': {'error': '调用失败，未返回有效结果',
                                                         'error_type': type(exc).__name__}})})
                ERRORS.labels('assistant_model').inc()
                yield sse({'error': (
                    f'模型调用失败（阶段：{stage}，类型：{type(exc).__name__}），'
                    f'请检查 {cfg.model_provider} 模型服务配置、模型名称和网络连接'
                )})

        return StreamingResponse(
            events(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    return application


app = create_app()

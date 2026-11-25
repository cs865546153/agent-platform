"""意图路由公共契约：领域清单、请求、决策与分类器接口。

该模块不依赖 LangChain/FastAPI，领域包、评测脚本和测试都可以直接复用。
"""
import re
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLevel = Literal["low", "medium", "high"]
DomainMode = Literal["conversation", "case"]
# 能力名是"对话编排处理器"的键：内核按名字分派，不认识任何领域概念。
CAPABILITY_PATTERN = r"^[a-z][a-z0-9_]*$"
# 槽位名由领域清单声明（如 policy_id）；内核只按名字核对是否缺失。
SLOT_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

# 意图路由子系统的统一 logger 命名空间：同一个子系统的日志必须落在一个名字下，
# 否则按 logger 名做 grep / 告警会漏掉 jev 与 hybrid 的记录。
INTENT_LOGGER_NAME = "agent_core.intent"


def fallback_capability(mode: str, risk: str) -> str:
    """意图未声明 ``capability`` 时由 ``mode`` / ``risk`` 推导内核内置能力。

    ``case`` 模式与 ``high`` 风险意图先进人工确认门（guard），其余走直接回答
    （answer）。这样清单里已声明的字段真正驱动分流，而不是在图代码里比字符串。
    """
    if mode == "case" or risk == "high":
        return "guard"
    return "answer"


class ConfirmationSpec(BaseModel):
    """写操作意图的确认契约：对话链路只引导，真正执行交给独立 API。

    领域在 manifest 里声明，内核据此做确定性裁决，不调用 LLM 判断"要不要确认"：

    - ``api``：承载该写操作的确定性 HTTP 入口（相对路径，如 ``/api/claims/process``）；
    - ``requires_human_review``：是否需要授权人员二次审批；
    - ``summary``：给用户看的引导话术；留空时用内核的中性兜底文案。

    字段名/值都属于领域事实源，内核只读取形状，不内置任何业务路径。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    api: str = Field(min_length=1, max_length=200)
    requires_human_review: bool = True
    summary: str | None = Field(default=None, max_length=500)


class IntentSpec(BaseModel):
    """领域内的一个意图及其路由元数据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)
    # JEV 是英文优先模型；提供英文描述可显著提升跨语言路由质量。
    description_en: str | None = Field(default=None, max_length=500)
    examples: list[str] = Field(default_factory=list, max_length=20)
    risk: RiskLevel = "low"
    mode: DomainMode = "conversation"
    # 命中该意图时调用的编排能力名，由领域插件在装配期注册实现。
    # 留空时回落到内核内置能力（见 ``fallback_capability``）。
    capability: str | None = Field(default=None, pattern=CAPABILITY_PATTERN, max_length=64)
    # 声明该意图会改变业务状态。内核的 ConfirmGate 据此拦截并引导到独立 API，
    # 即使领域已注册对话能力也不会在 SSE 链路里直接执行写操作。
    confirmation: ConfirmationSpec | None = None
    # 命中该意图前必须补齐的槽位名；内核的 ClarifyGate 缺哪个问哪个。
    # 槽位值仍由领域槽位抽取器/分类模型填充，内核不解释字段含义。
    required_slots: list[str] = Field(default_factory=list, max_length=8)
    # 面向用户的槽位文案（如 product_id -> 产品代码）；缺失时回退到槽位名。
    slot_labels: dict[str, str] = Field(default_factory=dict, max_length=8)

    @model_validator(mode="after")
    def validate_slots(self) -> "IntentSpec":
        for name in self.required_slots:
            if not re.fullmatch(SLOT_NAME_PATTERN, name):
                raise ValueError(f"非法槽位名：{name}")
        if len(set(self.required_slots)) != len(self.required_slots):
            raise ValueError(f"意图 {self.name} 存在重复 required_slots")
        unknown = sorted(set(self.slot_labels) - set(self.required_slots))
        if unknown:
            raise ValueError(f"意图 {self.name} 的 slot_labels 含未声明槽位：{'、'.join(unknown)}")
        return self

    def criteria_text(self) -> str:
        """返回给意图引擎的意图描述，优先英文。"""
        return self.description_en or self.description

    @property
    def resolved_capability(self) -> str:
        """该意图落到对话图的编排能力名（未声明时为内核内置能力）。"""
        return self.capability or fallback_capability(self.mode, self.risk)


class DomainManifest(BaseModel):
    """一个可插拔领域的清单；新增领域只需提供该清单和图实现。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=32)
    description: str = Field(min_length=1, max_length=500)
    description_en: str | None = Field(default=None, max_length=500)
    version: str = Field(default="1.0.0", max_length=32)
    intents: list[IntentSpec] = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list, max_length=32)
    # 领域专属的意图判别边界，拼进内核的兜底分类提示词；内核不写死任何领域话术。
    classification_hint: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def unique_intents(self) -> "DomainManifest":
        names = [item.name for item in self.intents]
        if len(names) != len(set(names)):
            raise ValueError(f"领域 {self.name} 存在重复意图")
        # 意图绑定的能力必须在领域声明的能力集合里，否则拼错的capability会静默退回内核兜底。
        declared = set(self.capabilities)
        missing = sorted({item.capability for item in self.intents if item.capability} - declared)
        if missing:
            raise ValueError(f"领域 {self.name} 的意图绑定了未声明的能力：{'、'.join(missing)}")
        return self

    def criteria_text(self) -> str:
        return self.description_en or self.description

    def intent(self, name: str) -> IntentSpec | None:
        return next((item for item in self.intents if item.name == name), None)

    def capability_of(self, intent: str, *, default: str = "answer") -> str:
        """该领域里某个意图绑定的编排能力；未注册的意图落回 ``default``。"""
        spec = self.intent(intent)
        return spec.resolved_capability if spec else default


class Turn(BaseModel):
    """最近对话中的一轮。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class IntentRequest(BaseModel):
    """一次意图识别请求；领域清单由分类器持有的注册表提供。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    active_domain: str | None = Field(default=None, max_length=32)
    attachments: list[str] = Field(default_factory=list, max_length=6)


class RouteDecision(BaseModel):
    """统一的路由结果；不同引擎必须收敛到该结构。"""

    model_config = ConfigDict(extra="forbid")

    domain: str
    intent: str
    confidence: float = Field(ge=0, le=1)
    margin: float | None = Field(default=None, ge=0)
    domain_probabilities: dict[str, float] = Field(default_factory=dict)
    intent_probabilities: dict[str, float] = Field(default_factory=dict)
    slots: dict[str, str] = Field(default_factory=dict)
    needs_clarification: bool = False
    engine: str = "unknown"
    latency_ms: int | None = Field(default=None, ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    shadow: dict[str, Any] | None = None


class IntentClassificationError(RuntimeError):
    """意图识别失败；调用方应决定回落或向用户澄清。"""


@runtime_checkable
class IntentClassifier(Protocol):
    """可替换的意图识别引擎（JEV / LLM / 其他）。"""

    async def classify(self, request: IntentRequest) -> RouteDecision:
        ...

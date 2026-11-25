"""工作台的单证识别与问答契约。

``IntentResult`` / ``ChatTurn`` 是领域无关的内核契约：意图名来自领域清单，
业务编号统一进 ``slots``。``claim_id``、``material_round`` 与
``DemoClaimRunResponse`` 仍随示例理赔工作台保留——它们是示例前端的请求形态，
不是内核的路由词汇，自定义前端可以裁掉。
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.schemas import ClaimRequest, ClaimResponse


FieldName = Annotated[str, Field(min_length=1, max_length=50)]
FieldValue = Annotated[str, Field(max_length=300)]
WarningText = Annotated[str, Field(min_length=1, max_length=500)]
# 业务槽位由领域清单声明、由领域能力处理器解释，内核只约束形态。
SLOT_VALUE_LIMIT = 64
SlotValue = Annotated[str, Field(min_length=1, max_length=SLOT_VALUE_LIMIT)]
# 意图名由领域清单（插件 manifest.yaml）声明，内核契约不硬编码具体业务意图，
# 否则新增领域必须改这里。"是不是合法意图"由 DomainRegistry 在运行期判定。
IntentName: TypeAlias = Annotated[str, Field(min_length=1, max_length=64)]


class DocumentAnalysis(BaseModel):
    """单份附件的结构化识别结果。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    document_id: str = Field(min_length=1, max_length=64)
    file_name: str = Field(min_length=1, max_length=255)
    document_type: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=2000)
    fields: dict[FieldName, FieldValue] = Field(default_factory=dict, max_length=20)
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[WarningText] = Field(default_factory=list, max_length=10)


class DocumentBatchResponse(BaseModel):
    """批量附件识别响应。"""

    documents: list[DocumentAnalysis]


class ChatTurn(BaseModel):
    """浏览器传回的有限对话历史。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)
    intent: IntentName | None = None


class AssistantRequest(BaseModel):
    """基于案件和已识别单证的问答请求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    claim_id: str | None = Field(default=None, max_length=64)
    documents: list[DocumentAnalysis] = Field(default_factory=list, max_length=6)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    # 示例理赔工作台的材料补充轮次（旧字段，保留兼容）。
    material_round: int = Field(default=0, ge=0, le=3)
    # 通用澄清轮次（新字段）；为空时回落到 material_round，旧前端无需改动。
    round: int | None = Field(default=None, ge=0, le=3)


class IntentResult(BaseModel):
    """模型对当前用户消息的意图识别结果。

    除 ``intent`` / ``confidence`` 外，模型还可能按领域清单输出业务编号字段
    （如保单号）。内核不枚举这些字段名：多出来的键统一收进 ``slots``，
    由对应领域的能力处理器解释。这样新增领域不需要改契约模型。
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    intent: IntentName
    confidence: float = Field(ge=0, le=1)
    slots: dict[FieldName, SlotValue] = Field(default_factory=dict, max_length=10)

    @model_validator(mode="after")
    def fold_extra_slots(self) -> "IntentResult":
        extra = self.model_extra or {}
        for key in list(extra):
            value = extra.pop(key)
            if isinstance(value, str) and value.strip():
                self.slots.setdefault(key, value.strip()[:SLOT_VALUE_LIMIT])
        return self


class DemoClaimRunResponse(BaseModel):
    """本地合成案件的一次完整状态机运行结果。"""

    claim: ClaimRequest
    documents: list[DocumentAnalysis]
    result: ClaimResponse
    trace: list[dict[str, Any]]

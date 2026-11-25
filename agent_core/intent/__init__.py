"""意图路由公共层：契约、领域注册表与可替换分类器。"""

from .adapter import AssistantIntentRouter, AssistantRoute
from .contracts import (
    CAPABILITY_PATTERN,
    INTENT_LOGGER_NAME,
    DomainManifest,
    IntentClassificationError,
    IntentClassifier,
    IntentRequest,
    IntentSpec,
    RouteDecision,
    Turn,
    fallback_capability,
)
from .registry import DomainRegistry

__all__ = [
    "AssistantIntentRouter",
    "AssistantRoute",
    "CAPABILITY_PATTERN",
    "INTENT_LOGGER_NAME",
    "DomainManifest",
    "DomainRegistry",
    "IntentClassificationError",
    "IntentClassifier",
    "IntentRequest",
    "IntentSpec",
    "RouteDecision",
    "Turn",
    "fallback_capability",
]

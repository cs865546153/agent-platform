"""领域注册表：从 YAML 加载领域清单，供意图路由与领域解析使用。"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .contracts import DomainManifest, RouteDecision

# 目录名用 configs 而不是 config，避免与项目根目录的 config.py 同名冲突。
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "domains.yaml"


class DomainRegistry:
    """启动时构建、运行时只读的领域注册表。"""

    def __init__(self, domains: Iterable[DomainManifest] = ()) -> None:
        self._domains: dict[str, DomainManifest] = {}
        # 意图名 → 领域名。JEV / LLM 引擎返回的是扁平意图名，重名会静默歧义。
        self._intent_owners: dict[str, str] = {}
        for manifest in domains:
            self.register(manifest)

    def register(self, manifest: DomainManifest) -> None:
        if manifest.name in self._domains:
            raise ValueError(f"领域重复注册：{manifest.name}")
        for spec in manifest.intents:
            owner = self._intent_owners.get(spec.name)
            if owner is not None and owner != manifest.name:
                raise ValueError(
                    f"意图「{spec.name}」同时由领域 {owner} 与 {manifest.name} 声明："
                    "路由结果只有意图名，重名无法判定归属，请改名或归并领域"
                )
            self._intent_owners[spec.name] = manifest.name
        self._domains[manifest.name] = manifest

    def intent_owner(self, intent: str) -> str | None:
        """意图名归属的领域；未注册返回 ``None``。"""
        return self._intent_owners.get(intent)

    def get(self, name: str) -> DomainManifest:
        try:
            return self._domains[name]
        except KeyError:
            raise KeyError(f"未注册领域：{name}") from None

    def all(self) -> list[DomainManifest]:
        return list(self._domains.values())

    def names(self) -> list[str]:
        return list(self._domains)

    def resolve(self, decision: RouteDecision) -> DomainManifest:
        """校验路由结果确实落在已注册领域和意图内。"""
        manifest = self.get(decision.domain)
        if manifest.intent(decision.intent) is None:
            raise ValueError(f"领域 {decision.domain} 不存在意图：{decision.intent}")
        return manifest

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "DomainRegistry":
        config_path = Path(path or os.getenv("DOMAINS_CONFIG") or DEFAULT_CONFIG_PATH)
        if not config_path.exists():
            raise FileNotFoundError(f"领域配置不存在：{config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        items = raw.get("domains")
        if not isinstance(items, list) or not items:
            raise ValueError(f"领域配置必须包含非空 domains 列表：{config_path}")
        return cls(DomainManifest.model_validate(item) for item in items)

    @classmethod
    def default(cls) -> "DomainRegistry":
        return cls.from_yaml()

    @classmethod
    def from_plugins(
        cls,
        plugins: Iterable[Any],
        *,
        fallback: "DomainRegistry | None" = None,
    ) -> "DomainRegistry":
        """从领域插件的 ``intent_manifest()`` 聚合领域清单。

        插件声明优先；未声明意图的插件被跳过。若传了 ``fallback``（通常是
        ``DomainRegistry.from_yaml()``），则把 fallback 中未被插件覆盖的领域补齐——
        这样没有对应插件的领域（例如助手图内分流的 general）仍可被意图路由识别。

        插件声明与 fallback 同名时内容必须完全一致，否则抛 ``ValueError``
        （避免两份事实源悄悄分叉）；若插件未声明某领域而 fallback 有，则以 fallback 为准。
        """
        registry = cls()
        for plugin in plugins:
            hook = getattr(plugin, "intent_manifest", None)
            manifest = hook() if callable(hook) else None
            if manifest is None:
                continue
            if manifest.name in registry.names():
                raise ValueError(f"领域插件重复声明领域：{manifest.name}")
            registry.register(manifest)
        if fallback is not None:
            for name in fallback.names():
                if name not in registry.names():
                    registry.register(fallback.get(name))
                    continue
                # 同名必须内容一致，否则两份事实源会悄悄分叉：改了插件 manifest
                # 却忘改 YAML（或反过来）时，运行时"插件优先"会静默忽略其中一份。
                declared = registry.get(name)
                if declared.model_dump() != fallback.get(name).model_dump():
                    raise ValueError(
                        f"领域 {name} 同时由插件与 fallback 声明但内容不一致："
                        "请以插件 manifest 为唯一事实源，或让两份清单完全一致"
                    )
        return registry


__all__ = ["DEFAULT_CONFIG_PATH", "DomainRegistry"]

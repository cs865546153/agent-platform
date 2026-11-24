"""通用接入安全：API Key 认证与审核员认证。

从 app.py 抽出，领域无关。仅依赖环境变量，不认识任何业务主键。
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

API_KEY_HEADER = "X-API-Key"
REVIEWER_KEY_HEADER = "X-Reviewer-Key"


def authenticate(x_api_key: str = Header(default="")) -> None:
    """普通接口认证；开发环境未配置密钥时放行，生产环境强制。"""
    expected = os.getenv("API_KEY", os.getenv("CLAIMS_API_KEY", ""))
    if os.getenv("APP_ENV") == "production" and not expected:
        raise HTTPException(503, "生产环境尚未配置API认证")
    if expected and not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "API认证失败")


def authenticate_reviewer(x_reviewer_key: str = Header(default="")) -> None:
    """审核接口二次认证；未配置审核员密钥时拒绝，避免降级为普通认证。"""
    expected = os.getenv("REVIEWER_API_KEY", "")
    if not expected:
        raise HTTPException(503, "尚未配置审核员认证")
    if not hmac.compare_digest(x_reviewer_key, expected):
        raise HTTPException(403, "需要授权审核员")


def validate_production_keys() -> None:
    """生产启动校验：密钥必须存在、够长且非弱口令。"""
    if os.getenv("APP_ENV") != "production":
        return
    api_key = os.getenv("API_KEY", os.getenv("CLAIMS_API_KEY", ""))
    reviewer_key = os.getenv("REVIEWER_API_KEY", "")
    if not api_key or not reviewer_key:
        raise RuntimeError("生产环境必须配置API_KEY(或CLAIMS_API_KEY)和REVIEWER_API_KEY")
    weak = {"changeme", "secret", "password", "test", "dev", "admin", "123456"}
    for name, value in (("API_KEY", api_key), ("REVIEWER_API_KEY", reviewer_key)):
        if len(value) < 16 or value.lower() in weak:
            raise RuntimeError(f"生产环境 {name} 过弱：至少 16 位且不能是常见口令")


__all__ = ["API_KEY_HEADER", "REVIEWER_KEY_HEADER", "authenticate", "authenticate_reviewer", "validate_production_keys"]

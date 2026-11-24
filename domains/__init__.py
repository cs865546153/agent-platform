"""示例领域插件包。

每个子目录是一个可插拔领域，实现 ``agent_core.plugins.DomainPlugin``：

- ``claims``：保险理赔（报案 → 材料 → 三专家 → 判责）
- ``investment``：投资咨询与交易指令（确认 + 人工审批）

新增领域：复制 ``example`` 骨架，实现插件，然后在 ``registry.py`` 登记。
"""
from domains.registry import default_registry

__all__ = ["default_registry"]

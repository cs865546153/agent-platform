"""最小示例领域：演示如何用 60 行代码接入一个新业务域。

默认不启用；在 ``domains/registry.py`` 的 ``default_registry()`` 里加入
``ExamplePlugin()`` 即可生效。完整指南见 TEMPLATE.md。
"""
from domains.example.plugin import ExamplePlugin

__all__ = ["ExamplePlugin"]

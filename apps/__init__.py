"""按助手裁剪的应用入口。

每个模块只装配自己的领域插件，不 import 其他领域的实现；共享内核与通用装配在
``app_factory.py``。启动示例：

    uvicorn apps.claims_app:app --port 8001
    uvicorn apps.investment_app:app --port 8002
"""

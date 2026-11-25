"""生成新领域骨架：manifest + 插件 + 对话能力 + 路由 + 冒烟测试。

用法::

    python -m scripts.create_domain wealth \\
        --description "财富管理" \\
        --intent "财富咨询" \\
        --example "帮我看看资产配置"

生成后把 ``domains/wealth/plugin.py`` 里的占位服务替换成真实实现，
再在 ``domains/registry.py`` 登记插件即可。内核不需要改动。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / 'templates' / 'domain'
NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,39}$')
RESERVED = frozenset({'general', 'app', 'agent', 'core', 'domains', 'tests', 'main'})


class DomainScaffoldError(ValueError):
    """输入不合法或目标已存在。"""


def _json_text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _python_text(value: str) -> str:
    return value.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')


def _class_name(name: str) -> str:
    return ''.join(part[:1].upper() + part[1:] for part in name.split('_') if part)


def _validate_name(name: str, *, label: str) -> None:
    if not NAME_RE.match(name):
        raise DomainScaffoldError(f'{label} 必须是 snake_case（小写字母开头，仅含小写字母/数字/下划线）：{name}')
    if name in RESERVED:
        raise DomainScaffoldError(f'{label} 不能使用保留名：{name}')


def _substitutions(
    *,
    name: str,
    description: str,
    description_en: str,
    capability: str,
    intent: str,
    intent_description: str,
    intent_description_en: str,
    example: str,
) -> dict[str, str]:
    return {
        'name': name,
        'class_name': _class_name(name),
        'capability': capability,
        'description': _python_text(description),
        'description_yaml': _json_text(description),
        'description_en_yaml': _json_text(description_en),
        'intent_yaml': _json_text(intent),
        'intent_description_yaml': _json_text(intent_description),
        'intent_description_en_yaml': _json_text(intent_description_en),
        'example_yaml': _json_text(example),
    }


def _render(template: Path, values: dict[str, str]) -> str:
    return string.Template(template.read_text(encoding='utf-8')).substitute(values)


def create_domain(
    root: Path,
    *,
    name: str,
    description: str,
    description_en: str | None = None,
    capability: str | None = None,
    intent: str = '咨询',
    intent_description: str | None = None,
    intent_description_en: str | None = None,
    example: str | None = None,
    force: bool = False,
) -> list[Path]:
    """把模板渲染到 ``root`` 下，返回写入的文件列表。"""
    _validate_name(name, label='领域名')
    capability = capability or f'{name}_answer'
    _validate_name(capability, label='能力名')
    description = description.strip()
    if not description:
        raise DomainScaffoldError('description 不能为空')
    description_en = (description_en or f'{description} assistant').strip()
    intent = intent.strip() or '咨询'
    intent_description = (intent_description or f'{description}的典型请求').strip()
    intent_description_en = (intent_description_en or f'Typical {description_en} request').strip()
    example = (example or f'我想咨询{description}').strip()

    values = _substitutions(
        name=name,
        description=description,
        description_en=description_en,
        capability=capability,
        intent=intent,
        intent_description=intent_description,
        intent_description_en=intent_description_en,
        example=example,
    )
    destinations = {
        'manifest.yaml.tmpl': root / 'domains' / name / 'manifest.yaml',
        'plugin.py.tmpl': root / 'domains' / name / 'plugin.py',
        'assistant_handlers.py.tmpl': root / 'domains' / name / 'assistant_handlers.py',
        'routes.py.tmpl': root / 'domains' / name / 'routes.py',
        'test_domain.py.tmpl': root / 'tests' / f'test_{name}_domain.py',
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not force:
        raise DomainScaffoldError('目标已存在（如需覆盖请加 --force）：' + '、'.join(str(p) for p in existing))

    written: list[Path] = []
    for template_name, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_render(TEMPLATE_DIR / template_name, values), encoding='utf-8')
        written.append(destination)
    init_file = root / 'domains' / name / '__init__.py'
    if not init_file.exists():
        init_file.write_text(f'"""{_python_text(description)}领域包。"""\n', encoding='utf-8')
        written.append(init_file)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='生成新的 Agent 领域骨架')
    parser.add_argument('name', help='领域名（snake_case），如 wealth')
    parser.add_argument('--description', required=True, help='领域中文描述')
    parser.add_argument('--description-en', default=None, help='领域英文描述（JEV 使用）')
    parser.add_argument('--capability', default=None, help='对话能力名，默认 <name>_answer')
    parser.add_argument('--intent', default='咨询', help='意图名，默认“咨询”')
    parser.add_argument('--intent-description', default=None, help='意图描述')
    parser.add_argument('--intent-description-en', default=None, help='意图英文描述')
    parser.add_argument('--example', default=None, help='意图示例问题')
    parser.add_argument('--root', default=str(ROOT), help='仓库根目录，默认当前项目')
    parser.add_argument('--force', action='store_true', help='覆盖已存在的文件')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        written = create_domain(
            Path(args.root),
            name=args.name,
            description=args.description,
            description_en=args.description_en,
            capability=args.capability,
            intent=args.intent,
            intent_description=args.intent_description,
            intent_description_en=args.intent_description_en,
            example=args.example,
            force=args.force,
        )
    except DomainScaffoldError as exc:
        print(f'生成失败：{exc}', file=sys.stderr)
        return 2
    for path in written:
        print(f'已生成：{path}')
    print('\n下一步：')
    print(f'  1. 实现 domains/{args.name}/plugin.py 里的领域服务')
    print(f'  2. 在 domains/registry.py 登记 {_class_name(args.name)}Plugin')
    print(f'  3. 跑 pytest -q tests/test_{args.name}_domain.py')
    print('  4. 需要不同循环/上下文时在 build_services 写 ctx.harnesses.register(...)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

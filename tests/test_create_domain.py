"""create_domain 脚手架：渲染、校验与安全转义。"""
from pathlib import Path

import pytest
import yaml

from agent_core.intent.contracts import DomainManifest
from scripts.create_domain import DomainScaffoldError, create_domain


def test_create_domain_writes_complete_skeleton(tmp_path: Path) -> None:
    written = create_domain(
        tmp_path,
        name='wealth',
        description='财富管理',
        description_en='Wealth management',
        intent='财富咨询',
        example='帮我看看资产配置',
    )
    domain_dir = tmp_path / 'domains' / 'wealth'
    assert (domain_dir / 'manifest.yaml').exists()
    assert (domain_dir / 'plugin.py').exists()
    assert (domain_dir / 'assistant_handlers.py').exists()
    assert (domain_dir / 'routes.py').exists()
    assert (domain_dir / '__init__.py').exists()
    assert (tmp_path / 'tests' / 'test_wealth_domain.py').exists()
    assert len(written) == 6

    plugin = (domain_dir / 'plugin.py').read_text(encoding='utf-8')
    assert 'class WealthPlugin(DomainPluginBase)' in plugin
    assert "name = 'wealth'" in plugin
    handlers = (domain_dir / 'assistant_handlers.py').read_text(encoding='utf-8')
    assert 'wealth_answer' in handlers

    manifest = DomainManifest.model_validate(
        yaml.safe_load((domain_dir / 'manifest.yaml').read_text(encoding='utf-8'))
    )
    assert manifest.name == 'wealth'
    assert manifest.description == '财富管理'
    assert manifest.intents[0].capability == 'wealth_answer'
    assert manifest.intents[0].description_en == 'Typical Wealth management request'


def test_create_domain_escapes_template_and_python_metacharacters(tmp_path: Path) -> None:
    create_domain(
        tmp_path,
        name='payments',
        description="支付$金额's 管理",
        intent_description='处理 $100 的 "支付"',
    )
    manifest_text = (tmp_path / 'domains' / 'payments' / 'manifest.yaml').read_text(encoding='utf-8')
    manifest = DomainManifest.model_validate(yaml.safe_load(manifest_text))
    assert manifest.description == "支付$金额's 管理"
    assert manifest.intents[0].description == '处理 $100 的 "支付"'

    plugin = (tmp_path / 'domains' / 'payments' / 'plugin.py').read_text(encoding='utf-8')
    compile(plugin, 'plugin.py', 'exec')
    for name in ('assistant_handlers.py', 'routes.py', '__init__.py'):
        source = (tmp_path / 'domains' / 'payments' / name).read_text(encoding='utf-8')
        compile(source, name, 'exec')
    test_source = (tmp_path / 'tests' / 'test_payments_domain.py').read_text(encoding='utf-8')
    compile(test_source, 'test_payments_domain.py', 'exec')


def test_create_domain_rejects_duplicate_without_force(tmp_path: Path) -> None:
    create_domain(tmp_path, name='wealth', description='财富管理')
    with pytest.raises(DomainScaffoldError, match='目标已存在'):
        create_domain(tmp_path, name='wealth', description='财富管理')
    create_domain(tmp_path, name='wealth', description='财富管理', force=True)


def test_create_domain_rejects_bad_names(tmp_path: Path) -> None:
    with pytest.raises(DomainScaffoldError, match='snake_case'):
        create_domain(tmp_path, name='Wealth-Management', description='财富管理')
    with pytest.raises(DomainScaffoldError, match='保留名'):
        create_domain(tmp_path, name='general', description='通用')
    with pytest.raises(DomainScaffoldError, match='能力名'):
        create_domain(tmp_path, name='wealth', description='财富管理', capability='Wealth Answer')

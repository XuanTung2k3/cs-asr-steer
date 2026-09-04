"""Regression tests for split source identity and the production/test boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from csasr.lss import manifest as manifest_mod
from csasr.utils import provenance


def _fake_repo(root: Path) -> Path:
    files = {
        "src/demo/__init__.py": "",
        "src/demo/worker.py": "VALUE = 1\n",
        "configs/demo.yaml": "value: 1\n",
        "tests/test_demo.py": "def test_demo():\n    assert True\n",
        "cs_asr_e1_e5.sh": "#!/bin/bash\n",
        "cs_asr_nat5h.sh": "#!/bin/bash\n",
        "cs_asr_lss.sh": "#!/bin/bash\n",
        "pyproject.toml": "[project]\nname = 'demo'\n",
        "environment.lock.txt": "python=3.11\n",
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _bind_snapshot_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        manifest_mod, "source_snapshot_hash",
        lambda: provenance.source_snapshot_hash(root))
    monkeypatch.setattr(
        manifest_mod, "code_config_snapshot_hash",
        lambda: provenance.code_config_snapshot_hash(root))
    monkeypatch.setattr(
        manifest_mod, "test_snapshot_hash",
        lambda: provenance.test_snapshot_hash(root))


def _cfg(root: Path) -> dict:
    return {
        "experiment": {"output_root": str(root / "artifacts")},
        "model": {"id": "fake", "revision": "fixed"},
    }


def _publish(root: Path, cfg: dict) -> Path:
    path = root / "artifacts" / "sample.parquet"
    manifest_mod.publish_frame(
        path, pd.DataFrame({"utterance_id": ["u1"]}),
        stage="identity_test", cfg=cfg, key_columns=("utterance_id",))
    return path


def _replace_identity(path: Path, identity: dict) -> None:
    sidecar = manifest_mod.manifest_path(path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["identity"] = identity
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _common_identity(identity: dict) -> dict:
    source_fields = {"source_sha256", "code_config_sha256", "test_sha256"}
    return {key: value for key, value in identity.items()
            if key not in source_fields}


def test_new_manifest_records_split_source_identity(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo")
    _bind_snapshot_root(monkeypatch, root)
    cfg = _cfg(root)
    path = _publish(root, cfg)

    recorded = manifest_mod.load(path)["identity"]
    assert "source_sha256" not in recorded
    assert recorded["code_config_sha256"] == provenance.code_config_snapshot_hash(root)
    assert recorded["test_sha256"] == provenance.test_snapshot_hash(root)
    assert manifest_mod.verify(path, cfg=cfg, require_identity=True)["ok"] is True


def test_legacy_identity_verifies_with_the_original_combined_hash(
        tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo")
    _bind_snapshot_root(monkeypatch, root)
    cfg = _cfg(root)
    path = _publish(root, cfg)
    current = manifest_mod.load(path)["identity"]
    legacy = {
        "source_sha256": provenance.source_snapshot_hash(root),
        **_common_identity(current),
    }
    _replace_identity(path, legacy)

    assert manifest_mod.verify(path, cfg=cfg, require_identity=True)["ok"] is True

    # The grandfathered branch is intentionally exact: tests were gated in the
    # legacy format, and an immutable old manifest keeps that meaning.
    (root / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    verdict = manifest_mod.verify(path, cfg=cfg, require_identity=True)
    assert verdict["verdict"] == manifest_mod.IDENTITY_MISMATCH
    assert "source_sha256" in verdict["detail"]


def test_test_only_edit_is_recorded_but_not_gated(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo")
    _bind_snapshot_root(monkeypatch, root)
    cfg = _cfg(root)
    path = _publish(root, cfg)
    before_code = provenance.code_config_snapshot_hash(root)
    before_tests = provenance.test_snapshot_hash(root)
    recorded_tests = manifest_mod.load(path)["identity"]["test_sha256"]

    (root / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert 2 + 2 == 4\n", encoding="utf-8")

    assert provenance.code_config_snapshot_hash(root) == before_code
    assert provenance.test_snapshot_hash(root) != before_tests
    verdict = manifest_mod.verify(path, cfg=cfg, require_identity=True)
    assert verdict["ok"] is True
    assert verdict["manifest"]["identity"]["test_sha256"] == recorded_tests


@pytest.mark.parametrize("relative", ["src/demo/worker.py", "configs/demo.yaml"])
def test_code_or_config_edit_is_gated(relative, tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo")
    _bind_snapshot_root(monkeypatch, root)
    cfg = _cfg(root)
    path = _publish(root, cfg)
    before = provenance.code_config_snapshot_hash(root)

    changed = root / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "# changed\n",
                       encoding="utf-8")

    assert provenance.code_config_snapshot_hash(root) != before
    verdict = manifest_mod.verify(path, cfg=cfg, require_identity=True)
    assert verdict["verdict"] == manifest_mod.IDENTITY_MISMATCH
    assert "code_config_sha256" in verdict["detail"]


@pytest.mark.parametrize("source_fields", [
    {},
    {"code_config_sha256": "1" * 64},
    {"test_sha256": "2" * 64},
    {
        "source_sha256": "0" * 64,
        "code_config_sha256": "1" * 64,
        "test_sha256": "2" * 64,
    },
])
def test_malformed_source_identity_is_refused(
        source_fields, tmp_path, monkeypatch):
    root = _fake_repo(tmp_path / "repo")
    _bind_snapshot_root(monkeypatch, root)
    cfg = _cfg(root)
    path = _publish(root, cfg)
    current = manifest_mod.load(path)["identity"]
    _replace_identity(path, {**_common_identity(current), **source_fields})

    verdict = manifest_mod.verify(path, cfg=cfg, require_identity=True)
    assert verdict["verdict"] == manifest_mod.IDENTITY_MISMATCH
    assert verdict["detail"].startswith("malformed source identity:")


def _module_name(path: Path, source_root: Path) -> tuple[str, bool]:
    relative = path.relative_to(source_root)
    is_package = relative.name == "__init__.py"
    parts = list(relative.with_suffix("").parts)
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _imports(path: Path, module: str, is_package: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module.split(".") if is_package else module.split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package) - (node.level - 1)
            prefix = package[:max(0, keep)]
        else:
            prefix = []
        base_parts = prefix + (node.module.split(".") if node.module else [])
        base = ".".join(part for part in base_parts if part)
        if base:
            imported.add(base)
        for alias in node.names:
            if alias.name != "*":
                imported.add(".".join(part for part in (base, alias.name) if part))
    return imported


def _forbidden_test_import_chains(source_root: Path) -> list[str]:
    modules: dict[str, tuple[Path, bool]] = {}
    for path in sorted(source_root.rglob("*.py")):
        module, is_package = _module_name(path, source_root)
        if module:
            modules[module] = (path, is_package)

    graph: dict[str, set[str]] = {}
    for module, (path, is_package) in modules.items():
        graph[module] = _imports(path, module, is_package)

    failures: set[str] = set()

    def visit(origin: str, current: str, chain: tuple[str, ...]) -> None:
        for imported in sorted(graph.get(current, ())):
            next_chain = (*chain, imported)
            if imported == "tests" or imported.startswith("tests."):
                failures.add(" -> ".join((origin, *next_chain)))
            elif imported in modules and imported not in chain:
                visit(origin, imported, next_chain)

    for module in sorted(modules):
        visit(module, module, ())
    return sorted(failures)


def test_import_guard_reports_direct_and_transitive_offenders(tmp_path):
    source = tmp_path / "src"
    (source / "demo").mkdir(parents=True)
    (source / "demo" / "__init__.py").write_text("", encoding="utf-8")
    (source / "demo" / "entry.py").write_text(
        "from demo import helper\n", encoding="utf-8")
    (source / "demo" / "helper.py").write_text(
        "from tests.fixtures import secret\n", encoding="utf-8")

    failures = _forbidden_test_import_chains(source)
    assert "demo.helper -> tests.fixtures" in failures
    assert "demo.entry -> demo.helper -> tests.fixtures" in failures


def test_production_source_does_not_import_tests_directly_or_transitively():
    source = Path(__file__).resolve().parents[1] / "src"
    failures = _forbidden_test_import_chains(source)
    assert not failures, "production imports test-only code:\n" + "\n".join(failures)

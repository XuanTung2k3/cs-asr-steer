"""RC7-0C continuation safety: separate root, immutable inputs, recorded overrides, 2 GPU-h ceiling."""
import hashlib
import json

import pytest

from reft.rc7.n4_continuation import continuation_plan, fingerprint, prepare


def _archive(tmp_path, decision='N2_NO_GO'):
    src = tmp_path / 'core'; (src / 'B0' / 'predictions').mkdir(parents=True); (src / 'jobs' / 'N4').mkdir(parents=True)
    (src / 'data_manifest.json').write_text('{"roles": {}}')
    data = hashlib.sha256((src / 'data_manifest.json').read_bytes()).hexdigest()
    plan = {'campaign_id': 'c', 'created_epoch': 1.0, 'source_commit': 'abc', 'config_sha256': 'cfg', 'data_sha256': data,
            'config': {'campaign_root': str(src), 'resources': {'gpu_hours': 72, 'reserve_gpu_hours': 3, 'wall_hours': 120}}}
    (src / 'frozen_plan.json').write_text(json.dumps(plan))
    for f in ['N1_selection.json', 'N2_selection.json']:
        (src / f).write_text('{}')
    (src / 'N2_decision.json').write_text(json.dumps({'decision': decision}))
    for f in ['identity.json', 'SEALED.json', 'metrics.json']:
        (src / 'B0' / f).write_text('{}')
    (src / 'B0' / 'predictions' / 'u.json').write_text('{}')
    (src / 'jobs' / 'N4' / 'result.json').write_text('{"status": "FAILED"}')
    return src, plan


def test_plan_overrides_are_engineering_only_and_recorded():
    plan = {'created_epoch': 1.0, 'source_commit': 's', 'data_sha256': 'd',
            'config': {'campaign_root': '/old', 'resources': {'gpu_hours': 72, 'reserve_gpu_hours': 3, 'wall_hours': 120},
                       'gate': {'epsilon': .25}, 'training': {'lr': 1}}}
    new, o = continuation_plan(plan, '/new', 1.0, 2.0, 5.0)
    assert new['source_commit'] == 's' and new['data_sha256'] == 'd'
    assert new['config']['gate'] == plan['config']['gate'] and new['config']['training'] == plan['config']['training']
    assert new['config']['campaign_root'] == '/new' and new['created_epoch'] == 5.0
    assert set(o) == {'created_epoch', 'config.campaign_root', 'config.resources.gpu_hours',
                      'config.resources.reserve_gpu_hours', 'config.resources.wall_hours'}
    assert plan['config']['campaign_root'] == '/old'  # input not mutated


@pytest.mark.parametrize('hours', [0, 2.5])
def test_gpu_ceiling(hours):
    with pytest.raises(ValueError, match='ceiling'):
        continuation_plan({'created_epoch': 0, 'config': {'resources': {'gpu_hours': 1, 'reserve_gpu_hours': 0, 'wall_hours': 1}}},
                          '/r', hours, 1, 0)


def test_prepare_copies_inputs_and_leaves_archive_untouched(tmp_path):
    src, plan = _archive(tmp_path)
    before = fingerprint(src)
    rec = prepare(src, tmp_path / 'cont')
    assert fingerprint(src) == before == rec['archived_fingerprint_before']
    root = tmp_path / 'cont'
    assert json.loads((root / 'frozen_plan.json').read_text())['config']['campaign_root'] == str(root.resolve())
    assert (root / 'B0' / 'predictions' / 'u.json').exists() and not (root / 'B0').is_symlink()
    assert rec['data_sha256'] == plan['data_sha256'] and rec['archived_source_commit'] == 'abc'


def test_prepare_refuses_nested_existing_or_go_roots(tmp_path):
    src, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        prepare(src, src / 'cont')
    with pytest.raises(ValueError, match='separate'):
        prepare(src, src)
    prepare(src, tmp_path / 'once')
    with pytest.raises(FileExistsError):
        prepare(src, tmp_path / 'once')
    src2, _ = _archive(tmp_path / 'go', decision='N2_EXPLORATORY_GO')
    with pytest.raises(ValueError, match='NO_GO'):
        prepare(src2, tmp_path / 'go_cont')


def test_prepare_refuses_confirmation_package(tmp_path):
    src, _ = _archive(tmp_path)
    (src / 'confirmation_freeze.json').write_text('{}')
    with pytest.raises(ValueError, match='confirmation'):
        prepare(src, tmp_path / 'c2')

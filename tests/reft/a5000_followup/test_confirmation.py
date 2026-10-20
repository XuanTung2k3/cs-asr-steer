import json

import pytest

from reft.a5000_followup.confirm import require_completed


def test_confirmation_refuses_unfinished_or_partial_primary(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'RUNNING'}))
    with pytest.raises(RuntimeError):require_completed(tmp_path)
    (tmp_path/'status.json').write_text(json.dumps({'status':'COMPLETE'}))
    with pytest.raises(RuntimeError):require_completed(tmp_path)


def test_confirmation_binds_exact_selected_checkpoint_bytes(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'COMPLETE'}));(tmp_path/'comparison.json').write_text('{}')
    checkpoint=tmp_path/'ad.pt';checkpoint.write_bytes(b'checkpoint')
    from reft.a5000.data import sha
    d={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
    (tmp_path/'selection.json').write_text(json.dumps({'ereft':d,'lra':d}))
    assert set(require_completed(tmp_path))=={'ereft','lra'}
    checkpoint.write_bytes(b'changed')
    with pytest.raises(AssertionError):require_completed(tmp_path)

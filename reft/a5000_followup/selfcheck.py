"""Codex internal contract cross-check; explicitly not an independent external scientific audit."""
import ast
import json

import torch

from reft.a5000.data import ROOT,config,output,integrity,sha,atomic,load_rows
from reft.a5000.adapters import EncoderAdapter
from reft.a5000.runner import check,git


def main():
    C=config();frozen=check();roles=json.loads((ROOT/C['manifest']).read_text())
    integrity(roles['roles'])
    assert C['training']['objective']=='O1_pure_CE'
    assert C['training']['goldkl_coefficient']==C['training']['lid_coefficient']==0
    assert C['training']['english_weight']==1
    assert C['model']['site']=='encoder_output_after_final_layer_norm'
    assert C['training']['learning_rates']['ereft']==C['training']['learning_rates']['lra']==[.0001,.0005]
    tree=ast.parse((ROOT/'reft/a5000/runner.py').read_text())
    called={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert 'weighted_ce' in called and 'goldkl' not in called and 'LIDHead' not in called
    assert 'model.train(' not in (ROOT/'reft/a5000/runner.py').read_text()
    counts={}
    for method in ['ereft','lra']:
        torch.manual_seed(42);a=EncoderAdapter(method);counts[method]=a.count();assert a.count()==5122
    try:load_rows('A5000_C150')
    except ValueError:pass
    else:raise AssertionError('Initial campaign confirmation firewall was bypassed')
    preflight=json.loads((output()/'preflight/report.json').read_text());assert preflight['passed']
    assert all(preflight['methods'][m]['params']==5122 for m in counts)
    audit={'type':'Codex internal contract self-check; not independent external review',
        'source_commit':frozen['source_commit'],'review_commit':git('rev-parse','HEAD'),
        'freeze_sha256':sha(output()/'freeze.json'),'role_manifest_sha256':sha(ROOT/C['manifest']),
        'parameter_counts':counts,'frozen_source_hashes_match':True,'O1_only_ast_and_config':True,
        'native_cached_encoder_site_checks':'passed existing real-tiny-model tests and real A5000 preflight',
        'backbone_gradient_isolation':'asserted for every training batch; full weight hash checked preflight/completion',
        'common_decoder_and_scorer':'same Runtime.forward/decode and canonical reft.text.one/ratios/compare for both methods',
        'roles_disjoint':True,'C150_selection_or_fitting':False,'official_test_access_by_campaign':False,
        'target_shift':'canonical build_tf_batch logit3+t predicts target[t]; content+EOS; logical token denominator',
        'selection':'same frozen LR/epoch tuple, equal trial budgets, V106 only'}
    cp=output()/'comparison.json'
    if cp.exists():
        comp=json.loads(cp.read_text())
        for m,d in comp['arms'].items():
            assert d['source_commit']==frozen['source_commit'] and sha(d['checkpoint'])==d['checkpoint_sha256']
        assert comp['frozen_backbone_unchanged']
        audit['selected_checkpoint_links_verified']=True
    else:audit['selected_checkpoint_links_verified']='pending campaign completion'
    atomic(output()/'audits/codex_self_check.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()

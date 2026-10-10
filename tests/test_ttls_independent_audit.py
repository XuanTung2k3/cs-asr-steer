"""Independent audit regression tests: scoring and reproduced defects, not method tuning."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1_independent_analysis import alignment,one,compare,category

def test_alignment_against_canonical_all_saved_hypotheses():
    from csasr.evaluation.mer import align_tokens
    from csasr.data.normalize import normalize_text,segment_units
    rows=json.loads((ROOT/'results/inference_cf/ttls_r1_independent_audit/independent_metrics.json').read_text())['rows']
    for row in rows:
        a=[u.surface for u in segment_units(normalize_text(row['reference']))]
        for s,h in row['texts'].items():
            b=[u.surface for u in segment_units(normalize_text(h))]
            assert alignment(a,b)==[(o.op,o.ref_idx,o.hyp_idx) for o in align_tokens(a,b)]

def test_independent_counts_sum_and_pier_identity():
    d=json.loads((ROOT/'results/inference_cf/ttls_r1_independent_audit/independent_metrics.json').read_text())
    assert d['published_mismatches']==[]
    for s,m in d['metrics'].items():
        assert sum(r['systems'][s]['counts'][0] for r in d['rows'])==m['poi_errors']
        assert sum(r['systems'][s]['counts'][2] for r in d['rows'])==m['mixed_errors']
        assert m['mixed_errors']==m['sub']+m['del']+m['ins']
        t=d['transitions'][s]
        assert d['metrics']['B0']['poi_errors']-m['poi_errors']==t['corrections']-t['corruptions']

def test_category_differential_all_raw_transcripts():
    from csasr.evaluation.pier import evaluate_pois
    d=json.loads((ROOT/'results/inference_cf/ttls_r1_independent_audit/independent_metrics.json').read_text())
    for row in d['rows']:
        for s,h in row['texts'].items():
            o=one(row['reference'],h)
            for p in evaluate_pois(row['reference'],h):
                assert category(o,p.poi_index)==(p.category,p.hyp_surface)

def test_ttls_word_boundary_repairs_are_not_new_lexical_substitutions():
    d=json.loads((ROOT/'results/inference_cf/ttls_r1_independent_audit/independent_metrics.json').read_text())
    e=[e for e in d['events']['T1'] if e['type']=='correction']
    assert len(e)==8
    assert sum(x['canonical_category']=='same_language_substitution' for x in e)==3
    assert all(x['audit_kind']!='lexical' for x in e)

def test_zero_direction_normpreserve_is_not_identity_bf16():
    import torch
    from csasr.models.hooks import apply_steering
    torch.manual_seed(100)
    h=torch.randn(1,20,1280,dtype=torch.bfloat16)
    z=torch.zeros(1280)
    edited=apply_steering(h,z,1.,1.,torch.ones(1,20,dtype=torch.bfloat16),True)
    assert not torch.equal(h,edited) # records existing defect; does not accept it as a scientific invariant
    assert torch.equal(h,apply_steering(h,z,0.,1.,None,True))

def test_saved_initial_preservation_claim_is_false_for_ttls():
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'results/inference_cf/ttls_r1/run1/rows').glob('*.json'))]
    assert any(r['T2']['log']['parts'][0]['P']>0 for r in rows)
    assert all(r['T2A']['log']['parts'][0]['P']==0 for r in rows)

def test_replay_selection_has_every_changed_t1_a2_and_accepted_ac():
    sel=json.loads((ROOT/'results/inference_cf/ttls_r1_independent_audit/replay_selection.json').read_text())
    assert len(sel['indices'])==24
    for i,p in enumerate(sorted((ROOT/'results/inference_cf/ttls_r1/run1/rows').glob('*.json'))):
        r=json.loads(p.read_text())
        if not r['T1']['equal_B0'] or not r['B2']['equal_B0'] or r['candidates']['t_star'] is not None:assert i in sel['indices']

def test_proposed_ratio_repair_zero_identity_and_gradient():
    # Engineering-only proposal, never installed on historical experiment code.
    import torch
    for dtype in [torch.float64,torch.bfloat16]:
        torch.manual_seed(100);h=torch.randn(1,20,1280,dtype=dtype);z=torch.zeros(1280,dtype=dtype,requires_grad=True)
        u=h+z;out=u*(h.norm(dim=-1,keepdim=True)/u.norm(dim=-1,keepdim=True).clamp_min(1e-6))
        assert torch.equal(out,h)
        out[...,0].sum().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.norm()>0

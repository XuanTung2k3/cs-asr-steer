#!/usr/bin/env python
"""Independent audit completion: integrity, seals, stratified effects and defect scope."""
from pathlib import Path
import json,hashlib,sys,collections,subprocess,inspect
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1_independent_analysis import one,compare,ratios
OUT=ROOT/'results/inference_cf/ttls_r1_independent_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(o):return 'sha256:'+hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
    d=json.loads((OUT/'independent_metrics.json').read_text());plan=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());man=json.loads((ROOT/'results/inference_cf/ttls_r1/run1/manifest.json').read_text());seal=json.loads((ROOT/'results/inference_cf/ttls_r1/output_seal.json').read_text())
    hash_checks={}
    for name,obj,key in [('plan',plan,'plan_hash'),('manifest',man,'manifest_hash'),('seal',seal,'seal_hash')]:hash_checks[name]=canonical_hash({k:v for k,v in obj.items() if k!=key})==obj[key]
    for r in plan['rows']:
        for nm,expected in [('audio_path','audio_full_sha256')]:assert sha(r[nm])==r[expected]
        assert sha(ROOT/r['A2']['row'])==r['A2']['row_sha256'];assert sha(ROOT/r['A2']['masters_npz'])==r['A2']['masters_npz_sha256']
    assert all(hash_checks.values())
    summary={'hash_checks':hash_checks,'audio100_verified':True,'A2_archives100_verified':True,'panel_sha256':sha(ROOT/'docs/inference_cf/P2_SEL_MINI_PANEL.json'),'config_sha256':sha(ROOT/'configs/inference_cf/ttls_r1.json'),'original_source_commit':man['git_commit'],'completed_commit':'800fee26a55651c307a1a33b066fdb47ece22b05'}
    replay=json.loads((OUT/'replay/summary.json').read_text());summary['replay']={'utterances':24,'dialogues':replay['selection']['dialogues'],'elapsed_seconds':replay['elapsed_seconds'],'job':replay['job'],'comparisons':len(replay['comparisons']),'exact_matches':all(x.get('tokens_equal',True) and x.get('termination_equal',True) and x.get('equal',True) for x in replay['comparisons']),'loss_max_abs':max(x.get('loss_max_abs',0) for x in replay['comparisons']),'gradient_max_abs':max(x.get('gradient_max_abs',0) for x in replay['comparisons']),'z_max_abs':max(x.get('z_max_abs',0) for x in replay['comparisons']),'A2_reconstruction24':all(x['effective_archive_equal'] and x['tokens_equal'] and x['termination_equal'] for x in replay['ce_reconstruction']),'reset':replay['reset_final'],'nonln_unchanged':replay['nonln_unchanged'],'model_grads_none':replay['model_grads_none'],'zero_vector_identity_failures':[x['id'] for x in replay['zero_controls'] if not x['tokens_equal_B0']],'zero_vector_logit_difference_max':max(x['max_abs_logit'] for x in replay['zero_controls']),'zero_vector_native_chord_max':max(x['ffn_max_chord'] for x in replay['zero_controls']),'native_chord_max':max(x['native_ffn_chord_max'] for x in replay['consumption']),'norm_relative_error_max':max(x['native_norm_relative_error_max'] for x in replay['consumption']),'outside_direct_edit_max':max(x['outside_direct_chord_max'] for x in replay['consumption'])}
    sub={}
    for s in d['metrics']:
        events=d['events'][s];corr=[e for e in events if e['type']=='correction']
        arr=np.array([r['systems'][s]['counts'] for r in d['rows']]);brr=np.array([r['systems']['B0']['counts'] for r in d['rows']])
        eos=np.array([r['systems'][s]['eos_recovery'] for r in d['rows']]);delta=brr-arr
        bydlg={z:int(sum(delta[i,0] for i,r in enumerate(d['rows']) if r['dialogue']==z)) for z in sorted({r['dialogue'] for r in d['rows']})}
        dlgrepair={z:int(sum(delta[i,4] for i,r in enumerate(d['rows']) if r['dialogue']==z)) for z in bydlg}
        sub[s]={'correction_dialogues':len({e['dialogue'] for e in corr}),'correction_kind':dict(collections.Counter(e['audit_kind'] for e in corr)),'canonical_genuine':sum(e['canonical_category'] in ['wrong_language_substitution','phonetic_transliteration_or_script','same_language_substitution','other'] for e in corr),'canonical_genuine_dialogues':len({e['dialogue'] for e in corr if e['canonical_category'] in ['wrong_language_substitution','phonetic_transliteration_or_script','same_language_substitution','other']}),'eos_recovery_rows':int(eos.sum()),'net_poi_rescue_in_eos_rows':int(delta[eos,0].sum()),'net_zh_rescue_in_eos_rows':int(delta[eos,4].sum()),'net_deletion_rescue_in_eos_rows':int(delta[eos,9].sum()),'net_poi_by_dialogue':bydlg,'net_zh_by_dialogue':dlgrepair,'lodo_poi_rescue_min':int(delta[:,0].sum()-max(bydlg.values())),'lodo_zh_rescue_min':int(delta[:,4].sum()-max(dlgrepair.values()))}
    summary['effect_decomposition']=sub
    # Only sealed census outputs may receive reference-based scoring.
    zseal=json.loads((OUT/'zero_census_seal.json').read_text());assert all(sha(ROOT/p)==h for p,h in zseal['files'].items())
    cnt=[];tr=collections.Counter();ev=[];zero_rows=[]
    for i,r in enumerate(d['rows']):
        raw=json.loads((OUT/f'zero_census/{i:03d}.json').read_text());z=raw['zero'];assert raw['baseline_match']
        b=one(r['reference'],r['texts']['B0']);m=one(r['reference'],z['text']);cnt.append(m['counts']);e=compare(b,m,r['id'],r['dialogue'],False);ev.extend(e.pop('events'));tr.update(e)
        zero_rows.append({'id':r['id'],'tokens_equal':raw['zero_matches_B0'],'normalized_units_equal':b['hyp']==m['hyp'],'count_delta_vs_B0':[a-b for a,b in zip(m['counts'],b['counts'])],'text':z['text']})
    c=np.array(cnt).sum(0);base=np.array([r['systems']['B0']['counts'] for r in d['rows']]).sum(0)
    summary['zero_census']={'gpu':json.loads((OUT/'zero_census/summary.json').read_text()),'metrics':ratios(c),'counts':c.tolist(),'delta_counts_vs_B0':(c-base).tolist(),'transitions':dict(tr),'events':ev,'nonidentity':[r for r in zero_rows if not r['tokens_equal']],'normalized_nonidentity':[r['id'] for r in zero_rows if not r['normalized_units_equal']]}
    # Recompute the original frozen label independently from raw transition accounting.
    qualified=[];mixed=[];matched={'T1':'B2','T2':'T2A','T4':'T3','T6':'T5'}
    for s,y in matched.items():
        x,t=d['metrics'][s],d['transitions'][s];q=sub[s];corr,corrupt=t['corrections'],t['corruptions']
        safe=(x['mer']-d['metrics']['B0']['mer']<=.01 and x['zh_cer']-d['metrics']['B0']['zh_cer']<=.015 and (1-t['matrix_loss']/t['matrix_den'])>=.98 and t['outside_damage']/t['outside_den']<=.03 and corrupt/t['correct_en']<=.05 and t['caps']-d['transitions']['B0']['caps']<=1 and t['severe_truncation']==0)
        if q['canonical_genuine']>=8 and q['canonical_genuine_dialogues']>=4 and safe and q['canonical_genuine']>=sub[y]['canonical_genuine'] and t['new_zh']<=d['transitions'][y]['new_zh']:qualified.append(s)
        if q['canonical_genuine']>=3 and corr-corrupt>0:mixed.append(s)
    summary['original_frozen_label_independent']={'label':'TTLS_R1_PROMISING' if qualified else 'TTLS_R1_MIXED' if mixed else 'TTLS_R1_NOT_SUPPORTED','qualifying_promising':qualified,'qualifying_mixed':mixed,'caveat':'canonical taxonomy includes word-boundary repairs'}
    summary['audit_verdict']='AUDIT_FAIL_INVALID_RESULTS'
    summary['invalid_scope']='TTLS zero-activation control and clean-start causal attribution; corpus metrics and A2 comparator outputs remain reproduced. No corruption of original artifacts alleged.'
    summary['published_metric_mismatches']=d['published_mismatches']
    (OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['effect_decomposition','zero_census']},indent=2));print('ZERO',json.dumps(summary['zero_census'],ensure_ascii=False,indent=1))
if __name__=='__main__':main()

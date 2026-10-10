#!/usr/bin/env python
"""Independent repaired-run audit: own DP aggregation, lexical classes, clusters and provenance.
Reuses ONLY the earlier independent auditor's normalization/DP/count primitives, not Claude's evaluator.
"""
from pathlib import Path
import sys,json,hashlib,collections,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1_independent_analysis import one,compare,ratios
OUT=ROOT/'results/inference_cf/ttls_r1r_independent_audit';ARMS=['T1','T2','T4','T6']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ch(o):return 'sha256:'+hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def lexical(e):
    cat=e['canonical_category']
    if cat in ('deletion','boundary_error','insertion_near_poi'):return 'deletion_or_boundary'
    if cat in ('wrong_language_substitution','phonetic_transliteration_or_script'):return 'genuine_wrong_language'
    a=''.join(e['reference'].lower().split());b=''.join(e['before'].lower().split())
    return 'word_boundary_repair' if a and a in b else 'genuine_same_language'
def main():
    OUT.mkdir(parents=True,exist_ok=True);base=ROOT/'results/inference_cf/ttls_r1r';plan=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());man=json.loads((base/'run1/manifest.json').read_text());seal=json.loads((base/'output_seal.json').read_text())
    assert ch({k:v for k,v in man.items() if k!='manifest_hash'})==man['manifest_hash'];assert ch({k:v for k,v in seal.items() if k!='seal_hash'})==seal['seal_hash']
    assert all(sha(ROOT/p)==h.removeprefix('sha256:') for p,h in man['sources'].items());assert all(sha(ROOT/p)==h.removeprefix('sha256:') for p,h in seal['files'].items())
    assert ch({k:v for k,v in plan.items() if k!='plan_hash'})==plan['plan_hash']==man['plan_hash']
    original=json.loads((ROOT/'results/inference_cf/ttls_r1/run1/manifest.json').read_text());assert all(sha(ROOT/p)==h.removeprefix('sha256:') for p,h in original['sources'].items())
    import pyarrow.parquet as pq
    refpath=Path('/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet')
    refs={r['utterance_id']:r for r in pq.read_table(refpath,columns=['utterance_id','dialogue_id','role','transcript_raw'],filters=[('utterance_id','in',man['ids'])]).to_pylist()};assert set(refs)==set(man['ids'])
    systems=['B0','B1','B2','T1','T2','T2A','T3','T4','T5','T6','ACSUB','CD','Z0','Z0H']+[a+'_R1' for a in ARMS]
    C={a:[] for a in systems};TR={a:collections.Counter() for a in systems};EV={a:[] for a in systems};records=[];integ=[];selection=set();repair={a:[] for a in ARMS}
    for i,s in enumerate(plan['rows']):
        old=json.loads((ROOT/f'results/inference_cf/ttls_r1/run1/rows/{i:03d}.json').read_text());new=json.loads((base/f'run1/rows/{i:03d}.json').read_text());zero=json.loads((base/f'run1/integrity/{i:03d}.json').read_text());hist=json.loads((ROOT/f'results/inference_cf/ttls_r1_independent_audit/zero_census/{i:03d}.json').read_text())
        assert old['identity']==new['identity']==zero['identity']==s['utterance_id']==man['ids'][i];assert new['status']==zero['status']=='ok';assert refs[s['utterance_id']]['role']=='D-dev-select';assert refs[s['utterance_id']]['dialogue_id']==s['dialogue_id'];assert sha(s['audio_path'])==s['audio_full_sha256']
        h={'B0':{'text':s['y_B_text'],'tokens':s['y_B'],'terminated':s['y_B_terminated']},'B1':{'text':s['y_A_text'],'tokens':s['y_A'],'terminated':s['y_A_terminated']},**{a:old[a] for a in ['B2','T2A','T3','T5','ACSUB','CD']},**{a:new[a] for a in ARMS},**{a+'_R1':old[a] for a in ARMS},'Z0':zero['zero']['ALL'],'Z0H':hist['zero']}
        assert all(h['Z0'][k]==h['B0'][k] for k in ('tokens','text','terminated'));assert all(v for v in zero['candidates_equal_R1'].values())
        for mask,z in zero['zero'].items():assert z['kl0_stable']==0 and z['probe']['max_abs_logit_diff']==0 and z['probe']['ffn_input_bitwise_equal'];assert z['tokens']==s['y_B'] and z['terminated']==s['y_B_terminated']
        for a in ARMS:
            if new[a]['status']=='ok':
                p=new[a]['probe'];assert p['outside_chord_max']==0 and p['edited_norm_rel_err_max']<=.02 and p['edited_chord_max']>0;assert new[a]['log']['master_delta_l2']<=1.1260757575454359*(1+1e-6);assert new[a]['log']['grad_l2'][0]>0
                if a in ['T2','T6']:assert new[a]['log']['parts'][0]['P']==0
            else:assert new[a]['tokens']==s['y_B']
            if h[a]['tokens']!=h[a+'_R1']['tokens']:repair[a].append(s['utterance_id'])
        integ.append({'id':s['utterance_id'],'zero_masks':list(zero['zero']),'initial_P':{a:new[a]['log']['parts'][0].get('P') for a in ARMS if new[a]['status']=='ok'}})
        ref=refs[s['utterance_id']]['transcript_raw'];scores={a:one(ref,h[a]['text']) for a in systems};r={'id':s['utterance_id'],'dialogue':s['dialogue_id'],'reference':ref,'texts':{a:h[a]['text'] for a in systems},'systems':{}}
        for a in systems:
            b,t=h['B0']['tokens'],h[a]['tokens'];eos=h['B0']['terminated']=='eos' and len(t)>len(b) and t[:len(b)]==b
            e=compare(scores['B0'],scores[a],s['utterance_id'],s['dialogue_id'],eos);ev=e.pop('events')
            for v in ev:v['lexical_kind']=lexical(v)
            EV[a].extend(ev);TR[a].update(e);TR[a].update({'corrections':sum(v['type']=='correction' for v in ev),'corruptions':sum(v['type']=='corruption' for v in ev),'caps':int(h[a]['terminated']=='cap'),'changed':int(t!=b),'eos_recovery':int(eos),'premature_eos':int(h[a]['terminated']=='eos' and len(t)<len(b) and b[:len(t)]==t),'severe_truncation':int(h[a]['terminated']=='eos' and len(b)>=10 and len(t)<=len(b)//2)})
            c=scores[a]['counts'];C[a].append(c);r['systems'][a]={'counts':c,**e,'eos_recovery':eos,'step0_changed':bool(t and b and t[0]!=b[0]),'normalized_text_equal_ACSUB':one(ref,h[a]['text'])['hyp']==scores['ACSUB']['hyp'],'normalized_string_equal_ACSUB':__import__('csasr.data.normalize',fromlist=['normalize_text']).normalize_text(h[a]['text'])==__import__('csasr.data.normalize',fromlist=['normalize_text']).normalize_text(h['ACSUB']['text'])}
        records.append(r)
        if any(new[a]['tokens']!=s['y_B'] for a in ARMS) or old['B2']['tokens']!=s['y_B'] or old['candidates']['t_star'] is not None:selection.add(i)
    matrices={a:np.array(c,dtype=np.int64) for a,c in C.items()};metrics={a:{**ratios(c.sum(0)),**dict(zip(['poi_errors','poi','mixed_errors','ref_units','zh_errors','zh_ref','en_errors','en_ref','sub','del','ins'],map(int,c.sum(0))))} for a,c in matrices.items()}
    lex={a:dict(collections.Counter(e['lexical_kind'] for e in EV[a] if e['type']=='correction')) for a in systems};breadth={a:len({e['dialogue'] for e in EV[a] if e['type']=='correction' and e['lexical_kind'].startswith('genuine')}) for a in systems}
    pub=json.loads((base/'evaluation.json').read_text());mism=[]
    for a in systems:
        for k in ['pier','mer','zh_cer','en_wer','num_poi_errors','substitutions','deletions','insertions']:
            ours=metrics[a][{'num_poi_errors':'poi_errors','substitutions':'sub','deletions':'del','insertions':'ins'}.get(k,k)]
            if abs(pub['metrics'][a][k]-ours)>1e-12:mism.append([a,k,ours,pub['metrics'][a][k]])
        assert metrics['B0']['poi_errors']-metrics[a]['poi_errors']==TR[a]['corrections']-TR[a]['corruptions']
        if a!='B0':
            for x,y in [('corrections','corrections'),('corruptions','corruptions')]:
                if TR[a][x]!=pub['vs_B0'][a]['transitions'][y]:mism.append([a,x,TR[a][x],pub['vs_B0'][a]['transitions'][y]])
    dlgs=sorted({r['dialogue'] for r in records});ix={d:np.array([i for i,r in enumerate(records) if r['dialogue']==d]) for d in dlgs};rng=np.random.default_rng(240924);draws=[np.concatenate([ix[dlgs[k]] for k in rng.integers(20,size=20)]) for _ in range(2000)];boot={}
    pairs=[('T1','B0'),('T1','B2'),('T2','T2A'),('T4','T3'),('T6','T5'),('T1','B1'),('T2','T1'),('T2A','B2'),('T4','ACSUB')]+[(a,a+'_R1') for a in ARMS]
    for a,b in pairs:
        vals={k:[] for k in ('pier','mer','en_wer','zh_cer')}
        for inds in draws:
            ca,cb=ratios(matrices[a][inds].sum(0)),ratios(matrices[b][inds].sum(0))
            for k in vals:vals[k].append(ca[k]-cb[k])
        boot[a+'-'+b]={k:{'point':metrics[a][k]-metrics[b][k],'ci95':np.quantile(v,[.025,.975]).tolist()} for k,v in vals.items()}
    covered={records[i]['dialogue'] for i in selection};control=[i for i,r in enumerate(records) if i not in selection and r['dialogue'] not in covered];selection.update(control[:24-len(selection)]);assert len(selection)==24
    sel={'schema':'ttls_r1r_independent_replay_v1','indices':sorted(selection),'ids':[records[i]['id'] for i in sorted(selection)],'dialogues':len({records[i]['dialogue'] for i in selection}),'rule':'union every changed repaired TTLS row, every changed A2 row, every accepted AC row; add first canonical unchanged controls from unrepresented dialogues until24','outcome_selected_for_audit_only':True,'source_revision':'f1573198866cdb3edd2b3b1acf3a3b08b627be82'}
    out={'manifest_hash':man['manifest_hash'],'manifest_source_commit':man['git_commit'],'panel_sha256':sha(ROOT/'docs/inference_cf/P2_SEL_MINI_PANEL.json'),'config_sha256':sha(ROOT/'configs/inference_cf/ttls_r1r.json'),'reference_file_sha256':sha(refpath),'ids':man['ids'],'model':man['model'],'seal_verified_files':len(seal['files']),'historical_pins_unchanged':True,'metrics':metrics,'transitions':{a:dict(t) for a,t in TR.items()},'lexical':lex,'genuine_dialogues':breadth,'events':EV,'rows':records,'integrity100':integ,'repaired_vs_original_token_changes':repair,'bootstrap':boot,'published_mismatches':mism}
    (OUT/'independent_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(OUT/'replay_selection.json').write_text(json.dumps(sel,indent=2)+'\n');print(json.dumps({'metrics':metrics,'lexical':lex,'genuine_dialogues':breadth,'mismatches':mism,'repair_changes':repair,'replay':sel},indent=2))
if __name__=='__main__':main()

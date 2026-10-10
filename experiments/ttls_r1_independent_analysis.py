#!/usr/bin/env python
"""Independent post-seal TTLS audit. No primary experiment/evaluator aggregation imports.
Uses only the audited canonical normalization/tagging primitives. DP, counts,
transitions, retention, EOS classification and dialogue resampling are independent.
"""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from csasr.data.normalize import normalize_text,segment_units,is_han
from csasr.data.language_tags import tag_unit
SYSTEMS=['B0','B1','B2','T1','T2','T2A','T3','T4','T5','T6','CD','ACSUB']
OUT=ROOT/'results/inference_cf/ttls_r1_independent_audit'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alignment(a,b):
    # Independent Wagner-Fischer with canonical diagonal/deletion/insertion ties.
    d=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
    d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i,j]=min(d[i-1,j-1]+(a[i-1]!=b[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    ops=[];i,j=len(a),len(b)
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(a[i-1]!=b[j-1]):
            ops.append(('match' if a[i-1]==b[j-1] else 'sub',i-1,j-1));i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:
            ops.append(('del',i-1,None));i-=1
        else: ops.append(('ins',None,j-1));j-=1
    return ops[::-1]
def tokenize(s): return [u.surface for u in segment_units(normalize_text(s))]
def one(ref,hyp):
    a,b=tokenize(ref),tokenize(hyp);at,bt=list(map(tag_unit,a)),list(map(tag_unit,b))
    ops=alignment(a,b);by={i:(o,j) for o,i,j in ops if i is not None}; c=collections.Counter(o for o,i,j in ops)
    en=sum(at[i]=='EN' for o,i,j in ops if o in ('sub','del'))+sum(bt[j]=='EN' for o,i,j in ops if o=='ins')
    zh=sum(at[i]=='ZH' for o,i,j in ops if o in ('sub','del'))+sum(bt[j]=='ZH' for o,i,j in ops if o=='ins')
    counts=[sum(at[i]=='EN' and by[i][0]!='match' for i in range(len(a))),at.count('EN'),c['sub']+c['del']+c['ins'],len(a),zh,at.count('ZH'),en,at.count('EN'),c['sub'],c['del'],c['ins']]
    return {'counts':counts,'ref':a,'hyp':b,'tags':at,'ops':ops,'by':by}
def category(o,i):
    op,j=o['by'][i]; a,b=o['ref'],o['hyp']
    if op=='match':return 'correct',b[j]
    near=[k for p,q,k in o['ops'] if p=='ins' and any(q2 in (i-1,i,i+1) and k2 in (k-1,k+1) for p2,q2,k2 in o['ops'])]
    if op=='del':
        loc=min(i,len(b)-1)
        if a[i] in b[max(0,loc-3):min(len(b),loc+4)]:return 'boundary_error',''
        return ('insertion_near_poi',' '.join(b[k] for k in near)) if near else ('deletion','')
    if any(map(is_han,b[j])):
        extra=[k for p,q,k in o['ops'] if p=='ins' and abs(k-j)==1 and any(map(is_han,b[k]))]
        return ('phonetic_transliteration_or_script',b[j]+''.join(b[k] for k in extra)) if extra else ('wrong_language_substitution',b[j])
    return ('same_language_substitution' if tag_unit(b[j])=='EN' else 'other'),b[j]
def ratios(c):return {n:float(c[i]/c[j]) if c[j] else None for n,i,j in [('pier',0,1),('mer',2,3),('zh_cer',4,5),('en_wer',6,7)]}
def compare(b,m,uid,dlg,eos):
    ev=[];znew=zfix=outdamage=outcorrect=encorrect=zhcorrect=0
    cs='EN' in b['tags'] and 'ZH' in b['tags'];mzden=mzloss=0
    for i,(tok,tag) in enumerate(zip(b['ref'],b['tags'])):
        ok=b['by'][i][0]=='match';mk=m['by'][i][0]=='match'
        if tag=='ZH':
            zhcorrect+=ok;znew+=ok and not mk;zfix+=not ok and mk
            if cs:mzden+=ok;mzloss+=ok and not mk
        if tag!='EN':outcorrect+=ok;outdamage+=ok and not mk
        else:
            encorrect+=ok
            if ok!=mk:
                cat,surf=category(b if mk else m,i)
                kind='lexical'
                if cat in ('deletion','boundary_error','insertion_near_poi'):kind='deletion_or_boundary'
                elif cat=='same_language_substitution' and (tok in surf and tok!=surf):kind='word_boundary_repair'
                ev.append(dict(id=uid,dialogue=dlg,index=i,reference=tok,before=category(b,i)[1],after=category(m,i)[1],type='correction' if mk else 'corruption',canonical_category=cat,audit_kind=kind,eos_recovery=eos))
    return dict(events=ev,new_zh=znew,zh_repairs=zfix,correct_zh=zhcorrect,matrix_den=mzden,matrix_loss=mzloss,outside_den=outcorrect,outside_damage=outdamage,correct_en=encorrect)
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    base=ROOT/'results/inference_cf/ttls_r1';p=json.loads((base/'plan_sealed.json').read_text());man=json.loads((base/'run1/manifest.json').read_text());seal=json.loads((base/'output_seal.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in seal['files'].items())
    assert all(sha(ROOT/k)==v.removeprefix('sha256:') for k,v in man['sources'].items())
    import pyarrow.parquet as pq
    path=Path('/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet')
    refs={r['utterance_id']:r for r in pq.read_table(path,columns=['utterance_id','dialogue_id','role','transcript_raw'],filters=[('utterance_id','in',man['ids'])]).to_pylist()}
    assert set(refs)==set(man['ids']) and all(r['role']=='D-dev-select' for r in refs.values())
    records=[];C={s:[] for s in SYSTEMS};EV={s:[] for s in SYSTEMS};TR={s:collections.Counter() for s in SYSTEMS};dlg=[]
    original=[]
    for i,rp in enumerate(p['rows']):
        u=rp['utterance_id'];r=json.loads((base/f'run1/rows/{i:03d}.json').read_text());original.append(r)
        assert r['identity']==u and r['manifest_hash']==man['manifest_hash'] and r['status']=='ok';assert refs[u]['dialogue_id']==rp['dialogue_id']
        dlg.append(rp['dialogue_id']);ref=refs[u]['transcript_raw']
        h={'B0':rp['y_B_text'],'B1':rp['y_A_text'],'B2':r['B2']['text'],**{s:r[s]['text'] for s in SYSTEMS[3:]}}
        tok={'B0':rp['y_B'],'B1':rp['y_A'],**{s:r[s]['tokens'] for s in SYSTEMS[2:]}}
        term={'B0':rp['y_B_terminated'],'B1':rp['y_A_terminated'],**{s:r[s]['terminated'] for s in SYSTEMS[2:]}}
        scores={s:one(ref,h[s]) for s in SYSTEMS};entry={'id':u,'dialogue':dlg[-1],'reference':ref,'texts':h,'systems':{}}
        for s in SYSTEMS:
            C[s].append(scores[s]['counts']);a,b=tok['B0'],tok[s]
            er=term['B0']=='eos' and len(b)>len(a) and b[:len(a)]==a
            premature=term[s]=='eos' and len(b)<len(a) and a[:len(b)]==b
            severe=term[s]=='eos' and len(a)>=10 and len(b)<=len(a)//2
            z=compare(scores['B0'],scores[s],u,dlg[-1],er)
            EV[s].extend(z.pop('events'));TR[s].update(z)
            TR[s].update(dict(changed=int(a!=b),eos_recovery=int(er),premature_eos=int(premature),severe_truncation=int(severe),caps=int(term[s]=='cap')))
            entry['systems'][s]={'counts':scores[s]['counts'],**z,'eos_recovery':er,'premature_eos':premature}
        records.append(entry)
    matrices={s:np.asarray(v,dtype=np.int64) for s,v in C.items()};totals={s:v.sum(0) for s,v in matrices.items()}
    metrics={s:{**ratios(t),**dict(zip(['poi_errors','poi','mixed_errors','ref_units','zh_errors','zh_ref','en_errors','en_ref','sub','del','ins'],map(int,t)))} for s,t in totals.items()}
    for s,ev in EV.items():
        TR[s].update(dict(corrections=sum(e['type']=='correction' for e in ev),corruptions=sum(e['type']=='corruption' for e in ev)))
        assert totals['B0'][0]-totals[s][0]==TR[s]['corrections']-TR[s]['corruptions']
    groups=sorted(set(dlg));ix={d:np.where(np.asarray(dlg)==d)[0] for d in groups};rng=np.random.default_rng(240924)
    sampled=[np.concatenate([ix[groups[k]] for k in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    pairs=[('T1','B2'),('T2','T2A'),('T4','T3'),('T6','T5'),('T1','B1'),('T2A','B2'),('T2','T1'),('T6','T4'),('T5','T3'),('T4','ACSUB'),('T1','B0'),('T2A','B1')]
    boot={}
    for a,b in pairs:
        samples={k:[] for k in ('pier','mer','zh_cer','en_wer')}
        for inds in sampled:
            ra,rb=ratios(matrices[a][inds].sum(0)),ratios(matrices[b][inds].sum(0))
            for k in samples:
                if ra[k] is not None and rb[k] is not None:samples[k].append(ra[k]-rb[k])
        boot[a+'-'+b]={k:{'point':metrics[a][k]-metrics[b][k],'ci95':np.quantile(v,[.025,.975]).tolist(),'valid_draws':len(v)} for k,v in samples.items()}
    pub=json.loads((base/'evaluation.json').read_text()); mism=[]
    names={'num_poi_errors':'poi_errors','substitutions':'sub','deletions':'del','insertions':'ins'}
    for s in SYSTEMS:
        for k in ['pier','mer','en_wer','zh_cer',*names]:
            if abs(pub['metrics'][s][k]-metrics[s][names.get(k,k)])>1e-12:mism.append([s,k,pub['metrics'][s][k],metrics[s][names.get(k,k)]])
        if s!='B0':
            for a,b in [('corrections','corrections'),('corruptions','corruptions')]:
                if TR[s][a]!=pub['vs_B0'][s]['transitions'][b]:mism.append([s,a,TR[s][a],pub['vs_B0'][s]['transitions'][b]])
            for a,b in [('new_zh','new_zh_errors'),('zh_repairs','zh_repairs')]:
                if TR[s][a]!=pub['vs_B0'][s]['zh'][b]:mism.append([s,a,TR[s][a],pub['vs_B0'][s]['zh'][b]])
    # Targeted audit replay: union of all changed T1, B2 and accepted AC, plus controls.
    chosen={i for i,r in enumerate(original) if not r['T1']['equal_B0'] or not r['B2']['equal_B0'] or r['candidates']['t_star'] is not None}
    covered={dlg[i] for i in chosen}
    control=[i for i in range(100) if i not in chosen and dlg[i] not in covered]
    chosen.update(control[:3]);assert len(chosen)==24
    replay={'schema':'ttls_r1_independent_replay_selection_v1','completed_commit':'800fee26a55651c307a1a33b066fdb47ece22b05','rule':'all changed T1/B2 plus all accepted AC; then first three canonical unchanged rows from not-yet-covered dialogues','indices':sorted(chosen),'ids':[man['ids'][i] for i in sorted(chosen)],'dialogues':len({dlg[i] for i in chosen}),'selected_using_outcomes':True,'purpose':'audit coverage, never an effectiveness estimate','primary_manifest_hash':man['manifest_hash']}
    result={'schema':'ttls_r1_independent_analysis_v1','source_commit':replay['completed_commit'],'primary_sources_verified':True,'seal_files_verified':len(seal['files']),'reference_source':str(path),'reference_source_sha256':sha(path),'references100_hash':hashlib.sha256(json.dumps({u:refs[u]['transcript_raw'] for u in sorted(refs)},ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'metrics':metrics,'transitions':{s:dict(v) for s,v in TR.items()},'events':EV,'bootstrap':boot,'rows':records,'published_mismatches':mism,'replay_selection':replay}
    (OUT/'independent_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'replay_selection.json').write_text(json.dumps(replay,indent=2)+'\n')
    print(json.dumps({'metrics':metrics,'mismatches':mism,'replay':replay},indent=2))
if __name__=='__main__':main()

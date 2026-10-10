"""Post-replay audit crosschecks; primary JSON is compared, never used to calculate counts."""
from pathlib import Path
import json, hashlib, sys, collections
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1_independent_analysis import one,compare
from csasr.inference_cf.exposure_registry import exposed_ids
OUT=ROOT/'results/inference_cf/a2p_dev200_independent_audit'
def projection(o):
    p={i:[None,[]] for i in range(-1,len(o['ref']))};last=-1
    for op,i,j in o['ops']:
        if op=='ins':p[last][1].append(o['hyp'][j])
        else:p[i][0]=o['hyp'][j] if j is not None else None;last=i
    return p
def main():
    a=json.loads((OUT/'independent_metrics.json').read_text());pub=json.loads((ROOT/'results/inference_cf/a2p_dev200/evaluation.json').read_text())
    assert not a['published_mismatches']
    ids={r['id'] for r in a['FULL300']['rows']};registry=exposed_ids(ROOT);assert ids<=registry['ids']
    checks=[]
    for pop,key in [('NEW200','NEW200'),('FIXED100','FIXED100_secondary')]:
        v=a[pop]
        for arm in ['B1','B2','B3']:
            t=v['transitions'][arm];p=pub[key]['vs_B0'][arm]
            for ours,section,theirs in [('corrections','transitions','corrections'),('corruptions','transitions','corruptions'),('new_zh','zh','new_zh_errors'),('zh_repairs','zh','zh_repairs'),('changed','events','changed_rows'),('eos_recovery','events','eos_recovery_rows'),('premature_eos','events','premature_eos_rows'),('severe_truncation','events','new_severe_truncations')]:
                assert t[ours]==p[section][theirs],(pop,arm,ours)
            os=collections.Counter()
            for r in v['rows']:
                b,m=one(r['reference'],r['texts']['B0']),one(r['reference'],r['texts'][arm]);pb,pm=projection(b),projection(m)
                outside=[i for i,tag in enumerate(b['tags']) if tag!='EN']
                os.update(outside_units=len(outside),outside_edits=sum(pb[i]!=pm[i] for i in outside),leading_insertion_change=int(pb[-1]!=pm[-1]),baseline_correct_outside=sum(b['by'][i][0]=='match' for i in outside),outside_harm=sum(b['by'][i][0]=='match' and m['by'][i][0]!='match' for i in outside))
            assert dict(os)==p['outside'],(pop,arm,dict(os),p['outside'])
            lex=v['lexical'][arm];pl=pub[key]['lexical'][arm]['by_kind']
            assert all(lex.get(k,0)==pl.get(k,0) for k in ['genuine_wrong_language','genuine_same_language','deletion_or_boundary','word_boundary_repair'])
            checks.append([pop,arm,'metrics/transitions/outside/lexical agree'])
    # Compare precise harm identities, not only equal totals.
    n=a['NEW200'];harm={arm:set() for arm in ['B2','B3']}
    for r in n['rows']:
        b=one(r['reference'],r['texts']['B0'])
        for arm in harm:
            m=one(r['reference'],r['texts'][arm])
            harm[arm]|={(r['id'],i) for i,tag in enumerate(b['tags']) if tag=='ZH' and b['by'][i][0]=='match' and m['by'][i][0]!='match'}
    assert harm['B2']==harm['B3'] and len(harm['B2'])==9
    seal=json.loads((OUT/'replay_output_seal.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in seal['files'].items())
    replay=json.loads((OUT/'replay/summary.json').read_text());rc=[]
    for rr in replay['rows']:
        r=json.loads((OUT/f"replay/rows/{rr['index']:03d}.json").read_text());old=n['rows'][r['index']]
        for arm in ['B2','B3']:
            scores=one(old['reference'],r[arm]['text']);assert scores['counts']==old['systems'][arm]['counts']
            e=compare(one(old['reference'],old['texts']['B0']),scores,old['id'],old['dialogue'],old['systems'][arm]['eos_recovery'])
            assert e['new_zh']==old['systems'][arm]['new_zh'] and e['zh_repairs']==old['systems'][arm]['zh_repairs']
        rc.append({'id':r['id'],'counts_and_harm_exact':True})
    D=n['transitions']['B2']['new_zh']-n['transitions']['B3']['new_zh'];assert D==0
    branch_changes={}
    for pop in ['NEW200','FIXED100']:
        changes=[]
        for r in a[pop]['rows']:
            def branch(arm):
                b,m=r['tokens']['B0'],r['tokens'][arm]
                if b==m and r['termination']['B0']==r['termination'][arm]:return None
                i=next((j for j in range(min(len(b),len(m))) if b[j]!=m[j]),min(len(b),len(m)))
                return [i,b[i] if i<len(b) else r['termination']['B0'],m[i] if i<len(m) else r['termination'][arm]]
            b2,b3=branch('B2'),branch('B3')
            if b2!=b3:changes.append({'id':r['id'],'B2_first_divergence':b2,'B3_first_divergence':b3})
        branch_changes[pop]=changes
    result={'schema':'a2p_dev200_independent_review_v1','technical_verdict':'A2P_DEV200_AUDIT_PASS','scientific_verdict':'A2P_DEV200_NOT_SUPPORTED','checks':checks,'published_count_mismatches':[],'exact_NEW200_harm_identity':sorted(harm['B2']),'independent_frozen_decision_trigger':{'D':D,'NOT_SUPPORTED_D_le_0':D<=0},'replay_rescoring':rc,'replay_seal':seal['seal_hash'],'exposure_registry':{k:v for k,v in registry.items() if k!='ids'},'FULL300_without_CSD0012':a['FULL300']['lodo']['CSD0012'],'NEW200_EOS_contribution':n['termination'],'first_branch_changes':branch_changes}
    (OUT/'review_crosschecks.json').write_text(json.dumps(result,indent=2)+'\n');print('Independent aggregate crosschecks and replay rescoring: PASS')
if __name__=='__main__':main()

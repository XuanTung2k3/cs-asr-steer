#!/usr/bin/env python
"""Independent DP aggregation and preservation/breadth audit; no primary evaluator imports."""
from pathlib import Path
import sys,json,hashlib,collections,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1_independent_analysis import one,compare,ratios
from experiments.ttls_r1r_independent_analysis import lexical
OUT=ROOT/'results/inference_cf/a2p_dev200_independent_audit';BASE=ROOT/'results/inference_cf/a2p_dev200';ARMS=('B0','B1','B2','B3')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o):return 'sha256:'+hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def verify_json(p,key):
 o=json.loads(p.read_text());assert digest({k:v for k,v in o.items() if k!=key})==o[key];return o
def groups(events):
 out=[]
 for (uid,typ),es in __import__('itertools').groupby(sorted(events,key=lambda e:(e['id'],e['type'],e['index'])),key=lambda e:(e['id'],e['type'])):
  run=[]
  for e in es:
   if run and e['index']!=run[-1]['index']+1:out.append(run);run=[]
   run.append(e)
  if run:out.append(run)
 return [{'id':r[0]['id'],'dialogue':r[0]['dialogue'],'type':r[0]['type'],'units':len(r),'genuine_units':sum(e['lexical_kind'].startswith('genuine') for e in r),'words':' '.join(e['reference'] for e in r),'before':' '.join(e['before'] for e in r),'after':' '.join(e['after'] for e in r)} for r in out]
def shares(values):
 positive=sorted([(k,v) for k,v in values.items() if v>0],key=lambda kv:(-kv[1],kv[0]));tot=sum(v for k,v in positive)
 return {'positive_total':tot,'net':sum(values.values()),'largest':positive[0] if positive else None,'largest_share':positive[0][1]/tot if tot else None,'top3':positive[:3],'top3_share':sum(v for k,v in positive[:3])/tot if tot else None}
def analyse(items,refs):
 rows=[];events={a:[] for a in (*ARMS,'B3_vs_B2')};tr={a:collections.Counter() for a in events}
 for index,(p,h) in enumerate(items):
  uid=p['utterance_id'];dlg=p['dialogue_id'];r=refs[uid];assert r['role']=='D-dev-select' and r['dialogue_id']==dlg
  scores={a:one(r['transcript_raw'],h[a]['text']) for a in ARMS}
  row={'index':index,'id':uid,'dialogue':dlg,'reference':r['transcript_raw'],'texts':{a:h[a]['text'] for a in ARMS},'tokens':{a:h[a]['tokens'] for a in ARMS},'termination':{a:h[a]['terminated'] for a in ARMS},'systems':{}}
  for a,comparator in [(a,'B0') for a in ARMS]+[('B3','B2')]:
   key=a if comparator=='B0' else 'B3_vs_B2';bt,mt=h[comparator]['tokens'],h[a]['tokens']
   eos=h[comparator]['terminated']=='eos' and len(mt)>len(bt) and mt[:len(bt)]==bt
   e=compare(scores[comparator],scores[a],uid,dlg,eos);ev=e.pop('events')
   for v in ev:v['lexical_kind']=lexical(v)
   events[key].extend(ev);tr[key].update(e);tr[key].update({'corrections':sum(x['type']=='correction' for x in ev),'corruptions':sum(x['type']=='corruption' for x in ev),'caps':int(h[a]['terminated']=='cap'),'changed':int(mt!=bt),'eos_recovery':int(eos),'premature_eos':int(h[a]['terminated']=='eos' and len(mt)<len(bt) and bt[:len(mt)]==mt),'severe_truncation':int(h[a]['terminated']=='eos' and len(bt)>=10 and len(mt)<=len(bt)//2)})
   row['systems'][key]={'counts':scores[a]['counts'],**e,'eos_recovery':eos,'changed':mt!=bt,'step0_changed':bool(mt and bt and mt[0]!=bt[0])}
  rows.append(row)
 matrices={a:np.array([r['systems'][a]['counts'] for r in rows],dtype=np.int64) for a in ARMS}
 metrics={a:{**ratios(c.sum(0)),**dict(zip(['poi_errors','poi','mixed_errors','ref_units','zh_errors','zh_ref','en_errors','en_ref','sub','del','ins'],map(int,c.sum(0))))} for a,c in matrices.items()}
 for a in ARMS:assert metrics['B0']['poi_errors']-metrics[a]['poi_errors']==tr[a]['corrections']-tr[a]['corruptions']
 assert metrics['B2']['poi_errors']-metrics['B3']['poi_errors']==tr['B3_vs_B2']['corrections']-tr['B3_vs_B2']['corruptions']
 lexical_counts={a:dict(collections.Counter(e['lexical_kind'] for e in ev if e['type']=='correction')) for a,ev in events.items()};evgroups={a:groups(ev) for a,ev in events.items()}
 dlgs=sorted({r['dialogue'] for r in rows});didx={d:[i for i,r in enumerate(rows) if r['dialogue']==d] for d in dlgs}
 positive={r['id']:r['systems']['B2']['new_zh']-r['systems']['B3']['new_zh'] for r in rows};mixed={r['id']:r['systems']['B0']['counts'][2]-r['systems']['B3']['counts'][2] for r in rows};zh={r['id']:r['systems']['B0']['counts'][4]-r['systems']['B3']['counts'][4] for r in rows}
 dlg={d:{a:{'counts':matrices[a][idx].sum(0).tolist(),'new_zh':sum(rows[i]['systems'][a]['new_zh'] for i in idx)} for a in ARMS} for d,idx in didx.items()}
 sums=lambda v:{d:sum(v[rows[i]['id']] for i in idx) for d,idx in didx.items()}
 wlt=lambda vals:{'improved':sum(v<0 for v in vals),'worsened':sum(v>0 for v in vals),'tied':sum(v==0 for v in vals)}
 pairs=[('B3','B2'),('B3','B0'),('B3','B1'),('B2','B0'),('B2','B1'),('B1','B0')]
 breadth={a+'-'+b:{'utterances':wlt(matrices[a][:,2]-matrices[b][:,2]),'dialogues':wlt([dlg[d][a]['counts'][2]-dlg[d][b]['counts'][2] for d in dlgs])} for a,b in pairs}
 lodo={};rng=np.random.default_rng(240924);boot={a+'-'+b:{k:[] for k in ('pier','mer','en_wer','zh_cer')} for a,b in pairs};nb={'B2':[],'B3':[],'D':[]}
 for d in dlgs:
  keep=[i for i,r in enumerate(rows) if r['dialogue']!=d];means={a:ratios(c[keep].sum(0)) for a,c in matrices.items()}
  lodo[d]={'D':sum(positive[rows[i]['id']] for i in keep),'pairs':{a+'-'+b:{k:means[a][k]-means[b][k] for k in means[a]} for a,b in pairs},'net_zh_B0_minus_B3':sum(zh[rows[i]['id']] for i in keep),'net_mixed_B0_minus_B3':sum(mixed[rows[i]['id']] for i in keep)}
 for _ in range(2000):
  indices=[i for k in rng.integers(len(dlgs),size=len(dlgs)) for i in didx[dlgs[k]]];means={a:ratios(c[indices].sum(0)) for a,c in matrices.items()}
  for a,b in pairs:
   for k in means[a]:boot[a+'-'+b][k].append(means[a][k]-means[b][k])
  for a in ('B2','B3'):nb[a].append(sum(rows[i]['systems'][a]['new_zh'] for i in indices))
  nb['D'].append(nb['B2'][-1]-nb['B3'][-1])
 bootstrap={key:{k:{'point':metrics[key.split('-')[0]][k]-metrics[key.split('-')[1]][k],'ci95':np.quantile(vals,[.025,.975]).tolist()} for k,vals in v.items()} for key,v in boot.items()};bootstrap['new_zh']={k:{'point':sum(positive.values()) if k=='D' else tr[k]['new_zh'],'ci95':np.quantile(v,[.025,.975]).tolist()} for k,v in nb.items()}
 key=lambda e:(e['id'],e['index']);c2={key(e) for e in events['B2'] if e['type']=='correction'};c3={key(e) for e in events['B3'] if e['type']=='correction'};g2={key(e) for e in events['B2'] if e['type']=='correction' and e['lexical_kind'].startswith('genuine')}
 termination={a:{'eos_rows':[r['id'] for r in rows if r['systems'][a]['eos_recovery']],'eos_delta_counts_B0_minus_system':np.sum([np.array(r['systems']['B0']['counts'])-r['systems'][a]['counts'] for r in rows if r['systems'][a]['eos_recovery']],axis=0).tolist()} for a in ('B2','B3')}
 return {'n':len(rows),'dialogues':len(dlgs),'metrics':metrics,'transitions':{a:dict(v) for a,v in tr.items()},'lexical':lexical_counts,'events':events,'phrase_events':evgroups,'rows':rows,'dialogue_counts':dlg,'breadth':breadth,'concentration':{'ZH_protection_utt':shares(positive),'ZH_protection_dlg':shares(sums(positive)),'mixed_gain_utt':shares(mixed),'mixed_gain_dlg':shares(sums(mixed)),'zh_gain_utt':shares(zh),'zh_gain_dlg':shares(sums(zh))},'per_utterance_protection':positive,'lodo':lodo,'bootstrap':bootstrap,'termination':termination,'retention':{'corrections':len(c2),'retained':len(c2&c3),'genuine':len(g2),'genuine_retained':len(g2&c3),'extra':len(c3-c2)}}
def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'configs/inference_cf/a2p_dev200.json').read_text());p=verify_json(BASE/'plan_sealed.json','plan_hash');m=verify_json(BASE/'run1/manifest.json','manifest_hash');s=verify_json(BASE/'output_seal.json','seal_hash')
 assert p['plan_hash']==m['plan_hash']==s['plan_hash'];assert sha(ROOT/'configs/inference_cf/a2p_dev200.json')==m['config_sha256']==s['config_sha256']
 for f,h in {**m['sources'],**s['files']}.items():assert sha(ROOT/f)==h.removeprefix('sha256:')
 for path in ['results/inference_cf/p2path5/output_seal.json','results/inference_cf/ttls_r1/output_seal.json']:
  seal=verify_json(ROOT/path,'seal_hash')
  assert all(sha(ROOT/f)==h.removeprefix('sha256:') for f,h in seal['files'].items())
 panel=json.loads((ROOT/cfg['population']['panel']).read_text());assert sha(ROOT/cfg['population']['panel'])==cfg['population']['panel_sha256']
 dlg={r['utterance_id']:r['dialogue_id'] for r in panel['rows']};parts={k:panel['partitions'][k]['ids'] for k in ('NEW200','FIXED100','FULL300')}
 assert parts['NEW200']==[u for u in parts['FULL300'] if u not in set(parts['FIXED100'])]==p['ids']==m['ids'];assert len(set(parts['NEW200']))==200 and len(set(dlg[u] for u in parts['NEW200']))==20
 for name,ids in parts.items():assert digest({'partition':name,'rows':[{'utterance_id':u,'dialogue_id':dlg[u]} for u in ids]}).removeprefix('sha256:')==cfg['population']['membership_sha256'][name]
 import pyarrow.parquet as pq
 refpath=Path('/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet');refs={r['utterance_id']:r for r in pq.read_table(refpath,columns=['utterance_id','dialogue_id','role','transcript_raw'],filters=[('utterance_id','in',parts['FULL300'])]).to_pylist()};assert set(refs)==set(parts['FULL300'])
 items=[];raw=[]
 for i,pr in enumerate(p['rows']):
  r=json.loads((BASE/f'run1/rows/{i:03d}.json').read_text());assert r['identity']==pr['utterance_id'] and r['status']=='ok' and all(r['gates'].values()) and r['manifest_hash']==m['manifest_hash'];assert sha(pr['audio_path'])==pr['audio_full_sha256'];assert all(r['B1'][k] for k in ('replay_tokens_equal','replay_terminated_equal','replay_text_equal'))
  c=r['candidates'];assert c['S']==[t for t,(prob,valid) in enumerate(zip(c['p0_top'],pr['y_B_valid_mask'])) if valid and prob>=.5 and t not in c['M']];assert r['B3']['log']['parts'][0]['P']==0 and r['B3']['log']['parts'][0]['CE']==pr['A2']['loss0']
  original=json.loads((ROOT/pr['source_row']).read_text());assert sha(ROOT/pr['source_row'])==pr['source_row_sha256'];assert pr['A2']['tokens']==original['A2_free']['tokens'] and pr['A2']['state_hash']==original['A2_state_hash'];raw.append(r)
  items.append((pr,{'B0':{'tokens':pr['y_B'],'text':pr['y_B_text'],'terminated':pr['y_B_terminated']},'B1':pr['AUTO'],'B2':pr['A2'],'B3':r['B3']}))
 rt=json.loads((BASE/'run1/runtime.json').read_text());assert rt['status']=='completed' and rt['rows_done']==200 and all(rt[k] for k in ('reset_final_ok','nonln_unchanged','model_grads_none'));assert rt['nonln_hash_start']==rt['nonln_hash_end']
 fp=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());fixed=[]
 for i,pr in enumerate(fp['rows']):
  r=json.loads((ROOT/f'results/inference_cf/ttls_r1/run1/rows/{i:03d}.json').read_text());fixed.append((pr,{'B0':{'tokens':pr['y_B'],'text':pr['y_B_text'],'terminated':pr['y_B_terminated']},'B1':{'tokens':pr['y_A'],'text':pr['y_A_text'],'terminated':pr['y_A_terminated']},'B2':r['B2'],'B3':r['T2A']}))
 result={'schema':'a2p_dev200_independent_analysis_v1','experiment_revision':'4bbc0f83bd8f440a196c62a64a65025b9fc07420','manifest_hash':m['manifest_hash'],'seal_hash':s['seal_hash'],'plan_hash':p['plan_hash'],'panel_sha256':sha(ROOT/cfg['population']['panel']),'memberships':cfg['population']['membership_sha256'],'model':m['model'],'reference_file_sha256':sha(refpath),'NEW200':analyse(items,refs),'FIXED100':analyse(fixed,refs),'FULL300':analyse(fixed+items,refs),'published_mismatches':[]}
 published=json.loads((BASE/'evaluation.json').read_text())
 for pop,pubkey in [('NEW200','NEW200'),('FIXED100','FIXED100_secondary')]:
  for arm,metrics in result[pop]['metrics'].items():
   for k in ('pier','mer','en_wer','zh_cer','poi_errors','sub','del','ins'):
    pk={'poi_errors':'num_poi_errors','sub':'substitutions','del':'deletions','ins':'insertions'}.get(k,k);v=published[pubkey]['metrics'][arm][pk]
    if abs(metrics[k]-v)>1e-12:result['published_mismatches'].append([pop,arm,k,metrics[k],v])
 (OUT/'independent_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
 a=result['NEW200'];changed=[i for i,r in enumerate(a['rows']) if r['systems']['B3_vs_B2']['changed']];sel=set(changed)
 for predicate in [lambda r:any(e['type']=='correction' and e['lexical_kind'].startswith('genuine') and e['id']==r['id'] for e in a['events']['B3']),lambda r:r['systems']['B3']['new_zh']>0,lambda r:r['systems']['B3']['eos_recovery']]:sel.update(i for i,r in enumerate(a['rows']) if predicate(r))
 # Audit-only outcome-stratified core, then one untouched canonical control per unrepresented dialogue, then order.
 covered={a['rows'][i]['dialogue'] for i in sel}
 for i,r in enumerate(a['rows']):
  if len(sel)>=24:break
  if i not in sel and r['dialogue'] not in covered and not r['systems']['B3']['changed']:sel.add(i);covered.add(r['dialogue'])
 for i,r in enumerate(a['rows']):
  if len(sel)>=24:break
  if i not in sel and not r['systems']['B3']['changed']:sel.add(i)
 assert len(sel)==24
 selection={'indices':sorted(sel),'ids':[a['rows'][i]['id'] for i in sorted(sel)],'dialogues':len({a['rows'][i]['dialogue'] for i in sel}),'rule':'union all B3-vs-A2 changes, all genuine-recovery utterances, all B3 Mandarin-harm rows, all EOS recoveries; then first canonical unchanged control per unrepresented dialogue until24, then canonical unchanged controls','audit_only_outcome_stratified':True,'categories_absent':{'entirely_correct_baseline':not any(r['systems']['B0']['counts'][2]==0 for r in a['rows'])}}
 (OUT/'replay_selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps({'metrics':a['metrics'],'transitions':a['transitions'],'lexical':a['lexical'],'retention':a['retention'],'concentration':a['concentration'],'mismatches':result['published_mismatches'],'selection':selection},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
if __name__=='__main__':main()

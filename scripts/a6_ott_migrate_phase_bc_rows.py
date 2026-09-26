#!/usr/bin/env python3
"""Re-home early Phase-B/C rows after decode-mode namespace correction."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO),str(REPO/'src')]
from csasr.experiments.a6_ott_phase_bc import row_path, canonical_key
ROOT=REPO/'results/a6_ott_upper_bound'
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False,default=str)+'\n'); t.replace(p)
def migrate(phase):
 base=ROOT/('confirm' if phase=='B' else 'transfer')/'rows'; moved=0
 for path in sorted(base.glob('**/*.json')):
  if path.name.startswith('.') or '/greedy/' in str(path) or '/official_standard/' in str(path): continue
  try: x=json.loads(path.read_text())
  except Exception: continue
  if x.get('phase')!=phase or x.get('regime')!='oracle_tt': continue
  model=x.get('model'); mode=x.get('decode_mode');
  if mode not in ('greedy','official_standard'): continue
  key=canonical_key(phase=phase,model=model,dataset=x['dataset'],uid=x['utterance_id'],family=x['method_family'],side=x['side'],layer=x['layer'],rho=x['rho'],decode_mode=mode)
  x['canonical_key']=key; atomic(row_path(phase,key),x)
  if model=='qwen3_asr_1p7b':
   other='official_standard' if mode=='greedy' else 'greedy'; ok=canonical_key(phase=phase,model=model,dataset=x['dataset'],uid=x['utterance_id'],family=x['method_family'],side=x['side'],layer=x['layer'],rho=x['rho'],decode_mode=other)
   y=dict(x); y['decode_mode']=other; y['canonical_key']=ok; y['provenance']=dict(x.get('provenance',{}),decode_equivalence='official_standard == greedy',reconstructed_from=key,equivalence_hash='sha256:'+__import__('hashlib').sha256(json.dumps({'baseline':x.get('baseline_hypothesis'),'steered':x.get('steered_hypothesis'),'direction_hash':x.get('direction_hash')},sort_keys=True,separators=(',',':')).encode()).hexdigest()); atomic(row_path(phase,ok),y)
  legacy=path.parent/'legacy_flat'; legacy.mkdir(exist_ok=True); shutil.move(str(path),str(legacy/path.name)); moved+=1
 print(phase,'migrated',moved)
if __name__=='__main__':
 migrate('B'); migrate('C')

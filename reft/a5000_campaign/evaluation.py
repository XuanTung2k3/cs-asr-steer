"""Shared canonical MER, explicit unavailable word/POI metrics, frozen GO screen."""
import math
import numpy as np
from reft.a5000.runner import score
from reft.text import ratios


def metrics(rows,predictions,baseline=None):
    if set(predictions)!={r['utterance_id'] for r in rows}: raise ValueError('Missing/extra predictions')
    m=score(rows,predictions,baseline)
    # The inherited script proxy must NEVER be relabeled verified linguistic PIER.
    m.pop('pier',None); m.pop('pier_definition',None)
    m['pier']=None; m['pier_unavailable']='No verified POI annotation/evaluator for new roles'
    m['overall_wer']=None; m['wer_cs']=None; m['wer_noncs']=None
    m['word_segmentation_unavailable']='No frozen Chinese word tokenizer; MER is Han-character/Latin-word mixed units'
    for stratum in ['EN_ONLY','ZH_ONLY']:
        rr=[r['counts'] for r in m['per_utterance'] if r['stratum']==stratum]
        m[stratum.lower()]=ratios(np.sum(rr,axis=0)) if rr else None
    noncs=[r['counts'] for r in m['per_utterance'] if r['stratum']!='CS']
    m['noncs_mer']=ratios(np.sum(noncs,axis=0))['mer'] if noncs else None
    m['audio_seconds']=sum(r['duration_sec'] for r in rows)
    m['rtf']=m['seconds']/m['audio_seconds']
    return m


def pilot_decision(panel,rule):
    need=['G1','G3','G4','G5','G8']
    if any(k not in panel for k in need): return {'decision':'N2_INCONCLUSIVE','reason':'Incomplete controls'}
    g=panel['G5']; cs=[r for r in g['per_utterance'] if r['stratum']=='CS']
    noncs=[r for r in g['per_utterance'] if r['stratum']!='CS']
    if len(cs)<rule['min_cs_utterances'] or len(noncs)<rule['min_noncs_utterances']:
        return {'decision':'N2_INCONCLUSIVE','reason':'Insufficient group support'}
    if g.get('allocation_std',0)<rule['min_allocation_std']:
        return {'decision':'N2_NO_GO','reason':'Constant or collapsed allocation'}
    controls=['G1','G3','G4','G8']; gains={k:panel[k]['genuine_cs_mer']-g['genuine_cs_mer'] for k in controls}
    passed=all(v>=rule['cs_min_gain'] for v in gains.values())
    passed &= all(g['loops']<=panel[k]['loops'] and g['nontermination']<=panel[k]['nontermination'] for k in controls)
    if any(panel[k].get('noncs_mer') is None for k in controls) or g.get('noncs_mer') is None:
        return {'decision':'N2_INCONCLUSIVE','reason':'Non-CS support unavailable'}
    passed &= all(g['noncs_mer']-panel[k]['noncs_mer']<=rule['noncs_max_degradation'] for k in controls)
    # Positive gain must span dialogues, relative to strongest control; do not
    # use a positive total to hide single-dialogue concentration.
    k=min(controls,key=lambda k:panel[k]['genuine_cs_mer']); old={r['id']:r for r in panel[k]['per_utterance']}
    gains_by={}
    for r in cs:
        a=old[r['id']]['counts']; b=r['counts']; diff=a[2]-b[2]
        gains_by[r['dialogue']]=gains_by.get(r['dialogue'],0)+diff
    positive=[v for v in gains_by.values() if v>0]
    concentrated=not positive or len(positive)<rule['min_gain_dialogues'] or max(positive)/sum(positive)>rule['max_positive_gain_dialogue_share']
    passed &= not concentrated
    return {'decision':'N2_EXPLORATORY_GO' if passed else 'N2_NO_GO','gains':gains,'dialogue_gains':gains_by,
        'pier_available':g.get('pier') is not None,'claim':'Exploratory screen; no independent confirmation/novelty claim'}


def pareto(points,x,y):
    valid=[p for p in points if p.get(x) is not None and p.get(y) is not None and math.isfinite(p[x]) and math.isfinite(p[y])]
    return [p for p in valid if not any(q[x]<=p[x] and q[y]<=p[y] and (q[x]<p[x] or q[y]<p[y]) for q in valid)]

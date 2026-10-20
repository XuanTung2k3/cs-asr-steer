"""Dialogue-cluster inference, no utterance-independence assumption."""
import numpy as np


def paired_clusters(a,b,draws=2000,seed=42,alpha=.05,cs_only=True):
    aa={r['id']:r for r in a['per_utterance']};bb={r['id']:r for r in b['per_utterance']}
    if set(aa)!=set(bb):raise ValueError('Unpaired evaluation populations')
    clusters={}
    for k,x in aa.items():
        y=bb[k]
        if x['dialogue']!=y['dialogue'] or x['counts'][3]!=y['counts'][3]:raise ValueError('Incompatible reference alignment')
        if cs_only and x['stratum']!='CS':continue
        key=x['dialogue']; v=clusters.setdefault(key,np.zeros(3))
        v += [x['counts'][2],y['counts'][2],x['counts'][3]]
    if len(clusters)<3:return {'status':'INSUFFICIENT_CLUSTERS','n':len(clusters)}
    vals=np.array(list(clusters.values()));total=vals.sum(0)
    if total[2]<=0:raise ValueError('No reference units')
    observed=(total[0]-total[1])/total[2]
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(vals),(draws,len(vals)))
    sums=vals[idx].sum(1);samples=(sums[:,0]-sums[:,1])/sums[:,2]
    ci=np.quantile(samples,[alpha/2,1-alpha/2]).tolist()
    leave=[]
    for key,v in zip(clusters,vals):
        t=total-v
        leave.append({'dialogue_removed':key,'difference':float((t[0]-t[1])/t[2]) if t[2]>0 else None})
    return {'status':'COMPLETE','difference_a_minus_b':float(observed),'interval':ci,'alpha':alpha,
        'clusters':len(clusters),'resamples':draws,'seed':seed,'leave_one_dialogue_out':leave,
        'claim':'Paired dialogue-cluster development uncertainty; not independent final evidence'}

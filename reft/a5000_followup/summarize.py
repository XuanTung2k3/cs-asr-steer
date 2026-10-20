"""Read-only pilot aggregation: actual metrics, dialogue sensitivity, costs and provenance.

No checkpoint selection or new evaluation is performed here. Run after comparison.json exists.
"""
import json
from pathlib import Path

import numpy as np

from reft.a5000.data import output,atomic,sha


def paired_bootstrap(base,adapted,draws=2000):
    dialogues=sorted({r['dialogue'] for r in base})
    def table(rows):
        d={k:np.zeros(4) for k in dialogues}
        for r in rows:
            c=r['counts'];d[r['dialogue']][:2]+=[c[2],c[3]]
            if r['stratum']=='CS':d[r['dialogue']][2:]+=[c[2],c[3]]
        return np.stack([d[k] for k in dialogues])
    b,a=table(base),table(adapted)
    def diff(x,y):return [float(y[0]/y[1]-x[0]/x[1]),float(y[2]/y[3]-x[2]/x[3])]
    point=diff(b.sum(0),a.sum(0));rng=np.random.default_rng(42);samples=[]
    for _ in range(draws):
        i=rng.integers(0,len(dialogues),size=len(dialogues))
        bs,ass=b[i].sum(0),a[i].sum(0)
        if bs[1] and bs[3]:samples.append(diff(bs,ass))
    ci=np.quantile(np.asarray(samples),[.025,.975],axis=0)
    return {'dialogues':len(dialogues),'draws':draws,'delta_mer':point[0],'delta_cs_mer':point[1],
            'mer_ci95':[float(ci[0,0]),float(ci[1,0])],'cs_mer_ci95':[float(ci[0,1]),float(ci[1,1])],
            'interpretation':'descriptive paired dialogue bootstrap; one seed, development-selected; not a confirmatory superiority test'}


def summarize():
    root=output();comparison=json.loads((root/'comparison.json').read_text())
    baseline=json.loads((root/'B0/metrics.json').read_text())
    report={'status':'A5000_E1_PURE_CE_PILOT_COMPLETE',
        'replication_status':'A5000_NEW_IMPLEMENTATION_VALIDATED_NOT_HISTORICAL_REPLICATION',
        'source_commit':comparison['source_commit'],'freeze_sha256':sha(root/'freeze.json'),
        'role_manifest_sha256':json.loads((root/'freeze.json').read_text())['role_manifest_sha256'],
        'B0':{k:v for k,v in baseline.items() if k!='per_utterance'},'arms':{},
        'budget':json.loads((root/'budget.json').read_text()),'C150_evaluated':False}
    for method,sel in comparison['arms'].items():
        path=Path(sel['checkpoint'])
        assert sha(path)==sel['checkpoint_sha256']
        metrics=json.loads((path.parent/'validation/metrics.json').read_text())
        logs=[];logged_training_seconds=0.
        for f in sorted((root/'arms'/method).glob('*/training.jsonl')):
            rows=[json.loads(l) for l in f.read_text().splitlines()]
            assert len({(x['epoch'],x['batch']) for x in rows})==len(rows),'duplicate optimizer log entries'
            logs+=rows
            last_epoch=None;last_seconds=0.
            for row in rows:
                if row['epoch']!=last_epoch or row['seconds']<last_seconds:
                    logged_training_seconds+=last_seconds
                last_epoch=row['epoch'];last_seconds=row['seconds']
            logged_training_seconds+=last_seconds
        report['arms'][method]={'selection':sel,'contrast_vs_B0':paired_bootstrap(baseline['per_utterance'],metrics['per_utterance']),
            'training_steps_all_lr_trials':len(logs),'training_tokens_all_lr_trials':sum(x['tokens'] for x in logs),
            'logged_training_seconds':logged_training_seconds,
            'timing_caveat':'First E-ReFT trial also warms the shared exact encoder cache; not a pure method-speed comparison',
            'training_peak_allocated_bytes':max(x['allocated_peak_bytes'] for x in logs),
            'training_peak_reserved_bytes':max(x['reserved_peak_bytes'] for x in logs),
            'checkpoint_bytes':path.stat().st_size,'selected_decode_seal':json.loads((path.parent/'validation/SEALED.json').read_text())}
    a,b=[report['arms'][m]['selection'] for m in ['ereft','lra']]
    am=json.loads((Path(a['checkpoint']).parent/'validation/metrics.json').read_text())
    bm=json.loads((Path(b['checkpoint']).parent/'validation/metrics.json').read_text())
    report['lra_minus_ereft']=paired_bootstrap(am['per_utterance'],bm['per_utterance'])
    atomic(root/'pilot_summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))


if __name__=='__main__':summarize()

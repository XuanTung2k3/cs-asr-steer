"""Actual-data performance/efficiency exports, including incomplete arm status."""
import argparse
import csv
import json
from pathlib import Path
from reft.a5000_campaign.state import read,atomic,now
from reft.a5000_campaign.evaluation import pareto
from .jobs import parent_records


def screening_costs(parent):
    costs={}
    for p in (parent/'jobs').glob('*/runs/*/run.json'):
        spec=read(p);u=p.parent/'usage.json'
        if spec['seed']!=42 or spec['tag'] not in [None,'N3_WCE'] or not u.exists():continue
        key=spec['method']+('_CE' if spec['omega']==1 else '_WCE')
        costs[key]=costs.get(key,0)+read(u)['allocated_seconds']
    return costs


def point(name,rec,cost,status='COMPLETE',incremental=0.):
    m=rec['metrics'];peak=rec.get('training_peak') or {}
    return {'system':name,'status':status,'genuine_cs_mer':m['genuine_cs_mer'],'mer':m['mer'],
        'noncs_mer':m.get('noncs_mer'),'en_wer':m['en_wer'],'zh_cer':m['zh_cer'],
        'wer_cs':None,'wer_noncs':None,'pier':None,
        'phase_trainable_parameters':rec.get('trainable_parameters',rec.get('total_adapter_gate_parameters',0)),
        'total_adapter_parameters':rec.get('total_adapter_parameters',rec.get('total_adapter_gate_parameters',0)),
        'adapter_checkpoint_bytes':rec.get('checkpoint_bytes',0),'checkpoint_sha256':rec.get('checkpoint_sha256'),
        'incremental_fit_gpu_hours':rec.get('fit_seconds',0)/3600,
        'incremental_allocated_gpu_hours':incremental/3600,'shared_screening_allocated_gpu_hours':cost/3600,
        'attributed_total_pipeline_gpu_hours':(cost+incremental)/3600,
        'peak_allocated_bytes':peak.get('allocated_peak_bytes'),'peak_reserved_bytes':peak.get('reserved_peak_bytes'),
        'decoding_seconds':m['seconds'],'rtf':m['rtf'],'insertions':m['insertions'],
        'allocation_std':m.get('allocation_std',0.),
        'loops':m['loops'],'nontermination':m['nontermination'],'optimizer_updates':rec.get('updates',rec.get('steps',0))}


def decision(rows,spec):
    by={r['system']:r for r in rows if r['status']=='COMPLETE'}
    if any(k not in by for k in ['L0','L1','L2','L3','L4']):return {'outcome':'INCONCLUSIVE_INCOMPLETE','architecture_expansion':False}
    proposed=by['L4'];controls=spec['decision']['controls'];gains={k:by[k]['genuine_cs_mer']-proposed['genuine_cs_mer'] for k in controls}
    if any(by[k].get('noncs_mer') is None for k in controls+['L4']):
        return {'outcome':'INCONCLUSIVE_MISSING_PRESERVATION','architecture_expansion':False}
    useful=all(v>=spec['decision']['min_absolute_cs_gain'] for v in gains.values())
    useful &= all(proposed['noncs_mer']-by[k]['noncs_mer']<=spec['decision']['max_noncs_mer_degradation'] for k in controls)
    useful &= all(proposed['loops']<=by[k]['loops'] and proposed['nontermination']<=by[k]['nontermination'] for k in controls)
    useful &= proposed.get('allocation_std',0)>=spec['decision']['minimum_allocation_std']
    return {'outcome':'EXPLORATORY_POSITIVE_SIGNAL' if useful else 'NO_MEANINGFUL_L4_BENEFIT',
        'gains_absolute_cs_mer':gains,'minimum_gain':spec['decision']['min_absolute_cs_gain'],
        'allocation_std':proposed.get('allocation_std',0),'minimum_allocation_std':spec['decision']['minimum_allocation_std'],
        'architecture_expansion':False,'claim':'Post-hoc one-seed V300 exploration, not independent confirmation'}


def report(root,final=False):
    root=Path(root);plan=read(root/'frozen_plan.json');spec=plan['spec'];parent=Path(spec['parent_root'])
    dest=root/'reports';dest.mkdir(exist_ok=True);costs=screening_costs(parent);records=parent_records(parent)
    rows=[point(k,rec,costs.get(k,0)) for k,rec in records.items()]
    incomplete={};audit_ok=False
    for arm in ['L0','L1','L2','L3','L4','AUDIT']:
        p=root/'jobs'/arm/'result.json'
        if not p.exists():incomplete[arm]='PENDING_OR_RUNNING';continue
        rec=read(p)
        if rec['status']!='COMPLETE':incomplete[arm]=rec;continue
        if arm=='AUDIT':audit_ok=rec['result'].get('verified',False);continue
        rows.append(point(arm,rec['result'],costs['L_CE'],incremental=rec['allocated_seconds']))
    status=read(root/'status.json') if (root/'status.json').exists() else {}
    result=decision(rows,spec)
    if final and not audit_ok:result={'outcome':'INVALID_OR_INCOMPLETE_BASELINE_AUDIT','architecture_expansion':False}
    accounting={'sprint_controller_allocated_gpu_hours':status.get('gpu_seconds',0)/3600,
        'parent_controller_recorded_gpu_hours':plan['parent_receipt']['status']['gpu_seconds']/3600,
        'method_screening_attributed_gpu_hours':{k:v/3600 for k,v in costs.items()},
        'not_additive':'Warm/screening costs are attributed to each dependent arm; NEVER sum table rows as actual campaign GPU usage',
        'timing_scopes':'incremental_fit is synchronized optimizer-loop wall time; allocated includes model setup/normalization/decode/checkpoint I/O; screening is archived per-arm GPU residence'}
    atomic(dest/'accounting.json',accounting);atomic(dest/'decision.json',result)
    with (dest/'combined.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=['|System|CS-MER|MER|Non-CS MER|EN-WER|ZH-CER|Total params|Own GPUh|Attributed total GPUh|Loops/cap|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        table.append('|'+ '|'.join([r['system']]+[f"{r[k]*100:.3f}%" for k in ['genuine_cs_mer','mer','noncs_mer','en_wer','zh_cer']]+
            [str(r['total_adapter_parameters']),f"{r['incremental_allocated_gpu_hours']:.3f}",f"{r['attributed_total_pipeline_gpu_hours']:.3f}",f"{r['loops']}/{r['nontermination']}"])+'|')
    (dest/'combined.md').write_text('\n'.join(table)+'\n')
    tex=['\\begin{tabular}{lrrrr}','System & CS-MER & MER & Non-CS MER & Params \\\\','\\hline']
    for r in rows:
        tex.append(f"{r['system'].replace('_',' ')} & {r['genuine_cs_mer']*100:.3f} & {r['mer']*100:.3f} & {r['noncs_mer']*100:.3f} & {r['total_adapter_parameters']} \\\\")
    tex.append('\\end{tabular}');(dest/'combined.tex').write_text('\n'.join(tex)+'\n')
    axes=['total_adapter_parameters','attributed_total_pipeline_gpu_hours','noncs_mer']
    frontiers={x:[r['system'] for r in pareto(rows,x,'genuine_cs_mer')] for x in axes}
    atomic(dest/'pareto.json',{'measured_points_only':True,'axes':frontiers,'incomplete':incomplete,
        'cost_axis_is_attributed_pipeline_cost_not_additive_campaign_usage':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=root/'figures';figs.mkdir(exist_ok=True)
    for x in axes:
        fig,ax=plt.subplots(figsize=(7,4.6))
        for r in rows:
            ax.scatter(r[x],r['genuine_cs_mer']*100,s=25)
            ax.annotate(r['system'],(r[x],r['genuine_cs_mer']*100),fontsize=7,xytext=(3,3),textcoords='offset points')
        if x=='total_adapter_parameters':ax.set_xscale('symlog',linthresh=100)
        ax.set_xlabel(x);ax.set_ylabel('Genuine-CS MER (%)');fig.tight_layout()
        fig.savefig(figs/(x+'.pdf'));fig.savefig(figs/(x+'.png'),dpi=220);plt.close(fig)
    by={r['system']:r for r in rows}
    measured_contrasts=['No complete L0–L4 comparison yet.']
    if all(k in by for k in ['L0','L1','L2','L3','L4']):
        measured_contrasts=[f"L4 minus {k} CS-MER: {by['L4']['genuine_cs_mer']-by[k]['genuine_cs_mer']:+.6f} absolute; "
            f"non-CS MER: {by['L4']['noncs_mer']-by[k]['noncs_mer']:+.6f}." for k in ['L0','L1','L2','L3']]
    lines=['# Gated-LRA exploratory sprint','',f'Generated {now()}; final={final}',f"Outcome: {result['outcome']}",
        '',*table,'','## Evidence and answers','',
        '1. Same-site LRA is a strong compact overall/preservation baseline. CE ReFT has marginally lower CS-MER but worse overall/non-CS results with a retained loop. A universal ~5K winner is not established.',
        '2. New gated/continued/scalar contrasts are available only when all L0–L4 complete. The table preserves their actual outcomes and equal125update/2000example fitting budget.',
        '3. L4 must improve actual CS-MER by>=0.005 absolute versus L3 and every other control, with frozen preservation/failure conditions. A smaller gap is a negative useful-benefit outcome; no architecture expansion.',
        '4. Judge extra289gate parameters, serialized projector/normalization buffers, own GPU time, shared screening and model-inference RTF together. Nondominance is axis-specific, not a universal efficiency claim.',
        '5. Designed after viewing prior V300 outcomes; seed42 only. Multi-seed and independent confirmation remain required. C500 and locked D-test were not opened.',
        '',*measured_contrasts,
        '',f'Incomplete/unavailable arms: {json.dumps(incomplete)}',
        f'Baseline audit complete: {audit_ok}',
        'PIER unavailable without verified POIs; WER-CS/non-CS unavailable without canonical Chinese word segmentation. MER/CER remain canonical.',
        'Inherited E-ReFT CE looping/nontermination remains in headline scores. Earlier N2_NO_GO is unchanged. Full N4/N5 was not resumed.',
        'The energy constraint applies before BF16 interface casting; cast-energy drift is recorded in prediction diagnostics.',
        'See accounting.json: attributed method totals reuse shared warm/search cost and must not be summed as actual campaign GPU-hours.']
    (dest/('FINAL_REPORT.md' if final else 'PRELIMINARY_REPORT.md')).write_text('\n'.join(lines)+'\n')
    atomic(dest/'report_status.json',{'utc':now(),'final':final,'rows':len(rows),'outcome':result,'incomplete':list(incomplete)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--final',action='store_true');a=p.parse_args();report(a.root,a.final)
if __name__=='__main__':main()

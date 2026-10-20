"""Actual-data tables/plots; absence of results is reported explicitly."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from .state import atomic,read,now
from .evaluation import pareto


def report(root):
    root=Path(root); dest=root/'reports';dest.mkdir(exist_ok=True); points=[]
    for stage,file in [('N1','N1_selection.json'),('N3','N3_selection.json'),('N2','N2_selection.json')]:
        p=root/file
        if not p.exists():continue
        for name,r in read(p).items():
            m=r['metrics'];points.append({'stage':stage,'method':name,'seed':r['spec']['seed'],
                'objective':'CE' if r['spec']['omega']==1 else 'WCE',
                'genuine_cs_mer':m['genuine_cs_mer'],'mer':m['mer'],'noncs_mer':m.get('noncs_mer'),
                'en_wer':m['en_wer'],'zh_cer':m['zh_cer'],'pier':None,
                'parameters':r['total_adapter_gate_parameters'],'training_gpu_hours':r['seconds']/3600,
                'peak_reserved_bytes':r['training_peak']['reserved_peak_bytes'],
                'decode_seconds':m['seconds'],'rtf':m['rtf'],'checkpoint':r['checkpoint'],
                'checkpoint_sha256':r['checkpoint_sha256'],'checkpoint_bytes':r['checkpoint_bytes'],
                'examples':r['examples'],'steps':r['steps'],
                'cost_limitation':'Per-arm residence excludes shared warm start/LR search; NOT total-method cost frontier'})
    if points:
        with (dest/'development_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
        # Capacity/memory/latency frontiers are legitimate per-arm measurements.
        # Full-method GPU efficiency cannot be computed from selected-arm cost
        # alone: retain points but mark total-cost Pareto unavailable.
        frontiers={x:[p['method'] for p in pareto(points,x,'genuine_cs_mer')] for x in ['parameters','peak_reserved_bytes','decode_seconds']}
        atomic(dest/'pareto.json',{'frontiers':frontiers,'total_gpu_hours_frontier':None,
            'reason':'Warm-start/search/recovery overhead not yet aggregated per independent pipeline'})
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figs=root/'figures';figs.mkdir(exist_ok=True)
            for x in ['parameters','peak_reserved_bytes','decode_seconds']:
                fig,ax=plt.subplots(figsize=(6,4))
                for p in points:
                    ax.scatter(p[x],p['genuine_cs_mer']);ax.annotate(p['method']+' '+p['objective'],(p[x],p['genuine_cs_mer']),fontsize=7)
                ax.set_xlabel(x);ax.set_ylabel('Genuine-CS MER');fig.tight_layout()
                fig.savefig(figs/(x+'.pdf'));fig.savefig(figs/(x+'.png'),dpi=200);plt.close(fig)
        except ImportError: pass
        table=['\\begin{tabular}{llrr}','Method & Objective & CS-MER & MER \\\\','\\hline']
        table += [f"{p['method'].replace('_',' ')} & {p['objective']} & {p['genuine_cs_mer']:.4f} & {p['mer']:.4f} \\\\" for p in points]
        table+=['\\end{tabular}'];(dest/'development_table.tex').write_text('\n'.join(table)+'\n')
    # Fixed primary holdout contrasts; Bonferroni family-wise percentile
    # intervals. No significance language is inferred from seed42 selection.
    heldpath=root/'confirmation/results.json'
    if heldpath.exists():
        from .statistics import paired_clusters
        import numpy as np
        held=read(heldpath); contrasts={}; pairs=[('B0','R'),('L','R'),('LS','R')]
        for a,b in pairs:
            if a in held and b in held:
                contrasts[a+'_minus_'+b]=paired_clusters(held[a],held[b],alpha=.05/len(pairs))
        atomic(dest/'paired_dialogue_intervals.json',contrasts)
        groups={}
        for name,m in held.items():
            base=name.split('_seed')[0]
            if base in ['R','L','LS','G3','G4','G5','G8']:
                groups.setdefault(base,[]).append({'name':name,'genuine_cs_mer':m['genuine_cs_mer'],'mer':m['mer']})
        summary={k:{'n':len(v),'mean_cs_mer':float(np.mean([x['genuine_cs_mer'] for x in v])),
            'seed_sd':float(np.std([x['genuine_cs_mer'] for x in v],ddof=1)) if len(v)>1 else None,
            'per_seed':v,'confirmation':len(v)==3} for k,v in groups.items()}
        atomic(dest/'seed_summary.json',summary)
    decision=read(root/'N2_decision.json') if (root/'N2_decision.json').exists() else {'decision':'NOT_MEASURED'}
    confirmed=(root/'confirmation/results.json').exists()
    status=read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['# A5000 campaign scientific findings','',f'Generated {now()}',f"Campaign status: {status.get('status','UNKNOWN')}",
        f"Gate decision: {decision['decision']}",f'Complete selected development cells: {len(points)}',
        f'One-time C500 package completed: {confirmed}','',
        'All populations are drawn from original TRAIN. These results do not reproduce historical H100 roles.',
        'Verified linguistic PIER and Chinese word WER are unavailable. Script proxies are not PIER.',
        'No official locked test or external final evaluation was authorized or run.',
        'One-seed method selection is exploratory; no superiority/novelty conclusion is generated from engineering PASS.',
        'D3 remains unimplemented; full cost aggregation and uncertainty-based primary claims remain limitations.',
        '', '## Available selected results','', '|Method|Objective|CS-MER|MER|','|---|---|---|---|']
    lines += [f"|{p['method']}|{p['objective']}|{p['genuine_cs_mer']:.5f}|{p['mer']:.5f}|" for p in points]
    lines += ['','## Remaining evidence','',
        'Independent official/external testing: BLOCKED pending separate scoped authorization.',
        'No causal conclusion is drawn from gate correlation, script probes, or a non-significant contrast.']
    for name in ['SCIENTIFIC_FINDINGS.md','MANUSCRIPT_RESULTS.md','CAMPAIGN_SUMMARY.md']:
        (dest/name).write_text('\n'.join(lines)+'\n')
    atomic(root/'stage_results'/'REPORT.json',{'status':'COMPLETE','finished':now(),'result':{'selected_cells':len(points),'holdout_complete':confirmed}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)

if __name__=='__main__':main()

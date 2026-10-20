"""Frozen arm-level DAG: scientific recipes unchanged, independent jobs parallel."""
from .state import read


def graph(cfg):
    jobs={}
    def add(name,kind,deps=(),priority=10,**args):
        jobs[name]={'id':name,'kind':kind,'dependencies':list(deps),'priority':priority,'args':args}
    add('B0','bootstrap',priority=0)
    for objective,omega,priority in [('CE',1,10),('WCE',cfg['training']['weighted_omegas'][0],30)]:
        selects=[]
        for method in ['R','L','LP','LS']:
            trials=[]
            for i,lr in enumerate(cfg['training']['lr'][method]):
                name=f'{objective}_{method}_lr{i}_s42'; trials.append(name)
                add(name,'train',['B0'],priority,method=method,lr=lr,omega=omega,seed=42,
                    horizon=cfg['training']['epochs'],tag=None if objective=='CE' else 'N3_WCE')
            sel=f'select_{objective}_{method}';selects.append(sel)
            add(sel,'select',trials,priority=1,method=method,objective=objective)
        add('N1' if objective=='CE' else 'N3','assemble',selects,priority=1,objective=objective)
    add('scalar','scalar',['select_CE_R','B0'],priority=20)
    for kind in ['G3','G4','G5','G8']:
        add(kind,'gate',['select_CE_R','B0'],priority=20,gate=None if kind=='G8' else kind,tag=kind)
    add('N2','decision',['scalar','G3','G4','G5','G8'],priority=1)
    add('G5_WCE','weighted_gate',['N2','select_WCE_R','B0'],priority=31)
    jobs['N3']['dependencies'].append('G5_WCE')
    add('CORE_REPORT','core_report',['N1','N2','N3','G5_WCE'],priority=1)
    # Existing serial downstream workers retain the C500 freeze/firewall. Their
    # starts are controlled by this DAG, after all core work is complete.
    add('N4','legacy',['CORE_REPORT'],priority=40,stage='N4')
    add('N5','legacy',['N4'],priority=50,stage='N5')
    add('REPORT','report',['N5'],priority=0)
    # Planning only: measured warm (not first-call kernel startup) teacher-
    # forced times plus actual 80-utterance baseline mean. Safety factor1.5
    # covers serialization/shared storage. No recipe is shortened for this ETA.
    decode=cfg.get('prior_observed_decode_seconds_mean',1.7242)
    speeds={'R':.25,'L':.13,'LP':.4,'LS':.42}
    for j in jobs.values():
        if j['kind']=='bootstrap': seconds=(2300*.12+300*decode)*1.5
        elif j['kind']=='train':seconds=(6000*speeds[j['args']['method']]+900*decode)*1.5
        elif j['kind'] in ['gate','weighted_gate']:seconds=(2000*.25+300*decode+150)*1.5
        elif j['kind']=='scalar':seconds=1500*decode*1.5
        else:seconds=1.
        j['estimate_seconds']=seconds
    return jobs


def satisfied(job,state):
    return all(state[d]['status'] in ['COMPLETE','SKIPPED'] for d in job['dependencies'])


def pick_ready(jobs,state):
    return sorted([k for k,v in jobs.items() if state[k]['status']=='PENDING' and satisfied(v,state)],
        key=lambda k:(jobs[k]['priority'],list(jobs).index(k)))


def selection(records):
    return min(records,key=lambda h:(h['metrics']['genuine_cs_mer'],h['metrics']['mer'],
        h['metrics']['zh_cer'],h['epoch'],h['spec']['lr']))


def estimate_core(jobs,gpus=6):
    state={k:{'status':'PENDING'} for k in jobs};clock=0.;running={};slots=list(range(gpus));gpu_seconds=0.
    wanted=set()
    def include(k):
        if k in wanted:return
        wanted.add(k)
        for d in jobs[k]['dependencies']:include(d)
    include('CORE_REPORT')
    while state['CORE_REPORT']['status']!='COMPLETE':
        ready=[k for k in pick_ready(jobs,state) if k in wanted]
        changed=False
        for k in ready:
            if jobs[k]['kind'] not in ['bootstrap','train','gate','weighted_gate','scalar']:
                state[k]['status']='COMPLETE';changed=True;continue
            if slots:
                slot=slots.pop(0);duration=jobs[k]['estimate_seconds'];gpu_seconds+=duration
                running[k]=(clock+duration,slot);state[k]['status']='RUNNING';changed=True
        if changed:continue
        if not running:raise ValueError('Invalid/cyclic core graph')
        clock=min(t[0] for t in running.values())
        for k,(end,slot) in list(running.items()):
            if end<=clock:state[k]['status']='COMPLETE';slots.append(slot);del running[k]
    return {'core_wall_hours':clock/3600,'aggregate_gpu_hours':gpu_seconds/3600,'gpus':gpus,
        'scope':'N1/N2/N3, before confirmation; estimate not a deadline guarantee','safety_factor':1.5}

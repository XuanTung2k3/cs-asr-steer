"""Fail-closed preparation CLI. This is admission scaffolding, not a complete R0–R6 controller."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid
from .admission import admit,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','launch','status','report','resume','stop'])
    p.add_argument('--config',default='configs/reft/a5000_ce_only_v1.yaml')
    p.add_argument('--run-id');args=p.parse_args()
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
    root=Path(config['output_root']).expanduser().resolve()
    if args.command in ['prepare','launch']:
        problems=admit(config)
        # No configuration switch can activate an absent scientific executor.
        if not problems:problems=['R0–R6 executor absent: admission scaffold cannot launch scientific training']
        ident=args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        if Path(ident).name!=ident:raise ValueError('run-id must be one path component')
        dest=root/ident;dest.mkdir(parents=True,exist_ok=False)
        (dest/'frozen_plan.json').write_text(json.dumps(config,indent=2)+'\n')
        source={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'config_sha256':digest(config_path),'files':{str(f):digest(f) for f in sorted(Path('reft/a5000_ce_only').glob('*.py'))}}
        (dest/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
        state={'status':'BLOCKED_MISSING_ARTIFACT','admission_status':'A5000_CE_ONLY_PREFLIGHT_BLOCKED','reasons':problems,
               'run_id':ident,'scientific_process_started':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (dest/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        (dest/'events.jsonl').write_text(json.dumps(state)+'\n')
        (dest/'campaign.log').write_text('Admission blocked; no GPU process started.\n'+'\n'.join(problems)+'\n')
        (dest/'reports').mkdir()
        (dest/'reports/A5000_CE_ONLY_IMPLEMENTATION_REPORT.md').write_text(
            '# A5000 CE-only admission blocked\n\nOnly metadata admission is implemented. '
            'R0–R6, temporal gates, encoder LoRA, GPU preflight and autonomous training are not implemented or launched.\n\n'
            +'\n'.join('- '+x for x in problems)+'\n')
        print(json.dumps({'directory':str(dest),**state},indent=2));return 2
    if not args.run_id or Path(args.run_id).name!=args.run_id:raise ValueError('A valid --run-id is required')
    dest=root/args.run_id
    if args.command in ['status','report']:print((dest/'status.json').read_text());return 0
    if args.command=='stop':(dest/'STOP').touch();print('No controller was launched; stop sentinel recorded.');return 0
    print('Resume refused: admission blocker must be resolved; this scaffold has no training executor.',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())

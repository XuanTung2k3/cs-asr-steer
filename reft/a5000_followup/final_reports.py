"""Generate completion artifacts from saved evidence, without changing tracked source."""
import json
import subprocess
from pathlib import Path
from reft.a5000.data import output, sha


def main():
    root=output(); dest=root/'reports';dest.mkdir(exist_ok=True)
    def read(name):
        p=root/name
        return json.loads(p.read_text()) if p.exists() else {'status':'NOT_AVAILABLE'}
    primary=read('pilot_summary.json')
    assert primary.get('status')=='A5000_E1_PURE_CE_PILOT_COMPLETE'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    evidence={name:read(name) for name in ['comparison.json','selection.json','freeze.json','budget.json',
        'confirmation/comparison.json','objective_engineering_preflight/report.json','audits/data_and_target_integrity.json']}
    evidence['pilot_summary']=primary;evidence['report_source_commit']=commit
    (dest/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    prefix=(f'# A5000 E0/E1 completed evidence\n\nPrimary status: {primary["status"]}\n\n'
        'A5000_NEW_IMPLEMENTATION_VALIDATED_NOT_HISTORICAL_REPLICATION\n\n'
        f'Primary scientific source: `{primary["source_commit"]}`. Reporting source: `{commit}`.\n\n'
        f'Role manifest SHA256: `{primary["role_manifest_sha256"]}`.\n\n'
        'This is a one-seed exploratory comparison on new splits from the original TRAIN partition. '
        'It is not historical H100 replication or official test evaluation. No final test was decoded.\n\n')
    sections={
      'A5000_E0_SCIENTIFIC_SETUP.md':('Data protocol',{'roles':'T394 / V106 / C150; speaker/dialogue disjoint',
          'source':'Versioned configs/reft/a5000_manifests/roles_v1.json','freeze':evidence['freeze.json'],
          'data_audit':evidence['audits/data_and_target_integrity.json']}),
      'A5000_E1_IMPLEMENTATION_REPORT.md':('Engineering evidence',{
          'site':'Encoder output after final layer normalization; frozen backbone; rank 2; 5122 scalars each',
          'CPU':'1030 distinct passed; one private historical artifact test blocked/skipped',
          'arms':primary['arms'],'objective_preflight':evidence['objective_engineering_preflight/report.json']}),
      'A5000_E1_PILOT_RESULTS.md':('Scientific evidence',{
          'baseline':primary['B0'],'selection':evidence['comparison.json'],
          'paired_development_contrast':primary['lra_minus_ereft'],
          'confirmation':evidence['confirmation/comparison.json'],
          'limitations':'Single seed, selected on V106. Both chosen LRs may sit at grid upper boundary. No superiority claim; script groups are language proxies.'}),
      'A5000_OBJECTIVE_MATRIX_READINESS.md':('Follow-up readiness',{
          'engineering_preflight':evidence['objective_engineering_preflight/report.json'],
          'ready_tools':'O2/O3 and practical decoder weight-LoRA; D1/D2 CPU foundations; separate GoldKL bridge',
          'not_executed':'No full O2/O3 objective matrix; no large D1/D2 study; no GoldKL campaign',
          'rule':'An engineering pass is not scientific evidence of recognition gains.'}),
      'A5000_REPRODUCIBILITY_LEDGER.md':('Provenance and cost',{
          'source_commit':commit,'primary_source':primary['source_commit'],'freeze_sha256':sha(root/'freeze.json'),
          'budget':evidence['budget.json'],'results_root':str(root),
          'controls':'scripts/a5000/finish_control.sh status|logs|stop|resume|fetch',
          'audit':'Codex self-check is internal, not an independent external scientific audit.'})}
    for name,(title,body) in sections.items():
        (dest/name).write_text(prefix+f'## {title}\n\n```json\n'+json.dumps(body,indent=2)+'\n```\n')
    print(str(dest))


if __name__=='__main__':main()

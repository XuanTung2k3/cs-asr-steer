import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('admission',Path(__file__).resolve().parents[3]/'reft/a5000_ce_only/admission.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AdmissionTests(unittest.TestCase):
    def test_missing_historical_roles_block(self):
        self.assertEqual(len(a.validate_roles({})),4)
    def test_a5000_roles_not_substituted(self):
        self.assertEqual(len(a.validate_roles({'A5000_T394':{'count':394}})),4)
    def test_hash_mismatch_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('not json')
            self.assertIn('SHA256 mismatch',a.validate_roles({'Train394':{'path':str(p),'sha256':'wrong'}})[0])
    def test_objective_and_test_firewall(self):
        errors=a.admit({'objective':'GoldKL','locked_test_access':True})
        self.assertIn('Objective is not CE-only',errors)
        self.assertIn('Locked test firewall missing',errors)
    def test_full500_union_and_disjointness(self):
        with tempfile.TemporaryDirectory() as d:
            def rows(prefix,n):return [{'utterance_id':f'{prefix}{i}','dialogue_id':prefix,'speaker_id':prefix,
                'audio_sha256':f'{prefix}-audio-{i}','partition':'train'} for i in range(n)]
            t,v,dev=rows('t',394),rows('v',106),rows('d',150)
            def specs(t,v,dev):
                out={}
                for role,rs in [('Train394',t),('Inner106',v),('Full500',t+v),('Dev150',dev)]:
                    p=Path(d)/f'{role}.json';p.write_text(json.dumps({'role':role,'provenance':'synthetic test only','rows':rs}))
                    out[role]={'path':str(p),'sha256':a.digest(p)}
                return out
            self.assertEqual(a.validate_roles(specs(t,v,dev)),[])
            v[0]['dialogue_id']='t'
            self.assertIn('Train394/Inner106 overlap: dialogue_id',a.validate_roles(specs(t,v,dev)))

if __name__=='__main__':unittest.main()

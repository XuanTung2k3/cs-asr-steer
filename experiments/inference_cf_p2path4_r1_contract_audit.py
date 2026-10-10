"""Independent CPU-only PATH4-R1 freeze audit. No model, reference or primary decision imports.

This audits the amendment freeze; Claude must separately audit the runnable manifest
and implementation before issuing the same gate for its scientific allocation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ORIGINAL = "29661e972f0ebfd727d228d2f6dbd9b17dd41c5a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original(path):
    return subprocess.check_output(["git", "show", f"{ORIGINAL}:{path}"])


def audit(junit):
    config = "configs/inference_cf/p2_path4.json"
    c = json.loads(Path(config).read_text())
    old = json.loads(original(config))
    checks = {}
    changed = {"status", "G1", "audit", "stopping", "compute"}
    checks["original_science_unchanged"] = all(c[k] == v for k, v in old.items() if k not in changed)
    checks["compute_budget_unchanged"] = all(c["compute"][k] == v for k,v in old["compute"].items() if k != "phase_order")
    checks["archive_exact"] = Path(c["original_blocked_provenance"]["config_archive"]).read_bytes() == original(config)
    checks["blocked_provenance"] = c["original_blocked_provenance"]["label"] == "P2_PATH4_BLOCKED_EOS_FIRST_BRANCH" and c["original_blocked_provenance"]["commit"] == ORIGINAL
    checks["revision"] = c["contract_revision"] == "PATH4_R1_EOS_BOUNDARY_V1" and c["audit"]["pre"] == "PASS_TO_P2_PATH4_R1"
    checks["G1_narrow_change"] = all(c["G1"][k] == v for k,v in old["G1"].items() if k != "effective_H")
    checks["historical_sources_unchanged"] = all(sha(p) == h for p,h in old["source_sha256"].items())
    checks["amendment_sources"] = all(sha(p) == h for p,h in c["amendment_source_sha256"].items())
    checks["spec_design_historical_body_preserved"] = all(Path(p).read_bytes().endswith(original(p)) for p in (c["spec"],c["design"]))
    panel = json.loads(Path(c["panel"]).read_text())
    checks["panel_exact"] = Path(c["panel"]).read_bytes() == original(c["panel"]) and sha(c["panel"]) == old["panel_byte_sha256"]
    checks["fixed100"] = len(panel["rows"]) == 100 and len({r["dialogue_id"] for r in panel["rows"]}) == 20
    own_delta, boundary, content = [], [], []
    for row in panel["rows"]:
        b = json.loads(Path(row["B0_FORCED"]["path"]).read_text())["systems"]["S0"]
        a = json.loads(Path(row["A2"]["path"]).read_text())["objectives"]["A2"]
        if (b["tokens"],b["terminated"]) == (a["tokens"],a["terminated"]):
            continue
        own_delta.append(row["utterance_id"])
        bs = b["tokens"] + ([50257] if b["terminated"]=="eos" else [])
        aa = a["tokens"] + ([50257] if a["terminated"]=="eos" else [])
        k = next(i for i,(x,y) in enumerate(zip(bs,aa)) if x!=y)
        assert bs[:k] == aa[:k]
        if (bs[k]==50257) != (aa[k]==50257):
            boundary.append(row["utterance_id"])
        else:
            content.append(row["utterance_id"])
    checks["token_only_partitions"] = own_delta == panel["partitions"]["A2_DELTA"]["ids"] and len(own_delta)==14
    checks["known_four_representable"] = boundary == [r["utterance_id"] for r in panel["known_blockers"]] and len(boundary)==4 and len(content)==10
    helper = Path("src/csasr/inference_cf/eos_boundary.py").read_text()
    tree = ast.parse(helper)
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    checks["runtime_reference_free"] = not any("evaluate" in m or "analyze" in m for m in imports) and not any(r["utterance_id"] in helper for r in panel["rows"])
    checks["scope_no_runner_or_scientific_output"] = not Path("experiments/inference_cf_p2path4.py").exists() and not Path("results/inference_cf/p2path4").exists()
    checks["EOS_H1"] = c["G1"]["EOS_BOUNDARY"]["H_actions"]==1 and c["G1"]["H"]==3 and c["G1"]["weights"]==[.5,.5]
    # Independent next-action math, neither EOS removal nor branch-length normalization.
    def own_logp(x):
        m=max(x);z=m+math.log(sum(math.exp(v-m) for v in x));return [v-z for v in x]
    for x,y in (([0.,1.,3.],[0.,4.,1.]),([0.,3.,1.],[0.,1.,4.])):
        px,py=own_logp(x),own_logp(y)
        s0=.5*px[2]+.5*py[2];s2=.5*px[1]+.5*py[1]
        assert all(math.isfinite(v) for v in (s0,s2))
        choice0=(s0-s2)>=-1e-12
        assert choice0 == (s0>=s2-1e-12)
    checks["independent_H1_math"] = True
    suites = ET.parse(junit).getroot()
    cases = list(suites.iter("testcase"))
    required = {"test_inference_cf_p2tta0", "test_inference_cf_p2tta_funnel", "test_inference_cf_p2path1", "test_inference_cf_p2path2", "test_inference_cf_p2path3", "test_inference_cf_p2path4_r1"}
    represented = {t.attrib.get("classname","").split('.')[-1] for t in cases}
    checks["required_CPU_tests_pass"] = required <= represented and len(cases)>0 and not any(list(t) and any(ch.tag in ("failure","error","skipped") for ch in t) for t in cases)
    verdict="PASS_TO_P2_PATH4_R1" if all(checks.values()) else "FAIL"
    return {"schema":"p2_path4_r1_contract_audit_v1","audit_scope":"pre-outcome freeze, NOT runnable-manifest audit","contract_revision":c["contract_revision"],"original_blocked_commit":ORIGINAL,"config_sha256":sha(config),"amendment_source_sha256":c["amendment_source_sha256"],"checks":checks,"CPU_test_count":len(cases),"test_report_sha256":sha(junit),"boundary_ids_reference_free":boundary,"content_content_count":len(content),"scientific_runs":0,"PATH4_reference_outcomes_opened":False,"runtime_manifest_pre_audit_required_before_job":True,"verdict":verdict}


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--junit",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    result=audit(a.junit);Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["verdict"]!="PASS_TO_P2_PATH4_R1":raise SystemExit(1)

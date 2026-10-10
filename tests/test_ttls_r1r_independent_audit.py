"""Independent numerical/alignment checks; no primary evaluator imports."""
from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.ttls_r1r_independent_analysis import lexical
from experiments.ttls_r1_independent_analysis import one
from csasr.inference_cf.ttls_r1r import apply_steering_ratio_first as repaired
from csasr.models.hooks import apply_steering as historical
OUT=ROOT/'results/inference_cf/ttls_r1r_independent_audit'
def test_nonempty_zero_reproduces_original_failure_and_corrected_identity():
    torch.manual_seed(240924);h=torch.randn(2,20,1280,dtype=torch.bfloat16);g=torch.ones(2,20,dtype=h.dtype);g[:,:4]=0;z=torch.zeros(1280)
    assert not torch.equal(h,historical(h,z,1.,1.,g,True))
    assert torch.equal(h,repaired(h,z,1.,1.,g,True))
    assert torch.equal(h,repaired(h,z,0.,1.,g,True))
    assert torch.equal(h,repaired(h,z,1.,1.,torch.zeros_like(g),True))
def test_independent_analytic_zero_gradient_and_finite_difference():
    torch.manual_seed(24);h=torch.randn(2,5,16,dtype=torch.float64);w=torch.randn_like(h);z=torch.zeros(16,dtype=h.dtype,requires_grad=True);mask=torch.tensor([[0,0,1,1,1],[0,1,1,0,1]],dtype=h.dtype)
    loss=(repaired(h,z,1.,1.,mask,True)*w).sum();actual=torch.autograd.grad(loss,z)[0]
    u=h/h.norm(dim=-1,keepdim=True);expected=((w-u*(w*u).sum(-1,keepdim=True))*mask.unsqueeze(-1)).sum((0,1))
    assert torch.allclose(actual,expected,rtol=1e-12,atol=1e-12) and actual.norm()>0
    d=torch.randn_like(z);d/=d.norm();eps=1e-5
    def f(v):return (repaired(h,v,1.,1.,mask,True)*w).sum()
    fd=(f(z.detach()+eps*d)-f(z.detach()-eps*d))/(2*eps)
    assert abs(float(fd-actual@d))<1e-7

def test_native_gradient_exists_without_zero_bypass():
    torch.manual_seed(100);h=torch.randn(1,20,1280,dtype=torch.bfloat16);z=torch.zeros(1280,requires_grad=True);out=repaired(h,z,1.,1.,torch.ones(1,20),True);assert torch.equal(h,out)
    g=torch.autograd.grad(out[...,0].float().sum(),z)[0];assert g.norm()>0 and torch.isfinite(g).all()

def test_numerical_forward_independent_native_expression():
    torch.manual_seed(3);h=torch.randn(1,12,1280,dtype=torch.bfloat16);z=torch.randn(1280)*.01;g=torch.ones(1,12);g[:,:4]=0;v=h+z.to(h.dtype).view(1,1,-1)*g.to(h.dtype).unsqueeze(-1)
    norm=lambda v:torch.sqrt((v.double()**2).sum(-1,keepdim=True)).to(h.dtype)
    # Different reduction implementation should agree within one native rounding step, not a lexical tolerance.
    expected=v*(norm(h)/norm(v).clamp_min(1e-6));expected=torch.where(g.unsqueeze(-1)>0,expected,h)
    assert torch.equal(expected,repaired(h,z,1.,1.,g,True))

def test_saved100_zero_identity_actual_tokens_not_only_flags():
    p=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text())
    for i,s in enumerate(p['rows']):
        r=json.loads((ROOT/f'results/inference_cf/ttls_r1r/run1/integrity/{i:03d}.json').read_text())
        for m,z in r['zero'].items():assert z['tokens']==s['y_B'] and z['text']==s['y_B_text'] and z['terminated']==s['y_B_terminated'] and z['kl0_stable']==0

def test_independent_metrics_count_and_alignment():
    d=json.loads((OUT/'independent_metrics.json').read_text());assert d['published_mismatches']==[]
    for a,m in d['metrics'].items():
        assert m['mixed_errors']==sum(r['systems'][a]['counts'][2] for r in d['rows'])==m['sub']+m['del']+m['ins']
        assert m['poi_errors']==sum(r['systems'][a]['counts'][0] for r in d['rows'])
        assert d['metrics']['B0']['poi_errors']-m['poi_errors']==d['transitions'][a]['corrections']-d['transitions'][a]['corruptions']
    assert d['lexical']['T1']['genuine_wrong_language']==4 and d['genuine_dialogues']['T1']==1

def test_historical_kernels_and_results_untouched():
    original=json.loads((ROOT/'results/inference_cf/ttls_r1/run1/manifest.json').read_text());seal=json.loads((ROOT/'results/inference_cf/ttls_r1/output_seal.json').read_text())
    for p,h in {**original['sources'],**seal['files']}.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h.removeprefix('sha256:')

def test_lexical_separation_not_promoted_boundary_repairs():
    e=lambda cat,ref,before:{'canonical_category':cat,'reference':ref,'before':before}
    assert lexical(e('same_language_substitution','people','peoplethey'))=='word_boundary_repair'
    assert lexical(e('wrong_language_substitution','cooking','烧烤'))=='genuine_wrong_language'
    assert lexical(e('deletion','topic',''))=='deletion_or_boundary'

def test_independent_complete_site_gradient_includes_ffn_residual_skip():
    sys.path.insert(0,str(ROOT/'tests'))
    from test_ttls_r1 import _bundle,_encode,LAYER,CB
    from csasr.inference_cf import ttls as L,ttls_r1r as R
    from csasr.inference_cf.episodic_tta import teacher_logits
    from experiments.ttls_r1r_independent_replay import independent_repair
    b=_bundle();_,enc=_encode(b);y=[20,21,22,23];ep=R.EpisodeR1R(b,enc,L.ALL,layer=LAYER)
    _,loss=L.ce_terms(ep,y,[True]*4,[],[],2,prompt=CB);g=torch.autograd.grad(loss,ep.z)[0]
    z=torch.zeros(b.d_model,requires_grad=True);layer=b.model.model.decoder.layers[LAYER];cap={}
    def pre(mod,args):cap['q']=args[0]
    def intervention(mod,args,output):
        u=output[0];h=cap['q']+u;gain=(torch.arange(h.shape[1])>=4).to(h.dtype).view(1,-1);out=u+(independent_repair(h,z,gain)-h)
        return (out,)+output[1:]
    class Direct:
        params={'z':z}
        def forward(self,prompt,y):
            hooks=[layer.encoder_attn_layer_norm.register_forward_pre_hook(pre),layer.encoder_attn.register_forward_hook(intervention)]
            try:return teacher_logits(b.model,enc,prompt,y,None)
            finally:
                for hook in hooks:hook.remove()
    _,li=L.ce_terms(Direct(),y,[True]*4,[],[],2,prompt=CB);gi=torch.autograd.grad(li,z)[0]
    assert torch.equal(loss,li) and torch.equal(g,gi)

import copy
import io
import pytest
import torch
from transformers import WhisperConfig,WhisperForConditionalGeneration
from reft.a5000_campaign.methods import Allocation,RepresentationAdapter,EncoderWeightLoRA,energy_coefficients
from reft.auto_policy import build_tf_batch
from reft.a5000.objectives import weighted_ce


def tiny():
    c=WhisperConfig(vocab_size=32,num_mel_bins=4,d_model=16,encoder_layers=2,decoder_layers=2,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=10,max_target_positions=24,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        decoder_start_token_id=1,suppress_tokens=[],begin_suppress_tokens=[])
    m=WhisperForConditionalGeneration(c).eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


@pytest.mark.parametrize('method',['ereft','lra'])
def test_identity_count_gradient(method):
    a=RepresentationAdapter(method);a.lengths=[7]
    h=torch.randn(1,10,1280)
    assert a.count()==5122 and torch.equal(a(h),h)
    a(h).square().sum().backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in a.parameters())
    if method=='ereft':assert torch.allclose(a.reft.rot.weight@a.reft.rot.weight.T,torch.eye(2),atol=1e-6)


@pytest.mark.parametrize('epsilon',[0,.25,.99])
@pytest.mark.parametrize('magnitude',[0,1e-15,1,1e15])
def test_energy_and_padding(epsilon,magnitude):
    q=torch.randn(2,3,requires_grad=True); e=torch.tensor([[1.,2.,0.],[3.,2.,1.]],dtype=torch.float64)*magnitude
    n=torch.tensor([[3,1,0],[3,3,2]])
    a=energy_coefficients(q,e,n,epsilon,alpha=1.5)
    assert torch.isfinite(a).all() and (a[n>0]>0).all() and (a[n==0]==0).all()
    assert torch.allclose((a.double().square()*e).sum(-1),2.25*e.sum(-1),rtol=1e-6,atol=1e-29)
    a.sum().backward();assert torch.isfinite(q.grad).all()
    z=energy_coefficients(torch.zeros_like(q),e,n,epsilon,1)
    assert torch.equal(z[n>0],torch.ones_like(z[n>0]))


def test_invalid_lengths_energy():
    for e,n in [(torch.tensor([[-1.]]),torch.ones(1,1)),(torch.ones(1,1),torch.zeros(1,1))]:
        with pytest.raises(ValueError):energy_coefficients(torch.zeros(1,1),e,n)
    a=RepresentationAdapter(d=16);a.lengths=[0]
    with pytest.raises(ValueError):a(torch.randn(1,10,16))


@pytest.mark.parametrize('kind',['G2','G3','G4','G5','G6','G7','G9'])
def test_gate_warm_identity_padding_gradient_serialization(kind):
    torch.manual_seed(42);g=Allocation(kind,block_frames=3)
    assert sum(p.numel() for p in g.parameters())==289
    h=torch.randn(1,10,16);d=torch.randn_like(h)*.01;p=torch.randn(1,10,2)
    y=g(h,d,p,[7]);assert torch.equal(y[:,:7],d[:,:7]);assert torch.equal(y[:,7:],torch.zeros_like(y[:,7:]))
    (y*torch.randn_like(y)).sum().backward()
    assert g.net[-1].weight.grad.norm()>0
    h2=h.clone();h2[:,7:]=1000
    assert torch.equal(g(h2,d,p,[7]),y)
    b=io.BytesIO();torch.save(g.state_dict(),b);b.seek(0);copy=Allocation(kind,3);copy.load_state_dict(torch.load(b,weights_only=True))
    assert torch.equal(copy(h,d,p,[7]),y)


def test_post_final_norm_hook_free_generation_frozen():
    m=tiny();a=RepresentationAdapter('ereft',d=16);a.lengths=[7]
    features=torch.randn(1,4,20);seen={}
    norm=m.model.encoder.layer_norm.register_forward_hook(lambda mod,args,out:seen.update(norm=out.detach().clone()))
    enc=m.model.encoder(features).last_hidden_state
    assert torch.equal(enc,seen['norm'])
    X=torch.tensor([[1,4,5,6]])
    with a.active(m):
        with pytest.raises(RuntimeError):
            with a.active(m):pass
        o=m(input_features=features,decoder_input_ids=X);o.logits.square().sum().backward()
    assert a.calls==1 and all(p.grad is None for p in m.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in a.parameters())
    a.calls=0
    with a.active(m):m.generate(features,num_beams=2,max_new_tokens=3)
    assert a.calls==1
    norm.remove()


def test_loss_shift_and_accumulation():
    X,Y=build_tf_batch([[1,3,4,5]],[[8,9,2]],2,0)
    assert X.tolist()==[[1,3,4,5,8,9]] and Y.tolist()==[[-100,-100,-100,8,9,2]]
    z=torch.randn(2,6,12,requires_grad=True); y=Y.expand(2,-1)
    plain=weighted_ce(z,y);ones=weighted_ce(z,y,torch.ones_like(y),1)
    assert torch.equal(plain,ones)
    accum=sum(weighted_ce(z[i:i+1],y[i:i+1],denominator=6) for i in range(2))
    assert torch.allclose(accum,plain)
    g=torch.ones_like(y);weighted=weighted_ce(z,y,g,2)
    assert torch.allclose(weighted,plain)


def test_lora_encoder_targets_identity_gradient():
    m=tiny();a=EncoderWeightLoRA(m);f=torch.randn(1,4,20);old=m.model.encoder(f).last_hidden_state
    with a.active():
        out=m.model.encoder(f).last_hidden_state
        assert torch.equal(out,old);out.square().sum().backward()
    assert all('l1_' in k for k in a.wrapped)
    assert all(w.B.grad is not None and w.B.grad.norm()>0 for w in a.wrapped.values())
    assert all(p.grad is None for p in m.parameters())

@pytest.mark.parametrize('method',['ereft','lra'])
def test_exact_optimizer_boundary_resume(method):
    torch.manual_seed(8);a=RepresentationAdapter(method,d=16);a.lengths=[7]
    opt=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=0)
    def step(m,o,x):
        m.lengths=[7];o.zero_grad(set_to_none=True);m(x).square().mean().backward();o.step()
    x=torch.randn(1,10,16);step(a,opt,x)
    buf=io.BytesIO();torch.save({'adapter':a.state_dict(),'optimizer':opt.state_dict()},buf);buf.seek(0)
    s=torch.load(buf,weights_only=True);b=RepresentationAdapter(method,d=16)
    b.load_state_dict(s['adapter']);ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=0);ob.load_state_dict(s['optimizer'])
    step(a,opt,x);step(b,ob,x)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_padding_independent_with_nonconstant_gate_and_single_block():
    torch.manual_seed(7);g=Allocation('G5',3)
    with torch.no_grad():g.net[-1].weight.normal_()
    h=torch.randn(1,10,16);d=torch.randn_like(h);p=torch.randn(1,10,2)
    y=g(h,d,p,[7]);h[:,7:]=float('1e5');p[:,7:]=-500;d[:,7:]=40
    assert torch.equal(y,g(h,d,p,[7]))
    short=g(h[:,:2],d[:,:2],p[:,:2],[2])
    assert torch.allclose(short,d[:,:2],atol=1e-6)


def test_weighted_denominator_matches_logical_batch():
    z=torch.randn(2,5,20);y=torch.tensor([[-100,1,2,3,4],[-100,5,6,7,8]])
    g=torch.tensor([[-100,1,0,-100,1],[-100,0,0,1,-100]])
    den=float(((y!=-100)*torch.where(g==1,3.,1.)).sum())
    full=weighted_ce(z,y,g,3)
    micro=sum(weighted_ce(z[i:i+1],y[i:i+1],g[i:i+1],3,denominator=den) for i in range(2))
    assert torch.allclose(full,micro)

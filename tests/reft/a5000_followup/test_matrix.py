import pytest
import torch
from transformers import WhisperConfig,WhisperForConditionalGeneration

from reft.a5000_followup.matrix import PracticalWeightLoRA,logical_loss
from reft.a5000.objectives import LIDHead,weighted_ce


def test_weight_lora_capacity_identity_gradient_and_restore():
    c=WhisperConfig(vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=1,decoder_layers=2,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=8,max_target_positions=16,pad_token_id=2,eos_token_id=2,decoder_start_token_id=1)
    m=WhisperForConditionalGeneration(c).eval();m.requires_grad_(False)
    a=PracticalWeightLoRA(m,rank=2)
    assert a.count()==2*4*2*16*2
    assert 32*4*2*1280*8==2621440
    enc=torch.randn(1,8,16);X=torch.tensor([[1,5,7,8]])
    base=m(encoder_outputs=(enc,),decoder_input_ids=X,use_cache=False).logits
    with a.active():z=m(encoder_outputs=(enc,),decoder_input_ids=X,use_cache=False).logits
    assert torch.equal(base,z)
    z.sum().backward()
    assert all(p.grad is None for p in m.parameters())
    assert any(w.B.grad.norm()>0 for w in a.wrapped.values())
    assert all('.base.' not in k for k in a.adaptation_state())
    b=PracticalWeightLoRA(m,rank=2);b.restore(a.adaptation_state())
    assert all(torch.equal(v,b.adaptation_state()[k]) for k,v in a.adaptation_state().items())


@pytest.mark.parametrize('objective',['O1','O2','O3'])
def test_logical_objective_normalization_matches_unsplit_batch(objective):
    torch.manual_seed(1)
    z=torch.randn(2,5,7,requires_grad=True);y=torch.randint(0,7,(2,5));y[1,3:]=-100
    g=torch.tensor([[-100,0,1,0,-100],[-100,1,0,-100,-100]])
    states=torch.randn(2,5,4,requires_grad=True);head=LIDHead(4)
    td=int((y!=-100).sum());ld=int((g!=-100).sum());wd=float(((y!=-100)*torch.where(g==1,2.,1.)).sum())
    full=logical_loss(z,y,g,objective,td,wd,ld,states,head,.1,2.)
    split=sum(logical_loss(z[i:i+1],y[i:i+1],g[i:i+1],objective,td,wd,ld,
        states[i:i+1],head,.1,2.) for i in range(2))
    assert torch.allclose(full,split,atol=1e-7)
    full.backward();assert torch.isfinite(z.grad).all()
    if objective=='O2':assert states.grad is not None and states.grad.norm()>0


@pytest.mark.parametrize('method',['ereft','lra','practical_weight_lora'])
@pytest.mark.parametrize('objective',['O1','O2','O3'])
def test_real_tiny_training_path_for_each_matrix_cell(method,objective):
    from reft.a5000.adapters import EncoderAdapter
    torch.manual_seed(42)
    c=WhisperConfig(vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=1,decoder_layers=2,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=8,max_target_positions=16,pad_token_id=2,eos_token_id=2,decoder_start_token_id=1)
    c._attn_implementation='eager'
    m=WhisperForConditionalGeneration(c).eval();m.requires_grad_(False)
    enc=torch.randn(1,8,16);X=torch.tensor([[1,5,6,7,9,11]])
    Y=torch.tensor([[-100,-100,-100,9,11,2]]);groups=torch.tensor([[-100,-100,-100,0,1,-100]])
    a=PracticalWeightLoRA(m,rank=2) if method=='practical_weight_lora' else EncoderAdapter(method,16)
    head=LIDHead(16)
    if method=='practical_weight_lora':
        with a.active():o=m(encoder_outputs=(enc,),decoder_input_ids=X,use_cache=False,output_hidden_states=True)
    else:o=m(encoder_outputs=(a(enc),),decoder_input_ids=X,use_cache=False,output_hidden_states=True)
    loss=logical_loss(o.logits,Y,groups,objective,3,4,2,o.decoder_hidden_states[-1],head,.1,2.)
    assert torch.isfinite(loss)
    loss.backward()
    adaptation=[p for p in a.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adaptation)
    assert any(p.grad.norm()>0 for p in adaptation)
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('method',['ereft','lra'])
def test_optimizer_boundary_resume_matches_uninterrupted(method,tmp_path):
    from reft.a5000.adapters import EncoderAdapter
    torch.manual_seed(42)
    h=torch.randn(1,5,16);target=torch.randn(1,5,16)
    def make():
        # The published Householder parametrization also has a full orthogonal buffer.
        # Match the runner's global seed immediately before construction, not only the QR generator seed.
        torch.manual_seed(42)
        a=EncoderAdapter(method,16)
        return a,torch.optim.AdamW(a.parameters(),lr=.001,betas=(.9,.99),eps=1e-6,weight_decay=0.)
    a,opt=make();b,opb=make()
    def step(ad,optimizer):
        optimizer.zero_grad(set_to_none=True)
        ((ad(h)-target)**2).mean().backward();optimizer.step()
    for _ in range(4):step(a,opt)
    for _ in range(2):step(b,opb)
    path=tmp_path/'resume.pt';torch.save({'adapter':b.state_dict(),'optimizer':opb.state_dict()},path)
    c,opc=make();state=torch.load(path,weights_only=True)
    c.load_state_dict(state['adapter']);opc.load_state_dict(state['optimizer'])
    for _ in range(2):step(c,opc)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())

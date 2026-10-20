import copy

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from reft.a5000.adapters import EncoderAdapter
from reft.a5000.data import integrity
from reft.a5000.diagnostics import decompose, controlled_readout
from reft.a5000.objectives import LIDHead, objective, script_group, goldkl
from reft.auto_policy import build_tf_batch, tf_loss


@pytest.fixture
def tiny():
    torch.manual_seed(42)
    c = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_source_positions=8, max_target_positions=32, pad_token_id=2, eos_token_id=2,
        bos_token_id=1, decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[])
    c._attn_implementation = 'eager'
    m = WhisperForConditionalGeneration(c).eval()
    m.requires_grad_(False)
    g = m.generation_config
    g.no_timestamps_token_id = 45
    g.task_to_id = {'transcribe': 46}
    g.lang_to_id = {'<|en|>': 47}
    g.is_multilingual = True
    return m, torch.randn(1, 8, 16), torch.tensor([[1,47,46,45,9,11]])


@pytest.mark.parametrize('method', ['ereft', 'lra'])
def test_parameter_count_identity_and_alpha_zero(method):
    a = EncoderAdapter(method)
    assert a.count() == 5122
    h = torch.randn(2,13,1280)
    assert torch.equal(a(h),h)
    with torch.no_grad():
        (a.reft.learned.bias if method == 'ereft' else a.c).add_(1)
    assert not torch.equal(a(h),h)
    assert torch.equal(a(h,0),h)
    if method == 'ereft':
        r = a.reft.rot.weight
        assert torch.allclose(r@r.T,torch.eye(2),atol=1e-6)


@pytest.mark.parametrize('method', ['ereft','lra'])
def test_site_exactly_after_final_norm_and_single_application(tiny,method):
    m,x,X = tiny
    a = EncoderAdapter(method,16)
    with torch.no_grad():
        (a.reft.learned.bias if method=='ereft' else a.c).fill_(0.3)
    seen = {}
    norm = m.model.encoder.layer_norm.register_forward_hook(lambda mod,args,out: seen.update(norm=out.detach().clone()))
    decoder = m.model.decoder.register_forward_pre_hook(lambda mod,args,kw: seen.update(dec=kw['encoder_hidden_states'].detach().clone()),with_kwargs=True)
    with torch.no_grad(),a.active(m):
        m(input_features=x,attention_mask=torch.ones(1,16,dtype=torch.long),decoder_input_ids=X)
        assert a.calls==1
        assert torch.equal(seen['dec'],a(seen['norm']))
        # Would fail for last encoder-block output or decoder ReFT: exact normalized encoder interface is required.
        assert not torch.equal(seen['dec'],seen['norm'])
        with pytest.raises(RuntimeError):
            with a.active(m):pass
    norm.remove(); decoder.remove()
    assert not m.model.encoder._forward_hooks


@pytest.mark.parametrize('method',['ereft','lra'])
def test_gradient_isolation_cached_native_and_serialization(tiny,method,tmp_path):
    m,x,X=tiny
    a=EncoderAdapter(method,16)
    Y=torch.tensor([[-100]*3+[9,11,2]])
    with torch.no_grad(): enc=m.model.encoder(x).last_hidden_state
    with a.active(m): z_native=m(input_features=x,decoder_input_ids=X,use_cache=False).logits
    z_cached=m(encoder_outputs=(a(enc),),decoder_input_ids=X,use_cache=False).logits
    assert torch.equal(z_native,z_cached)
    opt=torch.optim.AdamW(a.parameters(),lr=0.01,weight_decay=0)
    for step in range(2):
        opt.zero_grad()
        z=m(encoder_outputs=(a(enc),),decoder_input_ids=X,use_cache=False).logits
        loss=tf_loss(z,Y)[0]
        loss.backward()
        assert all(p.grad is None for p in m.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        if step or method=='ereft': assert all(p.grad.norm()>0 for p in a.parameters())
        else: assert a.A.grad.norm()>0 and a.c.grad.norm()>0
        opt.step()
    if method=='ereft':assert torch.allclose(a.reft.rot.weight@a.reft.rot.weight.T,torch.eye(2),atol=1e-5)
    path=tmp_path/'a.pt'; torch.save(a.state_dict(),path)
    b=EncoderAdapter(method,16); b.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(a(enc),b(enc))


@pytest.mark.parametrize('method',['ereft','lra'])
def test_native_free_generation_once_and_cached_consistency(tiny,method):
    m,x,X=tiny
    a=EncoderAdapter(method,16)
    kw=dict(language='en',task='transcribe',return_timestamps=False,max_new_tokens=4,num_beams=2,do_sample=False)
    with torch.no_grad():
        plain=m.generate(input_features=x,**kw)
        with a.active(m): native=m.generate(input_features=x,**kw)
        assert a.calls==1 and torch.equal(plain,native)
        with torch.no_grad(): (a.reft.learned.bias if method=='ereft' else a.c).fill_(0.5)
        with a.active(m): native=m.generate(input_features=x,**kw)
        enc=m.model.encoder(x).last_hidden_state
        from transformers.modeling_outputs import BaseModelOutput
        cached=m.generate(input_features=x,encoder_outputs=BaseModelOutput(last_hidden_state=a(enc)),**kw)
        assert torch.equal(native,cached)


def test_objectives_masks_reduction_and_lid_path(tiny):
    m,x,X=tiny
    Y=torch.tensor([[-100]*3+[9,11,2]])
    groups=torch.tensor([[-100]*3+[0,1,-100]])
    a=EncoderAdapter('lra',16)
    with a.active(m): o=m(input_features=x,decoder_input_ids=X,use_cache=False,output_hidden_states=True)
    ce=tf_loss(o.logits,Y)[0]
    assert torch.equal(objective(o.logits,Y,lid_lambda=0),ce)
    assert torch.equal(objective(o.logits,Y,groups=groups,english_weight=1),ce)
    head=LIDHead(16)
    aux=head(o.decoder_hidden_states[-1],groups)
    aux.backward(retain_graph=True)
    assert a.A.grad is not None and a.A.grad.norm()>0
    assert all(p.grad is None for p in m.parameters())
    assert goldkl(o.logits,o.logits,Y).abs()<1e-6


def test_exact_logical_batch_accumulation():
    from reft.a5000.objectives import weighted_ce
    z=torch.randn(2,7,13,requires_grad=True); y=torch.randint(0,13,(2,7)); y[1,4:]=-100
    total=(y!=-100).sum()
    full=weighted_ce(z,y)
    split=sum(weighted_ce(z[i:i+1],y[i:i+1],denominator=total) for i in range(2))
    assert torch.allclose(full,split,atol=1e-7)
    g=torch.tensor([[0,1,-100,0,1,-100,0]]*2)
    w=(y!=-100)*torch.where(g==1,2.,1.)
    assert torch.allclose(weighted_ce(z,y,g,2),sum(weighted_ce(z[i:i+1],y[i:i+1],g[i:i+1],2,denominator=w.sum()) for i in range(2)))


@pytest.mark.parametrize('s,expected',[('我们',0),(' hello',1),('123',-100),('model中文',-100),('，',-100),('\ufffd',-100)])
def test_script_proxy(s,expected):assert script_group(s)==expected


def test_d1_exact_decomposition_and_target_alignment():
    z=torch.randn(1,4,7); y=torch.tensor([[-100,-100,3,5]]); groups=torch.tensor([-100,0,0,1,1,1,-100])
    d=decompose(z,y,groups)
    assert len(d['gold_nll'])==2
    assert torch.allclose(d['gold_nll'],d['group_nll']+d['within_group_nll'])
    assert (d['within_group_nll']>=-1e-6).all()


def test_role_integrity_and_confirmation_firewall():
    from reft.a5000.data import load_rows
    with pytest.raises(ValueError):load_rows('A5000_C150')
    row=dict(utterance_id='x',speaker_id='s',dialogue_id='d',audio_full_sha256='h',official_split='train')
    with pytest.raises(AssertionError):integrity({'T':[row],'V':[row]})


def test_d2_disjointness_and_fixed_refit_prefix_controls():
    g=torch.Generator().manual_seed(42)
    x=torch.randn(40,4,generator=g);y=(x[:,0]>0).long()
    r=controlled_readout(x[:30],x[:30]*2,y[:30],x[30:],x[30:]*2,y[30:],
                         x[:30,1:],x[30:,1:],['train'],['held'])
    assert set(r)>={'base_heldout','fixed_probe_on_adapted','refitted_probe_on_adapted','prefix_only'}
    with pytest.raises(ValueError):controlled_readout(x,x,y,x,x,y,x,x,['same'],['same'])

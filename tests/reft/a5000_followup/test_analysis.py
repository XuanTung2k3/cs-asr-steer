import torch
from transformers import WhisperConfig,WhisperForConditionalGeneration

from reft.a5000_followup.analysis import predictive_states,prefix_only_states,language_lexical_report


def test_predictive_mask_keeps_notimestamps_state_not_current_target_embedding():
    states=torch.arange(6.).reshape(1,6,1)
    y=torch.tensor([[-100,-100,-100,9,11,2]])
    g=torch.tensor([[-100,-100,-100,0,1,-100]])
    s,z=predictive_states(states,y,g)
    assert s[:,0].tolist()==[3.,4.] and z.tolist()==[0,1]


def test_group_mass_and_lexical_probabilities_multiply_to_gold():
    z=torch.randn(1,4,7);y=torch.tensor([[-100,-100,3,5]]);groups=torch.tensor([-100,0,0,1,1,1,-100])
    r=language_lexical_report(z,y,groups)
    assert torch.allclose(r['group_probability_mass']*r['within_group_gold_probability'],(-r['gold_nll']).exp())


def test_prefix_only_decoder_uses_no_utterance_information():
    c=WhisperConfig(vocab_size=64,num_mel_bins=8,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=8,max_target_positions=16,pad_token_id=2,eos_token_id=2,decoder_start_token_id=1)
    m=WhisperForConditionalGeneration(c).eval();m.requires_grad_(False)
    X=torch.tensor([[1,5,6,7,9]])
    a=prefix_only_states(m,X,8);b=prefix_only_states(m,X,8)
    assert torch.equal(a,b) and a.shape==(1,5,16)
    X2=X.clone();X2[0,-1]=10
    altered=prefix_only_states(m,X2,8)
    assert torch.equal(a[:,:-1],altered[:,:-1])

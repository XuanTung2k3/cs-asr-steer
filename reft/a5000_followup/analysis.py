"""Bounded D1/D2 extraction foundations. No probes or GPU analyses are launched on import."""
import torch
from transformers.modeling_outputs import BaseModelOutput

from reft.a5000.diagnostics import decompose,controlled_readout
from reft.a5000.objectives import token_groups
from reft.a5000.runner import score


def vocabulary_groups(tokenizer,vocab_size):
    # Three exhaustive groups: conservative Han, Latin and OTHER (including partial bytes/specials).
    return token_groups(tokenizer,list(range(vocab_size)))


def language_lexical_report(logits,labels,vocab_groups):
    d=decompose(logits,labels,vocab_groups)
    return {**d,'group_probability_mass':(-d['group_nll']).exp(),
            'within_group_gold_probability':(-d['within_group_nll']).exp(),
            'proxy':'Tokenizer script groups, not semantic language identity'}


def predictive_states(states,labels,target_groups):
    """States must be decoder outputs for X=prompt+target[:-1]; never shift them onto y_t embeddings."""
    assert states.shape[:2]==labels.shape==target_groups.shape
    valid=(labels!=-100)&(target_groups!=-100)
    return states[valid],target_groups[valid]


def prefix_only_states(model,decoder_input_ids,encoder_length=1500):
    """Gold-prefix shortcut baseline with constant zero encoder memory; declare this ablation explicitly.

    It removes utterance-specific acoustic information, preserving decoder/prefix architecture.
    Returns states at predictive positions; caller applies the SAME label mask as audio-conditioned states.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Prefix baseline requires the frozen backbone with adaptation disabled')
    p=next(model.parameters())
    enc=torch.zeros(decoder_input_ids.shape[0],encoder_length,model.config.d_model,device=p.device,dtype=p.dtype)
    with torch.no_grad():
        o=model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),decoder_input_ids=decoder_input_ids,
                use_cache=False,output_hidden_states=True)
    return o.decoder_hidden_states[-1]


def linked_decoding_report(rows,baseline_predictions,adapted_predictions,checkpoint_sha256,source_commit,manifest_sha256):
    """Canonical decoded corrections/damage alongside D1, with explicit artifact identities."""
    assert set(baseline_predictions)==set(adapted_predictions)=={r['utterance_id'] for r in rows}
    return {'decoded':score(rows,adapted_predictions,baseline_predictions),
            'source_commit':source_commit,'checkpoint_sha256':checkpoint_sha256,'manifest_sha256':manifest_sha256,
            'interpretation':'Decode changes must be related empirically to group/lexical changes; readout alone is not causal evidence'}


__all__=['vocabulary_groups','language_lexical_report','predictive_states','prefix_only_states',
         'linked_decoding_report','controlled_readout']

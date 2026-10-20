"""Bounded engineering-only checks for the prepared nine-cell matrix, after the primary pilot.

One training-side utterance and one optimizer step per cell; no development decoding, quality selection,
epochs or objective-matrix scientific results. One backbone and one GPU only.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from transformers.modeling_outputs import BaseModelOutput

from csasr.models.whisper import load_whisper
from reft.provenance import weights_hash
from reft.a5000.adapters import EncoderAdapter
from reft.a5000.data import config,runtime,output,load_rows,sha,atomic
from reft.a5000.objectives import LIDHead
from reft.a5000.runner import Runtime,Budget,Paused,check,git,memory,save_torch
from reft.a5000_followup.matrix import PracticalWeightLoRA,groups_for_labels,logical_loss


def main():
    root=output();dest=root/'objective_engineering_preflight'
    assert (root/'comparison.json').exists(),'Finish the matched O1 pilot first'
    assert not git('status','--porcelain'),'Commit reviewed preflight source first'
    if (dest/'report.json').exists():print('Engineering matrix preflight already complete');return
    dest.mkdir(exist_ok=True)
    with (root/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0'),
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU occupied: '+occupied)
        budget=Budget('objective_engineering_preflight');start=time.monotonic();status='FAILED'
        try:
            R=object.__new__(Runtime);R.C=config();R.R=runtime();R.freeze=check();R.budget=budget
            R.b=load_whisper({'model':{'id':str(Path(R.R['model']['root']).expanduser()),'dtype':'bfloat16',
                'device':'cuda','local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
            assert R.b.revision=='sha256:'+R.R['model']['sha256']
            R.eos=R.b.processor.tokenizer.eos_token_id
            R.fingerprint=hashlib.sha256(json.dumps({'model':R.R['model'],'data':R.freeze['role_manifest_sha256'],
                'processor':sha(Path(R.R['model']['root']).expanduser()/'tokenizer_config.json')},sort_keys=True).encode()).hexdigest()
            row=next(r for r in load_rows('A5000_T394') if r['stratum']=='CS')
            w0=weights_hash(R.b.model);cells={}
            for method in ['ereft','lra','practical_weight_lora']:
                for objective in ['O1','O2','O3']:
                    budget.tick()
                    if time.monotonic()-start>600:raise Paused('Ten-minute engineering bound')
                    torch.manual_seed(42)
                    ad=(PracticalWeightLoRA(R.b.model) if method=='practical_weight_lora' else EncoderAdapter(method)).to('cuda')
                    torch.manual_seed(42);head=LIDHead(1280).to('cuda') if objective=='O2' else None
                    params=[p for p in ad.parameters() if p.requires_grad]+(list(head.parameters()) if head else [])
                    opt=torch.optim.AdamW(params,lr=.0001,betas=(.9,.99),eps=1e-6,weight_decay=0.)
                    d,enc,X,Y=R.inputs(row);groups=groups_for_labels(R.b.processor.tokenizer,d['targets'],Y)
                    td=int((Y!=-100).sum());ld=int((groups!=-100).sum())
                    wd=float(((Y!=-100)*torch.where(groups==1,2.,1.)).sum())
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                    ctx=ad.active() if method=='practical_weight_lora' else contextlib.nullcontext()
                    with ctx:
                        h=enc if method=='practical_weight_lora' else ad(enc)
                        o=R.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=h),decoder_input_ids=X,
                            attention_mask=d['mask'].to('cuda'),use_cache=False,output_hidden_states=head is not None)
                    loss=logical_loss(o.logits,Y,groups,objective,td,wd,ld,
                        states=o.decoder_hidden_states[-1] if head else None,head=head,lid_lambda=.1,english_weight=2.)
                    assert torch.isfinite(loss);loss.backward()
                    assert all(p.grad is None for p in R.b.model.parameters())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    assert any(p.grad.norm()>0 for p in ad.parameters() if p.requires_grad)
                    opt.step();assert all(torch.isfinite(p).all() for p in params)
                    metrics=memory();del o
                    # Prove that the auxiliary term itself reaches adaptation, independent of ASR CE.
                    aux_reaches=None
                    if head:
                        opt.zero_grad(set_to_none=True)
                        ctx=ad.active() if method=='practical_weight_lora' else contextlib.nullcontext()
                        with ctx:
                            h=enc if method=='practical_weight_lora' else ad(enc)
                            o=R.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=h),decoder_input_ids=X,
                                use_cache=False,output_hidden_states=True)
                        aux=head(o.decoder_hidden_states[-1],groups);aux.backward()
                        aux_reaches=any(p.grad is not None and p.grad.norm()>0 for p in ad.parameters() if p.requires_grad)
                        assert aux_reaches and torch.isfinite(aux)
                        del o,aux
                    state=ad.adaptation_state() if method=='practical_weight_lora' else {k:v.detach().cpu() for k,v in ad.state_dict().items()}
                    path=dest/f'{method}_{objective}.pt'
                    save_torch(path,{'adapter':state,'head':head.state_dict() if head else {}})
                    restored=torch.load(path,map_location='cpu',weights_only=True)
                    assert all(torch.equal(v.cpu(),restored['adapter'][k]) for k,v in state.items())
                    metrics=memory()
                    cells[f'{method}_{objective}']={'passed':True,'loss':float(loss.detach()),'optimizer_steps':1,
                        'adaptation_parameters':ad.count(),'auxiliary_parameters':sum(p.numel() for p in head.parameters()) if head else 0,
                        'auxiliary_gradient_reaches_adaptation':aux_reaches,'seconds':time.monotonic()-t,
                        'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':sha(path),**metrics}
                    del ad,head,opt,params,loss,state,restored,h,enc,X,Y
            assert weights_hash(R.b.model)==w0
            atomic(dest/'report.json',{'source_commit':git('rev-parse','HEAD'),'primary_source_commit':R.freeze['source_commit'],
                'scope':'engineering only; one training-side step per cell; not an objective study or quality result',
                'cells':cells,'backbone_unchanged':True,'seconds':time.monotonic()-start,'C150_evaluated':False})
            print(json.dumps(cells),flush=True);status='COMPLETE'
        finally:budget.close(status)


if __name__=='__main__':main()

"""CPU-only waveform identity and exact tokenizer/processor target parity; no model evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import WhisperProcessor,WhisperTokenizer

from csasr.models.whisper import load_audio
from reft.text import target_text
from reft.a5000.data import ROOT,config,runtime,output,atomic,sha,load_rows,original_manifest
from reft.a5000.runner import git


def main():
    model=str(Path(runtime()['model']['root']).expanduser())
    t=WhisperTokenizer.from_pretrained(model,local_files_only=True)
    p=WhisperProcessor.from_pretrained(model,local_files_only=True)
    n=0
    for role in ['A5000_T394','A5000_V106']:
        for row in load_rows(role):
            text=target_text(row['transcript_raw'])
            a=t.encode(text,add_special_tokens=False)+[t.eos_token_id]
            b=p.tokenizer.encode(text,add_special_tokens=False)+[p.tokenizer.eos_token_id]
            assert a==b and len(b)==row['target_tokens_incl_eos'];n+=1
    # C150 is touched only for audio identity/metadata, never target tokenization or model decoding here.
    spec=json.loads((ROOT/config()['manifest']).read_text())
    paths=pd.read_parquet(original_manifest(),columns=['utterance_id','audio_path']).set_index('utterance_id').audio_path.to_dict()
    seen={};records={}
    for role,rows in spec['roles'].items():
        for row in rows:
            pcm=load_audio(paths[row['utterance_id']],16000)
            digest=hashlib.sha256(np.asarray(pcm,dtype='<f4').tobytes()).hexdigest()
            assert digest not in seen or seen[digest][0]==role,(seen.get(digest),role)
            seen[digest]=(role,row['utterance_id']);records[row['utterance_id']]=digest
    audit={'review_commit':git('rev-parse','HEAD'),'source_commit':json.loads((output()/'freeze.json').read_text())['source_commit'],
        'target_sequences_verified':n,'processor_tokenizer':type(p.tokenizer).__name__,
        'role_manifest_sha256':sha(ROOT/config()['manifest']),'waveforms_checked':len(records),
        'cross_role_duplicate_waveforms':0,'waveform_hashes':records,'C150_model_evaluated_by_this_audit':False}
    atomic(output()/'audits/data_and_target_integrity.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='waveform_hashes'},indent=2))


if __name__=='__main__':main()

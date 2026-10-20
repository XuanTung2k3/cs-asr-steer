"""Post-final-LN adapters and energy-conserving audio-only allocation.

New A5000 implementation; not authenticated historical H100 code. FP32
adapter arithmetic, BF16 interface cast. Energy invariant is measured before
that cast; quantization drift is reported separately, never called exact.
"""
import contextlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from reft.a5000.adapters import EncoderAdapter
from reft.adapters import LoRALinear


def frame_mask(h, lengths):
    lengths = torch.as_tensor(lengths, device=h.device)
    if lengths.shape != (h.shape[0],) or bool(((lengths <= 0) | (lengths > h.shape[1])).any()):
        raise ValueError('Invalid encoder lengths')
    return torch.arange(h.shape[1], device=h.device)[None, :] < lengths[:, None]


def energy_coefficients(q, block_energy, counts, epsilon=.25, alpha=1., normalize=True):
    if not 0 <= epsilon < 1 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid allocation scalars')
    if q.shape != block_energy.shape or q.shape != counts.shape:
        raise ValueError('Block shape mismatch')
    if not bool(torch.isfinite(q).all() and torch.isfinite(block_energy).all()):
        raise FloatingPointError('Nonfinite gate or energy')
    if bool((block_energy < 0).any() or (counts < 0).any() or (counts.sum(-1) <= 0).any()):
        raise ValueError('Invalid/empty blocks')
    valid = counts > 0
    if bool(((~valid) & (block_energy != 0)).any()):
        raise ValueError('Padding energy must be zero')
    # Ratio in float64 avoids underflow for tiny, legitimate updates. No clipping
    # or additive epsilon changes the energy constraint.
    u = 1 + epsilon * q.double().tanh()
    E = block_energy.double().sum(-1, keepdim=True)
    D = (u.square()*block_energy.double()).sum(-1, keepdim=True)
    safe = E > 0
    ratio = torch.where(safe, E, torch.ones_like(E))/torch.where(safe, D, torch.ones_like(D))
    scale = ratio.sqrt() if normalize else torch.ones_like(ratio)
    # Zero-energy rows are exactly scalar identity; no gate learning opportunity.
    a = torch.where(safe, alpha*u*scale, torch.full_like(u, alpha))
    a = torch.where(valid, a, torch.zeros_like(a))
    if not bool(torch.isfinite(a).all()): raise FloatingPointError('Nonfinite allocation')
    return a.to(q.dtype)


class Allocation(nn.Module):
    """16 inputs, width16 =>289 gate parameters; feature-ablation capacity identical.

    0:2 projected means; 2:4 projected variances; 4 update log RMS;
    5 state/update cosine; 6:8 left contrasts; 8:10 right contrasts;
    10 state log RMS; 11 valid block fraction; 12:16 fixed zero padding.
    Generic control has the same input dimension, projection/state access and
    capacity, but masks update-derived indices4,5. G7 equals this stronger
    generic control by design, avoiding a deliberately weak generic comparator.
    """
    def __init__(self, kind='G5', block_frames=50, epsilon=.25, alpha=1.):
        super().__init__()
        if kind not in ['G2','G3','G4','G5','G6','G7','G9'] or block_frames < 1:
            raise ValueError('Unknown gate/block layout')
        self.kind, self.block_frames, self.epsilon, self.alpha = kind, block_frames, epsilon, alpha
        self.net = nn.Sequential(nn.Linear(16,16), nn.Tanh(), nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('mean', torch.zeros(16)); self.register_buffer('std', torch.ones(16))
        self.register_buffer('normalization_fitted', torch.tensor(False))
        self.last = {}; self.remove_mode=None

    def features(self, h, delta, projected, lengths):
        mask = frame_mask(h,lengths)
        B,T,d = h.shape; K = math.ceil(T/self.block_frames); pad = K*self.block_frames-T
        m=F.pad(mask,(0,pad)).reshape(B,K,self.block_frames)
        def blocks(x):
            return F.pad(x,(0,0,0,pad)).reshape(B,K,self.block_frames,x.shape[-1])
        p=blocks(projected.float()); x=blocks(h.float()); v=blocks(delta.float())
        n=m.sum(-1); den=n.clamp_min(1).unsqueeze(-1)
        mean=(p*m[...,None]).sum(-2)/den
        var=((p-mean[:,:,None]).square()*m[...,None]).sum(-2)/den
        # Compute energy in float64 before sum for numerical near-zero stability.
        energy=(v.double().square().sum(-1)*m).sum(-1)
        log_update=(energy/den.squeeze(-1)/d).sqrt().log1p().float()
        xnorm=x.square().sum(-1).sqrt(); vnorm=v.square().sum(-1).sqrt()
        cosine=((x*v).sum(-1)/(xnorm*vnorm).clamp_min(1e-20)*m).sum(-1)/n.clamp_min(1)
        left=mean-torch.cat([mean[:,:1],mean[:,:-1]],1)
        right=torch.cat([mean[:,1:],mean[:,-1:]],1)-mean
        # Do not compare the final valid block against padded blocks.
        left=left*((n>0)&torch.cat([n[:,:1]>0,n[:,:-1]>0],1))[...,None]
        right=right*((n>0)&torch.cat([n[:,1:]>0,n[:,-1:]>0],1))[...,None]
        logrms=((x.square().sum(-1)*m).sum(-1)/n.clamp_min(1)/d).sqrt().log1p()
        z=torch.cat([mean,var,log_update[...,None],cosine[...,None],left,right,
            logrms[...,None],(n/self.block_frames)[...,None],torch.zeros(B,K,4,device=h.device)],-1)
        return z,energy,n,mask

    def fit_normalization(self, feature_rows):
        if bool(self.normalization_fitted): raise ValueError('Normalization already frozen')
        z=torch.cat(feature_rows).float()
        if z.ndim!=2 or z.shape[-1]!=16 or not bool(torch.isfinite(z).all()):
            raise ValueError('Invalid training-only features')
        self.mean.copy_(z.mean(0)); self.std.copy_(z.std(0,unbiased=False).clamp_min(1e-4))
        self.normalization_fitted.fill_(True)

    def forward(self,h,delta,projected,lengths,shuffle_seed=None):
        z,energy,n,mask=self.features(h,delta,projected,lengths)
        z=(z-self.mean)/self.std
        if self.kind in ['G3','G4','G7']: z=z*torch.tensor([1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0],device=z.device)
        if self.kind=='G6': z=z*torch.tensor([0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0],device=z.device)
        q=self.net(z).squeeze(-1)
        if self.kind=='G2':
            q=(q*n).sum(-1,keepdim=True)/n.sum(-1,keepdim=True); q=q.expand_as(n)
        if shuffle_seed is not None:
            q=q.clone(); gen=torch.Generator(device='cpu').manual_seed(shuffle_seed)
            for i in range(len(q)):
                k=int((n[i]>0).sum()); order=torch.randperm(k,generator=gen).to(q.device)
                q[i,:k]=q[i,:k][order]
        a=energy_coefficients(q,energy,n,self.epsilon,self.alpha,self.kind in ['G4','G5','G6','G7','G9'])
        if self.remove_mode is not None:
            a=a.clone()
            for i in range(len(a)):
                valid=torch.nonzero(n[i]>0).flatten()
                if len(valid)<2: continue
                if self.remove_mode=='largest': j=valid[a[i,valid].argmax()]
                elif self.remove_mode=='random':
                    gen=torch.Generator().manual_seed(271828)
                    j=valid[int(torch.randint(len(valid),(1,),generator=gen))]
                else: raise ValueError('Unknown removal control')
                a[i,j]=0
            target=energy.sum(-1,keepdim=True)*self.alpha**2
            actual=(a.double().square()*energy).sum(-1,keepdim=True)
            if bool(((target>0)&(actual<=0)).any()): raise ValueError('No energy remains after block removal')
            scale=torch.where(target>0,target,torch.ones_like(target))/torch.where(actual>0,actual,torch.ones_like(actual))
            a=a*scale.sqrt().to(a.dtype)
        c=a.repeat_interleave(self.block_frames,1)[:,:h.shape[1]]*mask
        update=delta*c[...,None]
        self.last={'energy':energy.detach(),'allocated_energy':(a.double().square()*energy).detach(),
            'coefficients':a.detach(),'counts':n.detach(),'zero_energy':int((energy.sum(-1)==0).sum()),
            'cast_energy':((update.to(h.dtype).float().double().square().sum(-1))*mask).sum(-1).detach()}
        if self.kind in ['G4','G5','G6','G7','G9']:
            e=energy.sum(-1)*self.alpha**2; d=(a.double().square()*energy).sum(-1)
            if not torch.allclose(e,d,rtol=2e-6,atol=1e-25): raise FloatingPointError('Energy invariant violated')
        return update


class RepresentationAdapter(EncoderAdapter):
    def __init__(self,method='ereft',d=1280,rank=2,seed=42,gate=None,alpha=1.):
        super().__init__(method,d,rank,seed)
        self.gate=gate; self.alpha=alpha; self.lengths=None; self.shuffle_seed=None

    def delta(self,h):
        x=h.float()
        return self.reft(x) if self.method=='ereft' else F.linear(F.linear(x,self.A,self.c),self.B)

    def forward(self,h,alpha=None):
        if self.lengths is None: raise ValueError('Explicit valid-frame lengths required')
        mask=frame_mask(h,self.lengths)
        if h.shape[-1]!=self.d: raise ValueError('Hidden dimension mismatch')
        self.calls+=1
        d=self.delta(h)
        if self.gate is not None:
            p=F.linear(h.float(),self.reft.rot.weight) if self.method=='ereft' else F.linear(h.float(),self.B.T)
            d=self.gate(h,d,p,self.lengths,self.shuffle_seed)
        else: d=d*mask[...,None]
        return h+((self.alpha if alpha is None else alpha)*d).to(h.dtype)

    def freeze_base(self):
        for name,p in self.named_parameters(): p.requires_grad_(name.startswith('gate.'))


class EncoderWeightLoRA(nn.Module):
    """PM: last encoder q/v r1=5120. Strong: all32 encoder q/v r8=1,310,720.

    Dropout0, no bias adaptation. Full encoder recomputation/backprop required;
    frozen encoder-output caches must never be used for these arms.
    """
    def __init__(self,model,strong=False,seed=42):
        super().__init__(); self.wrapped=nn.ModuleDict(); self.locations=[]; self.installed=False
        layers=range(len(model.model.encoder.layers)) if strong else [len(model.model.encoder.layers)-1]
        rank=8 if strong else 1
        for i in layers:
            att=model.model.encoder.layers[i].self_attn
            for name in ['q_proj','v_proj']:
                key=f'l{i}_{name}'; self.wrapped[key]=LoRALinear(getattr(att,name),rank,rank*2,seed+len(self.locations))
                self.locations.append((att,name,key))
        for x in self.wrapped.values():
            for p in x.base.parameters(): p.requires_grad_(False)

    @contextlib.contextmanager
    def active(self):
        if self.installed: raise RuntimeError('Duplicate LoRA installation')
        self.installed=True
        for att,name,key in self.locations:
            if getattr(att,name) is not self.wrapped[key].base: raise ValueError('Changed LoRA site')
            setattr(att,name,self.wrapped[key])
        try: yield self
        finally:
            for att,name,key in self.locations: setattr(att,name,self.wrapped[key].base)
            self.installed=False

    def adaptation_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if '.base.' not in k}

    def load_adaptation(self,state):
        expected=set(self.adaptation_state())
        if set(state)!=expected: raise ValueError('LoRA checkpoint mismatch')
        self.load_state_dict(state,strict=False)

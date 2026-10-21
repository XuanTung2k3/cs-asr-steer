"""Gauge-invariant LRA features. No reference/decoder-language inputs.

Q spans the *effective affine update* [BA | Bc], not arbitrary coordinates of
B. Small SVD after QR avoids a dense1280x1281 factorization. All feature
statistics use norms/dot products so rotations/signs of Q do not matter.
Numerical rank: singular values >max(1e-10,1e-6*s_max). This defines the
numerically identifiable update subspace; no exact invariance claim for
arbitrarily ill-conditioned floating-point gauge transforms.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from reft.a5000_campaign.methods import Allocation,RepresentationAdapter,frame_mask,energy_coefficients


def update_basis(A,B,c,rtol=1e-6,atol=1e-10):
    if A.ndim!=2 or B.ndim!=2 or c.shape!=(A.shape[0],) or B.shape[1]!=A.shape[0]:
        raise ValueError('Invalid affine LRA dimensions')
    if not all(bool(torch.isfinite(v).all()) for v in [A,B,c]):raise ValueError('Nonfinite LRA')
    with torch.no_grad():
        q,_=torch.linalg.qr(B.detach().double(),mode='reduced')
        small=(q.T@B.detach().double())@torch.cat([A.detach().double(),c.detach().double()[:,None]],1)
        u,s,_=torch.linalg.svd(small,full_matrices=False)
        keep=s>max(atol,rtol*float(s.max()))
        basis=(q@u[:,keep]).float()
    return basis,s.float()


def norm(x):return x.square().sum(-1).clamp_min(0).sqrt()


def cosine(x,y):return (x*y).sum(-1)/(norm(x)*norm(y)).clamp_min(1e-20)


class LRAAllocation(Allocation):
    """Same289-parameter MLP in all gates;16 active inputs in each arm.

    Shared8: state RMS, mean norm, variance trace, frame velocity, neighbor
    contrasts, valid fraction, utterance duration fraction. Generic tail8:
    half-block energies, neighboring cosines, acceleration, coherence, RMS
    variation and peak norm. Proposed tail8: invariant projected-state mean/
    variance, update RMS, state/update alignment, mean alignment, update
    neighbor contrasts and update coherence. Normalization fitted on T only.
    """
    def __init__(self,arm='L4',block_frames=50,epsilon=.25):
        if arm not in ['L2','L3','L4']:raise ValueError('Unknown sprint gate')
        super().__init__('G3' if arm=='L2' else 'G5',block_frames,epsilon,1.)
        self.arm=arm

    def features(self,h,delta,projected,lengths):
        mask=frame_mask(h,lengths);B,T,d=h.shape;K=math.ceil(T/self.block_frames);pad=K*self.block_frames-T
        m=F.pad(mask,(0,pad)).reshape(B,K,self.block_frames)
        def blocks(x):
            if x.shape[-1]==0:return x.new_zeros(B,K,self.block_frames,0)
            return F.pad(x.float(),(0,0,0,pad)).reshape(B,K,self.block_frames,x.shape[-1])
        x=blocks(h);v=blocks(delta);p=blocks(projected)
        n=m.sum(-1);den=n.clamp_min(1);mf=m[...,None]
        def mean(z):return (z*mf).sum(-2)/den[...,None]
        def moments(z):
            mu=mean(z);energy=(z.square().sum(-1)*m).sum(-1)/den
            variance=((z-mu[:,:,None]).square().sum(-1)*m).sum(-1)/den
            return mu,energy,variance
        mu,en,var=moments(x);vm,ve,vvar=moments(v);pm,pe,pvar=moments(p)
        hn=norm(x);rms=(en/d).sqrt()
        valid=n>0
        left_valid=valid&torch.cat([valid[:,:1],valid[:,:-1]],1)
        right_valid=valid&torch.cat([valid[:,1:],valid[:,-1:]],1)
        def neighbors(z):
            left=z-torch.cat([z[:,:1],z[:,:-1]],1);right=torch.cat([z[:,1:],z[:,-1:]],1)-z
            return norm(left)*left_valid,norm(right)*right_valid
        left,right=neighbors(mu);vl,vr=neighbors(vm)
        pair=m[:,:,1:]&m[:,:,:-1]
        velocity=((x[:,:,1:]-x[:,:,:-1]).square().sum(-1)*pair).sum(-1)/pair.sum(-1).clamp_min(1)
        first=m.clone();first[:,:,self.block_frames//2:]=False;second=m&~first
        half1=((hn.square()*first).sum(-1)/first.sum(-1).clamp_min(1)/d).sqrt()
        half2=((hn.square()*second).sum(-1)/second.sum(-1).clamp_min(1)/d).sqrt()
        prev=torch.cat([mu[:,:1],mu[:,:-1]],1);nxt=torch.cat([mu[:,1:],mu[:,-1:]],1)
        triple=m[:,:,2:]&m[:,:,1:-1]&m[:,:,:-2]
        acceleration=((x[:,:,2:]-2*x[:,:,1:-1]+x[:,:,:-2]).square().sum(-1)*triple).sum(-1)/triple.sum(-1).clamp_min(1)
        log=lambda z:z.clamp_min(0).log1p()
        shared=[log(rms),log(norm(mu)/math.sqrt(d)),log((var/d).sqrt()),log((velocity/d).sqrt()),
            log(left/math.sqrt(d)),log(right/math.sqrt(d)),n/self.block_frames,n.sum(-1,keepdim=True).expand_as(n)/T]
        if self.arm in ['L2','L3']:
            hnmean=(hn*m).sum(-1)/den
            hvar=((hn-hnmean[:,:,None]).square()*m).sum(-1)/den
            peak=(hn*m).amax(-1)
            tail=[log(half1),log(half2),cosine(mu,prev)*left_valid,cosine(mu,nxt)*right_valid,
                log((acceleration/d).sqrt()),norm(mu)/en.sqrt().clamp_min(1e-20),log(hvar.sqrt()/math.sqrt(d)),log(peak/math.sqrt(d))]
        else:
            align=(cosine(x,v)*m).sum(-1)/den
            tail=[log(norm(pm)/math.sqrt(d)),log((pvar/d).sqrt()),log((ve/d).sqrt()),align,
                cosine(mu,vm),log(vl/math.sqrt(d)),log(vr/math.sqrt(d)),norm(vm)/ve.sqrt().clamp_min(1e-20)]
        z=torch.stack(shared+tail,-1)*valid[...,None]
        energy=(v.double().square().sum(-1)*m).sum(-1)
        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite gate feature')
        return z,energy,n,mask

    def forward(self,h,delta,projected,lengths,shuffle_seed=None):
        if shuffle_seed is not None:raise ValueError('No unplanned shuffle arm')
        z,e,n,mask=self.features(h,delta,projected,lengths)
        q=self.net((z-self.mean)/self.std).squeeze(-1)
        a=energy_coefficients(q,e,n,self.epsilon,1.,self.arm!='L2')
        coeff=a.repeat_interleave(self.block_frames,1)[:,:h.shape[1]]*mask
        update=delta*coeff[...,None]
        allocated=a.double().square()*e
        if self.arm in ['L3','L4'] and not torch.allclose(allocated.sum(-1),e.sum(-1),rtol=2e-6,atol=1e-25):
            raise FloatingPointError('LRA energy invariant violated')
        self.last={'energy':e.detach(),'allocated_energy':allocated.detach(),'coefficients':a.detach(),
            'counts':n.detach(),'zero_energy':int((e.sum(-1)==0).sum()),
            'cast_energy':(update.to(h.dtype).double().square().sum(-1)*mask).sum(-1).detach()}
        return update


class SprintLRA(RepresentationAdapter):
    def __init__(self,d=1280,rank=2,seed=42,gate=None,alpha=1.):
        super().__init__('lra',d,rank,seed,gate,alpha)
        self.register_buffer('basis',torch.empty(d,0))
        self.register_buffer('basis_singular_values',torch.zeros(rank))
        self._basis_ready=False

    def prepare_basis(self):
        self.basis,self.basis_singular_values=update_basis(self.A,self.B,self.c)
        self._basis_ready=True

    def load_state_dict(self,state,strict=True,assign=False):
        if 'basis' in state:self.basis=torch.empty_like(state['basis'],device=self.B.device)
        result=super().load_state_dict(state,strict=strict,assign=assign)
        self.prepare_basis();return result

    def forward(self,h,alpha=None):
        mask=frame_mask(h,self.lengths);self.calls+=1;delta=self.delta(h)
        if self.gate is not None:
            if not self._basis_ready:self.prepare_basis()
            projected=h.float()@self.basis
            delta=self.gate(h,delta,projected,self.lengths)
        else:delta=delta*mask[...,None]
        return h+((self.alpha if alpha is None else alpha)*delta).to(h.dtype)

    def load_warm(self,state):
        if set(state)!={'A','B','c'}:raise ValueError('Warm start must be the verified plain LRA state')
        nn.Module.load_state_dict(self,state,strict=False)
        self.prepare_basis()

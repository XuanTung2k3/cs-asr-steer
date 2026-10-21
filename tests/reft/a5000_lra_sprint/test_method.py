import io
import pytest
import torch
from reft.a5000_lra_sprint.method import SprintLRA,LRAAllocation,update_basis
from reft.a5000_campaign.methods import RepresentationAdapter
from reft.auto_policy import build_tf_batch
from reft.a5000.objectives import weighted_ce


def adapted(d=16):
    a=SprintLRA(d=d)
    with torch.no_grad():a.A.normal_(std=.03);a.c.normal_(std=.01)
    a.prepare_basis();a.lengths=[9];return a


def test_identity_and_warm_checkpoint():
    a=SprintLRA();a.lengths=[7];h=torch.randn(1,10,1280)
    assert a.count()==5122 and torch.equal(a(h),h)
    old=RepresentationAdapter('lra');old.lengths=[7]
    with torch.no_grad():old.A.normal_(std=.01)
    a.load_warm(old.state_dict());assert torch.equal(a(h),old(h))


@pytest.mark.parametrize('rank',[0,1,2])
def test_effective_subspace_rank_and_finite_features(rank):
    a=adapted()
    with torch.no_grad():
        if rank==0:a.A.zero_();a.c.zero_()
        if rank==1:a.A[1].zero_();a.c[1]=0
    a.prepare_basis();assert a.basis.shape[1]==rank
    if rank:assert torch.allclose(a.basis.T@a.basis,torch.eye(rank),atol=1e-6)
    g=LRAAllocation('L4',3);h=torch.randn(1,10,16);d=a.delta(h)
    assert torch.isfinite(g.features(h,d,h@a.basis,[9])[0]).all()


@pytest.mark.parametrize('transform',[torch.tensor([[2.,.3],[.1,.5]]),torch.diag(torch.tensor([1e3,1e-3])),torch.tensor([[0.,-1.],[1.,0.]])])
def test_features_and_gate_are_gauge_invariant(transform):
    a=adapted();b=adapted();M=transform
    with torch.no_grad():
        b.B.copy_(a.B@M);b.A.copy_(torch.linalg.solve(M,a.A));b.c.copy_(torch.linalg.solve(M,a.c))
    b.prepare_basis();h=torch.randn(1,10,16);ga=LRAAllocation('L4',3);gb=LRAAllocation('L4',3)
    with torch.no_grad():ga.net[-1].weight.normal_();gb.load_state_dict(ga.state_dict())
    da=a.delta(h);db=b.delta(h)
    za=ga.features(h,da,h@a.basis,[9])[0];zb=gb.features(h,db,h@b.basis,[9])[0]
    assert torch.allclose(da,db,atol=2e-6,rtol=2e-5)
    assert torch.allclose(za,zb,atol=2e-5,rtol=2e-5)
    assert torch.allclose(ga(h,da,h@a.basis,[9]),gb(h,db,h@b.basis,[9]),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('arm',['L2','L3','L4'])
def test_gate_identity_gradient_padding_serialization(arm):
    a=adapted();a.gate=LRAAllocation(arm,3);a.freeze_base();h=torch.randn(1,10,16)
    d=a.delta(h);p=h@a.basis;g=a.gate
    assert sum(x.numel() for x in g.parameters())==289
    u=g(h,d,p,[9]);assert torch.equal(u[:,:9],d[:,:9]);assert (u[:,9:]==0).all()
    (u*torch.randn_like(u)).sum().backward();assert g.net[-1].weight.grad.norm()>0
    assert all(x.grad is None for n,x in a.named_parameters() if not n.startswith('gate.'))
    with torch.no_grad():g.net[-1].weight.normal_()
    u=g(h,d,p,[9]);h[:,9:]=500;p[:,9:]=-500;d[:,9:]=100
    assert torch.equal(u,g(h,d,p,[9]))
    if arm!='L2':assert torch.allclose(u[:,:9].square().sum(),d[:,:9].square().sum(),rtol=1e-6)
    buf=io.BytesIO();torch.save(a.state_dict(),buf);buf.seek(0)
    b=SprintLRA(d=16,gate=LRAAllocation(arm,3));b.load_state_dict(torch.load(buf,weights_only=True));b.lengths=[9]
    a.lengths=[9];assert torch.equal(a(h),b(h))


def test_single_block_and_invalid_lengths_and_basis_orientation():
    a=adapted();g=LRAAllocation('L4',50);h=torch.randn(1,10,16);d=a.delta(h)
    with torch.no_grad():g.net[-1].weight.normal_()
    assert torch.allclose(g(h,d,h@a.basis,[9])[:,:9],d[:,:9],atol=1e-6)
    with pytest.raises(ValueError):g(h,d,h@a.basis,[0])
    rot=torch.tensor([[0.,-1.],[1.,0.]])
    assert torch.allclose(g.features(h,d,h@a.basis,[9])[0],g.features(h,d,h@(a.basis@rot),[9])[0],atol=1e-6)


def test_ce_shift_and_exact_logical_reduction():
    X,Y=build_tf_batch([[1,3,4,5]],[[8,9,2]],2,0)
    assert X.tolist()==[[1,3,4,5,8,9]] and Y.tolist()==[[-100,-100,-100,8,9,2]]
    z=torch.randn(2,6,12,requires_grad=True);y=Y.expand(2,-1)
    assert torch.allclose(weighted_ce(z,y),sum(weighted_ce(z[i:i+1],y[i:i+1],denominator=6) for i in range(2)))

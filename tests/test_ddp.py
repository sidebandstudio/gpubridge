"""DistributedDataParallel with gradients synced by gpubridge.ddp_comm_hook."""

from functools import partial

import pytest
import torch
from _harness import run_ranks
from test_ops import Exploding, Keep, gather
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpubridge
from gpubridge.config import SPLIT_TEST_ENV

STEPS = 3


def make_model() -> nn.Module:
    """Small integer weights, no bias, so every gradient below is a small integer."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5, bias=False), nn.ReLU(), nn.Linear(5, 3, bias=False))
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randint(-2, 3, param.shape))
    return model


def batch(rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 * step + rank)
    return (torch.randint(-3, 4, (4, 6), generator=g).float(),
            torch.randint(-2, 3, (4, 3), generator=g).float())


def loss_of(model: nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (model(x) * w).sum()  # integer inputs and weights: integer gradients


def _check_ddp_averages_exactly(topology, expected_policy):
    assert topology.policy_name == expected_policy
    rank, world = topology.rank, topology.world_size
    local = make_model()  # the same model without DDP, for this rank's own gradients
    # One bucket per parameter, so each step queues several async all_reduces.
    model = DDP(make_model(), bucket_cap_mb=1e-6)
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    keep = Keep()
    gpubridge.add_observer(keep)
    for step in range(STEPS):
        x, w = batch(rank, step)
        local.zero_grad()
        loss_of(local, x, w).backward()
        mine = [p.grad.clone() for p in local.parameters()]
        model.zero_grad()
        loss_of(model, x, w).backward()
        # Sums of small integers are exact in any order, so the average is exact too.
        everyone = gather(mine)
        totals = [sum(grads[i].double() for grads in everyone).float()
                  for i in range(len(mine))]
        for i, (param, total) in enumerate(zip(model.parameters(), totals, strict=True)):
            assert torch.equal(param.grad, total / world), (step, i)
        # The same integer update everywhere (an average over 3 or 5 ranks isn't
        # an integer, and the next step's sums must stay exact).
        with torch.no_grad():
            for param, twin, total in zip(model.parameters(), local.parameters(), totals,
                                          strict=True):
                param -= total.clamp(-1, 1)
                twin.copy_(param)
    # Every bucket went through gpubridge, as an AVG all_reduce under the active
    # policy. DDP regroups its buckets after the first step, so count at least one
    # per step, and the same records on every rank.
    summary = [(r.seq, r.op, r.reduce_op, r.policy) for r in keep.records]
    reduces = [s for s in summary if s[1] == "all_reduce"]
    assert len(reduces) >= STEPS
    assert {s[1:] for s in reduces} == {("all_reduce", "avg", expected_policy)}, summary
    # On a bridge other than Gloo the default group is gpubridge's own, so DDP's own
    # exchange on it (it broadcasts its rebuilt buckets after the first step) is carried
    # by gpubridge too, and observed; with bridge="gloo" that stays on the Gloo group.
    others = {s[1] for s in summary} - {"all_reduce"}
    assert others <= ({"broadcast"} if topology.gloo_free else set()), summary
    assert len({tuple(s) for s in gather(summary)}) == 1
    # And every rank ends with identical parameters.
    everyone = gather([p.detach().clone() for p in model.parameters()])
    assert all(torch.equal(a, b) for other in everyone for a, b in zip(
        everyone[0], other, strict=True))


@pytest.mark.parametrize(("vendors", "split", "policy", "expected"), [
    (["nvidia", "nvidia", "amd", "amd"], None, "auto", "reduce-bridge-broadcast"),
    (["nvidia", "amd", "nvidia", "amd", "amd"], None, "sharded-bridge", "sharded-bridge"),
    (["nvidia"] * 3 + ["amd"], None, "pipelined-reduce-bridge-broadcast",
     "pipelined-reduce-bridge-broadcast"),
    (["amd", "amd", "amd"], None, "auto", "native-only"),
    (["nvidia"] * 4, "half", "flat-gloo", "flat-gloo"),
], ids=["2+2-auto", "2+3-sharded", "3+1-pipelined", "single-vendor", "split-flat-gloo"])
def test_ddp_with_the_hook_averages_gradients_exactly(vendors, split, policy, expected, tmp_path,
                                                      monkeypatch):
    if split:
        monkeypatch.setenv(SPLIT_TEST_ENV, split)
    run_ranks(vendors, partial(_check_ddp_averages_exactly, expected_policy=expected), tmp_path,
              init_kwargs={"policy": policy})


def _check_failure_reaches_backward(topology):
    model = DDP(make_model())
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    x, w = batch(topology.rank, 0)
    with pytest.raises(RuntimeError, match="policy bug"):
        loss_of(model, x, w).backward()
    # Every later collective refuses to start: the groups may be inconsistent.
    with pytest.raises(RuntimeError, match="an earlier async gpubridge collective failed"):
        gpubridge.all_reduce(torch.ones(2))
    gather(None)  # the ranks still meet (through the store), and shut down cleanly


def test_a_failed_bucket_makes_backward_raise_on_every_rank(tmp_path):
    run_ranks(["nvidia", "amd"], _check_failure_reaches_backward, tmp_path,
              init_kwargs={"policy": Exploding})

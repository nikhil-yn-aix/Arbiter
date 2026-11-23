import copy

import torch

from arbiter.arms import noise_like
from arbiter.train import micro_batches, schedule, step_loss


def _groups(n, group=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randint(1, 64, (int(torch.randint(4, 12, (1,), generator=g)),), generator=g).tolist() for _ in range(group)] for _ in range(n)]


def test_accumulation_equals_full_batch(tiny_model):
    groups = _groups(8)
    a, b = tiny_model, copy.deepcopy(tiny_model)
    for mb in micro_batches(groups, 2):
        step_loss(a, mb, 0, len(groups), torch.device("cpu"))
    step_loss(b, groups, 0, len(groups), torch.device("cpu"))
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=1e-5), n


def test_micro_batches_keep_groups_whole():
    groups = _groups(8)
    mbs = micro_batches(groups, 4)
    assert sorted(map(id, [g for mb in mbs for g in mb])) == sorted(map(id, groups))
    assert all(len(mb) == 4 for mb in mbs)


def test_schedule_warmup_then_decay():
    f = schedule(100, 0.1)
    assert f(0) == 0.1 and f(9) == 1.0
    assert f(50) > f(90) > 0 and f(100) == 0.0


def test_noise_matches_per_tensor_norm():
    base = {"w": torch.ones(10, 10), "b": torch.zeros(5)}
    ref = {"w": torch.ones(10, 10) * 1.1, "b": torch.ones(5) * 0.2}
    out = noise_like(base, ref, 1)
    for k in base:
        assert torch.isclose((out[k] - base[k]).norm(), (ref[k] - base[k]).norm(), rtol=1e-5)
    assert not torch.allclose(out["w"] - base["w"], ref["w"] - base["w"])
    assert torch.equal(noise_like(base, ref, 1)["w"], out["w"])

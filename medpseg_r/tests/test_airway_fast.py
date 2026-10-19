"""Check that the airway-only forward gives exactly the same airway outputs, loss and gradients
as the released full forward (CPU, float64 for an exact comparison).

    $PY tests/test_airway_fast.py runs/medpseg_r_seed20251014/checkpoints/epoch_000.ckpt
"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from airway_fast import enable_airway_only_forward  # noqa: E402
from medpseg.poly_seg_2d_module import PolySeg2DModule  # noqa: E402


def main(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"]); hp["scratch"] = True; hp["cpu"] = True
    full = PolySeg2DModule(hp); full.load_state_dict(ck["state_dict"]); full = full.double()
    fast = copy.deepcopy(full); enable_airway_only_forward(fast)
    ok = True

    torch.manual_seed(0)
    x = torch.rand(2, 3, 128, 128, dtype=torch.float64)
    y1 = (torch.rand(2, 1, 128, 128) > 0.97).double()
    y = torch.cat([1 - y1, y1], 1)
    meta = {"preprocess": ["new_atm_hu", "new_atm_hu"]}

    for mode in ("eval", "train"):
        for m in (full, fast):
            m.train(mode == "train")
        torch.manual_seed(1); of = full(x, stacking=True)
        torch.manual_seed(1); oq = fast(x, stacking=True)
        keys = [k for k in of if k.startswith("atm")]
        assert set(oq) == set(keys), (sorted(oq), keys)
        for k in keys:
            d = (of[k] - oq[k]).abs().max().item()
            print(f"{mode} {k}: max |diff| = {d:.3e}")
            ok &= d == 0.0

    # loss + gradients in train mode
    for m in (full, fast):
        m.train(); m.zero_grad(set_to_none=True)
        m.log = lambda *a, **k: None          # no trainer attached
    torch.manual_seed(2); lf = full.supervised_loss(y=y, y_hat=full(x), meta=meta, prestr="")
    torch.manual_seed(2); lq = fast.supervised_loss(y=y, y_hat=fast(x), meta=meta, prestr="")
    lf.backward(); lq.backward()
    print(f"loss full={lf.item():.10f} fast={lq.item():.10f}")
    ok &= lf.item() == lq.item()
    gmax, n = 0.0, 0
    for (k, pf), (_, pq) in zip(full.named_parameters(), fast.named_parameters()):
        if pf.grad is None and pq.grad is None:
            continue
        if (pf.grad is None) != (pq.grad is None):
            print("grad presence differs:", k); ok = False; continue
        gmax = max(gmax, (pf.grad - pq.grad).abs().max().item()); n += 1
    print(f"{n} parameter gradients compared, max |diff| = {gmax:.3e}")
    ok &= gmax == 0.0
    print("AIRWAY-ONLY FORWARD IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

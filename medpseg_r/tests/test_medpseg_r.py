"""Correctness tests for MEDPSeg-R data handling (run on CPU with the medpseg env).

    $PY tests/test_medpseg_r.py smoke/index_smoke.json

1. Train-time 2.5D extraction (augmentation off) equals the released inference stacking
   (`read_preprocess(norm=True)` + `E2DStackDataset(extended_2d=1)`) bit for bit.
2. Prepared label is aligned with the image: airway voxels are air-like in HU.
3. Augmented samples keep image/label aligned and have the right shapes, dtypes and range.
4. Validation volumes equal the inference input.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from medpseg_r_data import AirwayTrainSlices, AirwayValVolumes, hu_norm, stack_25d  # noqa: E402
from medpseg.image_read import read_preprocess  # noqa: E402
from medpseg.eval_2d_utils import E2DStackDataset  # noqa: E402

PREP = os.path.join(os.environ.get("MEDPSEG_R_DATA", os.path.join(HERE, "data")), "prepared")


def main(index_json):
    index = json.load(open(index_json))
    ok = True

    # 1 + 4: inference stacking equivalence on one train and one val case
    for split in ("train", "val"):
        cid = sorted(index[split])[0]
        meta = json.load(open(f"{PREP}/{cid}_meta.json"))
        ref, *_ = read_preprocess(meta["image_path"], norm=True)        # [Z, H, W] in [0, 1]
        vol = ref.unsqueeze(0).unsqueeze(0)                               # [1, 1, Z, H, W]
        ds = E2DStackDataset(vol, extended_2d=1)
        img = np.load(f"{PREP}/{cid}_img.npy", mmap_mode="r")
        Z = img.shape[0]
        for z in (0, 1, Z // 2, Z - 2, Z - 1):
            mine = hu_norm(stack_25d(img, z).astype(np.float32)).astype(np.float32)
            theirs = ds[z].numpy()
            if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
                print(f"FAIL stacking {cid} z={z}: max abs diff {np.abs(mine - theirs).max()}")
                ok = False
        if split == "val":
            vx, vy, vm = AirwayValVolumes(PREP, index_json)[0]
            if not torch.equal(vx[0], ref):
                print(f"FAIL val volume {cid} differs from inference input"); ok = False
            if not (vy.sum(0) == 1).all():
                print("FAIL val one-hot"); ok = False
        print(f"stacking/{split} {cid}: checked")

    # 2: label alignment, airway lumen must be air (HU well below soft tissue)
    for cid in sorted(index["train"]):
        img = np.load(f"{PREP}/{cid}_img.npy", mmap_mode="r")
        lab = np.load(f"{PREP}/{cid}_lab.npy", mmap_mode="r")
        zs = index["train"][cid]["annotated_slices"][::25]
        hu_in = np.concatenate([np.asarray(img[z])[np.asarray(lab[z]) > 0] for z in zs])
        med = float(np.median(hu_in))
        print(f"alignment {cid}: median HU inside airway = {med:.0f}")
        if med > -600:
            print(f"FAIL alignment {cid}"); ok = False

    # 3: augmented samples
    ds = AirwayTrainSlices(PREP, index_json, augment=True)
    torch.manual_seed(0)
    n_pos, meds = 0, []
    for i in np.random.default_rng(0).integers(0, len(ds), 400):
        x, y, meta = ds[int(i)]
        assert x.shape == (3, 256, 256) and x.dtype == torch.float32, x.shape
        assert y.shape == (2, 256, 256) and torch.all(y.sum(0) == 1)
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
        assert meta["preprocess"] == "new_atm_hu"
        m = y[1] > 0
        if m.sum() > 20:
            n_pos += 1
            meds.append(float(x[1][m].median()))
    hu = np.median(meds) * 1624 - 1024
    print(f"augmented: {n_pos}/400 patches contain airway; median HU at airway voxels ~ {hu:.0f}")
    if hu > -500:
        print("FAIL augmented alignment"); ok = False

    # 3b: no-augmentation sample equals inference stack crop
    ds0 = AirwayTrainSlices(PREP, index_json, augment=False)
    x, y, meta = ds0[100]
    cid, z = ds0.items[100]
    full = hu_norm(stack_25d(np.load(f"{PREP}/{cid}_img.npy", mmap_mode="r"), z).astype(np.float32))
    found = any(np.array_equal(x.numpy(), full[:, r:r + 256, c:c + 256])
                for r in range(0, 257) for c in range(0, 257) if full[0, r, c] == x[0, 0, 0].item())
    print(f"no-aug crop is a sub-window of the inference stack: {found}")
    ok = ok and found

    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

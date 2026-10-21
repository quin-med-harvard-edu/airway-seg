"""Vote-pattern confidence: fit the lookup table on the validation split and evaluate it on test.

Inside the fused mask F, each voxel's pattern is (number of the five AirMorph-R networks voting
airway, raw MEDPSeg-R, raw TfeNet), coded as votes*4 + 2*m + t (24 patterns). Its confidence is the
fraction of validation voxels with that pattern that lie inside the ground truth. On test we report
the pooled AUROC for separating false-positive from true-positive voxels of F (95% CI from a
bootstrap over patients), the share of false positives among the 20% least confident voxels, and
the expected calibration error over patterns.

    python confidence/fit_confidence.py \\
        --val-airmorph ... --val-medpseg ... --val-tfenet ... --val-labels ... \\
        --test-airmorph ... --test-medpseg ... --test-tfenet ... --test-labels ... \\
        --out confidence/confidence_lut_medpseg_r.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "fusion"))
from anchored_union import anchored_union  # noqa: E402

MEMBERS = ["member_model1", "member_model2", "member_model3", "member_wingsnet", "member_break"]


def histograms(job):
    cid, d = job
    with np.load(os.path.join(d["airmorph"], f"ATM26_{cid}.npz")) as p:
        A = p["post"].astype(bool)
        votes = sum(p[k].astype(np.uint8) for k in MEMBERS)
    with np.load(os.path.join(d["medpseg"], f"ATM26_{cid}.npz")) as p:
        M = p["raw"].astype(bool)
    with np.load(os.path.join(d["tfenet"], f"ATM26_{cid}.npz")) as p:
        T = p["raw"].astype(bool)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d["labels"], cid + ".mha"))) > 0
    F = anchored_union(A, M, T)
    pattern = (votes[F] * 4 + 2 * M[F] + T[F]).astype(np.int64)
    return (np.bincount(pattern, minlength=24)[:24].astype(float),
            np.bincount(pattern[gt[F]], minlength=24)[:24].astype(float))


def auroc(fp, tp, order):
    """AUROC of 'false positive' scored by low confidence; fp/tp are per-pattern counts."""
    fp, tp = fp[order][::-1], tp[order][::-1]           # most confident first
    P, N = fp.sum(), tp.sum()
    below = np.cumsum(tp) - tp
    return float(((below + 0.5 * tp) * fp).sum() / (P * N))


def run_split(split, args, workers):
    ids = [c.split("/")[1] for c in json.load(open(os.path.join(REPO, "splits", "atm26_split.json")))[split]]
    d = {k: getattr(args, f"{split}_{k}") for k in ("airmorph", "medpseg", "tfenet", "labels")}
    with Pool(workers) as pool:
        h = pool.map(histograms, [(c, d) for c in sorted(ids)])
    return np.array([x[0] for x in h]), np.array([x[1] for x in h])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for split in ("val", "test"):
        for k in ("airmorph", "medpseg", "tfenet", "labels"):
            ap.add_argument(f"--{split}-{k}", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    va, vt = run_split("val", args, args.workers)
    p = np.where(va.sum(0) > 0, vt.sum(0) / np.maximum(va.sum(0), 1), np.nan)
    conf = np.nan_to_num(p, nan=0.5)
    order = np.argsort(conf)

    ta, tt = run_split("test", args, args.workers)
    tfp = ta - tt
    au = auroc(tfp.sum(0), tt.sum(0), order)
    rng = np.random.default_rng(0)
    boot = [auroc(tfp[i].sum(0), tt[i].sum(0), order) for i in rng.integers(0, len(ta), (2000, len(ta)))]
    al, fp = ta.sum(0), tfp.sum(0)
    share_fp_low20 = float(np.interp(0.2, np.r_[0, np.cumsum(al[order]) / al.sum()], np.r_[0, np.cumsum(fp[order]) / fp.sum()]))
    obs = np.where(al > 0, tt.sum(0) / np.maximum(al, 1), 0)
    ece = float((al / al.sum() * np.abs(conf - obs)).sum())
    print(f"test AUROC {au:.3f} (95% CI {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}); "
          f"FP in lowest 20%: {100 * share_fp_low20:.1f}%; ECE {100 * ece:.2f} points")

    votes, m, t = np.arange(24) // 4, (np.arange(24) // 2) % 2, np.arange(24) % 2
    lut = {"description": "P(voxel is airway | vote pattern) inside the anchored fused mask F, fitted on the 30 "
                          "ATM'26 validation volumes. pattern = airmorph_votes*4 + 2*medpseg_r + tfenet.",
           "table": [{"pattern": i, "airmorph_votes": int(votes[i]), "medpseg_r": int(m[i]), "tfenet": int(t[i]),
                      "p_airway": None if np.isnan(p[i]) else round(float(p[i]), 4),
                      "n_val_voxels": int(va.sum(0)[i])} for i in range(24)]}
    json.dump(lut, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

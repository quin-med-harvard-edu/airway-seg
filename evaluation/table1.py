"""Table 1 rows on the ATM'26 split: each member with its own post-processing, pairwise unions,
majority vote, three ablations that share the inputs of our method (the two anchored pairs and the
largest component of the three-model union), and the anchored union (ours). Metrics: DSC, clDice,
TLD, BND, sensitivity, precision and HD95.

Inputs (one file per case, as written by this repository's prediction scripts):
    --airmorph DIR   ATM26_<id>.npz from airmorph_r/dump_airmorph_masks.py (key --airmorph-key)
    --medpseg  DIR   ATM26_<id>.npz from medpseg_r/predict_medpseg_r.py (keys raw, post)
    --tfenet   DIR   ATM26_<id>.npz from tfenet/predict_tfenet.py (key raw; post is recomputed)
    --labels   DIR   ATM'26 labelsTr (<id>.mha)
    --centerlines DIR  per-case centerline.npy + meta.json from airmorph_r/data/prepare_dataset.py

Metrics come from the ATM'22 / AirMorph evaluation code (TLD, BND, DSC, sensitivity,
precision), `airmorph_r/eval_patch.py` (clDice) and `hd95.py`. Run inside an AirMorph clone with
`airmorph_r/` installed as `training/` (see airmorph_r/README.md):

    python evaluation/table1.py --split test --airmorph ... --medpseg ... --tfenet ... \\
        --labels ... --centerlines ... --out results/table1_test.csv
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "fusion"))
sys.path.insert(0, HERE)
from anchored_union import anchored_union, largest_component, majority, tfenet_post  # noqa: E402
from hd95 import hd95  # noqa: E402

ARGS = None


def load_case(cid):
    from training.eval_patch import parse_ground_truth
    from util.utils import load_itk_image, restore_image_via_box
    label, _, spacing, _ = load_itk_image(os.path.join(ARGS.labels, cid + ".mha"))
    gt = (label > 0).astype(np.uint8)
    cdir = os.path.join(ARGS.centerlines, cid)
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    skel = np.load(os.path.join(cdir, "centerline.npy")).astype(np.uint8)
    skel = (restore_image_via_box(tuple(meta["original_shape"]), skel, np.array(meta["bbox"])) > 0).astype(np.uint8)
    parsing, skeleton = parse_ground_truth(gt, skel)
    return gt, np.array(spacing, float), parsing, skeleton


def score(pred, gt, spacing, parsing, skeleton):
    from training.eval_patch import cldice_fixed
    from util.airwayatlas_eval import (branch_detected_calculation, dice_coefficient_score_calculation,
                                       precision_calculation, sensitivity_calculation, tree_length_calculation)
    pred = pred.astype(np.uint8)
    _, _, bnd = branch_detected_calculation(pred, parsing, skeleton, thresh=0.8)
    return {"DSC": float(dice_coefficient_score_calculation(pred, gt)),
            "clDice": float(cldice_fixed(v_p_pred=pred, v_l_gt=gt)),
            "TLD": float(tree_length_calculation(pred, skeleton)),
            "BND": float(bnd),
            "Sensitivity": float(sensitivity_calculation(pred, gt)),
            "Precision": float(precision_calculation(pred, gt)),
            "HD95": hd95(pred, gt, tuple(spacing))}


def npz(d, cid, key):
    with np.load(os.path.join(d, f"ATM26_{cid}.npz")) as p:
        return p[key].astype(bool)


def run(cid):
    gt, spacing, parsing, skeleton = load_case(cid)
    A = npz(ARGS.airmorph, cid, ARGS.airmorph_key)
    M_raw, M = npz(ARGS.medpseg, cid, "raw"), npz(ARGS.medpseg, cid, "post")
    T_raw = npz(ARGS.tfenet, cid, "raw")
    T = tfenet_post(T_raw)
    rows = {"AirMorph-R (A)": A, "MEDPSeg-R (M)": M, "TfeNet (T)": T,
            "A|M": A | M, "A|T": A | T, "M|T": M | T, "Majority": majority(A, M, T),
            "A|M~ anchored": anchored_union(A, M_raw), "A|T~ anchored": anchored_union(A, T_raw),
            "A|M~|T~ largest": largest_component(A | M_raw | T_raw),
            "Ours": anchored_union(A, M_raw, T_raw)}
    out = []
    for name, mask in rows.items():
        r = score(mask, gt, spacing, parsing, skeleton)
        r.update(case=cid, method=name)
        out.append(r)
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    for k in ("airmorph", "medpseg", "tfenet", "labels", "centerlines"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--airmorph-key", default="post", help="post: 3-of-5 vote + largest component (used in the paper); raw: vote only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ARGS = ap.parse_args()
    ids = [c.split("/")[1] for c in json.load(open(os.path.join(REPO, "splits", "atm26_split.json")))[ARGS.split]]
    with Pool(ARGS.workers) as pool:
        df = pd.DataFrame([r for rs in pool.map(run, sorted(ids)) for r in rs])
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
    df.to_csv(ARGS.out, index=False)
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "HD95"]
    print(df.groupby("method", sort=False)[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()

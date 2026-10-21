"""Table 2: adult per-generation DSC, TLD, BND and sensitivity for AirMorph-R, MEDPSeg-R and TfeNet
(each with its own post-processing) and our anchored union. Writes per-case rows and a summary;
generation_cldice.py adds clDice and build_generation_table.py writes the LaTeX table.

    export ATM26_LABELS=... ATM26_CENTERLINES=... AIRMORPH_MASKS=... MEDPSEG_MASKS=... TFENET_MASKS=...
    python evaluation/generation_table.py --out-dir results/generation
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
import per_generation as pg  # noqa: E402
from anchored_union import anchored_union, tfenet_post  # noqa: E402


def npz(directory, case, key):
    with np.load(os.path.join(directory, f"ATM26_{case}.npz")) as p:
        return p[key].astype(bool)


def strategies():
    A = lambda m: npz(pg.MASK_DIRS["airmorph"], m["_case"], "post")
    M = lambda m, key: npz(pg.MASK_DIRS["medpseg"], m["_case"], key)
    return {"airmorph": lambda m: A(m).astype(np.uint8),
            "medpseg_r_post": lambda m: M(m, "post").astype(np.uint8),
            "tfenet_post": lambda m: tfenet_post(m["tfenet"]).astype(np.uint8),
            "ours": lambda m: anchored_union(A(m), M(m, "raw"), m["tfenet"]).astype(np.uint8)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pg.STRATEGIES.clear(); pg.STRATEGIES.update(strategies())
    pg.LABEL.clear(); pg.LABEL.update({"airmorph": "AirMorph-R", "medpseg_r_post": "MEDPSeg-R",
                                       "tfenet_post": "TfeNet", "ours": "Ours"})
    os.makedirs(args.out_dir, exist_ok=True)
    o = args.out_dir
    sys.argv = [sys.argv[0], "--workers", str(args.workers), "--out", f"{o}/per_generation.csv",
                "--summary", f"{o}/summary.csv", "--tex", f"{o}/body.tex", "--diag", f"{o}/diagnostics.csv"]
    pg.main()


if __name__ == "__main__":
    main()

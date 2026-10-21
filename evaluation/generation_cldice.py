"""Per-generation clDice for Table 2, with the same masks as generation_table.py and the same
generation map as per_generation.py: every voxel takes the generation of its nearest ground-truth
branch. clDice follows Table 1 (Lee skeletons of prediction and ground truth):
  Tsens_g = |S_gt,g inside prediction| / |S_gt,g|,   Tprec_g = |S_pred,g inside ground truth| / |S_pred,g|
where S_x,g are the skeleton voxels whose nearest GT branch has generation g. WT is the whole-tree
clDice of Table 1. Same environment variables as generation_table.py.

    python evaluation/generation_cldice.py --out-dir results/generation
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fusion"))
import per_generation as pg
from anchored_union import anchored_union, tfenet_post

GENS = range(1, 11)


def load(case):
    def npz(d, key):
        with np.load(os.path.join(d, f"ATM26_{case}.npz")) as p:
            return p[key].astype(bool)
    a = npz(pg.MASK_DIRS["airmorph"], "post")
    mpost, mraw = npz(pg.MASK_DIRS["medpseg"], "post"), npz(pg.MASK_DIRS["medpseg"], "raw")
    t = npz(pg.MASK_DIRS["tfenet"], "raw")
    return {"airmorph": a, "medpseg_r_post": mpost, "tfenet_post": tfenet_post(t).astype(bool),
            "ours": anchored_union(a, mraw, t).astype(bool)}


def skel(mask):
    if not mask.any():
        return mask.copy()
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - 2), min(d, s.stop + 2)) for s, d in zip(sl, mask.shape))
    out = np.zeros(mask.shape, bool)
    out[sl] = skeletonize(mask[sl].astype(np.uint8), method="lee") > 0
    return out


def gen_map_of(case):
    """Generation of every voxel: copied from per_generation.run_case (GT part)."""
    with open(os.path.join(pg.CACHE, case, "meta.json")) as fh:
        meta = json.load(fh)
    label, _, spacing, _ = pg.load_itk_image(os.path.join(pg.GT_LABELS, case + ".mha"))
    label = (label > 0).astype(np.uint8)
    skel_crop = np.load(os.path.join(pg.CACHE, case, "centerline.npy"))
    skeleton = (pg.restore_image_via_box(tuple(meta["original_shape"]), skel_crop.astype(np.uint8),
                                         np.array(meta["bbox"])) > 0).astype(np.uint8)
    skeleton_parse, cd, num = pg.skeleton_parsing(skeleton)
    inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_distances=False, return_indices=True)
    full_parsing = cd[inds[0], inds[1], inds[2]]; del inds
    zc, yc, xc = np.nonzero(skeleton_parse)
    seg_of_vox = cd[zc, yc, xc].astype(np.int64)
    sl = ndimage.find_objects(label)[0]
    pad = tuple(slice(max(0, s.start - 1), min(d, s.stop + 1)) for s, d in zip(sl, label.shape))
    edt_crop = ndimage.distance_transform_edt(label[pad], sampling=tuple(spacing))
    edt_vals = np.zeros(seg_of_vox.shape)
    inside = ((zc >= pad[0].start) & (zc < pad[0].stop) & (yc >= pad[1].start) & (yc < pad[1].stop)
              & (xc >= pad[2].start) & (xc < pad[2].stop))
    edt_vals[inside] = edt_crop[zc[inside] - pad[0].start, yc[inside] - pad[1].start, xc[inside] - pad[2].start]
    n_vox = np.bincount(seg_of_vox, minlength=num + 1).astype(float)
    mean_radius = np.bincount(seg_of_vox, weights=edt_vals, minlength=num + 1) / np.maximum(n_vox, 1)
    mean_radius[0] = -1
    adj = pg.branch_adjacency(skeleton, skeleton_parse, cd, num)
    gen_of_branch, _ = pg.generations_from_root(adj, num, int(np.argmax(mean_radius)))
    return label.astype(bool), gen_of_branch.astype(np.int16)[full_parsing]


def run(case):
    G, gmap = gen_map_of(case)
    Sg = skel(G)
    rows = []
    for k, P in load(case).items():
        Sp = skel(P)
        gs, gp = gmap[Sg], gmap[Sp]
        sens_in, prec_in = P[Sg], G[Sp]
        def cl(ts, tp):
            return 0.0 if ts + tp == 0 else 200.0 * ts * tp / (ts + tp)
        wt = cl(sens_in.mean(), prec_in.mean() if Sp.any() else 0.0)
        for g in GENS:
            ng = int((gs == g).sum())
            if ng == 0:
                continue                                   # generation absent in this GT
            ts = sens_in[gs == g].mean()
            npg = int((gp == g).sum())
            tp = prec_in[gp == g].mean() if npg else 0.0
            rows.append({"case": case, "strategy": k, "generation": g, "clDice": cl(ts, tp),
                         "Tsens": 100 * ts, "Tprec": 100 * tp, "gt_skel": ng, "pred_skel": npg})
        rows.append({"case": case, "strategy": k, "generation": "WT", "clDice": wt,
                     "Tsens": 100 * sens_in.mean(), "Tprec": 100 * (prec_in.mean() if Sp.any() else 0),
                     "gt_skel": int(Sg.sum()), "pred_skel": int(Sp.sum())})
    return rows


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cases = sorted(d for d in os.listdir(pg.CACHE) if os.path.isdir(os.path.join(pg.CACHE, d)))
    with mp.Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(run, cases) for r in rs]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "per_case_cldice.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), "rows")

"""Anchored union fusion of three binary airway masks.

    U = A | M~ | T~
    F = the 6-connected component C of U that maximises |C & A|

A is the AirMorph-R ensemble mask (the anchor); M~ and T~ are the raw MEDPSeg-R and TfeNet
masks (before their own largest-component post-processing). If A touches no component of U,
the largest component is kept.

Also provides the members' released post-processing, used for the comparison rows:
TfeNet keeps the largest 6-connected component and fills holes (TfeNet/postprocessing.py);
MEDPSeg's `--post` keeps the largest 26-connected component (cc3d), which
`medpseg_r/predict_medpseg_r.py` already stores as `post`.

Command line (NIfTI or MHA, all on the same grid):
    python fusion/anchored_union.py --anchor A.nii.gz --others M.nii.gz T.nii.gz --out F.nii.gz
"""
import argparse

import numpy as np
from scipy import ndimage

SIX = ndimage.generate_binary_structure(3, 1)


def largest_component(mask, structure=SIX):
    lab, n = ndimage.label(mask.astype(bool), structure=structure)
    if n == 0:
        return np.zeros(mask.shape, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def keep_anchored_component(mask, anchor):
    """The 6-connected component of `mask` that holds most of `anchor`; the largest one when
    `anchor` misses every component."""
    lab, n = ndimage.label(mask.astype(bool), structure=SIX)
    if n == 0:
        return np.zeros(mask.shape, bool)
    hit = np.bincount(lab[anchor.astype(bool)], minlength=n + 1)
    hit[0] = 0
    if hit.sum() == 0:
        hit = np.bincount(lab.ravel(), minlength=n + 1)
        hit[0] = 0
    return lab == int(hit.argmax())


def anchored_union(anchor, *others):
    """F = anchored component of anchor | others[0] | others[1] | ..."""
    union = anchor.astype(bool).copy()
    for m in others:
        union |= m.astype(bool)
    return keep_anchored_component(union, anchor)


def tfenet_post(mask):
    """TfeNet's released post-processing: largest 6-connected component, then hole filling."""
    return ndimage.binary_fill_holes(largest_component(mask))


def majority(*masks):
    votes = sum(m.astype(np.uint8) for m in masks)
    return votes >= (len(masks) // 2 + 1)


def main():
    import SimpleITK as sitk
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", required=True, help="AirMorph-R mask")
    ap.add_argument("--others", nargs="+", required=True, help="raw MEDPSeg-R and TfeNet masks")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = sitk.ReadImage(args.anchor)
    anchor = sitk.GetArrayFromImage(ref) > 0
    others = [sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0 for p in args.others]
    for p, m in zip(args.others, others):
        if m.shape != anchor.shape:
            raise SystemExit(f"{p}: shape {m.shape} differs from the anchor {anchor.shape}")
    out = sitk.GetImageFromArray(anchored_union(anchor, *others).astype(np.uint8))
    out.CopyInformation(ref)
    sitk.WriteImage(out, args.out)


if __name__ == "__main__":
    main()

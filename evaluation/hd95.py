"""Symmetric 95th-percentile surface distance (mm).

Surfaces are the voxels removed by one 6-connected erosion; distances are Euclidean in mm from
each surface voxel of one mask to the nearest surface voxel of the other, pooled over both
directions. The distance transforms run on the bounding box of (pred | gt) + 2 voxels, which
leaves every nearest-surface distance unchanged and is much faster than the full volume."""
import numpy as np
from scipy import ndimage

ST6 = ndimage.generate_binary_structure(3, 1)


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, ST6, border_value=0)


def hd95(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, pred.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sp, sg = surface(pred[sl]), surface(gt[sl])
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([dt_g[sp], dt_p[sg]]), 95))

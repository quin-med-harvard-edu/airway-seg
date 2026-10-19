"""Datasets for MEDPSeg-R (airway-only retraining of MEDPSeg on ATM'26).

Every choice below follows docs/MEDPSEG_TRAINING_SPEC.md; the decision numbers (D#) refer
to its section 5.2 table.

Training sample (one per __getitem__):
  * an axial slice z drawn from the pooled list of slices that contain airway (D4, D5);
  * the 2.5D stack (z-1, z, z+1) with the neighbour index clamped at the volume edges,
    exactly as `medpseg.eval_2d_utils.E2DStackDataset(extended_2d=1)` does at inference (D16);
  * augmentation in HU on the full 512x512 stack (D10-D13), then the HU window
    [-1024, 600] -> [0, 1] (applied last, as the released `DefinitiveHUNorm`), then a uniform
    random 256x256 crop (D9);
  * target: one-hot [bg, airway] (D15); meta["preprocess"] = "new_atm_hu", which routes the
    sample to the airway head only inside the released PolySeg2DModule.

Validation sample: the whole volume, HU-normalised, [1, Z, 512, 512] plus one-hot target
[2, Z, 512, 512], like the released `eval` transform (D21).
"""
import json
import os

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

HU_MIN, HU_MAX = -1024.0, 600.0
PREPROCESS = "new_atm_hu"
PATCH = 256


def hu_norm(x):
    """`read_preprocess(norm=True)` / DefinitiveHUNorm: clip [-1024, 600] then scale to [0, 1]."""
    return (np.clip(x, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def stack_25d(vol, z):
    """(z-1, z, z+1), clamped, in this order: identical to E2DStackDataset(extended_2d=1)."""
    last = vol.shape[0] - 1
    idx = [min(max(z + d, 0), last) for d in (-1, 0, 1)]
    return np.stack([np.asarray(vol[i]) for i in idx], axis=0)


class _Volumes:
    """Lazily opened, per-worker memory maps of the prepared int16 images and uint8 labels."""

    def __init__(self, prepared_dir):
        self.dir = prepared_dir
        self._img, self._lab = {}, {}

    def img(self, cid):
        if cid not in self._img:
            self._img[cid] = np.load(os.path.join(self.dir, f"{cid}_img.npy"), mmap_mode="r")
        return self._img[cid]

    def lab(self, cid):
        if cid not in self._lab:
            self._lab[cid] = np.load(os.path.join(self.dir, f"{cid}_lab.npy"), mmap_mode="r")
        return self._lab[cid]


def augment_hu(x, y, rng):
    """nnU-Net 2D augmentation as listed in the released checkpoint's `transform_readable`.

    x: float32 [3, H, W] in HU; y: uint8 [H, W]. The same spatial transform is applied to all
    three slices and to the label. Returns new arrays.
    """
    # 1. Mirroring, p = 0.5 per in-plane axis (D11). The slice (channel) order is never reversed.
    if rng.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)

    # 2. Rotation and scaling, one joint event with p = 0.2 (D12). Angle U(-180, 180) degrees,
    #    isotropic scale U(0.7, 1.4) with batchgenerators semantics (sampling coordinates are
    #    multiplied by the scale). Image linear with fill -1024 HU, label nearest with fill 0.
    if rng.random() < 0.2:
        theta = np.deg2rad(rng.uniform(-180.0, 180.0))
        scale = rng.uniform(0.7, 1.4)
        c, s = np.cos(theta), np.sin(theta)
        m = scale * np.array([[c, -s], [s, c]])
        centre = (np.array(y.shape, dtype=np.float64) - 1.0) / 2.0
        offset = centre - m @ centre
        x = np.stack([ndimage.affine_transform(ch, m, offset=offset, order=1, mode="constant",
                                               cval=HU_MIN) for ch in x]).astype(np.float32)
        y = ndimage.affine_transform(y, m, offset=offset, order=0, mode="constant", cval=0).astype(np.uint8)
        # "Fix no label area": with a binary label, fill 0 already is background.

    # 3. Gaussian noise, p = 0.15, sigma ~ U(0, 0.1) * 1000 HU.
    if rng.random() < 0.15:
        sigma = rng.uniform(0.0, 0.1) * 1000.0
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

    # 4. Gaussian blur, p = 0.2, sigma ~ U(0.5, 1.5), in-plane, same sigma for the 3 slices (D13).
    if rng.random() < 0.2:
        sigma = rng.uniform(0.5, 1.5)
        x = np.stack([ndimage.gaussian_filter(ch, sigma) for ch in x]).astype(np.float32)

    # 5. Multiplicative brightness, p = 0.15, factor ~ U(0.7, 1.3).
    if rng.random() < 0.15:
        x = x * np.float32(rng.uniform(0.7, 1.3))

    # 6. Contrast, p = 0.15, factor ~ U(0.65, 1.5) around the mean, clipped to the previous range.
    if rng.random() < 0.15:
        f = np.float32(rng.uniform(0.65, 1.5))
        lo, hi, mean = x.min(), x.max(), x.mean()
        x = np.clip((x - mean) * f + mean, lo, hi)

    return x.astype(np.float32), y


class AirwayTrainSlices(Dataset):
    """Annotated axial slices of the 239 training volumes. Index i -> (case, z)."""

    def __init__(self, prepared_dir, index_json, augment=True, seed=0):
        index = json.load(open(index_json))
        self.seed = int(seed)
        self.items = [(cid, z) for cid in sorted(index["train"]) for z in index["train"][cid]["annotated_slices"]]
        self.shapes = {cid: tuple(v["shape"]) for cid, v in index["train"].items()}
        self.vols = _Volumes(prepared_dir)
        self.augment = augment
        self._rng = None

    def __len__(self):
        return len(self.items)

    def rng(self):
        # torch sets a distinct initial seed per worker and per epoch (derived from the
        # Lightning seed), so this is reproducible and decorrelated across workers.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 63))
        return self._rng

    def __getitem__(self, i):
        # `EpochSeededSampler` passes a code = draw_id * len(self) + slice_index, so that the
        # augmentation of every draw is a fixed function of (seed, epoch, position): identical
        # after a resume, and different in every epoch. Plain indices (< len) also work.
        code = int(i)
        i = code % len(self.items)
        if code >= len(self.items):
            self._rng = np.random.default_rng([self.seed, code])
        cid, z = self.items[i]
        x = stack_25d(self.vols.img(cid), z).astype(np.float32)   # [3, H, W] HU
        y = np.asarray(self.vols.lab(cid)[z]).astype(np.uint8)       # [H, W]
        rng = self.rng()
        if self.augment:
            x, y = augment_hu(x, y, rng)
        x = hu_norm(x).astype(np.float32)

        # "Crop to lowest side" (square) then a uniform random 256x256 patch (fullrandom, no
        # positive oversampling).
        h, w = y.shape
        side = min(h, w)
        if h != w:
            oh, ow = (h - side) // 2, (w - side) // 2
            x, y = x[:, oh:oh + side, ow:ow + side], y[oh:oh + side, ow:ow + side]
        if side < PATCH:
            raise ValueError(f"{cid}: in-plane size {side} < {PATCH}")
        r0 = int(rng.integers(0, side - PATCH + 1))
        c0 = int(rng.integers(0, side - PATCH + 1))
        x = x[:, r0:r0 + PATCH, c0:c0 + PATCH]
        y = y[r0:r0 + PATCH, c0:c0 + PATCH]

        yf = y.astype(np.float32)
        target = np.stack([1.0 - yf, yf], axis=0)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(target)),
                {"preprocess": PREPROCESS, "ID": f"{cid}:{z}"})


class AirwayValVolumes(Dataset):
    """The 30 validation volumes, whole, as the released `eval` transform produces them."""

    def __init__(self, prepared_dir, index_json, max_slices=None):
        index = json.load(open(index_json))
        self.cases = sorted(index["val"])
        self.vols = _Volumes(prepared_dir)
        self.max_slices = max_slices  # smoke tests only

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        img = np.asarray(self.vols.img(cid)).astype(np.float32)
        lab = np.asarray(self.vols.lab(cid)).astype(np.float32)
        if self.max_slices is not None:
            zs = np.where(lab.reshape(lab.shape[0], -1).any(axis=1))[0]
            z0 = int(zs[len(zs) // 2]) if len(zs) else 0
            img, lab = img[z0:z0 + self.max_slices], lab[z0:z0 + self.max_slices]
        x = torch.from_numpy(hu_norm(img).astype(np.float32))[None]          # [1, Z, H, W]
        y = torch.from_numpy(np.stack([1.0 - lab, lab], axis=0))              # [2, Z, H, W]
        if x.shape[-2:] != (512, 512):
            # eval transform: Resize(512, 512) bilinear for the image, nearest for the target
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
            y = torch.nn.functional.interpolate(y, size=(512, 512), mode="nearest")
        return x, y, {"preprocess": PREPROCESS, "ID": cid}


class EpochSeededSampler(torch.utils.data.Sampler):
    """Uniform sampling with replacement over the annotated slices (spec D5), reseeded from
    (seed, epoch) every epoch. Lightning calls `set_epoch` at the start of each epoch, also
    after resuming, so a resumed run does not replay an earlier epoch. Yields
    `draw_id * n + index` codes that `AirwayTrainSlices` decodes (and uses to seed the
    augmentation of that draw)."""

    def __init__(self, n, num_samples, seed):
        self.n, self.num_samples, self.seed, self.epoch = n, num_samples, int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        idx = torch.randint(self.n, (self.num_samples,), generator=g).tolist()
        base = (self.epoch + 1) * self.num_samples          # draw ids never collide across epochs
        return iter([(base + j) * self.n + k for j, k in enumerate(idx)])

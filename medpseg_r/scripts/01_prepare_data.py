"""Convert the 239 train and 30 val ATM'26 volumes into the arrays MEDPSeg-R trains on.

Run with the medpseg env (it provides SimpleITK/torch and the medpseg package):
    python scripts/01_prepare_data.py

What it does, per case:
  * image: read exactly as MEDPSeg inference does (`medpseg.image_read.read_preprocess`,
    norm=False), i.e. SimpleITK array [Z, Y, X] plus the lungmask "bed on bottom" flip.
    Stored as int16 HU (values are asserted to be integers in int16 range).
  * label: read with SimpleITK, the SAME flip applied, geometry asserted equal to the image,
    values asserted to be {0, 1}. Stored as uint8.
  * train only: the list of axial slices that contain at least one airway voxel
    (MEDPSeg trains on annotated slices only, paper Sec. 2.1).

Test cases are never read: the case lists come from splits.json["train"] / ["val"], and any
overlap with splits.json["test"] aborts the script.
"""
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk

from medpseg.image_read import read_preprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = os.path.join(REPO, "splits", "atm26_split.json")
DATA = os.environ.get("MEDPSEG_R_DATA", os.path.join(REPO, "medpseg_r", "data"))
TRAIN_DIR = os.path.join(DATA, "ATM26")                     # written by 00_extract_train.py
VAL_DIR = os.environ["ATM26_VAL_DIR"]                         # val volumes: imagesTr/, labelsTr/
OUT = os.path.join(DATA, "prepared")


def flip_axes(direction):
    """The axes `read_preprocess` flips (image_read.py:53-54)."""
    d = np.asarray(direction)
    if len(d) != 9:
        return ()
    return tuple(int(a) for a in np.where(d[[0, 4, 8]][::-1] < 0)[0])


def prepare(cid, img_path, lab_path, want_slices):
    img_out = f"{OUT}/{cid}_img.npy"
    lab_out = f"{OUT}/{cid}_lab.npy"
    meta_out = f"{OUT}/{cid}_meta.json"
    if os.path.exists(meta_out):
        return json.load(open(meta_out))

    data, original_shape, origin, spacing, directions, image = read_preprocess(img_path, norm=False)
    data = data.numpy()
    if not np.all(np.isfinite(data)) or not np.array_equal(data, np.round(data)):
        raise ValueError(f"{cid}: image is not integer HU")
    if data.min() < -32768 or data.max() > 32767:
        raise ValueError(f"{cid}: HU outside int16")

    lab_img = sitk.ReadImage(lab_path)
    if lab_img.GetSize() != image.GetSize():
        raise ValueError(f"{cid}: label size {lab_img.GetSize()} != image {image.GetSize()}")
    if not np.allclose(lab_img.GetSpacing(), image.GetSpacing(), atol=1e-4):
        raise ValueError(f"{cid}: label spacing differs")
    if not np.allclose(lab_img.GetOrigin(), image.GetOrigin(), atol=1e-3):
        raise ValueError(f"{cid}: label origin differs")
    if not np.allclose(lab_img.GetDirection(), image.GetDirection(), atol=1e-6):
        raise ValueError(f"{cid}: label direction differs")
    lab = sitk.GetArrayFromImage(lab_img)
    axes = flip_axes(directions)
    if axes:
        lab = np.flip(lab, axes).copy()
    vals = np.unique(lab)
    if not set(vals.tolist()) <= {0, 1}:
        raise ValueError(f"{cid}: label values {vals}")
    if lab.shape != data.shape:
        raise ValueError(f"{cid}: shapes {lab.shape} vs {data.shape}")

    np.save(img_out + ".part.npy", data.astype(np.int16))
    os.replace(img_out + ".part.npy", img_out)
    np.save(lab_out + ".part.npy", lab.astype(np.uint8))
    os.replace(lab_out + ".part.npy", lab_out)

    meta = {
        "case": cid,
        "shape": list(data.shape),
        "spacing_zyx": [float(s) for s in spacing],
        "direction": [float(d) for d in directions],
        "flip_axes": list(axes),
        "hu_min": float(data.min()), "hu_max": float(data.max()),
        "airway_voxels": int(lab.sum()),
        "image_path": img_path, "label_path": lab_path,
    }
    if want_slices:
        meta["annotated_slices"] = [int(z) for z in np.where(lab.reshape(lab.shape[0], -1).any(axis=1))[0]]
    with open(meta_out, "w") as f:
        json.dump(meta, f)
    return meta


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = json.load(open(SPLIT))
    ids = {k: sorted(c.split("/")[1] for c in sp[k]) for k in ("train", "val", "test")}
    assert len(ids["train"]) == 239 and len(ids["val"]) == 30 and len(ids["test"]) == 30
    assert not (set(ids["train"]) & set(ids["test"])) and not (set(ids["val"]) & set(ids["test"]))
    assert not (set(ids["train"]) & set(ids["val"]))

    index = {"split_file": SPLIT, "train": {}, "val": {}}
    t0 = time.time()
    jobs = [("train", c, f"{TRAIN_DIR}/imagesTr/{c}_0000.mha", f"{TRAIN_DIR}/labelsTr/{c}.mha") for c in ids["train"]]
    jobs += [("val", c, f"{VAL_DIR}/imagesTr/{c}_0000.mha", f"{VAL_DIR}/labelsTr/{c}.mha") for c in ids["val"]]
    for n, (split, cid, ip, lp) in enumerate(jobs, 1):
        for p in (ip, lp):
            if not os.path.exists(p):
                sys.exit(f"missing {p} (has 00_extract_train.py finished?)")
        m = prepare(cid, ip, lp, want_slices=(split == "train"))
        index[split][cid] = {k: m[k] for k in ("shape", "spacing_zyx", "flip_axes", "airway_voxels")}
        if split == "train":
            index[split][cid]["annotated_slices"] = m["annotated_slices"]
        print(f"[{n}/{len(jobs)}] {split} {cid} shape={m['shape']} flip={m['flip_axes']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    shapes = [tuple(v["shape"][1:]) for s in ("train", "val") for v in index[s].values()]
    non512 = [c for s in ("train", "val") for c, v in index[s].items() if tuple(v["shape"][1:]) != (512, 512)]
    flipped = [c for s in ("train", "val") for c, v in index[s].items() if v["flip_axes"]]
    n_slices = sum(len(v["annotated_slices"]) for v in index["train"].values())
    index["summary"] = {"n_train": len(index["train"]), "n_val": len(index["val"]),
                        "annotated_train_slices": n_slices, "non_512_inplane": non512,
                        "flipped_cases": flipped}
    with open(f"{OUT}/index.json", "w") as f:
        json.dump(index, f)
    print("SUMMARY", json.dumps(index["summary"]))


if __name__ == "__main__":
    main()

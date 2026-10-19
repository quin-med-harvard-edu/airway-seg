"""MEDPSeg-R inference on the ATM'26 val / test split or any image folder, saving binary masks AND the airway probability.

Reproduces the released inference path exactly for the airway output:
  read_preprocess(norm=True) -> (resize to 512x512 trilinear if needed) -> poly_stack_predict with
  extended_2d=1 -> softmax "atm" channel 1 -> (resize back nearest) -> > 0.5 -> undo the orientation flip
(poly_pipeline.py:246-322, pipeline.py:449-451). By default the network computes only its airway outputs
(airway_fast.py); airway outputs are bit-identical to the full forward (tests/test_airway_fast.py).
`--check-equivalence` compares this path with the released PolySegmentationPipeline on a few slices.

Outputs, per case, under --out-root/<split>/:
  ATM26_<cid>.npz (ATM'26) or <name>.npz (--image-dir): raw (uint8, = threshold 0.5), post (largest component),
  prob (float16 airway probability), original_shape; for --image-dir splits also <name>/medpseg_airway.nii.gz
  with the scan's geometry.

    python predict_medpseg_r_v2.py --weight runs/medpseg_r_seed20251014/checkpoints/epoch_035.ckpt \
        --splits test val --out-root outputs/epoch035
    python predict_medpseg_r_v2.py --weight ... --splits mycohort --image-dir /path/to/images --out-root ...
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from airway_fast import enable_airway_only_forward  # noqa: E402

REPO = os.path.dirname(HERE)
SPLIT = os.path.join(REPO, "splits", "atm26_split.json")
EXPECTED = "medseg_25d_a100_long_silver_gold_gdl"
logger = logging.getLogger("medpseg_r.predict_v2")


def unflip(array, dir_array):
    dir_array = np.asarray(dir_array)
    if len(dir_array) != 9:
        return array
    axes = np.where(dir_array[[0, 4, 8]][::-1] < 0)[0]
    return np.flip(array, axes).copy() if len(axes) else array


def jobs_for(split, image_dir=None):
    """(name, image path, NIfTI output dir or None) for every case of a split.
    val / test: ATM'26 IDs from splits.json, images in $ATM26_VAL_IMAGES / $ATM26_TEST_IMAGES
    (<id>_0000.mha). Any other split name: every *.nii.gz / *.mha in --image-dir."""
    if split in ("val", "test"):
        ids = [c.split("/")[1] for c in json.load(open(SPLIT))[split]]
        d = image_dir or os.environ[{"test": "ATM26_TEST_IMAGES", "val": "ATM26_VAL_IMAGES"}[split]]
        return [(f"ATM26_{c}", os.path.join(d, f"{c}_0000.mha"), None) for c in ids]
    import glob
    assert image_dir, f"--image-dir is required for split '{split}'"
    out = []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")) + glob.glob(os.path.join(image_dir, "*.mha"))):
        name = os.path.basename(path).replace(".nii.gz", "").replace(".mha", "")
        out.append((name, path, "nifti"))
    return out


@torch.no_grad()
def predict_volume(model, data, batch_size, device):
    """data: [Z, H, W] float in [0,1] (read_preprocess output, flipped frame). Returns P(airway) [Z, H, W]."""
    from medpseg.poly_pipeline import poly_stack_predict
    volume = data.unsqueeze(0).unsqueeze(0)
    pre_shape = volume.shape[2:]
    if pre_shape[1] != 512 or pre_shape[2] != 512:
        volume = F.interpolate(volume.to(device), (pre_shape[0], 512, 512), mode="trilinear").cpu()
    out = poly_stack_predict(model, volume, batch_size=batch_size, device=device, cli=True)
    atm = out["atm"]                                     # [1, 2, Z, H, W] softmax
    if atm.shape[2:] != pre_shape:
        atm = F.interpolate(atm.cpu(), pre_shape, mode="nearest")
    return atm[0, 1].float().cpu().numpy()


def load_model(weight, device, fast):
    from medpseg.poly_seg_2d_module import PolySeg2DModule
    model = PolySeg2DModule.load_from_checkpoint(weight, map_location="cpu").eval()
    assert EXPECTED in model.hparams.experiment_name, model.hparams.experiment_name
    if fast:
        enable_airway_only_forward(model)
    return model.to(device)


def check_equivalence(weight, image, device, nslices=12):
    """Released PolySegmentationPipeline vs this script on a few slices of one scan."""
    from medpseg.image_read import read_preprocess
    from medpseg.poly_pipeline import PolySegmentationPipeline

    class Q:
        def put(self, item):
            pass
    data, _, _, _, directions, original_image = read_preprocess(image, norm=True)
    z0 = data.shape[0] // 2
    sub = data[z0:z0 + nslices].contiguous()
    pipe = PolySegmentationPipeline(weight=weight, batch_size=2, cpu=device.type == "cpu", post=False, cli=True)
    _, airway_ref, _ = pipe(sub.unsqueeze(0).unsqueeze(0), Q(), None, original_image, "chk", np.asarray(directions), False)
    mine = predict_volume(load_model(weight, device, fast=True), sub, 2, device) > 0.5
    ref = np.asarray(airway_ref).astype(bool).reshape(mine.shape)
    print(f"equivalence on {nslices} slices: identical={np.array_equal(ref, mine)}, "
          f"differing voxels={int((ref != mine).sum())}, airway voxels={int(ref.sum())}", flush=True)
    return np.array_equal(ref, mine)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--image-dir", default=None, help="images for a non-ATM split (*.nii.gz / *.mha)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--full-forward", action="store_true")
    ap.add_argument("--check-equivalence", default=None, help="image path: compare with the released pipeline and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="(%(asctime)s) %(message)s")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    if args.check_equivalence:
        sys.exit(0 if check_equivalence(args.weight, args.check_equivalence, device) else 1)

    import SimpleITK as sitk
    from medpseg.image_read import read_preprocess
    from medpseg.poly_pipeline import get_connected_components

    model = load_model(args.weight, device, fast=not args.full_forward)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "WEIGHT.txt"), "w") as f:
        f.write(f"{os.path.abspath(args.weight)}\n{model.hparams.experiment_name}\nfast={not args.full_forward}\n")
    logger.info("device %s, weight %s", device, args.weight)

    for split in args.splits:
        jobs = jobs_for(split, args.image_dir)
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)
        for i, (name, path, nifti_out) in enumerate(jobs, 1):
            out_path = os.path.join(out_dir, f"{name}.npz")
            if os.path.exists(out_path):
                logger.info("[%s %d/%d] %s cached", split, i, len(jobs), name)
                continue
            t0 = time.time()
            data, original_shape, _, _, directions, original_image = read_preprocess(path, norm=True)
            dir_array = np.asarray(directions)
            prob = predict_volume(model, data, args.batch_size, device)
            raw = (prob > 0.5).astype(np.uint8)
            post = get_connected_components(raw, return_largest=1)[0].astype(np.uint8)
            raw_u, post_u, prob_u = unflip(raw, dir_array), unflip(post, dir_array), unflip(prob, dir_array)
            np.savez_compressed(out_path + ".part.npz", raw=raw_u, post=post_u, prob=prob_u.astype(np.float16),
                                original_shape=np.array(original_shape))
            os.replace(out_path + ".part.npz", out_path)
            if nifti_out is not None:
                os.makedirs(os.path.join(out_dir, name), exist_ok=True)
                img = sitk.GetImageFromArray(raw_u)
                img.CopyInformation(original_image)
                sitk.WriteImage(img, os.path.join(out_dir, name, "medpseg_airway.nii.gz"))
            logger.info("[%s %d/%d] %s done in %.0f s, airway voxels %d", split, i, len(jobs), name,
                        time.time() - t0, int(raw.sum()))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    logger.info("all done")


if __name__ == "__main__":
    main()

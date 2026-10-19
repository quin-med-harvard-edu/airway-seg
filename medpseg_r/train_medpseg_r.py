"""Train MEDPSeg-R: the released MEDPSeg architecture and training recipe, trained from
ImageNet initialisation on the 239 ATM'26 training volumes (airway labels only) and selected
on the 30 validation volumes. The 30 test volumes are never loaded.

The network, the loss, the deep supervision, the per-format loss routing, the optimiser and
the scheduler are the UNMODIFIED released code (`medpseg.poly_seg_2d_module.PolySeg2DModule`),
built from the hyper-parameters stored in the released checkpoint. This script only supplies
what the authors did not release: data, sampler, augmentation and the Lightning Trainer.
See docs/MEDPSEG_TRAINING_SPEC.md and README.md.

Run with the medpseg env:
    python train_medpseg_r.py --benchmark-steps 60 --micro-batch 12     # memory/speed check
    python train_medpseg_r.py --micro-batch 12                           # full training
"""
import argparse
import copy
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, RandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from medpseg_r_data import AirwayTrainSlices, AirwayValVolumes, EpochSeededSampler  # noqa: E402
from airway_fast import enable_airway_only_forward  # noqa: E402

import medpseg  # noqa: E402
REPO = os.path.dirname(HERE)
RELEASED_CKPT = os.path.join(os.path.dirname(medpseg.__file__), "poly_medseg_25d_fix.ckpt")  # hparams only
SPLIT = os.path.join(REPO, "splits", "atm26_split.json")
PREPARED = os.path.join(os.environ.get("MEDPSEG_R_DATA", os.path.join(HERE, "data")), "prepared")
EXPECTED_NAME = "medseg_25d_a100_long_silver_gold_gdl"   # asserted by poly_pipeline.py at inference

# Released recipe (checkpoint hyper_parameters / loops), spec section 3.1.
EFFECTIVE_BATCH = 48
STEPS_PER_EPOCH = 3312
SAMPLES_PER_EPOCH = EFFECTIVE_BATCH * STEPS_PER_EPOCH   # 158,976


# --------------------------------------------------------------------------- logging shim
class _KeyLog:
    def __init__(self, path, key):
        self.path, self.key = path, key

    def log(self, value):
        with open(self.path, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{self.key}\t{float(value)}\n")


class _ExperimentProxy:
    """Lets the released module call `self.logger.experiment["training/x"].log(v)` (Neptune
    syntax, poly_seg_2d_module.py:750-751) while everything else goes to the CSV writer."""

    def __init__(self, inner, path):
        self._inner, self._path = inner, path

    def __getitem__(self, key):
        return _KeyLog(self._path, key)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ShimCSVLogger(CSVLogger):
    @property
    def experiment(self):
        inner = super().experiment
        os.makedirs(self.log_dir, exist_ok=True)
        return _ExperimentProxy(inner, os.path.join(self.log_dir, "composite_metrics.tsv"))


class EpochReport(Callback):
    """Plain-text progress lines (the run lives in a screen session)."""

    def __init__(self, path):
        self.path = path
        self.t0 = None

    def _w(self, msg):
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        self._w(f"epoch {trainer.current_epoch} start, lr={lr:.4e}, global_step={trainer.global_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 500 == 0:
            loss = trainer.callback_metrics.get("loss_step", trainer.callback_metrics.get("loss"))
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
            el = time.time() - self.t0
            self._w(f"  epoch {trainer.current_epoch} micro-batch {batch_idx}/{trainer.num_training_batches} "
                    f"loss={float(loss) if loss is not None else float('nan'):.4f} "
                    f"elapsed={el/60:.1f} min peak_mem={mem:.1f} GiB")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        keys = [k for k in ("val_loss_3d", "val_loss_3d_epoch", "airway_dice_3d") if k in m]
        self._w(f"epoch {trainer.current_epoch} validation: " + ", ".join(f"{k}={float(m[k]):.4f}" for k in keys))

    def on_train_epoch_end(self, trainer, pl_module):
        self._w(f"epoch {trainer.current_epoch} done in {(time.time() - self.t0)/60:.1f} min")


# --------------------------------------------------------------------------- helpers
def released_hparams():
    ck = torch.load(RELEASED_CKPT, map_location="cpu", weights_only=False)
    hp = copy.deepcopy(dict(ck["hyper_parameters"]))
    assert ck["hparams_name"] == "hparams"
    del ck
    return hp


def check_split_and_data(index_json, partial=False):
    sp = json.load(open(SPLIT))
    test = {c.split("/")[1] for c in sp["test"]}
    train = {c.split("/")[1] for c in sp["train"]}
    val = {c.split("/")[1] for c in sp["val"]}
    index = json.load(open(index_json))
    got_train, got_val = set(index["train"]), set(index["val"])
    if partial:   # smoke test on a subset: still must be subsets of train/val
        assert got_train <= train and got_val <= val
    else:
        assert got_train == train, "prepared train set differs from splits.json"
        assert got_val == val, "prepared val set differs from splits.json"
    assert not (got_train | got_val) & test, "TEST CASE IN TRAINING DATA"
    for c in got_train | got_val:
        assert c not in test
    leaked = [f for f in os.listdir(PREPARED) if f.split("_img")[0].split("_lab")[0].split("_meta")[0] in test]
    assert not leaked, f"test files in prepared dir: {leaked[:3]}"
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "medpseg_r_seed20251014"))
    ap.add_argument("--micro-batch", type=int, default=12,
                    help="per-step batch; gradients are accumulated to an effective batch of 48")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20251014)
    ap.add_argument("--workers", type=int, default=12, help="training dataloader workers")
    ap.add_argument("--val-stack-workers", type=int, default=4,
                    help="workers of the released real_time_stack_predict during validation")
    ap.add_argument("--resume", default="auto", help="'auto' (use last.ckpt if present), 'none', or a path")
    ap.add_argument("--benchmark-steps", type=int, default=0,
                    help="run only this many training micro-batches (no validation) and report memory/speed")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test: tiny run, no GPU required")
    ap.add_argument("--index", default=None, help="index.json override (smoke tests only)")
    ap.add_argument("--full-forward", action="store_true",
                    help="compute all released outputs during training/validation (default: airway outputs only; "
                         "identical airway results, see airway_fast.py and tests/test_airway_fast.py)")
    args = ap.parse_args()

    if EFFECTIVE_BATCH % args.micro_batch:
        sys.exit(f"--micro-batch must divide {EFFECTIVE_BATCH}")
    acc = EFFECTIVE_BATCH // args.micro_batch
    index_json = args.index or os.path.join(PREPARED, "index.json")
    if args.index and not args.smoke:
        sys.exit("--index is for smoke tests only")
    index = check_split_and_data(index_json, partial=args.smoke)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")     # precision_str 'high'
    torch.backends.cudnn.benchmark = True

    if not args.smoke and not torch.cuda.is_available():
        sys.exit("No CUDA device visible. A GPU is required for training.")

    os.makedirs(args.out, exist_ok=True)

    # ---- hyper-parameters: released dict verbatim, only bookkeeping fields changed (D1)
    hp = released_hparams()
    hp.update({
        "experiment_name": f"{EXPECTED_NAME}__medpseg_r_atm26_airway_seed{args.seed}",
        "tag": f"{EXPECTED_NAME}__medpseg_r_atm26_airway_seed{args.seed}",
        "cli_args": " ".join(sys.argv),
        "batch_size": EFFECTIVE_BATCH,
        "acum": acc,
        "medpseg_r_micro_batch": args.micro_batch,
        "nworkers": args.val_stack_workers,
        "max_epochs": args.max_epochs,
        "ngpus": 1,
        "pytorch_version": torch.__version__,
        "pl_version": pl.__version__,
        "medpseg_r_seed": args.seed,
        "medpseg_r_split_file": SPLIT,
        "medpseg_r_train_cases": len(index["train"]),
        "medpseg_r_val_cases": len(index["val"]),
        "medpseg_r_init": "ImageNet EfficientNet-B7 (efficientnet_pytorch 0.7.1); NOT the released MEDPSeg weights",
        "medpseg_r_note": "airway-only training; main (lung/lesion) and vessel heads are untrained",
        "medpseg_r_airway_only_forward": not args.full_forward,
    })
    assert EXPECTED_NAME in hp["experiment_name"]
    for k, v in {"compound_coef": 7, "stem_replacement": True, "circulatory_branch": True,
                 "soft_circulatory": True, "self_attention": True, "deep_supervision": True,
                 "expand_bifpn": "upsample", "seg_nout": 4, "nin": 3, "bifpn_channels": 128,
                 "combined_loss": True, "celoss": True, "combined_gdl": True, "include_background": False,
                 "lr": 1e-4, "opt": "AdamW", "weight_decay": 1e-5, "scheduling": "step",
                 "scheduling_factor": 0.985, "val_3d": True, "extended_2d": 1, "eval_batch_size": 8,
                 "scratch": False, "pretrained_weights": None, "poly_level": 3, "pretraining": False}.items():
        assert hp[k] == v, (k, hp[k], v)
    if args.smoke:
        hp["cpu"] = True
        hp["eval_batch_size"] = 4

    from medpseg.poly_seg_2d_module import PolySeg2DModule
    model = PolySeg2DModule(hp)
    if not args.full_forward:
        enable_airway_only_forward(model)   # training/validation only; inference uses the released full forward

    # ---- data
    train_ds = AirwayTrainSlices(PREPARED, index_json, augment=True, seed=args.seed)
    val_ds = AirwayValVolumes(PREPARED, index_json, max_slices=8 if args.smoke else None)
    n_samples = SAMPLES_PER_EPOCH if not args.smoke else 4 * args.micro_batch
    sampler = EpochSeededSampler(len(train_ds), n_samples, args.seed)
    workers = 0 if args.smoke else args.workers
    train_dl = DataLoader(train_ds, batch_size=args.micro_batch, sampler=sampler, drop_last=True,
                          num_workers=workers, pin_memory=not args.smoke,
                          persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    val_dl = DataLoader(val_ds if not args.smoke else torch.utils.data.Subset(val_ds, [0]),
                        batch_size=1, shuffle=False, num_workers=0 if args.smoke else 2)

    # ---- trainer
    ck_dir = os.path.join(args.out, "checkpoints")
    # best checkpoint by val_loss_3d (as released)
    ckpt_cb = ModelCheckpoint(dirpath=ck_dir, monitor="val_loss_3d", mode="min", save_top_k=1, save_last=False,
                              filename="medpseg_r-{epoch:03d}-{val_loss_3d:.4f}", auto_insert_metric_name=False,
                              every_n_epochs=1)
    # every epoch + last.ckpt. A separate callback is needed: in Lightning >= 2.5 `save_last` of a monitored
    # callback only refreshes last.ckpt in epochs that set a new best.
    epoch_cb = ModelCheckpoint(dirpath=ck_dir, monitor=None, save_top_k=-1, save_last=True, every_n_epochs=1,
                               filename="epoch_{epoch:03d}", auto_insert_metric_name=False)
    callbacks = [ckpt_cb, epoch_cb, EarlyStopping(monitor="val_loss_3d", mode="min", patience=200),
                 LearningRateMonitor(logging_interval="epoch"), EpochReport(os.path.join(args.out, "progress.log"))]
    trainer_kw = dict(
        accelerator="cpu" if args.smoke else "gpu", devices=1,
        precision="32-true" if args.smoke else "16-mixed",
        max_epochs=2 if args.smoke else (1 if args.benchmark_steps else args.max_epochs),
        accumulate_grad_batches=acc, gradient_clip_val=None,
        num_sanity_val_steps=0 if args.benchmark_steps else 1,
        logger=ShimCSVLogger(args.out, name="logs"), callbacks=callbacks,
        log_every_n_steps=50, enable_progress_bar=False, use_distributed_sampler=False,
        deterministic=False, benchmark=True,
    )
    if args.benchmark_steps:
        trainer_kw.update(limit_train_batches=args.benchmark_steps, limit_val_batches=0, enable_checkpointing=False,
                          callbacks=[EpochReport(os.path.join(args.out, "benchmark.log"))])
    if args.smoke:
        trainer_kw.update(limit_train_batches=4)
    trainer = pl.Trainer(**trainer_kw)

    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "effective_batch": EFFECTIVE_BATCH, "accumulate": acc,
                   "samples_per_epoch": n_samples, "train_slices": len(train_ds), "val_cases": len(val_ds),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "hparams": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                               for k, v in hp.items()}}, f, indent=1)
    print(f"train slices={len(train_ds)} val volumes={len(val_ds)} micro_batch={args.micro_batch} "
          f"accumulate={acc} samples/epoch={n_samples}", flush=True)

    ckpt_path = None
    # newest last*.ckpt: Lightning writes last-v1.ckpt etc. when a stale last.ckpt already exists
    import glob
    lasts = sorted(glob.glob(os.path.join(ck_dir, "last*.ckpt")), key=os.path.getmtime)
    last = lasts[-1] if lasts else None
    if args.resume == "auto" and last and not args.benchmark_steps:
        ckpt_path = last
    elif args.resume not in ("auto", "none"):
        ckpt_path = args.resume
    if ckpt_path:
        print(f"Resuming from {ckpt_path}", flush=True)

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)

    if args.benchmark_steps:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        sps = args.benchmark_steps * args.micro_batch / dt
        msg = (f"BENCHMARK micro_batch={args.micro_batch} steps={args.benchmark_steps} time={dt:.0f}s "
               f"samples/s={sps:.1f} peak_mem={mem:.1f} GiB of "
               f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB; "
               f"est. training time/epoch (excl. validation) = {SAMPLES_PER_EPOCH / sps / 60:.0f} min")
        print(msg, flush=True)
        with open(os.path.join(args.out, "benchmark.log"), "a") as f:
            f.write(msg + "\n")
    else:
        print(f"best checkpoint: {ckpt_cb.best_model_path} (val_loss_3d={ckpt_cb.best_model_score})", flush=True)
        with open(os.path.join(args.out, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(f"{ckpt_cb.best_model_path}\n{ckpt_cb.best_model_score}\n")


if __name__ == "__main__":
    main()

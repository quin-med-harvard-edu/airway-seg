# MEDPSeg-R: MEDPSeg retrained on our ATM'26 training partition

**Why.** The released MEDPSeg weights (`poly_medseg_25d_fix.ckpt`) were trained on ATM airway
data (MEDPSeg paper, arXiv 2312.02365, Table 1: 399 volumes, 339/30/30). That data very likely
includes most of our 30 adult test volumes, which come from the ATM'26 (= ATM'22) training
pool. To get a member that has never seen the test set, we retrain MEDPSeg on exactly the
partition AirMorph-R used: 239 train / 30 val (checkpoint selection) / 30 test (never loaded).

**What "reimplementation" means here.** The MEDPSeg authors released the network and the
Lightning module (loss, deep supervision, per-format loss routing, optimiser, scheduler),
but not the training driver, data module or sampler. We reuse the released module
**unmodified** (`medpseg.poly_seg_2d_module.PolySeg2DModule`), built from the
hyper-parameters stored in the released checkpoint, and write only the missing parts
(data preparation, dataset/augmentation, sampler, Trainer).

Full specification with sources: `docs/MEDPSEG_TRAINING_SPEC.md`.

## Files

| file | role |
|---|---|
| `scripts/00_extract_train.py` | extracts the 239 train volumes from the ATM'26 Task-1 zip (val/test are never extracted here) |
| `scripts/01_prepare_data.py` | reads each volume exactly like MEDPSeg inference (`read_preprocess`), stores int16 HU + uint8 label, lists annotated slices, asserts geometry, label values and that no test ID is present |
| `medpseg_r_data.py` | training dataset (2.5D stack, nnU-Net 2D augmentation, HU normalisation, 256 crop), validation dataset (whole volumes) and the resume-safe `EpochSeededSampler` |
| `airway_fast.py` | airway-only forward pass (bit-identical airway outputs and gradients, ~40% less compute) |
| `train_medpseg_r.py` | builds the released module from the released hyper-parameters, trains with Lightning, selects on `val_loss_3d` |
| `predict_medpseg_r.py` | inference identical to the released pipeline for the airway output; saves `raw` (threshold 0.5), `post` (released `--post`: largest component) and the probability |
| `run_train.sh`, `run_infer.sh` | launchers |
| `tests/` | train-side stacking equals inference stacking (bit-exact), label alignment, augmentation sanity, airway-only forward equivalence |
| `docs/MEDPSEG_TRAINING_SPEC.md` | the full specification with sources |

## Decision register (deviations and choices)

Numbers refer to `docs/MEDPSEG_TRAINING_SPEC.md` §5.2.

| # | Item | Released MEDPSeg | MEDPSeg-R | Why |
|---|---|---|---|---|
| R1 | Training data | ~6,150 CTs; 5 label formats incl. ATM airway (339) | 239 ATM'26 volumes, airway labels only | Our partition; no lung/lesion/vessel labels. Non-airway heads receive no gradient and stay at initialisation (they are never used for airway). |
| R2 | Initialisation | ImageNet B7, silver→gold | ImageNet EfficientNet-B7 (efficientnet_pytorch 0.7.1), no silver stage | No silver data; **never** initialised from the released weights (test contamination). |
| R3 | Architecture, loss, DS weights, optimiser, LR schedule | as released | identical (released code + released hparams; only bookkeeping fields changed) | Fidelity; inference path unchanged. |
| R4 | Batch | 48 on one A100-80GB | effective 48 = micro-batch × gradient accumulation on one GPU | Our GPU has 24 GB. BatchNorm statistics see the micro-batch (e.g. 12) instead of 48. |
| R5 | Epoch | 3312 steps × 48 = 158,976 samples | same | Keeps the per-epoch LR decay (×0.985) on the same sample count. |
| R6 | Epochs | best at epoch 22 of a resumed run (paper: ~50 epochs, 120 h) | max 50 epochs, best `val_loss_3d` kept | Paper's budget. |
| R7 | Sampling | custom balanced sampler (not released) | uniform with replacement over pooled annotated slices | Paper §2.1 "uniform random sampling, repeating samples"; only annotated slices. |
| R8 | Augmentation | `256_aug_2d` (nnU-Net 2D, verbatim list in checkpoint) | same list and parameters; in-plane mirroring only; spatial aug on full 512² then random 256² crop; intensity aug identical for the 3 slices | Unrecorded details resolved as in spec D10–D13. |
| R9 | Validation | 127 volumes, `val_3d` | 30 val volumes, released `val_3d` code path | Our partition. |
| R10 | Logger | Neptune | CSV + a shim for the module's Neptune-style calls | Module not edited. |
| R11 | Software | torch 1.12.1, PL 2.0.5 | torch 2.5.1, PL 2.6.5, MONAI 1.3.0 (same MONAI as the released env) | The inference env must load the result. |
| R12 | Seed | not recorded | 20251014 | Reported. |
| R13 | Forward during training/validation | full network (all heads) | airway outputs only (`airway_fast.py`, default from epoch 2) | Airway outputs, loss and all 475 gradients are bit-identical to the full forward (`tests/test_airway_fast.py`, float64, max diff 0.0); ~40 % less compute. The saved checkpoint keeps every released module; inference uses the released full forward. Epochs 0-1 ran with the full forward. |
| R14 | Sampler seeding | custom sampler | `EpochSeededSampler`: indices and each draw's augmentation are a fixed function of (seed, epoch, position) | Resume-safe (a resumed run does not replay epoch 0). Epochs 0-1 used a single seeded `RandomSampler`. |
| R15 | Checkpoints | best only | best (`medpseg_r-<epoch>-<val_loss_3d>.ckpt`) + every epoch (`epoch_XXX.ckpt`) + rolling `last*.ckpt` | Lightning 2.6 refreshes `last.ckpt` only in epochs with a new best; a second ModelCheckpoint (monitor=None, save_top_k=-1) saves every epoch, and training resumes from the newest `last*.ckpt`. The run was restarted once from the end of epoch 1 to add this. |

## Run order

Environment: the released `medpseg` package (it provides the network, the Lightning module and the
released checkpoint whose hyper-parameters are reused; its weights are never loaded).

```
export ATM26_TASK1_ZIP=/path/to/ATM26-Task-1-Binary-Airway-Segmentation-TrainBatch.zip
export ATM26_VAL_DIR=/path/to/val          # imagesTr/ and labelsTr/ of the 30 val volumes
python scripts/00_extract_train.py
python scripts/01_prepare_data.py
bash run_train.sh                          # data tests, micro-batch benchmark, training
export ATM26_TEST_IMAGES=/path/to/test/images ATM26_VAL_IMAGES=/path/to/val/imagesTr
bash run_infer.sh 035                      # epoch 35 is the final model used in the paper
```

The final model is epoch 35 (val loss 0.0397, val airway Dice 0.933) of a 50-epoch run.

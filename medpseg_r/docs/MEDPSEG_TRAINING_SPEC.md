# MEDPSeg retraining specification (airway-only, ATM'26 adult 239/30/30)

Status: research and specification only. No training code is written here, and no existing file was modified.
Prepared 2026-10-14.

Confidence tags:
- **[CKPT]**: read directly from `poly_medseg_25d_fix.ckpt`, either its tensors or its saved Lightning metadata.
- **[CODE]**: the released source code. Paths are relative to `or_all3_run/code/medpseg/medpseg/` unless stated.
- **[PAPER]**: the arXiv 2312.02365 v2 text (`medpseg_r/docs/medpseg_paper_arxiv2312.02365v2.txt`, cited as `paper:L<line>`), or v1 where stated.
- **[INFERRED]**: reasoned from the above but not stated anywhere.
- **[UNKNOWN]**: no source; a choice is required.

---

## 0. Headline findings (read first)

1. **The released checkpoint is a full Lightning checkpoint, not a weights-only file.** Besides `state_dict` it contains `hyper_parameters` (all ~130 training hparams), the exact training command line (`cli_args`), human-readable train and eval transform pipelines, `optimizer_states`, `lr_schedulers`, the AMP GradScaler state, and the ModelCheckpoint/EarlyStopping callback state. Most of what the paper leaves out can therefore be read off the run that produced the weights. The verbatim dump is in Appendix A. **[CKPT]**
2. The command that trained the released weights was:
   ```
   starter.py poly_2d_train -en medseg_25d_a100_long_silver_gold_gdl --preprocess gold_poly_3levels_circulatory_balanced
   --transform_str 256_aug_2d --eval_transform_str eval -seg_nout 4 -nin 3 -bs 48 -ebs 8 -nw 1 -lr 1e-4 -o AdamW -me 1000
   --patience 200 --extended_2d 1 --orientation positive_0 --weight_decay 1e-5 --scheduling_factor 0.985 --expand_bifpn upsample
   --ngpus 1 --val_3d --poly_level 3 --compound_coef 7 --stem_replacement --precision 16 --precision_str high --combined_loss
   --celoss --combined_gdl --circulatory_branch --self_attention --deep_supervision --soft_circulatory --dont_shuffle -ckpt models/medseg_25d_a100_long_silver_gold_gdl-epoch=1-step=6624-...ckpt
   ```
   **[CKPT `hyper_parameters.cli_args`]** `starter.py` and the DLPT data module (v0.4.2.1) were never released. **[CODE README / GitHub]**
3. **Checkpoint versus paper discrepancies.** Where they disagree, the checkpoint wins, because the checkpoint is what the inference wrapper runs:
   - Batch size is **48** in the checkpoint, not the paper's 25 (`paper:L539`).
   - An epoch is **3312 steps × 48 = 158,976 samples**, not 150,000 (`paper:L276-278`).
   - The loss is the **sum** GDL + CE (`combined_loss.py:65`), not the "mean" in `paper:L537`. Under Adam this makes no practical difference, because the ratio between the two terms is the same.
4. **Airway head = 2-channel softmax `[bg, airway]`** (`soft_circulatory=True`). The released pipeline thresholds channel 1 at > 0.5 (`poly_pipeline.py:322`), which is equivalent to argmax. **[CKPT+CODE]**
5. **The architecture was verified.** A `MEDSeg` built from the checkpoint hparams loads the released `state_dict` with `strict=True` ("All keys matched"). It has 64,050,474 parameters in 1343 tensors and 73.1 GMACs per 512² slice, which matches paper Table 7 (73 GMACs, 64 M, 14 GB fwd/bwd). **[CKPT, verified by running]**
6. **Proof that all heads were trained.** The optimizer state holds 814 Adam states for 1343 parameter tensors. The 529 tensors that never received a gradient are exactly:
   - EfficientNet-B7 blocks 18–54 (481 tensors), which are never used by the feature extractor;
   - the BiFPN fast-attention weights `p*_w1/p*_w2` (8 per layer × 3 layers × 2 BiFPNs = 48), which are unused because `attention = compound_coef < 6` is False for B7.

   Every head, deep-supervision head and attention gate received gradients. **[CKPT, verified]**
7. **The released model may have seen our test cases.** It was trained on ATM'22 (339 train cases, `paper:L204`). Our ATM'26 adult cases include ATM'22 IDs. **Do not fine-tune from the released checkpoint.** Retrain from ImageNet initialisation, the paper's "Gold data" variant (Table III v1: airway Dice 0.90 ± 0.04, identical to silver > gold). **[PAPER v1 Table III, INFERRED]**
8. **No journal version exists.** The arXiv listing (checked 2026-10-14) shows only v1 (2023-12-04) and v2 (2024-03-25), with the comment "under review", no journal-ref and no journal DOI. v1 and v2 have identical training hyperparameters, and v1 additionally reports the ablations in Table III. The GitHub repo has 4 closed issues and none concern training. The README states "we provide the complete Lightning Module code (in poly_seg_2d_module.py) … future plans to provide training code". The upstream `main` branch differs from our copy only by `weights_only=False` and `torch_compile` loading tweaks; the loss, architecture and pipeline code are identical. **[web, diff run 2026-10-14]**

---

## 1. Architecture (exact reproduction of `poly_medseg_25d_fix.ckpt`)

### 1.1 Constructor arguments

`PolySeg2DModule(hparams)` builds the network at `poly_seg_2d_module.py:126-130`:

```python
MEDSeg(nin=3, nout=4, apply_sigmoid=False, backbone="effnet", expand_bifpn="upsample", pretrained=True,   # pretrained = not scratch(False)
       num_classes_atm=None, num_classes_vessel=None, num_classes_rec=None, stem_replacement=True, new_latent_space=False,
       compound_coef=7, imnet_norm=False, learnable_norm=None, circulatory_branch=True, bifpn_channels=128,
       sam_embedding=False, self_attention=True, deep_supervision=True, con_detecting=False, large=False, soft_circulatory=True)
```

All values are **[CKPT `hyper_parameters`]**. The code then derives the following.

| Component | Value | Source |
|---|---|---|
| Backbone | `efficientnet_pytorch` 0.7.1 `EfficientNet.from_pretrained('efficientnet-b7')` (ImageNet, advprop=False). Head layers (`_conv_head`, `_bn1`, `_avg_pooling`, `_dropout`, `_fc`) are deleted. | `edet/efficientdet/model.py:560-571` [CODE]; 55 blocks and stem width 64 [CKPT]; env has 0.7.1 |
| Stem | `EffNet3DStemReplacement`: `Conv3d(1,64,k=3,valid,bias=False)` over the 3 stacked slices (depth 3→1, H,W −2), then `ZeroPad2d(1)`, then `Conv2d(64,64,k=2,s=2,bias=False)` (→ /2). This replaces the pretrained stem, so it is **randomly initialised**; `_bn0` stays pretrained. | `architecture.py:117-120, 259-278` [CODE]; shapes `(64,1,3,3,3)`, `(64,64,2,2)` [CKPT]; `paper:L363-364` |
| Feature taps | P3 = output of block 3 (32 ch, /2), P4 = block 10 (48 ch, /4), P5 = block 17 (80 ch, /8). Block 18 is executed but unused; blocks 19-54 are skipped. | `model.py:583-596` [CODE]; `conv_channel_coef[7]=[32,48,80]` matches `p3/p4/p5_down_channel` weights `(128,32|48|80,1,1)` [CKPT] |
| Drop-connect | B7 global `drop_connect_rate=0.2`, scaled by `idx/55` per block (≤0.062 for blocks 0-17). Active in train mode. | `model.py:584-590` [CODE]; verified 0.2 |
| BiFPN | 2 independent BiFPN stacks (polymorphic + circulatory), each `repeat=3` layers, 128 ch, P3..P7 (P6 and P7 by max-pool from P5). Uses **unweighted `_forward`**: `attention=False` because compound_coef=7 ≥ 6. | `modeling_efficientdet.py:181-185, 203-207`; `architecture.py:70,74` [CODE+CKPT] |
| Attention gates | `SelfAttention(128)` on each of the 5 BiFPN outputs in **both** branches: 1×1 conv 128→64→32→1, leaky-ReLU, sigmoid, `x*att`. | `self_attention.py`; `modeling_efficientdet.py:157-158, 307-309`; `model.py:500-501, 521-523` [CKPT shapes] |
| Expansion | `expand_bifpn="upsample"` → `nn.UpsamplingBilinear2d(scale_factor=2)` (align_corners=True), applied to each of the 5 BiFPN outputs because deep supervision is on. Resolutions become /1, /2, /4, /8, /16. | `modeling_efficientdet.py:242-245, 316-323` [CODE] |
| Polymorphic heads (PH) | `SegmentationClasssificationHead(128, num_classes=4, num_layers=3)`: 3 × [SeparableConv(dw3×3 + pw1×1) → BN(momentum .01, eps 1e-3) → Swish], then a separable header conv to 4 channels. Main head at /1, plus 4 DS heads (`ds_seg_heads`) at /2../16. | `model.py:418-475` [CODE]; header `(4,128,1,1)` [CKPT] |
| Circulatory heads (CH) | Same head type, `num_classes = 1 + soft_circulatory = 2`. `airway_head`, `vessel_head`, `ds_airway_heads[4]`, `ds_vessel_heads[4]`. | `modeling_efficientdet.py:210`; `model.py:478-553` [CODE]; header `(2,128,1,1)` [CKPT] |
| Input normalisation inside net | None (`learnable_norm=None` → identity; `imnet_norm=False`). The input must already be clipped and normalised to [0,1]. | `architecture.py:101-111` [CODE+CKPT] |
| Params | 64,050,474 (1343 tensors). 529 tensors are dead (see §0.6). | [verified] |

### 1.2 Output channels and activations

The network returns logits. `PolySeg2DModule.forward` (`poly_seg_2d_module.py:161-186`) applies the activations: softmax for any key containing `main`, and softmax for `atm`/`vessel` because `soft_circulatory=True`.

| Key | Shape (input B×3×H×W) | Activation | Channels |
|---|---|---|---|
| `main` | B×4×H×W | softmax over 4 | 0 BG, 1 healthy lung, 2 GGO, 3 consolidation. Polymorphic reductions are built from it: lung = Σ1:3 (`L197`), unhealthy = GGO + CON (`L244`). [CODE `L197-198, 244-245`; `paper:L353-355, L385-390`] |
| `main1..main4` | B×4×H/2^i×W/2^i | softmax | Deep-supervision copies of `main` |
| `atm` | B×2×H×W | softmax over 2 (its own group) | 0 bg, **1 airway** |
| `atm1..atm4` | B×2×H/2^i | softmax | DS copies |
| `vessel` | B×2×H×W | softmax over 2 (its own group) | 0 bg, 1 pulmonary artery |
| `vessel1..vessel4` | B×2×H/2^i | softmax | DS copies |

There are three independent softmax groups: main(4), atm(2) and vessel(2). The paper notes that a single 3-channel softmax for airway and vessel performed worse (`paper:L640-648`). Verified output shapes at 256² input: `main/atm/vessel` are 256², and DS outputs are 128, 64, 32, 16. **[verified]**

---

## 2. Input representation

| Item | Training (released run) | Inference (`predict_medpseg.py` → `PolySegmentationPipeline`) | Source |
|---|---|---|---|
| 2.5D stacking | 3 channels = axial slices (i−1, i, i+1), "regardless of the original spacing" | `E2DStackDataset(volume, extended_2d=1)`: `[i-1, i, i+1]` in array-index order, with indices **clamped** at volume edges (edge slice repeated) | `extended_2d: 1`, `nin: 3` [CKPT]; `eval_2d_utils.py:21-50` [CODE]; `paper:L301-304` |
| Slicing axis | Axial; `orientation: positive_0` (meaning not documented) | Axis 0 of the SimpleITK array `[Z,Y,X]` (`volume[:, :, i]` on `[1,1,Z,H,W]`) | [CKPT]; [CODE]; exact meaning of `positive_0` [UNKNOWN] |
| Orientation fix | [INFERRED] the same lungmask trick | `np.flip(data, np.where(dir[[0,4,8]][::-1] < 0)[0])` gives "bed on bottom" | `image_read.py:53-54` [CODE] |
| HU window | clip [−1024, 600] → (x+1024)/1624 ∈ [0,1]. Applied **last**, after intensity augmentation ("DefinitiveHUNorm clip: True, -1024/600 -> 0/1"). | `read_preprocess(norm=True)`: identical clip and normalisation | [CKPT transform_readable]; `image_read.py:63-66`; `paper:L251-253` |
| In-plane size | Train: "Crop to lowest side" (square crop to min(H,W)), "min slice 256", then **random 256×256 patch**. Eval/val: `Resize(512,512, bilinear)` image and `nearest` target. | If H,W ≠ 512: `F.interpolate(..., (Z,512,512), trilinear)` (equivalent to bilinear in-plane), outputs mapped back with `nearest`. | [CKPT]; `poly_pipeline.py:259-299` |
| Z resampling | none | none | [CODE]; `paper:L302-303` |
| Lung crop | silver data only (Lungmask) | none | `paper:L253-255` |
| Slices used | gold data: **only slices with annotation** | all slices | `paper:L256-260` |

**Train/inference differences that matter.**
- The patch size differs: 256 crops in training versus full 512 at inference. The network is fully convolutional, so this is by design.
- Training uses only annotated slices; inference runs on all slices.
- The resize operator differs (torchvision bilinear vs trilinear). This is a no-op for ATM'26: all 30 val cases checked are 512×512 with identity direction, Z = 353-688 (median 557), and spacing 0.5-0.625 mm in Z.

The 239 train volumes were not available when this spec was written: only the val cache exists under `ATM26_VAL_RUN/data`. Their geometry must be asserted during data preparation.

---

## 3. Training procedure (released run, plus what our run must copy)

### 3.1 Optimisation

| Item | Value | Source |
|---|---|---|
| Optimizer | `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)`, betas (0.9, 0.999), eps 1e-8, amsgrad False | `utils/__init__.py:118-128`; `poly_seg_2d_module.py:758-759`; optimizer `param_groups` [CKPT]; `paper:L534-536` |
| Scheduler | `StepLR(step_size=1, gamma=0.985)`, stepped **per epoch** (Lightning default interval). At the saved checkpoint: `last_epoch=23`, lr = 1e-4·0.985²³ = 7.0637e-5. | `poly_seg_2d_module.py:768-771`; `lr_schedulers` [CKPT] |
| Batch size | **48** (all formats mixed). Paper says 25 (5 per format). | `batch_size: 48` [CKPT] vs `paper:L539` |
| Grad accumulation / clipping | `acum: 1`, `gradient_clip_val: None` | [CKPT] |
| Precision | `precision: 16`: Lightning 2.0.5 native AMP fp16 with GradScaler. GradScaler state at save: scale 16384, growth_interval 2000, backoff 0.5. 29 of 76,176 steps were skipped for inf grads (optimizer step 76,147). `precision_str: high` → `torch.set_float32_matmul_precision('high')` [INFERRED meaning]. | `MixedPrecisionPlugin`, optimizer `step` [CKPT] |
| Epoch | 3312 optimizer steps × 48 = 158,976 samples. The paper says 30,000 samples per format × 5 = 150,000 (unexplained 6% gap). Samples come from a custom balanced sampler (`dont_shuffle: True` because the sampler randomises). | `loops` [CKPT]; `paper:L274-282` |
| Epochs run | Released weights = **epoch 22, global step 76,176**, the best `val_loss_3d` (0.2500). The run was resumed (`-ckpt …epoch=1-step=6624`), `max_epochs 1000`, `patience 200` (EarlyStopping on `val_loss_3d`, min). Paper: "best validation loss weight after 120 hours (approximately 50 epochs)". | `callbacks`, `epoch`, `global_step` [CKPT]; `paper:L464-468` |
| Selection | `ModelCheckpoint(monitor='val_loss_3d', mode='min', save_top_k=1, every_n_epochs=1)` | [CKPT callbacks] |
| Hardware | 1 GPU (A100 80 GB; `ngpus: 1`, name "a100") | [CKPT]; `paper:L467-469` |
| Seeds | not recorded | [UNKNOWN] |
| Workers | `nworkers: 1` | [CKPT] |
| Init | ImageNet B7 (`scratch: False`, `pretrained_weights: None`). The released run is "silver > gold": silver-standard pretraining, then gold data. How the silver weights entered is not recorded (`pretrained_weights` is None, so probably via `-ckpt` resume). | [CKPT]; `paper:L251-261`, v1 Table III |

### 3.2 Augmentation (train transform `256_aug_2d`, verbatim from the checkpoint)

**Image transforms**, in this order:
1. nnUNet mirroring, p = 0.5 per axis.
2. RotationAndScaling, p = 0.2. Scale ~ U(0.7, 1.4); "degrees 2D 180" means rotation ~ U(−180°, 180°) [INFERRED from the nnU-Net 2D default]. Linear interpolation, **fill −1024**.
3. Gaussian noise, p = 0.15. "mean 0 and U(0, 0.1)·1000 std": with the image in HU, σ ~ U(0, 100) HU [INFERRED: batchgenerators uses the drawn value as the std].
4. Gaussian blur, p = 0.2, σ ~ U(0.5, 1.5).
5. Brightness (multiplicative), p = 0.15, factor ~ U(0.7, 1.3).
6. Contrast, p = 0.15, factor ~ U(0.65, 1.5), clipped to the previous min/max range.
7. DefinitiveHUNorm: clip [−1024, 600], then scale to [0, 1].

**Target transforms:**
1. The same mirroring and rotation/scale with **nearest** interpolation and fill 0.
2. "Fix no label area that might be added by transformations, setting it to BG". One-hot pixels that became all-zero after the rotation fill get channel 0 set to 1.
3. Morphological transform, `findings only: True`. This applies **to lesion labels only** (`paper:L288-290`), so it is **not applied to airway**.

**Then:**
- Crop to the lowest side (square), min slice 256.
- `ReturnPatch(patch_size=(256,256), ppositive=None, fullrandom=True, anyborder=True, segmentation=True)`. This is a uniform random crop with **no positive-patch oversampling**.

Evaluation transform: `Resize(512,512, bilinear)` + DefinitiveHUNorm on the image, and `Resize(512,512, nearest)` on the target.

Sources: [CKPT `transform_readable`, `eval_transform_readable`]; `paper:L282-292` ("parameters as Isensee et al.'s 2D nnUNet").

These points are not recorded:
- the per-sample/per-channel probabilities inside blur, noise, brightness and contrast;
- whether scaling is isotropic;
- which axes "each axis" mirroring covers;
- whether the spatial transforms run before or after the 256 crop.

All are [UNKNOWN] and are decided in §5.

### 3.3 Loss: exact formula and how missing targets are masked

**Per-sample routing.** Each batch item carries `meta["preprocess"][b]`, a string naming its label format (`poly_seg_2d_module.py:556-560`). Boolean index arrays are built per format: `simple` (lung), `detailed` (lesion), `separation` (GGO/CON), `atm` (`"new_atm"` or `"new_atm_hu"`), `vessel` (`"parse"`/`"parse_hu"`). Each format's loss is computed **only on its own sub-batch** (`y_hat[...][idx]`, `y[idx]`). A sample therefore contributes gradient only to the head and the channel reduction its label supports.

Missing targets are never masked pixel-wise. Instead, the sample is excluded from the other formats' loss terms. A format absent from the batch contributes 0 (`compute_loss`: `if index.sum() > 0 … else loss = 0`, `L514-524`).

**Airway term** (`atm_branch`, `L328-381`, `soft_circulatory=True`):

```python
y_airway     = y[atm, :2]            # one-hot [bg, airway]
y_hat_airway = y_hat["atm"][atm, :2] # softmax probabilities
atm_loss = CombinedLoss(include_background=False, cross_entropy=True, gdl=True)(y_hat_airway, y_airway)
```

**CombinedLoss** (`combined_loss.py:22-65`, MONAI 1.3.0):

```
L(ŷ, y) = GDL_MONAI(ŷ[:,1:], y[:,1:])  +  NLLLoss(log(ŷ + 1e-20), argmax_c y)
```

- GDL uses `GeneralizedDiceLoss(include_background=False)` with defaults: w_type square, smooth_nr = smooth_dr = 1e-5, batch=False, reduction mean.
- **Behaviour on our airway, verified by running MONAI 1.3.0.** With only one foreground channel, the weight w cancels and GDL equals the per-sample soft Dice, `1 − (2Σŷy + 1e-5)/(Σŷ + Σy + 1e-5)`, averaged over the airway sub-batch.
- **A patch with no airway contributes GDL = 0** (w becomes 0, so the result is 1 − 1e-5/1e-5), but it still counts in the batch mean. False positives on empty patches are penalised only by the CE term.
- CE is the mean NLL over all pixels of the sub-batch, including background.

**Deep supervision** (`compute_loss`, `L506-526`; `deep_supervision_fn`, `L474-504`), training only:

```
DS_L = 0.75·L(atm, y) + 1/8·L(atm1, y↓2) + 1/16·L(atm2, y↓4) + 1/32·L(atm3, y↓8) + 1/64·L(atm4, y↓16)
```

The targets are downsampled with `torchvision Resize(H//2^i, W//2^i, NEAREST)` on the one-hot tensor. Paper eq. (4): `paper:L452-458`.

**Total** (`supervised_loss` `L574-620`, `training_step` `L622-652`):

```
final_loss = (simple + detailed + separation + atm + vessel) / nlossterms
```

`nlossterms` is the number of format losses that are > 0 in this batch. This is paper eq. (1) with a dynamic denominator (fixed 5 in the paper when all formats are present). **With airway-only data, `final_loss = DS_L_airway` (nlossterms = 1).** `training_step` raises if `final_loss == 0`.

**Validation** (`validation_step`, `L654-699`; `val_3d=True`):
- The whole volume `[1,1,Z,512,512]` is predicted by `real_time_stack_predict` (eval batch 8, `extended_2d=1`).
- The same CombinedLoss is computed **on the full 3D volume, without DS**. It is logged as `val_atm_loss_3d` and `val_loss_3d`.
- `seg_metrics` (SimpleITK Dice, FPE, FNE, Jaccard, Hausdorff and others) runs on the argmax mask and is logged as `airway_dice_3d` and similar.
- Selection uses `val_loss_3d`.

---

## 4. Airway output to binary mask (must stay identical)

`poly_pipeline.py`:
1. `poly_stack_predict` (`L131-195`) runs `model(slice_batch, stacking=True)`, which returns the softmax dict. Slices are concatenated and permuted to `[1, C, Z, H, W]`.
2. `airway = poly_stack_output["atm"]` (`L284`). If the input was not 512², it is resized back with `nearest`.
3. `airway = (airway[:, 1:] > 0.5).astype(np.int32)` (`L322`). This is **channel 1 of the 2-channel softmax, threshold 0.5**, equivalent to argmax.
4. With `post=True`, the largest connected component is kept (`cc3d`, `L324-329`).

Our wrapper (`AirMorph/training/predict_medpseg.py:106-144`):
- calls `read_preprocess(norm=True)`, then `PolySegmentationPipeline(weight=POLY_WEIGHT, post=False)`;
- saves `raw` and `post` (LCC via `get_connected_components(return_largest=1)`);
- un-flips the orientation.

### Contract the retrained checkpoint must satisfy (so `predict_medpseg.py` runs unchanged)

1. It must be a Lightning `.ckpt` containing `hyper_parameters` and `state_dict`, with keys prefixed `model.`. `load_from_checkpoint` is strict, so the **full multitask key set** must be present.
2. `hparams.experiment_name` must **contain `"medseg_25d_a100_long_silver_gold_gdl"`**, because `poly_pipeline.py:214-215` asserts it.
3. `poly_stack_output["main"]` and `["vessel"]` must exist, because they are read unconditionally (`L283-285`).
4. hparams must rebuild the same net: `compound_coef 7, stem_replacement, circulatory_branch, soft_circulatory, self_attention, deep_supervision, expand_bifpn upsample, seg_nout 4, nin 3, bifpn_channels 128`. `debug` must be present (read in `forward`).
5. `load_from_checkpoint` calls `MEDSeg(pretrained=not scratch)`. With `scratch: False` it **downloads ImageNet B7 weights at every load**, inference included, before overwriting them with the state_dict. Pre-populate the torch hub cache on offline nodes, or set `scratch: True` in the saved hparams; the state_dict overwrites the weights either way.
6. **The weight path is the one real gap.** `predict_medpseg.py` hard-codes `POLY_WEIGHT = <installed medpseg>/poly_medseg_25d_fix.ckpt` and has no `--weight` flag. Do not overwrite the released file. Either:
   - add a backward-compatible `--weight` argument that defaults to `POLY_WEIGHT` (the recommended one-line change, made in a copy if the original must stay untouched); or
   - use a second editable medpseg install whose package directory holds the retrained file under that name.

---

## 5. Recommendation for our retraining

### 5.1 Option (a) versus option (b)

**Recommendation: option (a).**
- Instantiate the exact released architecture (`PolySeg2DModule` with the released `hyper_parameters`, §1).
- Feed only airway samples (`meta["preprocess"] = "new_atm_hu"`).
- Let the module's own polymorphic routing supervise only the `atm` head and its 4 DS heads.

The reasons:
- **Fidelity.** Every architectural component that touches the airway path is unchanged: the B7 backbone with 3D stem, the circulatory BiFPN with attention gates, the 2-class softmax airway head, deep supervision, and the GDL + CE loss with the same DS weights. The training code path is the released `training_step`/`validation_step`, with no reimplementation of the loss logic.
- **Inference unchanged.** The checkpoint satisfies the strict-load contract and the `main`/`vessel` keys exist (§4). Option (b) would fail both: removing heads changes `state_dict` keys and hparams, and `poly_pipeline.py` would hit a `KeyError` on `"main"`/`"vessel"`, so the inference code would have to change.
- **Missing targets need no new masking.** Samples without lung, lesion or vessel labels already contribute nothing to those terms (§3.3). Unsupervised parameters get `grad=None`, and PyTorch AdamW skips them, including weight decay, so they stay exactly at initialisation. The polymorphic BiFPN, main heads and vessel heads become inert passengers. Their outputs are meaningless and must be documented as such, but they are never used for airway.
- **Cost.** The polymorphic BiFPN, main heads and vessel heads still run in the forward pass but have no backward path. That is extra forward compute only.

Option (b), meaning one BiFPN plus an airway head without DS/SA, is the paper's **S-MEDSeg**. It is a different, named architecture (`paper:L470-485`), not "MEDPSeg". It scored airway Dice 0.92 versus MEDPSeg's 0.90 (Table 4), but it needs inference-code changes. Report it only as an optional extra baseline.

What cannot be reproduced:
- **Polymorphic and multitask regularisation** from lung, lesion and vessel data. We have no such labels. The paper reports multitasking helps lesion targets, but airway Dice is 0.90 for every MEDPSeg variant (v1 Table III).
- **Silver-standard pretraining.** We have no silver data.

The retrained model is therefore "MEDPSeg architecture and training recipe, trained on airway-only gold data". This is the paper's "Gold data" configuration restricted to the airway format. State this in the paper/README.

### 5.2 Decision table (gaps where paper and code are silent, and our choices)

| # | Decision | Choice | Basis / why most faithful |
|---|---|---|---|
| D1 | Architecture / hparams | Released `hyper_parameters` dict **verbatim** (Appendix A). Change only `experiment_name` to keep the asserted substring (e.g. `medseg_25d_a100_long_silver_gold_gdl__atm26_airway_retrain`), plus `cli_args`/`ckpt`/`tag` (informational). Keep `preprocess` (only used for an assert and the full_silver check). | [CKPT]; §4 contract |
| D2 | Initialisation | ImageNet EfficientNet-B7 from `efficientnet_pytorch` 0.7.1 (`scratch: False`). Stem and all BiFPN/heads use PyTorch default init. **No fine-tuning from the released checkpoint** (test contamination, §0.7). | [CKPT]; `paper:L390-392`; v1 Table III "Gold data" |
| D3 | Silver pretraining | Not done (no silver data). Documented deviation. | `paper:L251-261` |
| D4 | Training slices | Only axial slices with ≥ 1 airway voxel, across the 239 train volumes. The ATM'22 contribution got 103,623 such slices from 239 cases, so expect ~1e5. | `paper:L256-258`; arXiv 2209.15094 §2 |
| D5 | Sampling | Uniform random with replacement over the pooled annotated-slice list. Larger volumes contribute proportionally more. | "uniform random sampling, repeating samples" `paper:L279-282` [INFERRED pooled] |
| D6 | Epoch length | **3312 steps × 48 = 158,976 samples** per epoch. This keeps the per-step LR decay (0.985 per 3312 steps) identical to the released run. | [CKPT loops]; the paper's 150k is the alternative (5% difference) |
| D7 | Batch size | **48** [CKPT]. Not the paper's 25, which assumed 5 formats × 5 samples. Under Adam, the airway-only loss scale (nlossterms 1 vs ~5) has no effect. | [CKPT] |
| D8 | Multi-GPU | 1 GPU ≥ 80 GB: bs 48. 4 GPUs: DDP 4 × 12 with **SyncBatchNorm**, so BN statistics are computed over 48 like the original. Same lr 1e-4. If 12/GPU OOMs, use 6/GPU × `accumulate_grad_batches=2` (BN then sees 24). | [INFERRED]; fidelity to the bs-48 BN statistics |
| D9 | Patch | 256×256 uniform random crop, no foreground oversampling. Square crop to min side first (no-op at 512²). | [CKPT ReturnPatch] |
| D10 | Spatial-aug placement | Apply mirroring and rotate/scale to the full 512² slice, then crop 256² (avoids fill borders inside the patch). | [UNKNOWN] → nnU-Net-style; the CKPT string lists the transforms before ReturnPatch |
| D11 | Mirroring axes | In-plane H and W only, each with p = 0.5. **Do not reverse the slice (channel) order**: the 3D stem is order-sensitive and inference always feeds (i−1, i, i+1). | [INFERRED]; "2d" transform set, nnU-Net 2D mirrors 2 axes |
| D12 | Rotation / scale | One joint event with p = 0.2. Angle U(−180°, 180°); isotropic scale U(0.7, 1.4). Image bilinear with fill −1024 HU; target nearest with fill 0, then fix all-zero one-hot pixels to BG. | [CKPT] + nnU-Net 2D defaults [INFERRED] |
| D13 | Intensity augmentation | Applied in **HU** before the HU norm, all per sample: noise σ ~ U(0, 100) HU with p 0.15; blur σ ~ U(0.5, 1.5) with p 0.2 (same σ for all 3 channels); brightness ×U(0.7, 1.3) with p 0.15; contrast U(0.65, 1.5) around the mean, clipped to the original min/max, with p 0.15. | [CKPT strings]; per-channel behaviour [UNKNOWN] → same for all 3 slices, so the 2.5D context stays coherent |
| D14 | Morphological target aug | Off for airway. | [CKPT "findings only"]; `paper:L288-290` |
| D15 | Target encoding | One-hot float `[2,H,W]` = (1−airway, airway). Optionally zero-padded to 4 channels to mirror the paper's fixed representation; `atm_branch` reads `y[:, :2]` only. `meta["preprocess"] = "new_atm_hu"`. | `poly_seg_2d_module.py:341-342, 559`; `paper:L263-273` |
| D16 | Edge slices | Clamp the neighbour index (repeat the edge slice), as in `E2DStackDataset`. | `eval_2d_utils.py:41-46` |
| D17 | Orientation | Apply `read_preprocess`'s flip identically to image **and** label (import and reuse it). ATM'26 is identity, so this is a no-op. **Assert** identity/512² for all 299 cases in data prep. | `image_read.py:53-54` |
| D18 | HU / dtype | Store as int16 HU (memmap). Clip and normalise only at the end of the transform. Assert that each volume's min/max look like HU, as `read_preprocess` does. | [CKPT]; `image_read.py:58-62` |
| D19 | Loss | The released `CombinedLoss(include_background=False, cross_entropy=True, gdl=True, soft_circulatory=True)`, used via the module itself, with MONAI pinned to **1.3.0** (empty-patch GDL behaviour depends on it). | [CODE+CKPT] |
| D20 | Precision | `precision="16-mixed"` (AMP fp16 + GradScaler), `torch.set_float32_matmul_precision("high")`. | [CKPT] |
| D21 | Validation | The 30 val volumes, every epoch, full 512² stacking (`val_3d`, `eval_batch_size 8`), batch = one volume `[1,1,Z,512,512]`, target `[1,2,Z,512,512]`. Monitor `val_loss_3d` (min), `save_top_k=1`. **The 30 test volumes are never loaded by the training process.** | [CKPT] |
| D22 | Validation under DDP | `self.log` is called without `sync_dist`, so each rank would select on its own shard. Make every rank validate **all 30** volumes (non-distributed val sampler), or run validation on 1 GPU. | [CODE `L612-618`] → [INFERRED fix without editing the module] |
| D23 | Epoch budget | `max_epochs = 50` (≈ 8.0 M samples, 165,600 steps), best-`val_loss_3d` selection, EarlyStopping patience 200 as released (so effectively none). If the best epoch is the last, extend. | `paper:L464-468` (~50 epochs); released best was epoch 22 |
| D24 | Seeds | `pl.seed_everything(<fixed>, workers=True)`, with per-rank and per-worker sampler seeds derived from it. Report the seed. | [UNKNOWN] |
| D25 | Logger | `on_validation_epoch_end` calls `self.logger.experiment["training/..."].log(...)`, which is Neptune syntax (`L750-751`). Use a NeptuneLogger in offline mode or a tiny logger shim exposing that interface. **Do not edit the module.** | [CODE] |
| D26 | Grad clip / accumulation | None / 1 (unless D8 forces accumulation 2). | [CKPT] |
| D27 | Lightning / torch versions | Released: torch 1.12.1 + PL 2.0.5. Our env: torch 2.5.1, PL 2.6.5, MONAI 1.3.0, efficientnet_pytorch 0.7.1. Use our env (inference must load the result). Verify that `val_loss_3d` is present in `callback_metrics` (it is logged with on_step and on_epoch both True). | [CKPT]; env listing |
| D28 | Post-processing for reporting | Unchanged from the current MEDPSeg baseline: `raw` (threshold 0.5) and `post` (LCC). | `predict_medpseg.py` |
| D29 | Hausdorff in validation | Keep `seg_metrics` as released (it only affects logging time, not selection). It may be disabled for speed with no effect on the selected weight. | [CODE] |

---

## 6. Compute estimate

Measured or derived facts:
- 73.1 GMACs per 512² slice = 18.3 GMACs per 256² patch, forward (all heads and DS).
- torchinfo fwd/bwd activation estimate: 14.6 GB per 4 × 256² in fp32. This is identical to paper Table 7 "14 GB".
- Released run: bs 48 on one A100-80GB with fp16 AMP. ~120 h for ~50 epochs (`paper:L467-468`), about 2.4 h/epoch, including 127-volume 3D validation with **1 data worker**.
- Same group's ATM'22 MEDSeg: 84 h / 95 epochs (~103 k samples) on a 3080 Ti.

| Item | Estimate |
|---|---|
| FLOPs per training sample | ≈ 18 GMAC forward + ≈ 2×11 GMAC backward (airway path only) ≈ 40 GMAC ≈ 80 GFLOP |
| Per epoch (158,976 samples) | ≈ 1.3 × 10¹⁶ FLOP. On an A100 at a realistic 15-30 TFLOPS (EfficientNet depthwise convs are memory-bound): **~8-15 min pure GPU**. In practice **20-45 min** with ≥ 12 augmentation workers. The released 1-worker setup was ~2+ h. |
| Validation per epoch | 30 vols × ~540 slices ≈ 16 k slices × 73 GMAC ≈ 2.4 PFLOP: ~3-6 min GPU, plus CPU stacking, 3D loss and SimpleITK metrics/Hausdorff ≈ 5-15 min |
| GPU memory, bs 48, 256², AMP | torchinfo-proportional estimate ~88 GB, but the original fitted in 80 GB, so torchinfo over-counts. Expect ~50-75 GB. **Needs an A100/H100-80GB for bs 48 on one GPU.** |
| Per 24 GB GPU (4-GPU DDP) | bs 12/GPU: estimate ~15-22 GB, borderline. Measure; fall back to 6/GPU × accumulate 2. |
| Host RAM / disk | 239 vols × ~550 × 512² × int16 ≈ 69 GB. Memmap from disk; do not hold in RAM. |
| Wall-clock, 50 epochs, 1 × A100-80GB | ≈ 25-50 h (dataloader-adequate). Up to ~5 days if CPU-bound like the original. |
| Wall-clock, 50 epochs, 4 GPUs DDP | A100-class: ≈ 8-15 h. 24 GB consumer/pro cards (3090/4090/A5000): ≈ 15-30 h. Validation is not sped up if every rank runs all 30 volumes (D22). |
| Inference | < 8 GB VRAM, < 1 min/scan (`paper:L470-473, L733`) |

Before committing, run a 200-step benchmark: measure memory with `torch.cuda.max_memory_allocated` at bs 12 and 48, plus samples/s. These numbers are estimates, not measurements, because no GPU was available on this node.

---

## 7. Risks and pitfalls

1. **Train/inference preprocessing mismatch.** Training must use exactly:
   - the `read_preprocess` flip;
   - the [−1024, 600] → [0,1] window, applied last;
   - slice order (i−1, i, i+1) along array axis 0 with edge clamping.

   Unit-test by running the training-side 2.5D extraction and the inference `E2DStackDataset` on the same volume and asserting the tensors are equal (augmentation off).
2. **Label orientation.** Apply the image's flip to the label. Check that label geometry (size, spacing, origin, direction) equals the image. Assert binary {0,1}.
3. **Slice-order reversal.** The 3D stem makes the network sensitive to slice order. Never mirror the channel axis, and never stack (i+1, i, i−1).
4. **Empty patches.** A 256 crop of an annotated slice can miss all airway. Those patches give GDL = 0 but still count in the batch mean (dilution), and are trained only by CE. This is faithful (MONAI 1.3.0); do not "fix" it. Pin MONAI 1.3.0, because GDL empty-target handling has changed across versions.
5. **Class imbalance.** The airway is about 0.1-1% of a slice. The released recipe handles this with GDL (no background) + CE and annotated-slice-only sampling. There is no positive oversampling, so do not add it.
6. **Neptune-only logging** in `on_validation_epoch_end` crashes with other loggers (D25).
7. **DDP validation sharding** silently changes checkpoint selection (D22).
8. **The experiment-name assert and the hard-coded weight path** in the inference path (§4.2, §4.6).
9. **ImageNet weight download** at every `load_from_checkpoint` (§4.5). Offline nodes fail at inference.
10. **Inert heads.** `main`/`vessel` outputs of the retrained model are untrained noise. Never report them, and label them as such if activations are saved.
11. **Test leakage.** The test split must be absent from the training process (load only `splits.json["train"]` and `["val"]`). Do not initialise from the released checkpoint, which was trained on ATM'22 data that may include our test IDs.
12. **AMP.** `torch.log(y_hat + 1e-20)` is safe only because softmax outputs are fp32 under autocast. Do not cast head outputs to fp16 before the loss. Expect a few GradScaler-skipped steps (29 in the original).
13. **Version drift.** The original used torch 1.12.1 and PL 2.0.5. `efficientnet_pytorch` static-same-padding is computed for `image_size=600`; this is equivalent for even feature sizes (256/512 inputs), but do not change the input size away from multiples of 32.
14. **Paper versus checkpoint disagreements** (bs 25 vs 48, 150 k vs 159 k samples/epoch, "mean" vs sum loss). Follow the checkpoint and record it in the deviation register.
15. **Comparability.** Expect airway Dice around the paper's 0.90 ± 0.04 (ATM'22 split, `paper` Table 4). TD/BD were 91.45/84.15 on largest-component evaluation (Table 6). Airway-only training without silver data or multitask may differ. S-MEDSeg airway-only reached 0.92.

---

## Appendix A: released `hyper_parameters` (verbatim, `poly_medseg_25d_fix.ckpt`)

The `datamodule_hyper_parameters` are identical.

```
acum 1 | airway_ths 0.5 | apply_models False | arp_template None | atmbranch None | auto_report False | backbone 'effnet'
batch_size 48 | batchfy_e2d None | bdl False | bifpn_channels 128 | celoss True | circulatory_branch True
circulatory_regularization False | combined_gdl True | combined_loss True | compound_coef 7 | con_detect False
consistency False | cpu False | debug False | deep_supervision True | dim None | disable_classification False
display False | dlpt_version 'v0.4.2.1' | domain_fold False | dont_shuffle True | downstream_method None | dropout None
eval_batch_size 8 | eval_transform_str 'eval' | every_n_train_steps None | expand_bifpn 'upsample'
experiment_name 'medseg_25d_a100_long_silver_gold_gdl' | extended_2d 1 | feature_adapter None | findings_only False
focal False | fpn_c None | freeze_encoder False | gdl False | gradient_clip_val None | imnet_norm False
include_background False | init_channel 32 | isometric False | iss False | itksnap False | large False
learnable_norm None | log False | loss_weight None | lr 0.0001 | lung_extract False | max_epochs 1000 | mccl False
medseg3d False | metric_control None | model_size None | multitask False | new_latent_space False | ngpus 1 | nin 3
no_mask False | noatt False | nout None | nrdice_loss False | num_sanity_val_steps 2 | nworkers 1 | opt 'AdamW'
orientation 'positive_0' | output_path None | output_transform None | oversample None | patience 200
perceptual_loss False | phase None | pl_version '2.0.5' | poi_loss False | poly_level 3 | polyunet25d False
polyunet3d False | possible_formats None | precision 16 | precision_str 'high'
preprocess 'gold_poly_3levels_circulatory_balanced' | pretrained False | pretrained_weights None | pretraining False
project None | pytorch_version '1.12.1+cu113' | recbranch None | sam False | save_top_k 1 | scheduling 'step'
scheduling_factor 0.985 | scratch False | seg_nout 4 | select_data None | self_attention True | skip_to_mode None
soft_circulatory True | stem_replacement True | tag 'medseg_25d_a100_long_silver_gold_gdl' | task 'poly_2d_train'
test_domain None | test_phase None | transform_str '256_aug_2d' | tversky False | unet False | unettr False
val_3d True | vessel_ths 0.5 | vesselbranch None | volumetric False | weight_decay 1e-05
```

`transform_readable` (verbatim):

```
nnUNet transforms 2d
  Image Transform: Compose(
    8: nnUNet Mirroring with probability 0.5 for each axis
    1: RotationAndScaling with 0.2 application probability and scale sampling from (0.7, 1.4) and degrees 2D 180/3D 15. Interpolation: linear. Fill -1024
    2: nnUNet Gaussian Noise 0.15 probability with mean 0 and U(0, 0.1) *1000 if True standard deviation
    3: nnUNet Gaussian Blur 0.2 probability with mean 0 and U(0.5, 1.5) kernel standard deviation
    4: nnUNet brightness 0.15 probability with U(0.7, 1.3) brightness
    5: nnUNet contrast 0.15 probability with U(0.65, 1.5) factor clipped to previous min, max range
    DefinitiveHUNorm clip: True, -1024/600 -> 0/1
)
  Target Transform: Compose(
    8: nnUNet Mirroring with probability 0.5 for each axis
    1: RotationAndScaling with 0.2 application probability and scale sampling from (0.7, 1.4) and degrees 2D 180/3D 15. Interpolation: nearest. Fill 0
    Fix no label area that might be added by transformations, setting it to BG
    Morphological Transform with findings only: True, kernel: Random circle kernel from 1 to 8 size, mode: None
)
  Random Swap Code DEPRECATED
Crop to lowest side looking at last two dimensions. Minimum Z: None min slice 256
ReturnPatch: ppositive None, patch_size (256, 256), kernel_shape (3, 3), volumetric False, anyborder True, fullrandom True, segmentation True, reset_seed False
```

`eval_transform_readable`: `Resize(size=(512, 512), interpolation=bilinear)`, `DefinitiveHUNorm clip: True, -1024/600 -> 0/1`; target `Resize(size=(512, 512), interpolation=nearest)`.

Callback state:
- `ModelCheckpoint(monitor='val_loss_3d', mode='min')`, best `.../medseg_25d_a100_long_silver_gold_gdl-epoch=22-step=76176-val_loss_3d=0.25-healthy_dice_3d=0.93-unhealthy_dice_3d=0.71-ggo_dice_3d=0.71-con_dice_3d=0.62-airway_dice_3d=0.90-vessel_dice_3d=0.87.ckpt`
- `EarlyStopping(monitor='val_loss_3d', patience=200)`, wait_count 0.
- Validation loop: 127 val batches (volumes) per epoch.

## Appendix B: related source — ATM'22 contribution (arXiv 2209.15094, same authors)

This is the airway-only predecessor, **MEDSeg/CoEDet, not MEDPSeg**, and is used here only as supporting evidence for data choices. Its recipe:
- 239 train / 60 internal val;
- HU [−1024, 600] → [0,1];
- **only annotated axial 2.5D slices (103,623)**;
- random 256 crop;
- AdamW lr 1e-4, decay 0.985, wd 1e-5;
- 1-channel **sigmoid + Dice loss**;
- "epoch = one random crop from every annotated slice";
- 95 epochs, best val loss at epoch 75, 84 h on a 3080 Ti;
- inference threshold 0.5 + largest component.

Do not mix its sigmoid/Dice head into MEDPSeg: the released MEDPSeg airway head is a 2-class softmax trained with GDL + CE.

## Appendix C: verification performed for this spec

- Dumped all checkpoint metadata and tensor shapes with the MEDPSeg environment. Scripts and outputs are in the session scratchpad and are not part of the repo.
- Rebuilt `MEDSeg` from the checkpoint hparams (`pretrained=False`) and loaded the state_dict with `strict=True`: OK. Ran a train-mode forward at 2×3×256² (output shapes in §1.2) and backpropagated the airway-only `CombinedLoss`. The 972 parameter tensors without gradient are:
  - backbone blocks 18-54;
  - the polymorphic BiFPN, main heads and attention;
  - the vessel heads;
  - the airway DS heads (main-resolution-only test);
  - the BiFPN fast-attention weights.

  This confirms option (a) leaves the non-airway parts untouched.
- MONAI 1.3.0 `GeneralizedDiceLoss(include_background=False)` gives 0.0 on an empty target, and on a non-empty one it equals the soft Dice (0.87688 vs 0.87689).
- torchinfo: 73.14 GMACs and 14.59 GB fwd/bwd for both 4×3×256² and 1×3×512².
- Geometry of the 30 ATM'26 val cases: all identity direction and 512×512. The 239 train volumes were not available when this spec was written and must be checked during data preparation.

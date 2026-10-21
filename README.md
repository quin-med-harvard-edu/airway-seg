# Anchored union fusion for airway segmentation

Code for the ISBI 2027 paper "Automated Airway Segmentation: A Fusion Model for Both Pediatric and
Adult CT".

The method combines three airway segmentation models. AirMorph-R (our reimplementation of the
AirMorph ensemble: five networks, 3-of-5 vote, largest connected component) gives the mask A.
MEDPSeg-R (MEDPSeg retrained on our training split) and TfeNet (released BAS checkpoint) contribute
their raw outputs M~ and T~. The union U = A | M~ | T~ is reduced to the 6-connected component that
overlaps A most.

## Contents

- `fusion/anchored_union.py`: the fusion rule and the members' post-processing. Also usable from the
  command line on NIfTI/MHA masks.
- `airmorph_r/`: training code for AirMorph-R. It is used as the `training/` package of a clone of
  the original AirMorph repository (see `airmorph_r/README.md`).
- `medpseg_r/`: data preparation, training and inference for MEDPSeg-R (see `medpseg_r/README.md`).
- `tfenet/predict_tfenet.py`: TfeNet inference; saves the raw and post-processed masks.
- `evaluation/`
  - `table1.py`: Table 1 (members, pairwise unions, majority vote, the three ablations and ours).
  - `stats.py`: paired Wilcoxon tests, Holm correction and bootstrap CIs on the per-case output of
    `table1.py`.
  - `generation_table.py`, `generation_cldice.py`, `build_generation_table.py`: Table 2.
  - `per_generation.py`: generation assignment used by the Table 2 scripts.
  - `hd95.py`: symmetric 95th-percentile surface distance.
- `confidence/`: the vote-pattern confidence table used by the web application and the script that
  fits it on the validation split and evaluates it on test.
- `splits/atm26_split.json`: the 239/30/30 partition of the ATM'26 Task-1 training data.

## Reproducing the tables

Masks are stored as one `ATM26_<id>.npz` per case: AirMorph-R from
`airmorph_r/dump_airmorph_masks.py` (keys `raw`, `post` and the five member votes), MEDPSeg-R from
`medpseg_r/predict_medpseg_r.py` and TfeNet from `tfenet/predict_tfenet.py` (keys `raw`, `post`).
The evaluation scripts import the ATM'22-style metrics from the AirMorph clone, so run them from
there:

```
python evaluation/table1.py --split test --airmorph DIR --medpseg DIR --tfenet DIR \
    --labels ATM26/labelsTr --centerlines CENTERLINE_CACHE --out results/table1_test.csv
python evaluation/stats.py results/table1_test.csv

export ATM26_LABELS=... ATM26_CENTERLINES=... AIRMORPH_MASKS=... MEDPSEG_MASKS=... TFENET_MASKS=...
python evaluation/generation_table.py --out-dir results/generation
python evaluation/generation_cldice.py --out-dir results/generation
python evaluation/build_generation_table.py results/generation > table2.tex
```

The pediatric examinations are not public, so only the adult results can be reproduced from this
repository.

## Data and dependencies

The ATM'26 data are not redistributed; they are available from the challenge. AirMorph, MEDPSeg and
TfeNet are not included either: install `medpseg` as a package and clone the AirMorph and TfeNet
repositories. See `NOTICE.md`.

Trained weights will be added once the licence terms of the original models and the ATM'26 data
have been checked.

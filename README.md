# Anchored union fusion for airway segmentation

Code for the ISBI 2027 submission *Automated Airway Segmentation: A Fused-Model Web-Based Software
System Evaluated in Pediatric and Adult CT*.

Three airway segmentation models are combined by voxel-wise union, and the airway is kept as the
connected component that overlaps the AirMorph-R mask most:

```
U = A | M~ | T~          A: AirMorph-R, M~: raw MEDPSeg-R, T~: raw TfeNet (BAS checkpoint)
F = argmax over 6-connected components C of U of |C & A|
```

## Layout

| folder | contents |
|---|---|
| `fusion/` | the anchored union and the members' released post-processing (`anchored_union.py`, also a CLI) |
| `airmorph_r/` | training code for AirMorph-R, our reimplementation of the AirMorph segmentation ensemble (5 networks, 3-of-5 vote); see its README |
| `medpseg_r/` | MEDPSeg-R: the released MEDPSeg module retrained on our training partition (the released weights were trained on ATM data that likely include our test volumes); see its README |
| `tfenet/` | TfeNet inference wrapper (released BAS checkpoint; saves raw and post-processed masks) |
| `evaluation/` | `table1.py` (members, pairwise unions, majority vote, ours: DSC, clDice, TLD, BND, sensitivity, precision, HD95) and `hd95.py` |
| `confidence/` | vote-pattern confidence lookup table fitted on the 30 validation volumes (review information only; it does not change F) |
| `splits/` | the ATM'26 Task-1 partition: 239 train / 30 validation / 30 test case IDs |

## Data

Adult data are the 299 annotated ATM'26 Task-1 training volumes, partitioned as in
`splits/atm26_split.json`. The data are not redistributed here; obtain them from the challenge.

## Dependencies

- The original AirMorph repository (networks, tree parsing, ATM'22-style metrics). `airmorph_r/` is
  used as its `training/` package and `evaluation/` runs inside that clone.
- The released `medpseg` package (MEDPSeg-R reuses its Lightning module and hyper-parameters).
- The TfeNet repository and its BAS checkpoint.

See `NOTICE.md` for third-party code and licences.

## Status

Private research snapshot. Not yet included: trained checkpoints (pending licence checks), the
generation-wise evaluation script and the confidence fitting script.

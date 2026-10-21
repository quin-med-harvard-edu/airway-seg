# Third-party code and data

AirMorph (Zhang et al., arXiv:2412.11039). The networks, tree parsing and evaluation code are
imported from the original repository and are not included here. `airmorph_r/eval_patch.py`
contains two small adapted copies from it: a clDice function with a one-character fix, and three
scipy tree-parsing helpers that otherwise need pycuda to import. The original repository has no
licence file.

MEDPSeg (Carmo et al., arXiv:2312.02365). Imported as the installed `medpseg` package. MEDPSeg-R
reuses its released Lightning module without changes.

TfeNet (Wu et al., arXiv:2507.06581). Imported from its repository.

ATM'22/ATM'26 data. Not redistributed. Weights trained on these data are subject to the challenge's
data terms (CC BY-NC).

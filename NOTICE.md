# Third-party code

- **AirMorph** (Zhang et al., arXiv 2412.11039): the networks, tree parsing and evaluation code are
  imported from the original repository and are not included here. `airmorph_r/eval_patch.py`
  contains two small adapted copies from it (a clDice function with a one-character fix, and three
  scipy tree-parsing helpers that are otherwise only importable with pycuda); see that file's
  docstring. The original repository carries no licence; confirm terms with its authors before
  making this repository public.
- **MEDPSeg** (Carmo et al., arXiv 2312.02365): imported as the installed `medpseg` package; not
  included. MEDPSeg-R reuses its released Lightning module unmodified.
- **TfeNet** (Wu et al., arXiv 2507.06581): imported from its repository; not included.
- **ATM'26 / ATM'22 data**: not redistributed. Trained weights derived from these data are subject
  to the challenge's data terms (CC BY-NC).

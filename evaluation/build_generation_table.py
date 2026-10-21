"""LaTeX for Table 2 from generation_table.py (summary.csv, diagnostics.csv) and
generation_cldice.py (per_case_cldice.csv).

    python evaluation/build_generation_table.py results/generation > table_generation.tex
"""
import sys

import pandas as pd

D = sys.argv[1]
S = pd.read_csv(f"{D}/summary.csv")
W = pd.read_csv(f"{D}/diagnostics.csv")
C = pd.read_csv(f"{D}/per_case_cldice.csv")
ORDER = [("airmorph", "AirMorph-R"), ("medpseg_r_post", "MEDPSeg-R"), ("tfenet_post", "TfeNet"), ("ours", r"\textbf{Ours}")]
GENS = [str(g) for g in range(1, 11)]


def values(key, metric):
    if metric == "clDice":
        c = C[C.strategy == key].groupby("generation").clDice.mean()
        return [round(float(c[g]), 2) for g in GENS + ["WT"]]
    col = {"DSC": "DSC", "TLD": "TLD", "BND": "BranchesDetectedPct", "Sens.": "Sensitivity"}[metric]
    wt = {"DSC": "whole_DSC", "TLD": "whole_TLD", "BND": "whole_BND", "Sens.": "whole_Sensitivity"}[metric]
    s = S[S.strategy == key].set_index("generation")
    return [round(float(s.loc[int(g), col + "_mean"]), 2) for g in GENS] + [round(float(W[W.strategy == key][wt].mean()), 2)]


print(r"""\begin{table*}[t]
\centering
\caption{Adult results per generation (mean, \%). Generation 1 is the trachea; generations 8--10 occur in 29 of the 30 volumes. WT: whole tree. Members use their own post-processing, as in Table~\ref{tab:segmentation}; per-generation clDice restricts both skeletons to voxels nearest to ground-truth branches of that generation. Bold: best per metric and generation.}
\label{tab:generation}
\fontsize{9}{11}\selectfont
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrrrrr@{}}
\toprule
Metric & Strategy & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & WT\\""")
for metric in ("DSC", "clDice", "TLD", "BND", "Sens."):
    print(r"\midrule")
    vals = {k: values(k, metric) for k, _ in ORDER}
    best = [max(vals[k][j] for k, _ in ORDER) for j in range(11)]
    for i, (k, label) in enumerate(ORDER):
        cells = [(r"\textbf{%.2f}" if v == best[j] else "%.2f") % v for j, v in enumerate(vals[k])]
        print(f"{metric if i == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
print(r"""\bottomrule
\end{tabular*}
\end{table*}""")

"""Paired statistics on per-case results from table1.py: our method against chosen baselines.
Two-sided Wilcoxon signed-rank test (ties dropped), Holm correction over the baselines per metric,
paired bootstrap 95% CI of the mean difference (10,000 case resamples), and cases better/worse.
For HD95 the sign is flipped so that a positive difference always favors our method.

    python evaluation/stats.py results/table1_test.csv --exclude ATM_001 ATM_002
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASELINES = ["A|T~ anchored", "AirMorph-R (A)", "Majority"]
METRICS = ["BND", "TLD", "DSC", "HD95"]


def holm(p):
    order = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k])); out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--baselines", nargs="+", default=BASELINES)
    ap.add_argument("--exclude", nargs="*", default=[], help="case ids to leave out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    df = df[~df.case.isin(args.exclude)]
    wide = df.pivot_table(index="case", columns="method", values=METRICS)
    rows = []
    for m in METRICS:
        block = []
        for b in args.baselines:
            d = (wide[(m, "Ours")] - wide[(m, b)]).dropna().values
            if m == "HD95":
                d = -d
            nz = d[np.abs(d) > 1e-9]
            p = wilcoxon(nz).pvalue if len(nz) else 1.0
            boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            block.append(dict(metric=m, baseline=b, n=len(d), mean_diff=d.mean(), ci_low=lo, ci_high=hi,
                              better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()), p=p))
        for r, ph in zip(block, holm(np.array([r["p"] for r in block]))):
            r["p_holm"] = ph
        rows += block
    out = pd.DataFrame(rows)
    print(out.round({"mean_diff": 2, "ci_low": 2, "ci_high": 2, "p": 4, "p_holm": 4}).to_string(index=False))


if __name__ == "__main__":
    main()

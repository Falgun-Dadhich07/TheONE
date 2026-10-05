"""Tune the Hawkes parameters for live trading.

Same procedure as one walk-forward fold in research/short_lab.py: PSO on the
long/short Hawkes sleeve over the most recent TRAIN_DAYS of the six majors,
objective = competition composite net of costs. The result is written to
data/analysis/hawkes_live_params.json and copied by hand into settings.py
(so the deployed value is in git history).

Re-run roughly every 60 days (the tested walk-forward step).

Run:  python research/tune_hawkes.py
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")

import short_lab as X  # noqa: E402
import strategy_lab as L  # noqa: E402

TRAIN_DAYS = 180
LO, HI, INTS = [0.01, 24, 48, 0.01, 0.70], [1.0, 336, 720, 0.30, 0.99], [1, 2]


def main() -> None:
    t0 = time.time()
    ohlc = L.load_ohlc()
    c = ohlc["close"]
    idx = c.index
    fe = L.FastEval(ohlc["open"])
    b = len(idx)
    a = idx.searchsorted(idx[-1] - pd.Timedelta(days=TRAIN_DAYS))

    def fn(p):
        return X.hawkes_ls(ohlc, c, p[0], int(p[1]), int(p[2]), p[3], p[4])

    def obj(p):
        w = fn(p).to_numpy()
        if L.n_entries(w[a:b]) < 6:
            return -10.0
        return L.score(fe.returns(w)[a:b])["composite"]

    best, val, n_eval = L.pso(obj, LO, HI, INTS, seed=2026)
    out = {"kappa": best[0], "norm_lookback": int(best[1]), "signal_lookback": int(best[2]),
           "perc5": best[3], "perc95": best[4], "train": f"{idx[a]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d %H:%M}",
           "train_composite": round(val, 3), "evals": n_eval}
    path = os.path.join(ROOT, "data", "analysis", "hawkes_live_params.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

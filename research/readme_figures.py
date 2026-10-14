"""Figure and numbers for the README: the deployed v3 strategy in the backtest.

Rebuilds exactly what backtest/run_v3.py validates (STRATEGY=v3): the regime-
gated momentum sleeve, the walk-forward Hawkes sleeve and their 50/50 blend,
run through the real engine with fees, spreads and the kill switch. Compares
them with buy & hold BTC over the same dates and writes docs/img/v3_backtest.png.

matplotlib is needed for this figure only; the bot does not use it.

Run:  STRATEGY=v3 python research/readme_figures.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))

import regime  # noqa: E402
import run_v3 as R  # noqa: E402
import settings  # noqa: E402
import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img", "v3_backtest.png")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"v3 blend (deployed)": "#2a78d6", "Momentum sleeve": "#eb6834", "Hawkes sleeve": "#1baf7a",
          "Buy & hold BTC": "#8a8984"}


def build():
    d = R.load()
    c = d["close"]
    px = c.ffill()
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(px["BTC"])), c.index)
    el = S.liquid_universe(c.notna() & d["high"].notna() & d["low"].notna(), dqv, settings.V3_TOP_N)
    sma = S.btc_regime(px, settings.V3_REGIME_SMA)
    on, off = (prob > 0.5).fillna(False) & sma, (prob < 0.5).fillna(False) & ~sma
    mom = S.momentum_long_short(px, el, on, off, k=settings.V3_K)
    hawk = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for f in json.load(open(os.path.join(R.OUT_DIR, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]:
        k, n1, n2, p5, p95 = f["params"]
        a, b = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
        w = S.hawkes_long_short(d["high"].ffill(), d["low"].ffill(), px, el, k, int(n1), int(n2), p5, p95)
        m = (c.index >= a) & (c.index < b)
        hawk.loc[m] = w.loc[m]
    a, b = R.EVAL_START, c.index[-1]
    curves = {}
    for name, w in (("v3 blend (deployed)", S.v3_blend(mom, hawk, settings.V3_HAWKES_SHARE)),
                    ("Momentum sleeve", mom), ("Hawkes sleeve", hawk)):
        used = w.columns[(w.loc[a:b].abs() > 0).any()]
        res = engine.run(d["open"].ffill().loc[a:b, used], px.loc[a:b, used], w.loc[a:b, used],
                         half_spread=R.half_spreads(used), risk=RiskManager(),
                         daily_vol=(S.realized_vol(px) / np.sqrt(365))[used])
        curves[name] = res.equity / res.equity.iloc[0] * 100
    btc = px["BTC"].loc[a:b]
    curves["Buy & hold BTC"] = btc / btc.iloc[0] * 100
    return curves


def report(curves) -> None:
    for name, eq in curves.items():
        m = metrics.summary(eq)
        dl = eq.resample("1D").last()
        r10 = (dl.shift(-10) / dl - 1).dropna() * 100
        print(f"{name:<22} ret {m['total_ret_%']:6.1f}% dd {m['max_dd_%']:6.1f}% sharpe {m['sharpe']:.2f} "
              f"sortino {m['sortino']:.2f} calmar {m['calmar']:.2f} comp {m['composite']:.2f} | 10d mean "
              f"{r10.mean():.2f}% median {r10.median():.2f}% p10 {r10.quantile(.1):.2f}% p90 {r10.quantile(.9):.2f}% "
              f"worst {r10.min():.2f}% P(>0) {(r10 > 0).mean() * 100:.0f}%")


def plot(curves) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    for name, eq in curves.items():
        lw = 2.4 if name.startswith("v3") else 1.6
        ax1.plot(eq.index, eq.values, color=COLORS[name], linewidth=lw, label=name,
                 zorder=3 if name.startswith("v3") else 2)
        ax1.annotate(f"{eq.iloc[-1] - 100:+.0f}%", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=INK, fontsize=9)
    ax1.axhline(100, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    ax1.set_ylabel("Equity (start = 100)")
    ax1.legend(loc="upper left", frameon=False, ncol=2, labelcolor=INK)
    ax1.set_title("Deployed v3 in the backtest, 13 Jul 2025 to 4 Oct 2026 (after fees, spreads and kill switch)",
                  loc="left", color=INK, fontsize=11.5, pad=10)
    eq = curves["v3 blend (deployed)"]
    dd = (eq / eq.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=COLORS["v3 blend (deployed)"], alpha=0.25, linewidth=0)
    ax2.plot(dd.index, dd.values, color=COLORS["v3 blend (deployed)"], linewidth=1.2)
    ax2.set_ylabel("v3 drawdown (%)")
    ax2.annotate(f"max {dd.min():.1f}%", (dd.idxmin(), dd.min()), xytext=(8, 4), textcoords="offset points",
                 color=INK, fontsize=9)
    ax1.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=30))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print("saved", OUT)


if __name__ == "__main__":
    c = build()
    report(c)
    plot(c)

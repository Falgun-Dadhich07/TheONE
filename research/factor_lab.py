"""Factor lab: which signals actually predict cross-sectional crypto returns?

For ~25 candidate signals (price, volume/order-flow, futures funding, open
interest, positioning) this measures, on the causal top-30 liquidity universe:

  * rank IC: Spearman correlation across coins between the signal now and the
    return over the next 4h / 24h / 72h / 168h (non-overlapping samples)
  * stability: the same IC in the first and second half of the sample
  * a daily top-quintile minus bottom-quintile portfolio, gross and net of
    12.5 bp per unit of turnover

A signal is "kept" only if its 24h or 72h IC has the same sign in both halves
with |t| > 2 overall. Signs are then FIXED from the first half and the
combination is judged on the second half (see factor_strategy.py).

Run:  python research/factor_lab.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))
warnings.filterwarnings("ignore")

import run_v3 as R  # noqa: E402
import strategy as S  # noqa: E402

FAC = os.path.join(ROOT, "data", "factors")
OUT = os.path.join(ROOT, "data", "analysis")
HORIZONS = (4, 24, 72, 168)
TOP_N = 30
COST = 0.00125

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def load_panel():
    d = R.load()
    c = d["close"]
    px = c.ffill()
    idx, cols = c.index, c.columns
    taker, trades, qv = {}, {}, {}
    for coin in cols:
        p = os.path.join(FAC, f"spot_{coin}.pkl")
        if os.path.exists(p):
            s = pd.read_pickle(p)
            taker[coin], trades[coin], qv[coin] = s["taker_quote"], s["trades"], s["quote_volume"]
    taker = pd.DataFrame(taker).reindex(idx).reindex(columns=cols)
    trades = pd.DataFrame(trades).reindex(idx).reindex(columns=cols)
    qv = pd.DataFrame(qv).reindex(idx).reindex(columns=cols)
    fund = pd.read_pickle(os.path.join(FAC, "funding.pkl")).reindex(columns=cols)
    met = {}
    for coin in cols:
        p = os.path.join(FAC, f"metrics_{coin}.pkl")
        if os.path.exists(p):
            met[coin] = pd.read_pickle(p)
    return d, px, taker, trades, qv, fund, met


def build_features(d, px, taker, trades, qv, fund, met) -> dict:
    idx, cols = px.index, px.columns
    r = np.log(px).diff()
    f = {}
    for h in (24, 72, 168, 336, 720):
        f[f"mom_{h}h"] = px / px.shift(h) - 1
    for h in (1, 4):
        f[f"ret_{h}h"] = px / px.shift(h) - 1
    f["mom_168_skip24"] = px.shift(24) / px.shift(168) - 1
    f["vol_168h"] = r.rolling(168).std()
    f["vol_ratio_24_168"] = r.rolling(24).std() / r.rolling(168).std()
    f["dist_from_14d_high"] = px / d["high"].ffill().rolling(336).max() - 1
    f["skew_168h"] = r.rolling(168).skew()
    btc = r["BTC"]
    beta = r.rolling(720, min_periods=360).cov(btc).div(btc.rolling(720, min_periods=360).var(), axis=0)
    f["beta_btc"] = beta
    f["idio_mom_168h"] = f["mom_168h"].sub(beta.mul(f["mom_168h"]["BTC"], axis=0))
    v24 = qv.rolling(24).sum()
    f["volume_surge_24h_vs_30d"] = v24 / (qv.rolling(720).sum() / 30)
    f["taker_buy_imbalance_24h"] = (2 * taker.rolling(24).sum() - v24) / v24
    f["taker_buy_imbalance_168h"] = (2 * taker.rolling(168).sum() - qv.rolling(168).sum()) / qv.rolling(168).sum()
    f["avg_trade_size_chg"] = (v24 / trades.rolling(24).sum()) / (qv.rolling(720).sum() / trades.rolling(720).sum())
    f["illiquidity_amihud"] = (r.abs() / qv).rolling(168).mean()
    fh = fund.reindex(idx.union(fund.index)).sort_index().ffill().reindex(idx)   # last settled rate
    f["funding_last"] = fh
    per_day = fund.notna().rolling("7D").sum().reindex(idx, method="ffill") / 7
    f["funding_7d_avg"] = fund.rolling("7D").mean().reindex(idx, method="ffill")
    f["funding_3d_avg"] = fund.rolling("3D").mean().reindex(idx, method="ffill")
    f["funding_chg_3d_vs_14d"] = f["funding_3d_avg"] - fund.rolling("14D").mean().reindex(idx, method="ffill")
    if met:
        def mp(col):
            return pd.DataFrame({k: v[col] for k, v in met.items()}).reindex(idx).reindex(columns=cols).shift(1)
        oi = mp("oi")
        f["oi_chg_24h"] = oi / oi.shift(24) - 1
        f["oi_chg_168h"] = oi / oi.shift(168) - 1
        f["oi_chg_minus_price_chg_24h"] = f["oi_chg_24h"] - f["mom_24h"]
        f["top_trader_long_short"] = mp("top_pos_ls")
        f["global_long_short_accounts"] = mp("global_ls")
        f["global_ls_chg_24h"] = f["global_long_short_accounts"] / f["global_long_short_accounts"].shift(24) - 1
        f["futures_taker_ls_24h"] = mp("taker_ls").rolling(24).mean()
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def rank_ic(feat: pd.DataFrame, fwd: pd.DataFrame, elig: pd.DataFrame, times) -> pd.Series:
    out = {}
    fa, ra, ea = feat.to_numpy(), fwd.to_numpy(), elig.to_numpy()
    pos = feat.index.get_indexer(times)
    for t, i in zip(times, pos):
        m = ea[i] & np.isfinite(fa[i]) & np.isfinite(ra[i])
        if m.sum() < 8:
            continue
        x = pd.Series(fa[i][m]).rank().to_numpy()
        y = pd.Series(ra[i][m]).rank().to_numpy()
        out[t] = np.corrcoef(x, y)[0, 1]
    return pd.Series(out)


def quintile_ls(feat, px, elig, sign: float):
    """Daily: long top 20% / short bottom 20% by sign*feature, equal weight, 1 day hold."""
    times = feat.index[feat.index.hour == 0]
    fwd = px.shift(-24) / px - 1
    prev, rets, turn = None, [], []
    for t in times:
        m = elig.loc[t] & feat.loc[t].notna() & fwd.loc[t].notna()
        if m.sum() < 10:
            continue
        s = (sign * feat.loc[t][m]).rank(pct=True)
        w = pd.Series(0.0, index=feat.columns)
        lo, hi = s[s <= 0.2].index, s[s >= 0.8].index
        w[hi], w[lo] = 0.5 / len(hi), -0.5 / len(lo)
        rets.append((t, float((w * fwd.loc[t].fillna(0)).sum())))
        turn.append(float((w - prev).abs().sum()) if prev is not None else 1.0)
        prev = w
    r = pd.Series(dict(rets))
    tn = np.array(turn)
    return r, r - COST * tn, tn.mean()


def main() -> None:
    d, px, taker, trades, qv, fund, met = load_panel()
    idx = px.index
    print(f"panel: {px.shape[1]} coins, {len(idx)} hours; taker data {taker.notna().any().sum()} coins; "
          f"funding {fund.notna().any().sum()} coins; futures metrics {len(met)} coins "
          f"({min(v.index[0] for v in met.values()):%Y-%m-%d} on)" if met else "no metrics", flush=True)
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    elig = S.liquid_universe(d["close"].notna(), dqv, TOP_N)
    feats = build_features(d, px, taker, trades, qv, fund, met)
    start = idx[0] + pd.Timedelta(days=45)
    mid = start + (idx[-1] - start) / 2
    rows = []
    for name, feat in feats.items():
        row = {"signal": name}
        for h in HORIZONS:
            fwd = px.shift(-h) / px - 1
            times = idx[(idx >= start) & (idx.hour % min(h, 24) == 0)][:: max(h // 24, 1)]
            ic = rank_ic(feat, fwd, elig, times)
            if len(ic) < 30:
                row[f"IC_{h}h"], row[f"t_{h}h"] = np.nan, np.nan
                continue
            row[f"IC_{h}h"] = ic.mean()
            row[f"t_{h}h"] = ic.mean() / ic.std() * np.sqrt(len(ic))
            if h in (24, 72):
                a, b = ic[ic.index < mid], ic[ic.index >= mid]
                row[f"IC_{h}h_half1"] = a.mean() if len(a) > 15 else np.nan
                row[f"IC_{h}h_half2"] = b.mean() if len(b) > 15 else np.nan
                row[f"n_{h}h"] = len(ic)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("signal")
    df.to_csv(os.path.join(OUT, "factor_ic.csv"))
    show = ["IC_4h", "t_4h", "IC_24h", "t_24h", "IC_24h_half1", "IC_24h_half2", "IC_72h", "t_72h",
            "IC_72h_half1", "IC_72h_half2", "IC_168h", "t_168h", "n_24h"]
    print(f"\n=== RANK IC (top-{TOP_N} liquid universe; half split at {mid:%Y-%m-%d}) ===")
    print(df[show].round(3).sort_values("t_24h").to_string())

    keep = []
    for name, r in df.iterrows():
        for h in (24, 72):
            a, b, t = r.get(f"IC_{h}h_half1"), r.get(f"IC_{h}h_half2"), r.get(f"t_{h}h")
            if pd.notna(a) and pd.notna(b) and a * b > 0 and abs(t) > 2:
                keep.append((name, h, float(np.sign(r[f"IC_{h}h"])), round(float(t), 2)))
                break
    print("\nSignals with a stable IC (same sign in both halves, |t| > 2):")
    for k in keep:
        print("  ", k)

    print("\n=== Daily quintile long/short (top 20% vs bottom 20%), equal weight ===")
    q = []
    for name, h, sign, t in keep:
        g, n, tn = quintile_ls(feats[name], px, elig, sign)
        g = g[g.index >= start]
        n = n[n.index >= start]
        q.append({"signal": name, "sign": sign, "gross_ann_%": g.mean() * 365 * 100, "net_ann_%": n.mean() * 365 * 100,
                  "net_sharpe": n.mean() / n.std() * np.sqrt(365), "turnover/day": tn,
                  "net_half1_ann_%": n[n.index < mid].mean() * 365 * 100, "net_half2_ann_%": n[n.index >= mid].mean() * 365 * 100,
                  "days": len(n)})
    if q:
        qd = pd.DataFrame(q).set_index("signal")
        qd.to_csv(os.path.join(OUT, "factor_quintiles.csv"))
        print(qd.round(2).sort_values("net_sharpe", ascending=False).to_string())
    pd.to_pickle({"keep": keep, "mid": mid, "start": start}, os.path.join(OUT, "factor_keep.pkl"))


if __name__ == "__main__":
    main()

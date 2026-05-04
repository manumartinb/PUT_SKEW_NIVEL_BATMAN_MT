#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_evidence.py
====================
One-shot regenerator for the statistical evidence section of the
PUT_SKEW_NIVEL_BATMAN_MT dashboard.

Validates `skew_25d_vs50_pct_expanding` (PUT SKEW NIVEL, DTE 60, 10:30 ET)
against Batman MT PnL exclusively (DTE 40-200). NO Allantis. NO LT.
NO SPX filter as primary.

  - Spearman correlation + bootstrap CI95 by horizon d001..d049
  - Decile breakdown (D1..D10) at d020 (matches LIBERATION dashboard)
  - Year stability 2019..2025
  - Regime split FAVORABLE >=80 / NEUTRAL / ADVERSO <=20 at d020 + d050
  - Window-forward conditioning (HIGH/LOW PUT_SKEW at observation t),
    computed in-script from Batman MT trades. 3 SPX filter contexts
    (sin filtro / |SPX|<=3% / |SPX|<=2%) for the chart, but the rules block
    at the top is sin filtro (canonical).

Manual regen with:
    python generate_evidence.py            # local only
    python generate_evidence.py --push     # local + git push to GitHub Pages

Auth: env var GH_PUT_SKEW_TOKEN (User scope), Contents:write fine-grained PAT
scoped to manumartinb/PUT_SKEW_NIVEL_BATMAN_MT.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# ============================== CONFIG ==============================

DASHBOARD_DIR = Path(r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew\dashboards\PUT_SKEW_NIVEL_BATMAN_MT_DASHBOARD")
EVIDENCE_DIR = DASHBOARD_DIR / "evidence"

BATMAN_MT_CSV = Path(
    r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Batman\SPX\LIVE"
    r"\[MAIN RANKEO MT]_combined_BATMAN_mediana_w_stats_w_vix_OWN_ALLDAYS.csv"
)
SKEW_ENRICHED_CSV = Path(
    r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew\SKEW_PUT_ENRICHED.csv"
)

GH_REPO = "manumartinb/PUT_SKEW_NIVEL_BATMAN_MT"
GH_USER_NAME = "manumartinb"
GH_USER_EMAIL = "manuelmartinbarranco@gmail.com"
TOKEN_ENV = "GH_PUT_SKEW_TOKEN"
BRANCH = "main"
TZ = ZoneInfo("Europe/Madrid")

# Analysis params (consistent with LIBERATION dashboard rules block)
SCORE_COL = "PUT_SKEW_PCT"
DATE_COL = "trade_date"
WINDOWS = list(range(1, 50))
CHECKPOINTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
REGIME_FAV_MIN = 80.0
REGIME_ADV_MAX = 20.0
PNL_REF_HORIZON = 20  # d020 = canonical horizon (twin to LT/Allantis dashboards)

# Window-forward params (in-script computation, no external CSV)
WF_OBS_DAYS = [0, 10, 20, 30, 40]
WF_FORWARDS = [20, 50]
WF_SPX_FILTERS = ["sin filtro", "|SPX|<=3%", "|SPX|<=2%"]

# Dark theme
DARK_BG = "#0d1117"
DARK_PANEL = "#161b22"
DARK_TEXT = "#c9d1d9"
DARK_MUTED = "#8b949e"
DARK_BORDER = "#30363d"
DARK_GRID = "#21262d"
COLOR_TENSION = "#58a6ff"
COLOR_FAV = "#3fb950"
COLOR_NEU = "#d29922"
COLOR_ADV = "#f85149"
COLOR_ACCENT = "#a371f7"


def _setup_matplotlib_dark() -> None:
    plt.rcParams.update({
        "figure.facecolor": DARK_PANEL,
        "axes.facecolor": DARK_PANEL,
        "savefig.facecolor": DARK_PANEL,
        "savefig.edgecolor": DARK_PANEL,
        "text.color": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,
        "axes.titlecolor": DARK_TEXT,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "axes.edgecolor": DARK_BORDER,
        "grid.color": DARK_GRID,
        "axes.grid": True,
        "grid.alpha": 0.45,
        "axes.unicode_minus": False,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if spearmanr is not None:
        val = spearmanr(x, y, nan_policy="omit").correlation
        return float(val) if val is not None else float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _profit_factor(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    if pnl.empty:
        return float("nan")
    gw = float(pnl[pnl > 0].sum())
    gl = float((-pnl[pnl < 0]).sum())
    if gl <= 0:
        return float("nan")
    return gw / gl


def _winrate(pnl: pd.Series) -> float:
    p = pd.to_numeric(pnl, errors="coerce").dropna()
    if p.empty:
        return float("nan")
    return 100.0 * float((p > 0).mean())


def _fmt(v: float, prec: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{prec}f}"


def _fmt_int(v) -> str:
    if v is None:
        return "n/a"
    try:
        if not np.isfinite(v):
            return "n/a"
    except (TypeError, ValueError):
        pass
    return f"{int(v):,}"


def _fmt_pct(v: float, prec: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{prec}f}%"


# ============================== LOAD ==============================


@dataclass
class Dataset:
    df: pd.DataFrame
    n_trades: int
    n_days: int
    date_min: str
    date_max: str
    psk_daily: pd.DataFrame  # for window forward lookup


def _load_skew_daily() -> pd.DataFrame:
    cols = {"trade_date", "snapshot_time", "dte_target", "side",
            "skew_25d_vs50", "skew_25d_vs50_pct_expanding"}
    s = pd.read_csv(SKEW_ENRICHED_CSV, usecols=lambda c: c in cols, low_memory=False)
    s = s[(s["snapshot_time"] == "10:30:00")
          & (s["dte_target"] == 60)
          & (s["side"] == "PUT")].copy()
    s["trade_date"] = pd.to_datetime(s["trade_date"], errors="coerce")
    s = s.dropna(subset=["trade_date", "skew_25d_vs50_pct_expanding"]).copy()
    s["trade_date"] = s["trade_date"].dt.normalize()
    s = s.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    s = s.rename(columns={"skew_25d_vs50_pct_expanding": SCORE_COL,
                          "skew_25d_vs50": "PUT_SKEW_RAW"})
    return s[["trade_date", SCORE_COL, "PUT_SKEW_RAW"]].reset_index(drop=True)


def load_dataset() -> Dataset:
    """Load Batman MT trades, join with daily PUT_SKEW. NO filter applied."""
    if not BATMAN_MT_CSV.exists():
        raise FileNotFoundError(f"Batman MT CSV not found: {BATMAN_MT_CSV}")
    if not SKEW_ENRICHED_CSV.exists():
        raise FileNotFoundError(f"SKEW_PUT_ENRICHED not found: {SKEW_ENRICHED_CSV}")

    # Load ALL columns needed: horizon analysis (d001-d049), regime d050,
    # and window-forward up to t_max+x_max = 40+50 = d090
    horizon_pnl_cols = [f"PnL_d{d:03d}_mediana" for d in WINDOWS]
    horizon_spx_cols = [f"SPX_chg_pct_d{d:03d}" for d in WINDOWS]
    extra_days = list(range(50, max(WF_OBS_DAYS) + max(WF_FORWARDS) + 1))  # d050 .. d090
    extra_pnl_cols = [f"PnL_d{d:03d}_mediana" for d in extra_days]
    extra_spx_cols = [f"SPX_chg_pct_d{d:03d}" for d in extra_days]
    pnl_cols = horizon_pnl_cols + extra_pnl_cols
    spx_cols = horizon_spx_cols + extra_spx_cols
    needed = {DATE_COL} | set(pnl_cols) | set(spx_cols)
    print(f"[INFO] reading Batman MT {BATMAN_MT_CSV.name} (subset cols)")
    bm = pd.read_csv(BATMAN_MT_CSV, usecols=lambda c: c in needed, low_memory=False)
    bm[DATE_COL] = pd.to_datetime(bm[DATE_COL], errors="coerce").dt.normalize()
    for c in pnl_cols + spx_cols:
        if c in bm.columns:
            bm[c] = pd.to_numeric(bm[c], errors="coerce")
    bm = bm.dropna(subset=[DATE_COL]).copy()

    print(f"[INFO] reading SKEW_PUT_ENRICHED.csv (DTE=60/10:30/PUT)")
    s = _load_skew_daily()

    print(f"[INFO] joining Batman MT ({len(bm):,}) x PUT SKEW daily ({len(s):,})")
    df = bm.merge(s, on=DATE_COL, how="inner")
    df = df.dropna(subset=[SCORE_COL]).copy()
    print(f"[INFO] joined: {len(df):,} trades")

    return Dataset(
        df=df.reset_index(drop=True),
        n_trades=int(len(df)),
        n_days=int(df[DATE_COL].nunique()),
        date_min=str(df[DATE_COL].min().date()),
        date_max=str(df[DATE_COL].max().date()),
        psk_daily=s,
    )


# ============================== ANALYSIS CORE ==============================


def _attach_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["decile"] = np.nan
        return out
    try:
        dec = pd.qcut(out["score"], 10, labels=False, duplicates="drop")
        out["decile"] = dec.astype("float") + 1.0
    except Exception:
        out["decile"] = np.nan
    return out


def _decile_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "decile" not in df.columns:
        return pd.DataFrame(columns=["decile", "N", "mean", "median", "PF", "winrate"])
    rows = []
    for d in sorted(df["decile"].dropna().unique()):
        sub = df[df["decile"] == d]
        if sub.empty:
            continue
        pnl = sub["pnl"]
        rows.append({
            "decile": int(d),
            "N": int(pnl.notna().sum()),
            "mean": float(pnl.mean()),
            "median": float(pnl.median()),
            "PF": _profit_factor(pnl),
            "winrate": _winrate(pnl),
        })
    return pd.DataFrame(rows)


def _adjacent_non_decreasing_ratio(means: pd.Series) -> float:
    vals = means.sort_index().dropna().to_numpy(dtype=float)
    if vals.size < 2:
        return float("nan")
    return float(np.sum(np.diff(vals) >= 0) / (vals.size - 1))


def _bootstrap_ci(score: np.ndarray, pnl: np.ndarray, dec: np.ndarray,
                  n_boot: int, seed: int) -> Dict[str, float]:
    """Bootstrap CI95 with rank-once optimization."""
    n = score.size
    if n < 30:
        return {"sp_lo": float("nan"), "sp_hi": float("nan"),
                "delta_lo": float("nan"), "delta_hi": float("nan")}
    score_rank = pd.Series(score).rank().to_numpy(dtype=float)
    pnl_rank = pd.Series(pnl).rank().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sp_vals = np.full(n_boot, np.nan, dtype=float)
    delta_vals = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sr = score_rank[idx]
        pr = pnl_rank[idx]
        sx = sr.std(); sy = pr.std()
        if sx > 0 and sy > 0:
            sp_vals[b] = float(np.mean((sr - sr.mean()) * (pr - pr.mean())) / (sx * sy))
        p = pnl[idx]; d = dec[idx]
        d1 = p[d == 1]; d10 = p[d == 10]
        if d1.size > 0 and d10.size > 0:
            delta_vals[b] = float(np.mean(d10) - np.mean(d1))
    return {
        "sp_lo": float(np.nanpercentile(sp_vals, 2.5)),
        "sp_hi": float(np.nanpercentile(sp_vals, 97.5)),
        "delta_lo": float(np.nanpercentile(delta_vals, 2.5)),
        "delta_hi": float(np.nanpercentile(delta_vals, 97.5)),
    }


def compute_horizon_metrics(ds: Dataset) -> pd.DataFrame:
    """Spearman + bootstrap CI95 by horizon. Batman MT no filter."""
    rows = []
    for d in WINDOWS:
        pnl_col = f"PnL_d{d:03d}_mediana"
        if pnl_col not in ds.df.columns:
            continue
        sub = ds.df[[DATE_COL, SCORE_COL, pnl_col]].copy()
        sub = sub.rename(columns={SCORE_COL: "score", pnl_col: "pnl"}).dropna(subset=["score", "pnl"])
        if len(sub) < 100:
            continue
        sub_dec = _attach_deciles(sub)
        dec_tbl = _decile_table(sub_dec)
        means = dec_tbl.set_index("decile")["mean"] if not dec_tbl.empty else pd.Series(dtype=float)
        adj = _adjacent_non_decreasing_ratio(means)

        d1 = dec_tbl[dec_tbl["decile"] == 1]
        d10 = dec_tbl[dec_tbl["decile"] == 10]
        if not d1.empty and not d10.empty:
            d1_row = d1.iloc[0]; d10_row = d10.iloc[0]
            delta_mean = float(d10_row["mean"] - d1_row["mean"])
            pf_ratio = (float(d10_row["PF"] / d1_row["PF"])
                        if (np.isfinite(d10_row["PF"]) and np.isfinite(d1_row["PF"]) and d1_row["PF"] > 0)
                        else float("nan"))
        else:
            delta_mean = float("nan"); pf_ratio = float("nan")

        sp = _safe_spearman(sub["score"].to_numpy(dtype=float),
                            sub["pnl"].to_numpy(dtype=float))

        do_boot = (d in CHECKPOINTS)
        if do_boot:
            ci = _bootstrap_ci(
                sub_dec["score"].to_numpy(dtype=float),
                sub_dec["pnl"].to_numpy(dtype=float),
                sub_dec["decile"].to_numpy(dtype=float),
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + d,
            )
        else:
            ci = {"sp_lo": float("nan"), "sp_hi": float("nan"),
                  "delta_lo": float("nan"), "delta_hi": float("nan")}

        rows.append({
            "horizon_d": d,
            "N": int(len(sub)),
            "spearman": sp,
            "spearman_ci_lo": ci["sp_lo"],
            "spearman_ci_hi": ci["sp_hi"],
            "delta_mean_d10_d1": delta_mean,
            "delta_ci_lo": ci["delta_lo"],
            "delta_ci_hi": ci["delta_hi"],
            "pf_ratio_d10_d1": pf_ratio,
            "adjacent_non_decreasing": adj,
            "is_checkpoint": int(do_boot),
        })
    return pd.DataFrame(rows)


def compute_decile_table_ref(ds: Dataset) -> pd.DataFrame:
    pnl_col = f"PnL_d{PNL_REF_HORIZON:03d}_mediana"
    sub = ds.df[[DATE_COL, SCORE_COL, pnl_col]].copy()
    sub = sub.rename(columns={SCORE_COL: "score", pnl_col: "pnl"}).dropna(subset=["score", "pnl"])
    sub = _attach_deciles(sub)
    return _decile_table(sub)


def compute_year_stability(ds: Dataset) -> pd.DataFrame:
    pnl_col = f"PnL_d{PNL_REF_HORIZON:03d}_mediana"
    sub = ds.df[[DATE_COL, SCORE_COL, pnl_col]].copy()
    sub = sub.rename(columns={SCORE_COL: "score", pnl_col: "pnl"}).dropna(subset=["score", "pnl"])
    sub["year"] = sub[DATE_COL].dt.year
    rows = []
    for y, g in sub.groupby("year", sort=True):
        if len(g) < 50:
            continue
        sp = _safe_spearman(g["score"].to_numpy(dtype=float), g["pnl"].to_numpy(dtype=float))
        gd = _attach_deciles(g)
        dt = _decile_table(gd)
        d1 = dt[dt["decile"] == 1]; d10 = dt[dt["decile"] == 10]
        delta = (float(d10["mean"].iloc[0] - d1["mean"].iloc[0])
                 if (not d1.empty and not d10.empty) else float("nan"))
        rows.append({
            "year": int(y),
            "N": int(len(g)),
            "spearman": sp,
            "delta_mean_d10_d1": delta,
            "spearman_pos": int(np.isfinite(sp) and sp > 0),
            "delta_pos": int(np.isfinite(delta) and delta > 0),
        })
    return pd.DataFrame(rows)


def compute_regimes(ds: Dataset) -> pd.DataFrame:
    cols_needed = [SCORE_COL, f"PnL_d{PNL_REF_HORIZON:03d}_mediana", "PnL_d050_mediana"]
    available = [c for c in cols_needed if c in ds.df.columns]
    sub_d = ds.df[available].copy().dropna(subset=[SCORE_COL])

    def _bucket(v):
        if v >= REGIME_FAV_MIN:
            return "FAVORABLE"
        if v <= REGIME_ADV_MAX:
            return "ADVERSO"
        return "NEUTRAL"

    sub_d["regime"] = sub_d[SCORE_COL].apply(_bucket)

    horizons = [(f"PnL_d{PNL_REF_HORIZON:03d}_mediana", f"d{PNL_REF_HORIZON:03d}")]
    if "PnL_d050_mediana" in sub_d.columns:
        horizons.append(("PnL_d050_mediana", "d050"))

    rows = []
    for label in ["FAVORABLE", "NEUTRAL", "ADVERSO"]:
        g = sub_d[sub_d["regime"] == label]
        n = int(len(g))
        if n == 0:
            continue
        for hcol, hkey in horizons:
            p = pd.to_numeric(g[hcol], errors="coerce").dropna()
            if p.empty:
                continue
            mean = float(p.mean())
            if len(p) >= 30:
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                arr = p.to_numpy(dtype=float)
                boot = np.array([float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))]))
                                 for _ in range(800)])
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))
            else:
                ci_lo = float("nan"); ci_hi = float("nan")
            rows.append({
                "regime": label,
                "horizon": hkey,
                "N": n,
                "mean": mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "PF": _profit_factor(p),
                "winrate": _winrate(p),
            })
    return pd.DataFrame(rows)


def compute_continuous_curve(ds: Dataset, n_boot: int = 500) -> pd.DataFrame:
    """For each horizon x in 1..50, mean PnL Batman MT at d{x} for HIGH (PUT_SKEW>=80
    at entry) vs LOW (PUT_SKEW<=20 at entry) cohorts, with bootstrap CI95.

    This is the continuous version of Section 7 window-forward, fixed at t=0 (entry).
    """
    print(f"[INFO] computing continuous curve P80+ vs P20- at entry, horizons 1..50")
    score_arr = ds.df[SCORE_COL].to_numpy(dtype=float)
    high_mask = score_arr >= REGIME_FAV_MIN
    low_mask = score_arr <= REGIME_ADV_MAX
    rows = []
    for x in range(1, 51):
        col = f"PnL_d{x:03d}_mediana"
        if col not in ds.df.columns:
            continue
        pnl = pd.to_numeric(ds.df[col], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(pnl)
        h_pnl = pnl[high_mask & valid]
        l_pnl = pnl[low_mask & valid]
        if len(h_pnl) < 30 or len(l_pnl) < 30:
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED + x)
        h_boot = np.empty(n_boot, dtype=float)
        l_boot = np.empty(n_boot, dtype=float)
        nh, nl = len(h_pnl), len(l_pnl)
        for b in range(n_boot):
            h_boot[b] = float(np.mean(h_pnl[rng.integers(0, nh, size=nh)]))
            l_boot[b] = float(np.mean(l_pnl[rng.integers(0, nl, size=nl)]))
        rows.append({
            "x": x,
            "n_high": nh,
            "n_low": nl,
            "high_mean": float(np.mean(h_pnl)),
            "low_mean": float(np.mean(l_pnl)),
            "spread": float(np.mean(h_pnl) - np.mean(l_pnl)),
            "high_ci_lo": float(np.percentile(h_boot, 2.5)),
            "high_ci_hi": float(np.percentile(h_boot, 97.5)),
            "low_ci_lo": float(np.percentile(l_boot, 2.5)),
            "low_ci_hi": float(np.percentile(l_boot, 97.5)),
        })
    return pd.DataFrame(rows)


def compute_window_forward(ds: Dataset) -> pd.DataFrame:
    """Window-forward analysis on Batman MT trades.

    For each trade: at observation day t, look at PUT_SKEW value at trade_date+t.
    Classify HIGH (>=80) or LOW (<=20). Compute delta_PnL between t and t+x days.
    Apply optional SPX filter on |SPX_chg in window|.
    """
    print(f"[INFO] computing window-forward in-script (Batman MT, {len(WF_OBS_DAYS)} obs days x {len(WF_FORWARDS)} forwards x {len(WF_SPX_FILTERS)} filters)")
    psk_lookup = ds.psk_daily.set_index("trade_date")[SCORE_COL]

    def _ps_at(dt):
        idx = psk_lookup.index.searchsorted(dt, side="right") - 1
        return float(psk_lookup.iloc[idx]) if idx >= 0 else float("nan")

    # Pre-compute PUT_SKEW at each observation day for every trade
    obs_ps = {}
    for t in WF_OBS_DAYS:
        target_dates = ds.df[DATE_COL] + pd.Timedelta(days=t)
        obs_ps[t] = np.array([_ps_at(d) for d in target_dates])

    rows = []
    for t in WF_OBS_DAYS:
        ps_t = obs_ps[t]
        for x in WF_FORWARDS:
            # PnL change between t and t+x
            tx = t + x
            pnl_t_col = "ZERO_AT_T0" if t == 0 else f"PnL_d{t:03d}_mediana"
            pnl_tx_col = f"PnL_d{tx:03d}_mediana"
            spx_t_col = "ZERO_AT_T0" if t == 0 else f"SPX_chg_pct_d{t:03d}"
            spx_tx_col = f"SPX_chg_pct_d{tx:03d}"
            if pnl_tx_col not in ds.df.columns:
                continue
            if t == 0:
                pnl_t = np.zeros(len(ds.df))
                spx_t = np.zeros(len(ds.df))
            else:
                if pnl_t_col not in ds.df.columns:
                    continue
                pnl_t = pd.to_numeric(ds.df[pnl_t_col], errors="coerce").to_numpy()
                spx_t = pd.to_numeric(ds.df[spx_t_col], errors="coerce").to_numpy() if spx_t_col in ds.df.columns else np.zeros(len(ds.df))
            pnl_tx = pd.to_numeric(ds.df[pnl_tx_col], errors="coerce").to_numpy()
            spx_tx = pd.to_numeric(ds.df[spx_tx_col], errors="coerce").to_numpy() if spx_tx_col in ds.df.columns else np.zeros(len(ds.df))
            delta_pnl = pnl_tx - pnl_t
            spx_window_chg = spx_tx - spx_t
            for flt in WF_SPX_FILTERS:
                if flt == "sin filtro":
                    mask = np.ones(len(ds.df), dtype=bool)
                elif flt == "|SPX|<=3%":
                    mask = np.abs(spx_window_chg) <= 3.0
                elif flt == "|SPX|<=2%":
                    mask = np.abs(spx_window_chg) <= 2.0
                else:
                    continue
                ps_valid = ~np.isnan(ps_t)
                pnl_valid = ~np.isnan(delta_pnl)
                base_mask = mask & ps_valid & pnl_valid
                # HIGH cohort
                high_mask = base_mask & (ps_t >= REGIME_FAV_MIN)
                low_mask = base_mask & (ps_t <= REGIME_ADV_MAX)
                d_high = delta_pnl[high_mask]
                d_low = delta_pnl[low_mask]
                if len(d_high) < 10 and len(d_low) < 10:
                    continue
                rows.append({
                    "t": t, "x": x, "spx_filter": flt,
                    "N_high": int(len(d_high)),
                    "N_low": int(len(d_low)),
                    "high_mean": float(np.mean(d_high)) if len(d_high) > 0 else float("nan"),
                    "high_median": float(np.median(d_high)) if len(d_high) > 0 else float("nan"),
                    "high_WR": (100.0 * float((d_high > 0).mean())) if len(d_high) > 0 else float("nan"),
                    "low_mean": float(np.mean(d_low)) if len(d_low) > 0 else float("nan"),
                    "low_median": float(np.median(d_low)) if len(d_low) > 0 else float("nan"),
                    "low_WR": (100.0 * float((d_low > 0).mean())) if len(d_low) > 0 else float("nan"),
                    "spread": (float(np.mean(d_high) - np.mean(d_low))
                               if (len(d_high) > 0 and len(d_low) > 0) else float("nan")),
                })
    return pd.DataFrame(rows)


# ============================== PLOTS ==============================


def plot_spearman_curve(horizons: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    h = horizons.sort_values("horizon_d")
    ax.plot(h["horizon_d"], h["spearman"], "-",
            color=COLOR_TENSION, linewidth=2.0, label="Spearman r")
    ck = h[h["is_checkpoint"] == 1]
    ax.errorbar(
        ck["horizon_d"], ck["spearman"],
        yerr=[ck["spearman"] - ck["spearman_ci_lo"], ck["spearman_ci_hi"] - ck["spearman"]],
        fmt="o", color=COLOR_TENSION, ecolor=DARK_MUTED, elinewidth=1.4,
        capsize=4, markersize=6, markeredgecolor="white", markeredgewidth=0.6,
        label="Checkpoint + CI95",
    )
    ax.axhline(0, color=DARK_MUTED, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Horizonte (dias)")
    ax.set_ylabel("Spearman r (PUT SKEW NIVEL vs PnL)")
    ax.set_title("Predictividad de PUT SKEW NIVEL por horizonte (Batman MT, d001-d049)")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49])
    ax.legend(loc="lower right", framealpha=0.9, facecolor=DARK_PANEL, edgecolor=DARK_BORDER)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decile_bars(decs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    if decs.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=DARK_MUTED)
    else:
        d = decs.sort_values("decile")
        cmap = plt.cm.RdYlGn
        rng = max((d["mean"].max() - d["mean"].min()), 1e-9)
        norm = (d["mean"] - d["mean"].min()) / rng
        colors = [cmap(0.15 + 0.7 * v) for v in norm]
        bars = ax.bar(d["decile"].astype(int), d["mean"], color=colors,
                      edgecolor=DARK_BORDER, linewidth=0.8)
        for b, m in zip(bars, d["mean"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:+.1f}", ha="center", va="bottom" if m >= 0 else "top",
                    color=DARK_TEXT, fontsize=9)
    ax.axhline(0, color=DARK_MUTED, linewidth=0.8)
    ax.set_xlabel("Decil de PUT SKEW NIVEL (1=puts baratos, 10=puts caros)")
    ax.set_ylabel(f"PnL medio d{PNL_REF_HORIZON:03d} Batman MT (puntos)")
    ax.set_title(f"PnL d{PNL_REF_HORIZON:03d} Batman MT por decil de PUT SKEW NIVEL")
    ax.set_xticks(list(range(1, 11)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_year_stability(years: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    if years.empty:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=DARK_MUTED)
    else:
        y = years.sort_values("year")
        colors = [COLOR_FAV if v > 0 else COLOR_ADV for v in y["spearman"]]
        ax1.bar(y["year"].astype(str), y["spearman"], color=colors,
                edgecolor=DARK_BORDER, linewidth=0.8)
        ax1.axhline(0, color=DARK_MUTED, linewidth=0.8)
        ax1.set_title(f"Spearman r por anio (d{PNL_REF_HORIZON:03d}, Batman MT)")
        ax1.set_ylabel("Spearman r")
        for x, v in zip(y["year"].astype(str), y["spearman"]):
            ax1.text(x, v, f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     color=DARK_TEXT, fontsize=8)
        colors2 = [COLOR_FAV if v > 0 else COLOR_ADV for v in y["delta_mean_d10_d1"]]
        ax2.bar(y["year"].astype(str), y["delta_mean_d10_d1"], color=colors2,
                edgecolor=DARK_BORDER, linewidth=0.8)
        ax2.axhline(0, color=DARK_MUTED, linewidth=0.8)
        ax2.set_title(f"Delta D10 - D1 por anio (d{PNL_REF_HORIZON:03d}, pts)")
        ax2.set_ylabel("Delta PnL medio (pts)")
        for x, v in zip(y["year"].astype(str), y["delta_mean_d10_d1"]):
            ax2.text(x, v, f"{v:+.1f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     color=DARK_TEXT, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_pnl(regimes: pd.DataFrame, out_path: Path) -> None:
    horizons_present = regimes["horizon"].unique().tolist() if not regimes.empty else []
    n_panels = len(horizons_present) if horizons_present else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(11 if n_panels >= 2 else 7, 4.0))
    if n_panels == 1:
        axes = [axes]
    color_map = {"FAVORABLE": COLOR_FAV, "NEUTRAL": COLOR_NEU, "ADVERSO": COLOR_ADV}
    order = ["ADVERSO", "NEUTRAL", "FAVORABLE"]

    for ax, hkey in zip(axes, horizons_present):
        sub = regimes[regimes["horizon"] == hkey].set_index("regime").reindex(order).reset_index()
        if sub.empty or sub["mean"].isna().all():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=DARK_MUTED)
            continue
        means = sub["mean"].fillna(0).values
        lo = (sub["mean"] - sub["ci_lo"]).fillna(0).values
        hi = (sub["ci_hi"] - sub["mean"]).fillna(0).values
        colors = [color_map.get(r, DARK_MUTED) for r in sub["regime"]]
        bars = ax.bar(sub["regime"], means, color=colors, yerr=[lo, hi],
                      capsize=8, edgecolor=DARK_BORDER, linewidth=0.8,
                      ecolor=DARK_TEXT)
        for b, m, n in zip(bars, means, sub["N"].fillna(0).astype(int)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:+.1f}\nN={n:,}", ha="center",
                    va="bottom" if m >= 0 else "top",
                    color=DARK_TEXT, fontsize=9)
        ax.axhline(0, color=DARK_MUTED, linewidth=0.8)
        ax.set_title(f"PnL {hkey} Batman MT por regimen")
        ax.set_ylabel("PnL medio (pts)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_curve(horizons: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.0))
    h = horizons.sort_values("horizon_d")
    ax.plot(h["horizon_d"], h["delta_mean_d10_d1"], "-",
            color=COLOR_ACCENT, linewidth=2.0, label="Delta D10-D1")
    ck = h[h["is_checkpoint"] == 1]
    ax.errorbar(
        ck["horizon_d"], ck["delta_mean_d10_d1"],
        yerr=[ck["delta_mean_d10_d1"] - ck["delta_ci_lo"],
              ck["delta_ci_hi"] - ck["delta_mean_d10_d1"]],
        fmt="o", color=COLOR_ACCENT, ecolor=DARK_MUTED, elinewidth=1.4,
        capsize=4, markersize=6, markeredgecolor="white", markeredgewidth=0.6,
        label="Checkpoint + CI95",
    )
    ax.axhline(0, color=DARK_MUTED, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Horizonte (dias)")
    ax.set_ylabel("Delta PnL medio D10-D1 (pts)")
    ax.set_title("Spread D10 - D1 por horizonte (Batman MT)")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49])
    ax.legend(loc="lower right", framealpha=0.9, facecolor=DARK_PANEL, edgecolor=DARK_BORDER)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_continuous_curve(curve: pd.DataFrame, out_path: Path) -> None:
    """Continuous PnL trajectory for HIGH (P80+) vs LOW (P20-) at entry, horizons 1..50."""
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = curve["x"].values
    h = curve["high_mean"].values
    lo = curve["low_mean"].values
    h_ci_lo = curve["high_ci_lo"].values
    h_ci_hi = curve["high_ci_hi"].values
    l_ci_lo = curve["low_ci_lo"].values
    l_ci_hi = curve["low_ci_hi"].values

    ax.fill_between(x, h_ci_lo, h_ci_hi, color=COLOR_FAV, alpha=0.18,
                    label="HIGH CI95% (bootstrap n=500)")
    ax.plot(x, h, "-o", color=COLOR_FAV, linewidth=2.0, markersize=3,
            label=f"HIGH (PUT_SKEW>=80 al entry)  N={int(curve['n_high'].iloc[0]):,}")

    ax.fill_between(x, l_ci_lo, l_ci_hi, color=COLOR_ADV, alpha=0.18,
                    label="LOW CI95% (bootstrap n=500)")
    ax.plot(x, lo, "-o", color=COLOR_ADV, linewidth=2.0, markersize=3,
            label=f"LOW (PUT_SKEW<=20 al entry)  N={int(curve['n_low'].iloc[0]):,}")

    ax.axhline(0, color=DARK_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    # Mark the 2 horizons that the discrete chart uses
    for ref in [20, 50]:
        ax.axvline(ref, color=DARK_MUTED, linewidth=0.6, linestyle=":", alpha=0.5)
        ax.text(ref + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0,
                f"d{ref:03d}", color=DARK_MUTED, fontsize=8, alpha=0.7)

    ax.set_xlabel("Horizonte forward (dias desde entry)")
    ax.set_ylabel("Mean PnL Batman MT (puntos)")
    ax.set_title("Curva continua: PUT_SKEW HIGH (P80+) vs LOW (P20-) al entry  -  PnL Batman MT por horizonte d001-d050")
    ax.set_xlim(1, 50)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.legend(loc="upper left", framealpha=0.85, facecolor=DARK_PANEL, edgecolor=DARK_BORDER, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_forward(wf: pd.DataFrame, out_path: Path) -> None:
    """3 rows (SPX filter) x 2 cols (forward 20, 50)."""
    if wf.empty:
        return
    obs_days = sorted([int(t) for t in wf["t"].unique()])
    n_obs = len(obs_days)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey="col")
    for i, flt in enumerate(WF_SPX_FILTERS):
        for j, fwd in enumerate(WF_FORWARDS):
            ax = axes[i][j]
            sub = wf[(wf["spx_filter"] == flt) & (wf["x"] == fwd)].sort_values("t")
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", color=DARK_MUTED)
                continue
            x = np.arange(n_obs)
            w = 0.36
            ax.bar(x - w / 2, sub["high_mean"].values, w, color=COLOR_FAV,
                   edgecolor=DARK_BORDER, linewidth=0.7,
                   label="HIGH (PUT_SKEW P80+)" if (i == 0 and j == 0) else None)
            ax.bar(x + w / 2, sub["low_mean"].values, w, color=COLOR_ADV,
                   edgecolor=DARK_BORDER, linewidth=0.7,
                   label="LOW (PUT_SKEW P20-)" if (i == 0 and j == 0) else None)
            ax.axhline(0, color=DARK_MUTED, linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"t={t}" for t in obs_days], fontsize=9)
            ax.set_title(f"forward +{fwd}d  |  filtro: {flt}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Delta PnL en proximos {fwd}d (pts)", fontsize=9)
            if i == 2:
                ax.set_xlabel("Observation day t (cuando miramos PUT_SKEW)", fontsize=9)
            for k, (h, lo) in enumerate(zip(sub["high_mean"].values, sub["low_mean"].values)):
                ax.text(k - w / 2, h, f"{h:+.1f}",
                        ha="center", va="bottom" if h >= 0 else "top",
                        color=DARK_TEXT, fontsize=7.5)
                ax.text(k + w / 2, lo, f"{lo:+.1f}",
                        ha="center", va="bottom" if lo >= 0 else "top",
                        color=DARK_TEXT, fontsize=7.5)
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
                          facecolor=DARK_PANEL, edgecolor=DARK_BORDER)
    fig.suptitle(
        "PUT SKEW NIVEL en ventana alta (P80+) vs baja (P20-): cambio de PnL Batman MT en proximos x dias",
        fontsize=12, fontweight="bold", color=DARK_TEXT,
    )
    fig.text(0.5, 0.945,
             "Particion por regimen de PUT SKEW NIVEL en el dia de observacion t. Verde: HIGH (P80+).  Rojo: LOW (P20-).",
             ha="center", color=DARK_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================== TABLES ==============================


def _table_html(rows: List[List[str]], header: List[str], align: Optional[List[str]] = None) -> str:
    if align is None:
        align = ["right"] * len(header)
        if header:
            align[0] = "left"
    head_cells = "".join(
        f'<th style="text-align:{a}">{c}</th>' for c, a in zip(header, align)
    )
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>" + "".join(
                f'<td style="text-align:{a}">{c}</td>' for c, a in zip(r, align)
            ) + "</tr>"
        )
    return (
        '<div style="overflow-x:auto"><table class="ev-table">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def build_table_horizons_html(horizons: pd.DataFrame) -> str:
    ck = horizons[horizons["is_checkpoint"] == 1].sort_values("horizon_d")
    rows = []
    for _, r in ck.iterrows():
        sp_str = f'{_fmt(r["spearman"], 3)} [{_fmt(r["spearman_ci_lo"], 3)}, {_fmt(r["spearman_ci_hi"], 3)}]'
        delta_str = f'{_fmt(r["delta_mean_d10_d1"], 1)} [{_fmt(r["delta_ci_lo"], 1)}, {_fmt(r["delta_ci_hi"], 1)}]'
        rows.append([
            f'd{int(r["horizon_d"]):03d}',
            _fmt_int(r["N"]),
            sp_str,
            delta_str,
            _fmt(r["pf_ratio_d10_d1"], 2),
            _fmt(r["adjacent_non_decreasing"], 2),
        ])
    return _table_html(
        rows,
        header=["Horizonte", "N", "Spearman r [CI95]",
                "Delta D10-D1 [CI95]", "PF D10/D1", "Monotonia adj"],
    )


def build_table_deciles_html(decs: pd.DataFrame) -> str:
    rows = []
    for _, r in decs.sort_values("decile").iterrows():
        rows.append([
            f'D{int(r["decile"])}',
            _fmt_int(r["N"]),
            _fmt(r["mean"], 2),
            _fmt(r["median"], 2),
            _fmt(r["PF"], 2),
            _fmt_pct(r["winrate"], 1),
        ])
    return _table_html(
        rows,
        header=["Decil", "N", f"Mean d{PNL_REF_HORIZON:03d}", "Median", "PF", "Win Rate"],
    )


def build_table_years_html(years: pd.DataFrame) -> str:
    rows = []
    for _, r in years.sort_values("year").iterrows():
        sp_color = "#3fb950" if r["spearman_pos"] else "#f85149"
        delta_color = "#3fb950" if r["delta_pos"] else "#f85149"
        rows.append([
            str(int(r["year"])),
            _fmt_int(r["N"]),
            f'<span style="color:{sp_color}">{_fmt(r["spearman"], 3)}</span>',
            f'<span style="color:{delta_color}">{_fmt(r["delta_mean_d10_d1"], 1)}</span>',
        ])
    summary_rows = [[
        "<b>Total +</b>", "",
        f'<b>{int(years["spearman_pos"].sum())}/{len(years)}</b>',
        f'<b>{int(years["delta_pos"].sum())}/{len(years)}</b>',
    ]]
    return _table_html(
        rows + summary_rows,
        header=["Anio", "N", f"Spearman d{PNL_REF_HORIZON:03d}", "Delta D10-D1 (pts)"],
    )


def build_table_regimes_html(regimes: pd.DataFrame) -> str:
    rows = []
    order = ["FAVORABLE", "NEUTRAL", "ADVERSO"]
    horizon_order = sorted(regimes["horizon"].unique().tolist())
    for reg in order:
        for hkey in horizon_order:
            sub = regimes[(regimes["regime"] == reg) & (regimes["horizon"] == hkey)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            color = {"FAVORABLE": "#3fb950", "NEUTRAL": "#d29922",
                     "ADVERSO": "#f85149"}.get(reg, "#c9d1d9")
            mean_str = f'<span style="color:{color}"><b>{_fmt(r["mean"], 2)}</b></span>'
            ci_str = f'[{_fmt(r["ci_lo"], 1)}, {_fmt(r["ci_hi"], 1)}]'
            rows.append([
                f'<span style="color:{color}">{reg}</span>',
                hkey,
                _fmt_int(r["N"]),
                mean_str,
                ci_str,
                _fmt(r["PF"], 2),
                _fmt_pct(r["winrate"], 1),
            ])
    return _table_html(
        rows,
        header=["Regimen", "Horizonte", "N", "Mean PnL", "CI95", "PF", "Win Rate"],
    )


def build_table_window_forward_html(wf: pd.DataFrame) -> str:
    if wf.empty:
        return "<p style='color:#f85149'>window_forward sin datos.</p>"
    rows = []
    for t in [0, 20, 40]:
        for fwd in [20, 50]:
            for flt in WF_SPX_FILTERS:
                sub = wf[(wf["t"] == t) & (wf["x"] == fwd) & (wf["spx_filter"] == flt)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                spread = float(r["spread"])
                spread_color = "#3fb950" if spread > 0 else "#f85149"
                rows.append([
                    f't={int(t)}',
                    f'+{int(fwd)}d',
                    str(flt),
                    f'<span style="color:#3fb950">{float(r["high_mean"]):+.1f}</span>',
                    _fmt_int(r["N_high"]),
                    _fmt_pct(float(r["high_WR"]), 1),
                    f'<span style="color:#f85149">{float(r["low_mean"]):+.1f}</span>',
                    _fmt_int(r["N_low"]),
                    _fmt_pct(float(r["low_WR"]), 1),
                    f'<b style="color:{spread_color}">{spread:+.1f}</b>',
                ])
    return _table_html(
        rows,
        header=["t", "Fwd", "Filtro SPX",
                "HIGH mean", "N HIGH", "WR HIGH",
                "LOW mean", "N LOW", "WR LOW",
                "Spread"],
    )


# ============================== ORCHESTRATION ==============================


def build_evidence_json(
    ds: Dataset,
    horizons: pd.DataFrame,
    decs: pd.DataFrame,
    years: pd.DataFrame,
    regimes: pd.DataFrame,
    tables: Dict[str, str],
) -> dict:
    sp_by_h = {f'd{int(r.horizon_d):03d}': float(r.spearman) for r in horizons.itertuples()}
    delta_by_h = {f'd{int(r.horizon_d):03d}': float(r.delta_mean_d10_d1) for r in horizons.itertuples()}

    h_ref = horizons[horizons["horizon_d"] == PNL_REF_HORIZON]
    headline = {}
    if not h_ref.empty:
        r = h_ref.iloc[0]
        headline = {
            "horizon": f"d{PNL_REF_HORIZON:03d}",
            "spearman": float(r["spearman"]),
            "spearman_ci": [float(r["spearman_ci_lo"]), float(r["spearman_ci_hi"])],
            "delta_d10_d1": float(r["delta_mean_d10_d1"]),
            "delta_ci": [float(r["delta_ci_lo"]), float(r["delta_ci_hi"])],
            "pf_ratio_d10_d1": float(r["pf_ratio_d10_d1"]),
            "adjacent_non_decreasing": float(r["adjacent_non_decreasing"]),
        }

    images = {
        "spearman_curve": "evidence/put_skew_spearman_curve.png",
        "decile_bars": "evidence/put_skew_decile_bars.png",
        "year_stability": "evidence/put_skew_year_stability.png",
        "regime_pnl": "evidence/put_skew_regime_pnl.png",
        "delta_curve": "evidence/put_skew_delta_curve.png",
        "window_forward": "evidence/put_skew_window_forward.png",
        "continuous_curve": "evidence/put_skew_continuous_curve.png",
    }

    return {
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "input": {
            "batman_mt_csv": BATMAN_MT_CSV.name,
            "skew_csv": SKEW_ENRICHED_CSV.name,
            "n_trades": ds.n_trades,
            "n_days": ds.n_days,
            "date_min": ds.date_min,
            "date_max": ds.date_max,
            "spx_filter": "no_filter (Batman MT exclusive)",
        },
        "params": {
            "score_col": "skew_25d_vs50_pct_expanding (PUT, DTE 60, 10:30)",
            "horizons": [int(d) for d in WINDOWS],
            "checkpoints": CHECKPOINTS,
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "regime_favorable_min": REGIME_FAV_MIN,
            "regime_adverso_max": REGIME_ADV_MAX,
            "pnl_reference_horizon": PNL_REF_HORIZON,
        },
        "put_skew": {
            "headline": headline,
            "spearman_by_horizon": sp_by_h,
            "delta_by_horizon": delta_by_h,
            "deciles_d020": decs.to_dict(orient="records"),
            "year_stability": years.to_dict(orient="records"),
            "regimes": regimes.to_dict(orient="records"),
        },
        "tables_html": tables,
        "images": images,
    }


def main(push: bool) -> int:
    try:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        _setup_matplotlib_dark()

        ds = load_dataset()
        if ds.n_trades < 1000:
            raise RuntimeError(f"only {ds.n_trades} trades remain. Check data.")
        print(f"[INFO] dataset: {ds.n_trades:,} trades / {ds.n_days:,} days "
              f"({ds.date_min} -> {ds.date_max}). NO filter (Batman MT exclusive).")

        print("[INFO] computing horizon metrics (d001..d049)")
        horizons = compute_horizon_metrics(ds)

        print(f"[INFO] computing decile table at d{PNL_REF_HORIZON:03d}")
        decs = compute_decile_table_ref(ds)

        print("[INFO] computing year stability")
        years = compute_year_stability(ds)

        print("[INFO] computing regime split")
        regimes = compute_regimes(ds)

        print("[INFO] generating PNG plots (Batman MT)")
        plot_spearman_curve(horizons, EVIDENCE_DIR / "put_skew_spearman_curve.png")
        plot_decile_bars(decs, EVIDENCE_DIR / "put_skew_decile_bars.png")
        plot_year_stability(years, EVIDENCE_DIR / "put_skew_year_stability.png")
        plot_regime_pnl(regimes, EVIDENCE_DIR / "put_skew_regime_pnl.png")
        plot_delta_curve(horizons, EVIDENCE_DIR / "put_skew_delta_curve.png")

        print("[INFO] computing window-forward in-script (Batman MT trades)")
        wf = compute_window_forward(ds)
        plot_window_forward(wf, EVIDENCE_DIR / "put_skew_window_forward.png")

        print("[INFO] computing continuous curve (P80+ vs P20- at entry, x=1..50)")
        curve = compute_continuous_curve(ds, n_boot=500)
        plot_continuous_curve(curve, EVIDENCE_DIR / "put_skew_continuous_curve.png")

        print("[INFO] building HTML tables")
        tables = {
            "spearman": build_table_horizons_html(horizons),
            "deciles": build_table_deciles_html(decs),
            "years": build_table_years_html(years),
            "regimes": build_table_regimes_html(regimes),
            "window_forward": build_table_window_forward_html(wf),
        }

        print("[INFO] writing evidence/evidence.json")
        ev = build_evidence_json(ds, horizons, decs, years, regimes, tables)
        out_json = EVIDENCE_DIR / "evidence.json"
        out_json.write_text(json.dumps(ev, ensure_ascii=False, separators=(",", ":")),
                            encoding="utf-8")

        readme = EVIDENCE_DIR / "README.txt"
        readme.write_text(
            f"Evidence regenerated: {ev['generated_at']}\n"
            f"Batman MT input: {BATMAN_MT_CSV.name}\n"
            f"Skew input: {SKEW_ENRICHED_CSV.name}\n"
            f"N trades: {ds.n_trades:,}  N days: {ds.n_days:,}\n"
            f"Date range: {ds.date_min} to {ds.date_max}\n"
            f"Score: skew_25d_vs50_pct_expanding\n"
            f"Bootstrap: n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}\n"
            f"Reference horizon: d{PNL_REF_HORIZON:03d}\n"
            f"Headline Spearman r: {ev['put_skew']['headline'].get('spearman', float('nan')):.3f}\n",
            encoding="utf-8",
        )

        h = ev["put_skew"]["headline"]
        sp = h.get("spearman", float("nan"))
        ci = h.get("spearman_ci", [float("nan"), float("nan")])
        delta = h.get("delta_d10_d1", float("nan"))
        print(f"[OK] headline d{PNL_REF_HORIZON:03d}: r={sp:.3f} "
              f"CI95=[{ci[0]:.3f}, {ci[1]:.3f}]  delta_D10_D1={delta:.2f} pts")

        if push:
            return git_push()
        else:
            print("[INFO] run with --push to publish to GitHub Pages")
            return 0

    except Exception as exc:
        print(f"[X] generate_evidence failed: {exc}")
        traceback.print_exc()
        return 1


# ============================== GIT PUSH ==============================


def _git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(DASHBOARD_DIR), *args],
        capture_output=True, text=True, check=False,
    )


def git_push() -> int:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"[X] env var {TOKEN_ENV} not set; cannot push")
        return 1

    _git(["config", "user.name", GH_USER_NAME])
    _git(["config", "user.email", GH_USER_EMAIL])

    remote_url = f"https://x-access-token:{token}@github.com/{GH_REPO}.git"
    pull = subprocess.run(
        ["git", "-C", str(DASHBOARD_DIR), "pull", "--rebase", remote_url, BRANCH],
        capture_output=True, text=True,
    )
    if pull.returncode != 0:
        sanitized = pull.stderr.replace(token, "***")
        if "CONFLICT" in sanitized:
            print(f"[X] pull --rebase had conflicts: {sanitized.strip()}")
            return 1
        print(f"[WARN] pull --rebase output: {sanitized.strip()}")

    _git(["add", "evidence/", "generate_evidence.py", "index.html", "README.md"])
    status = _git(["status", "--porcelain"])
    if not status.stdout.strip():
        print("[INFO] no changes to commit")
        return 0

    today = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    commit = _git(["commit", "-m", f"evidence regen Batman MT exclusive {today}"])
    if commit.returncode != 0:
        print(f"[X] commit failed: {commit.stderr.strip()}")
        return 1

    push = subprocess.run(
        ["git", "-C", str(DASHBOARD_DIR), "push", remote_url, BRANCH],
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        sanitized = push.stderr.replace(token, "***")
        print(f"[X] push failed: {sanitized.strip()}")
        return 1

    print(f"[OK] pushed to https://manumartinb.github.io/PUT_SKEW_NIVEL_BATMAN_MT/")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate PUT_SKEW_NIVEL_BATMAN_MT dashboard evidence (Batman MT exclusive).")
    parser.add_argument("--push", action="store_true",
                        help="After regen, commit and push to GitHub Pages.")
    args = parser.parse_args()
    sys.exit(main(push=args.push))

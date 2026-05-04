#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_dashboard.py
===================
Genera data.json del dashboard PUT_SKEW_NIVEL_BATMAN_MT y hace push a GitHub Pages.

Lee SKEW_PUT_ENRICHED.csv (output de V8.0 SKEW PIPELINE Step 4) filtrado a
DTE=60 / snapshot=10:30 / side=PUT, y publica en
https://manumartinb.github.io/PUT_SKEW_NIVEL_BATMAN_MT/

Metric publicada: skew_25d_vs50_pct_expanding (percentil expanding del spread
IV puts 25d vs ATM, sin lookahead). Mismo score diario que el dashboard
LT/Allantis hermanos. La validacion empirica (seccion evidence) valida contra
trades Batman MT (DTE 40-200) en lugar de LT.

Token leido de env var GH_PUT_SKEW_TOKEN (User scope, set via
SetEnvironmentVariable). Disenado para ser invocado por
V0.[PERMA] MASTER_DAILY_PIPELINE.py como step adicional (paralelo a los
otros 5 dashboards).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

# ---------------- CONFIG ----------------
SOURCE_CSV = Path(
    r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew\SKEW_PUT_ENRICHED.csv"
)
DASHBOARD_DIR = Path(r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew\dashboards\PUT_SKEW_NIVEL_BATMAN_MT_DASHBOARD")
DATA_JSON = DASHBOARD_DIR / "data.json"

GH_REPO = "manumartinb/PUT_SKEW_NIVEL_BATMAN_MT"
GH_USER_NAME = "manumartinb"
GH_USER_EMAIL = "manuelmartinbarranco@gmail.com"
TOKEN_ENV = "GH_PUT_SKEW_TOKEN"
BRANCH = "main"

TZ = ZoneInfo("Europe/Madrid")

DTE_TARGET = 60
SNAPSHOT_TIME = "10:30:00"
SIDE = "PUT"

PCT_COL = "skew_25d_vs50_pct_expanding"
RAW_COL = "skew_25d_vs50"

# Standard bands (Batman LT convention).
# IMPORTANT: BWB usa banda invertida. Esta info se documenta en index.html
# (Seccion 1 Concepto + Seccion 8 Cross-strategy). Aqui solo publicamos el
# regimen base; el frontend pinta zonas segun convencion estandar.
FAV_MIN = 80.0
ADV_MAX = 20.0


# ---------------- HELPERS ----------------
def regime_label(v: float) -> str:
    if pd.isna(v):
        return "INDETERMINADO"
    if v >= FAV_MIN:
        return "FAVORABLE"
    if v <= ADV_MAX:
        return "ADVERSO"
    return "NEUTRAL"


def _round_or_none(v, prec: int = 2):
    if pd.isna(v):
        return None
    return round(float(v), prec)


def build_data_payload() -> dict:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    cols_needed = {
        "trade_date", "snapshot_time", "dte_target", "side",
        PCT_COL, RAW_COL,
    }
    df = pd.read_csv(
        SOURCE_CSV,
        usecols=lambda c: c in cols_needed,
        low_memory=False,
    )

    df = df[
        (df["snapshot_time"] == SNAPSHOT_TIME)
        & (df["dte_target"] == DTE_TARGET)
        & (df["side"] == SIDE)
    ].copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", PCT_COL]).copy()
    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid rows in SKEW_PUT_ENRICHED after filtering")

    last_v = float(df[PCT_COL].iloc[-1])
    last_raw = float(df[RAW_COL].iloc[-1]) if pd.notna(df[RAW_COL].iloc[-1]) else None
    last_date = str(df["trade_date"].iloc[-1])

    return {
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "source": SOURCE_CSV.name,
        "filter": {
            "dte_target": DTE_TARGET,
            "snapshot_time": SNAPSHOT_TIME,
            "side": SIDE,
        },
        "n_days": int(len(df)),
        "thresholds": {"favorable_min": FAV_MIN, "adverso_max": ADV_MAX},
        "latest": {
            "date": last_date,
            "pct": round(last_v, 2),
            "raw": _round_or_none(last_raw, 4),
            "regime": regime_label(last_v),
        },
        "dates": df["trade_date"].tolist(),
        "pct": [_round_or_none(v) for v in df[PCT_COL]],
        "raw": [_round_or_none(v, 4) for v in df[RAW_COL]],
    }


def _payload_data_changed(new_payload: dict) -> bool:
    if not DATA_JSON.exists():
        return True
    try:
        old = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    except Exception:
        return True
    keys_to_compare = ("dates", "pct", "raw", "latest", "n_days")
    for k in keys_to_compare:
        if old.get(k) != new_payload.get(k):
            return True
    return False


def write_data_json(payload: dict) -> None:
    DATA_JSON.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def _git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(DASHBOARD_DIR), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def push_to_github() -> int:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"[X] env var {TOKEN_ENV} not set; cannot push")
        return 1

    _git(["config", "user.name", GH_USER_NAME])
    _git(["config", "user.email", GH_USER_EMAIL])
    _git(["add", "-A"])

    status = _git(["status", "--porcelain"])
    if not status.stdout.strip():
        print("[INFO] no changes to commit, nothing to push")
        return 0

    today = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    commit = _git(["commit", "-m", f"daily update {today}"])
    if commit.returncode != 0:
        print(f"[X] commit failed: {commit.stderr.strip()}")
        return 1

    remote_url = f"https://x-access-token:{token}@github.com/{GH_REPO}.git"
    push = subprocess.run(
        ["git", "-C", str(DASHBOARD_DIR), "push", remote_url, BRANCH],
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        sanitized = push.stderr.replace(token, "***")
        print(f"[X] push failed: {sanitized.strip()}")
        return 1

    print(f"[OK] pushed to https://manumartinb.github.io/PUT_SKEW_NIVEL_BATMAN_MT/")
    return 0


# ---------------- MAIN ----------------
def main() -> int:
    try:
        if not DASHBOARD_DIR.exists():
            print(f"[X] dashboard dir not found: {DASHBOARD_DIR}")
            return 1

        payload = build_data_payload()
        changed = _payload_data_changed(payload)
        write_data_json(payload)

        latest = payload["latest"]
        print(
            f"[INFO] data.json {'updated' if changed else 'identical (timestamp refreshed)'} | "
            f"latest_date={latest['date']} pct={latest['pct']:.1f} raw={latest['raw']} regime={latest['regime']} | "
            f"n_days={payload['n_days']}"
        )

        if not changed:
            return 0

        return push_to_github()

    except Exception as exc:
        print(f"[X] update_dashboard failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

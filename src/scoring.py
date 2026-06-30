from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import normalize_0_100, score_distance_to_resistance, score_ma_slope_for_mode_b


MODE_B_WEIGHTS = {
    "rs_momentum_score": 0.20,
    "bbw_pctl_score": 0.20,
    "base_quality_score": 0.15,
    "dist_resistance_score": 0.15,
    "mansfield_rs_score": 0.10,
    "volume_score": 0.10,
    "ma_slope_score": 0.10,
}


def compute_mode_b_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rs_momentum_score"] = normalize_0_100(out["RS_Momentum_10D"], higher_is_better=True)
    out["bbw_pctl_score"] = normalize_0_100(out["BBW_Pctl"], higher_is_better=False)

    # Base quality: longer base and tighter closes.
    base_len_score = normalize_0_100(out["Base_Length_Weeks"], higher_is_better=True)
    tightness_score = normalize_0_100(out["Close_Tightness_Pct"], higher_is_better=False)
    out["base_quality_score"] = (base_len_score.fillna(0) * 0.6) + (tightness_score.fillna(0) * 0.4)

    out["dist_resistance_score"] = score_distance_to_resistance(out["Dist_To_Resistance_Pct"])
    out["mansfield_rs_score"] = normalize_0_100(out["Mansfield_RS"], higher_is_better=True)
    out["volume_score"] = normalize_0_100(out["Volume_Ratio"], higher_is_better=True)
    out["ma_slope_score"] = score_ma_slope_for_mode_b(out["SMA150_Slope_20D_Pct"])

    total = pd.Series(0.0, index=out.index)
    for col, weight in MODE_B_WEIGHTS.items():
        total += out[col].fillna(0) * weight

    out["ModeB_Technical_Score"] = total.round(2)
    return out


def classify_setup(row: pd.Series) -> str:
    if not row.get("Pass_Gates", False):
        return "REJECTED"

    dist = row.get("Dist_To_Resistance_Pct", np.nan)
    vol = row.get("Volume_Ratio", np.nan)
    bbw = row.get("BBW_Pctl", np.nan)
    rs_mom = row.get("RS_Momentum_10D", np.nan)

    if pd.notna(dist) and dist < 0 and pd.notna(vol) and vol >= 1.5:
        return "TRIGGERED_BREAKOUT"

    if pd.notna(dist) and 0 <= dist <= 3 and pd.notna(bbw) and bbw <= 25 and pd.notna(rs_mom) and rs_mom > 0:
        return "SETUP_READY"

    if pd.notna(dist) and 0 <= dist <= 7:
        return "WATCH_NEAR_RESISTANCE"

    return "WATCH"

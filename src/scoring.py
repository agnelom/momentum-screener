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

    base_len_score = normalize_0_100(out["Base_Length_Weeks"], higher_is_better=True)
    tightness_score = normalize_0_100(out["Close_Tightness_Pct"], higher_is_better=False)
    range_score = normalize_0_100(out.get("Base_Range_Pct", pd.Series(index=out.index)), higher_is_better=False)
    hl_score = normalize_0_100(out.get("Higher_Low_Count", pd.Series(index=out.index)), higher_is_better=True)

    out["base_quality_score"] = (
        base_len_score.fillna(0) * 0.35
        + tightness_score.fillna(0) * 0.30
        + range_score.fillna(0) * 0.20
        + hl_score.fillna(0) * 0.15
    )

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
    """
    More descriptive status labelling.

    WATCH now has reasoned sub-labels instead of acting as a vague bucket.
    """
    if not row.get("Pass_Gates", False):
        return "REJECTED"

    dist = row.get("Dist_To_Resistance_Pct", np.nan)
    vol = row.get("Volume_Ratio", np.nan)
    bbw = row.get("BBW_Pctl", np.nan)
    rs_mom = row.get("RS_Momentum_10D", np.nan)
    rsi = row.get("Composite_RSI", np.nan)
    prior_drop = row.get("Prior_Drop_Pct", np.nan)
    base_range = row.get("Base_Range_Pct", np.nan)
    tightness = row.get("Close_Tightness_Pct", np.nan)
    hl_count = row.get("Higher_Low_Count", 0)

    if pd.notna(dist) and dist < 0 and pd.notna(vol) and vol >= 1.5:
        return "TRIGGERED_BREAKOUT"

    weak_rs = pd.notna(rs_mom) and rs_mom <= 0
    weak_rsi = pd.notna(rsi) and rsi < 50
    too_far = pd.notna(dist) and dist > 5
    near_res = pd.notna(dist) and 0 <= dist <= 5
    compressed = pd.notna(bbw) and bbw <= 25
    constructive_base = (
        (pd.isna(base_range) or base_range <= 15)
        and (pd.isna(tightness) or tightness <= 5)
        and (pd.isna(prior_drop) or prior_drop <= 25)
    )

    if (
        pd.notna(dist) and 0 <= dist <= 3
        and compressed
        and pd.notna(rs_mom) and rs_mom > 0
        and pd.notna(rsi) and rsi >= 50
        and constructive_base
    ):
        return "SETUP_READY"

    if pd.notna(prior_drop) and prior_drop > 25:
        return "WATCH_POST_BREAKDOWN"

    if weak_rs and too_far:
        return "WATCH_WEAK_RS_TOO_FAR"

    if weak_rs:
        return "WATCH_WEAK_RS"

    if weak_rsi and near_res:
        return "WATCH_WEAK_RSI"

    if too_far:
        return "WATCH_TOO_FAR_FROM_RESISTANCE"

    if near_res and not compressed:
        return "WATCH_NEAR_RESISTANCE_NOT_TIGHT"

    if near_res and compressed and int(hl_count or 0) == 0:
        return "WATCH_NEEDS_HIGHER_LOW"

    if near_res:
        return "WATCH_NEAR_RESISTANCE"

    return "WATCH_REVIEW"


def setup_reason(row: pd.Series) -> str:
    status = str(row.get("Setup_Status", ""))
    pieces: list[str] = []

    if not bool(row.get("Pass_Gates", False)):
        return f"Failed gates: {row.get('Fail_Reasons', '')}"

    dist = row.get("Dist_To_Resistance_Pct", np.nan)
    rs_mom = row.get("RS_Momentum_10D", np.nan)
    rsi = row.get("Composite_RSI", np.nan)
    bbw = row.get("BBW_Pctl", np.nan)
    vol = row.get("Volume_Ratio", np.nan)
    prior_drop = row.get("Prior_Drop_Pct", np.nan)

    if pd.notna(dist):
        if dist < 0:
            pieces.append("above resistance")
        elif dist <= 3:
            pieces.append("within 3% of resistance")
        elif dist <= 5:
            pieces.append("within 5% of resistance")
        else:
            pieces.append(f"{dist:.1f}% below resistance")

    if pd.notna(rs_mom):
        pieces.append("RS improving" if rs_mom > 0 else "RS weakening")

    if pd.notna(rsi):
        pieces.append("RSI >= 50" if rsi >= 50 else "RSI < 50")

    if pd.notna(bbw):
        pieces.append("compressed" if bbw <= 25 else "not tightly compressed")

    if pd.notna(vol):
        pieces.append("volume confirmation" if vol >= 1.5 else "no volume confirmation")

    if pd.notna(prior_drop) and prior_drop > 25:
        pieces.append("large prior drop into base")

    return f"{status}: " + "; ".join(pieces)

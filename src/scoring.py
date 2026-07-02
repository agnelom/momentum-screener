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

MODE_C_WEIGHTS = {
    "mansfield_rs_score": 0.20,
    "rs_momentum_score": 0.15,
    "adx_sweet_score": 0.15,
    "rsi_sweet_score": 0.15,
    "volume_score": 0.15,
    "ema_distance_score": 0.10,
    "recent_high_score": 0.10,
}

NEAR_CLOSE_WEIGHTS = {
    "mansfield_rs_score": 0.15,
    "rs_momentum_score": 0.15,
    "rsi_sweet_score": 0.12,
    "adx_sweet_score": 0.10,
    "projected_rvol_score": 0.18,
    "close_location_score": 0.15,
    "ema_distance_score": 0.08,
    "day_range_score": 0.07,
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


def _adx_sweet_score(adx: pd.Series) -> pd.Series:
    x = pd.to_numeric(adx, errors="coerce")
    score = pd.Series(0.0, index=x.index)
    score[(x >= 25) & (x <= 35)] = 100
    score[(x > 35) & (x <= 45)] = 85
    score[(x >= 20) & (x < 25)] = 55
    score[(x > 45) & (x <= 55)] = 45
    score[(x > 55)] = 20
    return score


def _rsi_sweet_score(rsi: pd.Series) -> pd.Series:
    x = pd.to_numeric(rsi, errors="coerce")
    score = pd.Series(0.0, index=x.index)
    score[(x >= 55) & (x <= 65)] = 100
    score[(x > 65) & (x <= 72)] = 80
    score[(x >= 50) & (x < 55)] = 55
    score[(x > 72) & (x <= 78)] = 45
    score[(x < 50)] = 20
    score[(x > 78)] = 15
    return score


def _ema_distance_score(dist: pd.Series) -> pd.Series:
    x = pd.to_numeric(dist, errors="coerce")
    score = pd.Series(0.0, index=x.index)
    score[(x >= 0) & (x <= 3)] = 100
    score[(x > 3) & (x <= 6)] = 80
    score[(x > 6) & (x <= 8)] = 45
    score[(x > 8)] = 15
    score[(x < 0) & (x >= -2)] = 40
    score[(x < -2)] = 10
    return score


def _recent_high_score(dist_to_high: pd.Series) -> pd.Series:
    # Dist_To_20D_High_Pct: positive = below high, negative = above/breakout
    return score_distance_to_resistance(dist_to_high)


def _range_atr_score(day_range_atr: pd.Series) -> pd.Series:
    x = pd.to_numeric(day_range_atr, errors="coerce")
    score = pd.Series(0.0, index=x.index)
    score[(x >= 0.6) & (x <= 1.5)] = 100
    score[(x > 1.5) & (x <= 2.0)] = 75
    score[(x > 2.0) & (x <= 2.5)] = 40
    score[(x > 2.5)] = 15
    score[(x < 0.6)] = 50
    return score


def compute_mode_c_scores(df: pd.DataFrame, near_close: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["mansfield_rs_score"] = normalize_0_100(out["Mansfield_RS"], higher_is_better=True)
    out["rs_momentum_score"] = normalize_0_100(out["RS_Momentum_10D"], higher_is_better=True)
    out["adx_sweet_score"] = _adx_sweet_score(out["ADX14"])
    out["rsi_sweet_score"] = _rsi_sweet_score(out["RSI14"])
    out["volume_score"] = normalize_0_100(out["Volume_Ratio_20D"], higher_is_better=True)
    out["projected_rvol_score"] = normalize_0_100(out.get("Projected_RVol_20D", out["Volume_Ratio_20D"]), higher_is_better=True)
    out["ema_distance_score"] = _ema_distance_score(out["Dist_From_EMA10_Pct"])
    out["recent_high_score"] = _recent_high_score(out["Dist_To_20D_High_Pct"])
    out["close_location_score"] = pd.to_numeric(out.get("Close_Location_Value", 0), errors="coerce").clip(0, 100)
    out["day_range_score"] = _range_atr_score(out.get("Day_Range_ATR", pd.Series(index=out.index)))

    weights = NEAR_CLOSE_WEIGHTS if near_close else MODE_C_WEIGHTS
    total = pd.Series(0.0, index=out.index)
    for col, weight in weights.items():
        total += out[col].fillna(0) * weight

    out["ModeC_Technical_Score"] = total.round(2)
    return out


def classify_setup(row: pd.Series) -> str:
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


def classify_mode_c(row: pd.Series, near_close: bool = False, friday_strict: bool = False) -> str:
    if not row.get("Pass_Gates", False):
        return "REJECTED"

    rvol = row.get("Projected_RVol_20D", row.get("Volume_Ratio_20D", np.nan))
    clv = row.get("Close_Location_Value", np.nan)
    dist_ema10 = row.get("Dist_From_EMA10_Pct", np.nan)
    day_atr = row.get("Day_Range_ATR", np.nan)
    breakout = bool(row.get("Breakout_Hold_20D", False))
    vol_breakout = pd.notna(rvol) and rvol >= 2.0
    near_high = pd.notna(row.get("Dist_To_20D_High_Pct", np.nan)) and row.get("Dist_To_20D_High_Pct") <= 3

    if near_close:
        if pd.notna(clv) and clv < 50:
            return "FADING_INTO_CLOSE"
        if pd.notna(dist_ema10) and dist_ema10 > (5 if friday_strict else 8):
            return "EXTENDED_AVOID"
        if friday_strict:
            return "FRIDAY_STRONG_CLOSE" if breakout else "FRIDAY_ENTRY_READY"
        if breakout and vol_breakout:
            return "NEAR_CLOSE_BREAKOUT_READY"
        return "NEAR_CLOSE_ENTRY_READY"

    if pd.notna(dist_ema10) and dist_ema10 > 8:
        return "EXTENDED_AVOID"
    if breakout and vol_breakout:
        return "MOMENTUM_BREAKOUT_READY"
    if near_high:
        return "MOMENTUM_CONTINUATION_READY"
    return "MOMENTUM_WATCHLIST"


def setup_reason(row: pd.Series) -> str:
    if not row.get("Pass_Gates", False):
        return "Failed gates: " + str(row.get("Fail_Reasons", ""))

    status = row.get("Setup_Status", "")
    parts = []

    if status.startswith("SETUP") or status.startswith("TRIGGERED"):
        parts.append("Constructive early-stage setup.")
    elif "WEAK_RS" in status:
        parts.append("RS momentum is not yet improving.")
    elif "TOO_FAR" in status:
        parts.append("Price is still too far from resistance.")
    elif "POST_BREAKDOWN" in status:
        parts.append("Recent range may be post-breakdown consolidation, not accumulation.")
    elif "NOT_TIGHT" in status:
        parts.append("Near resistance but volatility/base is not tight enough.")
    elif "HIGHER_LOW" in status:
        parts.append("Needs clearer higher-low structure.")
    else:
        parts.append("Passed gates but needs manual chart review.")

    try:
        parts.append(f"Dist to resistance: {row.get('Dist_To_Resistance_Pct'):.2f}%.")
        parts.append(f"RS momentum: {row.get('RS_Momentum_10D'):.2f}.")
        parts.append(f"RSI: {row.get('Composite_RSI'):.1f}.")
    except Exception:
        pass

    return " ".join(parts)


def mode_c_reason(row: pd.Series) -> str:
    if not row.get("Pass_Gates", False):
        return "Failed gates: " + str(row.get("Fail_Reasons", ""))

    bits = []
    if row.get("Breakout_Hold_20D", False):
        bits.append("Holding above recent 20D resistance.")
    else:
        bits.append("Trend is aligned; not necessarily a fresh breakout.")

    try:
        bits.append(f"ADX {row.get('ADX14'):.1f}, RSI {row.get('RSI14'):.1f}.")
        bits.append(f"RS {row.get('Mansfield_RS'):.2f}, RS-Mom {row.get('RS_Momentum_10D'):.2f}.")
        bits.append(f"RVol {row.get('Volume_Ratio_20D'):.2f}x / projected {row.get('Projected_RVol_20D', row.get('Volume_Ratio_20D')):.2f}x.")
        bits.append(f"Close location {row.get('Close_Location_Value'):.0f}%, EMA10 distance {row.get('Dist_From_EMA10_Pct'):.2f}%.")
    except Exception:
        pass
    return " ".join(bits)

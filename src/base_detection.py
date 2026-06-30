from __future__ import annotations

import numpy as np
import pandas as pd


def detect_recent_base(
    df: pd.DataFrame,
    min_weeks: int = 4,
    max_weeks: int = 20,
    max_range_pct: float = 15.0,
) -> dict:
    """
    Detect the longest recent consolidation window ending at the latest bar.
    A valid base is a recent period where (highest high - lowest low) / lowest low < max_range_pct.

    This is intentionally simple for v0.1. It should be improved after chart validation.
    """
    if df is None or df.empty or len(df) < min_weeks * 5:
        return {
            "base_length_weeks": 0,
            "base_high": np.nan,
            "base_low": np.nan,
            "resistance": np.nan,
            "close_tightness_pct": np.nan,
            "base_valid": False,
        }

    best = None
    for weeks in range(min_weeks, max_weeks + 1):
        bars = weeks * 5
        if len(df) < bars:
            continue

        window = df.iloc[-bars:]
        high = float(window["High"].max())
        low = float(window["Low"].min())
        if low <= 0:
            continue

        range_pct = (high - low) / low * 100
        if range_pct <= max_range_pct:
            resistance = float(window["Close"].max())
            close_tightness = float(window["Close"].tail(20).std() / window["Close"].tail(20).mean() * 100)
            best = {
                "base_length_weeks": weeks,
                "base_high": high,
                "base_low": low,
                "resistance": resistance,
                "close_tightness_pct": close_tightness,
                "base_valid": True,
            }

    if best is None:
        recent = df.iloc[-min_weeks * 5 :]
        resistance = float(recent["Close"].max()) if not recent.empty else np.nan
        return {
            "base_length_weeks": 0,
            "base_high": float(recent["High"].max()) if not recent.empty else np.nan,
            "base_low": float(recent["Low"].min()) if not recent.empty else np.nan,
            "resistance": resistance,
            "close_tightness_pct": np.nan,
            "base_valid": False,
        }

    return best


def distance_to_resistance(close: float, resistance: float) -> float:
    if pd.isna(close) or pd.isna(resistance) or resistance == 0:
        return np.nan
    return (resistance - close) / resistance * 100

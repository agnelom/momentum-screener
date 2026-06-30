from __future__ import annotations

import numpy as np
import pandas as pd


def _count_higher_lows(lows: pd.Series, lookback_bars: int = 90) -> int:
    """Count successive swing lows where each swing low is higher than the previous one."""
    x = lows.dropna().tail(lookback_bars)
    if len(x) < 5:
        return 0

    swing_lows: list[float] = []
    vals = x.to_numpy(dtype=float)
    for i in range(2, len(vals) - 2):
        if vals[i] < vals[i - 1] and vals[i] < vals[i - 2] and vals[i] < vals[i + 1] and vals[i] < vals[i + 2]:
            swing_lows.append(float(vals[i]))

    if len(swing_lows) < 2:
        return 0

    count = 0
    for prev, curr in zip(swing_lows[:-1], swing_lows[1:]):
        if curr > prev:
            count += 1
    return count


def detect_recent_base(
    df: pd.DataFrame,
    min_weeks: int = 4,
    max_weeks: int = 20,
    max_range_pct: float = 18.0,
) -> dict:
    """
    Detect a recent consolidation window ending at the latest bar.

    The first MVP used only range width. This improved version also returns base-quality
    diagnostics so weak post-breakdown ranges are not treated the same as clean accumulation bases.
    """
    empty = {
        "base_length_weeks": 0,
        "base_high": np.nan,
        "base_low": np.nan,
        "resistance": np.nan,
        "close_tightness_pct": np.nan,
        "base_range_pct": np.nan,
        "higher_low_count": 0,
        "volume_dryup_ratio": np.nan,
        "prior_drop_pct": np.nan,
        "base_valid": False,
    }

    if df is None or df.empty or len(df) < min_weeks * 5:
        return empty

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
        if range_pct > max_range_pct:
            continue

        resistance = float(window["Close"].max())
        close_tail = window["Close"].tail(min(20, len(window)))
        close_tightness = float(close_tail.std() / close_tail.mean() * 100) if close_tail.mean() else np.nan
        higher_low_count = _count_higher_lows(window["Low"], lookback_bars=bars)

        if "Volume" in window.columns and len(window) >= 20:
            recent_vol = window["Volume"].tail(10).mean()
            base_vol = window["Volume"].mean()
            volume_dryup_ratio = float(recent_vol / base_vol) if base_vol else np.nan
        else:
            volume_dryup_ratio = np.nan

        start_pos = max(0, len(df) - bars)
        prior = df.iloc[max(0, start_pos - 60):start_pos]
        if not prior.empty:
            prior_high = float(prior["High"].max())
            prior_drop_pct = (prior_high - low) / prior_high * 100 if prior_high > 0 else np.nan
        else:
            prior_drop_pct = np.nan

        # Prefer longer bases, then tighter bases.
        candidate = {
            "base_length_weeks": weeks,
            "base_high": high,
            "base_low": low,
            "resistance": resistance,
            "close_tightness_pct": close_tightness,
            "base_range_pct": float(range_pct),
            "higher_low_count": int(higher_low_count),
            "volume_dryup_ratio": volume_dryup_ratio,
            "prior_drop_pct": float(prior_drop_pct) if pd.notna(prior_drop_pct) else np.nan,
            "base_valid": True,
        }

        if best is None:
            best = candidate
        else:
            if candidate["base_length_weeks"] > best["base_length_weeks"]:
                best = candidate
            elif candidate["base_length_weeks"] == best["base_length_weeks"]:
                if candidate["close_tightness_pct"] < best["close_tightness_pct"]:
                    best = candidate

    if best is None:
        recent = df.iloc[-min_weeks * 5:]
        resistance = float(recent["Close"].max()) if not recent.empty else np.nan
        out = empty.copy()
        out.update({
            "base_high": float(recent["High"].max()) if not recent.empty else np.nan,
            "base_low": float(recent["Low"].min()) if not recent.empty else np.nan,
            "resistance": resistance,
        })
        return out

    return best


def distance_to_resistance(close: float, resistance: float) -> float:
    if pd.isna(close) or pd.isna(resistance) or resistance == 0:
        return np.nan
    return (resistance - close) / resistance * 100

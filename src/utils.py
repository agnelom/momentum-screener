from __future__ import annotations

import re
import pandas as pd


def clean_symbol(symbol: str) -> str:
    """Clean NSE symbol input and remove exchange suffixes."""
    if symbol is None:
        return ""
    s = str(symbol).strip().upper()
    s = s.replace(".NS", "").replace(".BO", "")
    s = re.sub(r"[^A-Z0-9&\-_]", "", s)
    return s


def to_yahoo_symbol(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance NSE ticker."""
    s = clean_symbol(symbol)
    if not s:
        return ""
    return f"{s}.NS"


def tradingview_link(symbol: str) -> str:
    s = clean_symbol(symbol)
    return f"https://in.tradingview.com/chart/?symbol=NSE:{s}"


def normalize_0_100(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile-rank a series into 0-100 score."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(index=series.index, dtype=float)
    rank = x.rank(pct=True) * 100
    if not higher_is_better:
        rank = 100 - rank
    return rank


def score_distance_to_resistance(dist_pct: pd.Series) -> pd.Series:
    """
    Score distance to resistance.
    Best zone: 0% to 3% below resistance.
    Already above resistance gets a decent score but is considered triggered/late.
    Too far below resistance gets lower score.
    """
    d = pd.to_numeric(dist_pct, errors="coerce")
    score = pd.Series(0.0, index=d.index)

    # ideal: 0 to 3% below resistance
    score[(d >= 0) & (d <= 3)] = 100
    score[(d > 3) & (d <= 7)] = 75
    score[(d > 7) & (d <= 12)] = 45
    score[(d > 12)] = 15

    # already above resistance; useful but no longer "verge" setup
    score[(d < 0) & (d >= -5)] = 80
    score[(d < -5)] = 45
    return score


def score_ma_slope_for_mode_b(slope: pd.Series) -> pd.Series:
    """
    Mode B prefers flattening / early turn-up MA.
    Very steep already-rising MA may mean the move is no longer early.
    """
    s = pd.to_numeric(slope, errors="coerce")
    score = pd.Series(0.0, index=s.index)
    score[(s >= -0.5) & (s <= 1.5)] = 100
    score[(s > 1.5) & (s <= 4)] = 75
    score[(s > 4)] = 45
    score[(s < -0.5) & (s >= -2)] = 55
    score[(s < -2)] = 20
    return score

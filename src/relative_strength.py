from __future__ import annotations

import pandas as pd


def mansfield_rs(stock_close: pd.Series, benchmark_close: pd.Series, ma_window: int = 252, smooth_window: int = 10) -> pd.Series:
    """
    Mansfield-style relative strength:
    ((stock / benchmark) / SMA(stock / benchmark, 52w) - 1) * 100,
    smoothed using a 10-period SMA.
    """
    aligned = pd.concat([stock_close.rename("stock"), benchmark_close.rename("bench")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(index=stock_close.index, dtype=float)

    rs_ratio = aligned["stock"] / aligned["bench"]
    mansfield = ((rs_ratio / rs_ratio.rolling(ma_window, min_periods=max(60, ma_window // 2)).mean()) - 1) * 100
    mansfield = mansfield.rolling(smooth_window, min_periods=1).mean()
    return mansfield.reindex(stock_close.index)


def rs_momentum(mansfield_series: pd.Series, period: int = 10) -> pd.Series:
    return mansfield_series - mansfield_series.shift(period)

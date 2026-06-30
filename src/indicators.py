from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def composite_rsi(close: pd.Series) -> pd.Series:
    return pd.concat(
        [rsi(close, 5), rsi(close, 10), rsi(close, 15), rsi(close, 20)],
        axis=1,
    ).mean(axis=1)


def roc(close: pd.Series, period: int = 20) -> pd.Series:
    return close.pct_change(periods=period) * 100


def volume_ratio(volume: pd.Series, period: int = 50) -> pd.Series:
    return volume / volume.rolling(period, min_periods=period).mean()


def bollinger_band_width_percentile(close: pd.Series, window: int = 20, lookback: int = 252) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bbw = (upper - lower) / mid.replace(0, np.nan)

    def last_percentile(x):
        s = pd.Series(x).dropna()
        if len(s) == 0:
            return np.nan
        return (s <= s.iloc[-1]).mean() * 100

    pctl = bbw.rolling(lookback, min_periods=max(60, window)).apply(last_percentile, raw=False)
    return bbw, pctl


def macd_histogram(close: pd.Series) -> pd.Series:
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    return macd - signal


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

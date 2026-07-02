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


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ADX. Measures trend strength, not direction."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def close_location_value(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Where close/current price is within the day's range: 0 = low, 100 = high."""
    rng = (high - low).replace(0, np.nan)
    return ((close - low) / rng * 100).clip(lower=0, upper=100)

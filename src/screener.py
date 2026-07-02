from __future__ import annotations

import numpy as np
import pandas as pd

from .base_detection import detect_recent_base, distance_to_resistance
from .data_loader import download_benchmark, download_universe
from .gates import apply_mode_b_gates, apply_mode_c_eod_gates, apply_mode_c_near_close_gates
from .indicators import (
    adx,
    atr,
    bollinger_band_width_percentile,
    close_location_value,
    composite_rsi,
    ema,
    roc,
    rsi,
    sma,
    volume_ratio,
)
from .relative_strength import mansfield_rs, rs_momentum
from .scoring import (
    classify_mode_c,
    classify_setup,
    compute_mode_b_scores,
    compute_mode_c_scores,
    mode_c_reason,
    setup_reason,
)
from .utils import tradingview_link


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) and pd.notna(series.iloc[-1]) else np.nan


def compute_mode_b_latest_factors(symbol: str, df: pd.DataFrame, benchmark_close: pd.Series) -> dict:
    close = df["Close"]
    volume = df["Volume"]

    sma150 = sma(close, 150)
    sma150_slope = (sma150 - sma150.shift(20)) / sma150.shift(20) * 100
    price_vs_150 = close / sma150

    mrs = mansfield_rs(close, benchmark_close)
    rsm = rs_momentum(mrs, 10)

    vol_ratio = volume_ratio(volume, 50)
    bbw, bbw_pctl = bollinger_band_width_percentile(close)
    roc20 = roc(close, 20)
    crsi = composite_rsi(close)

    base = detect_recent_base(df)
    latest_close = float(close.iloc[-1])
    resistance = base["resistance"]
    dist_res = distance_to_resistance(latest_close, resistance)

    avg_traded_value_cr = float((close * volume).rolling(20, min_periods=10).mean().iloc[-1] / 1e7)

    return {
        "Symbol": symbol,
        "Close": latest_close,
        "Price_vs_150DMA": _latest(price_vs_150),
        "SMA150": _latest(sma150),
        "SMA150_Slope_20D_Pct": _latest(sma150_slope),
        "Mansfield_RS": _latest(mrs),
        "RS_Momentum_10D": _latest(rsm),
        "Volume_Ratio": _latest(vol_ratio),
        "BBW": _latest(bbw),
        "BBW_Pctl": _latest(bbw_pctl),
        "ROC_20D_Pct": _latest(roc20),
        "Composite_RSI": _latest(crsi),
        "Base_Length_Weeks": base["base_length_weeks"],
        "Base_High": base["base_high"],
        "Base_Low": base["base_low"],
        "Resistance": resistance,
        "Dist_To_Resistance_Pct": dist_res,
        "Close_Tightness_Pct": base["close_tightness_pct"],
        "Base_Range_Pct": base.get("base_range_pct", np.nan),
        "Higher_Low_Count": base.get("higher_low_count", 0),
        "Volume_Dryup_Ratio": base.get("volume_dryup_ratio", np.nan),
        "Prior_Drop_Pct": base.get("prior_drop_pct", np.nan),
        "Avg_Traded_Value_20D_Cr": avg_traded_value_cr,
        "TradingView": tradingview_link(symbol),
    }


def compute_mode_c_latest_factors(
    symbol: str,
    df: pd.DataFrame,
    benchmark_close: pd.Series,
    trading_day_progress_pct: float = 100.0,
) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema10 = ema(close, 10)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    adx14 = adx(high, low, close, 14)
    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14)

    mrs = mansfield_rs(close, benchmark_close)
    rsm = rs_momentum(mrs, 10)
    vol_ratio_20 = volume_ratio(volume, 20)
    vol_ratio_50 = volume_ratio(volume, 50)

    latest_close = float(close.iloc[-1])
    latest_ema10 = _latest(ema10)
    latest_atr = _latest(atr14)

    prev_20d_high_close = close.shift(1).rolling(20, min_periods=10).max()
    prev_50d_high_close = close.shift(1).rolling(50, min_periods=20).max()
    resistance_20d = _latest(prev_20d_high_close)
    resistance_50d = _latest(prev_50d_high_close)
    dist_to_20d_high = distance_to_resistance(latest_close, resistance_20d)
    dist_to_50d_high = distance_to_resistance(latest_close, resistance_50d)

    current_range = float(high.iloc[-1] - low.iloc[-1]) if pd.notna(high.iloc[-1]) and pd.notna(low.iloc[-1]) else np.nan
    day_range_atr = current_range / latest_atr if pd.notna(current_range) and pd.notna(latest_atr) and latest_atr != 0 else np.nan
    clv = _latest(close_location_value(close, high, low))
    avg_traded_value_cr = float((close * volume).rolling(20, min_periods=10).mean().iloc[-1] / 1e7)

    progress = max(min(float(trading_day_progress_pct), 100.0), 1.0) / 100.0
    projected_volume = float(volume.iloc[-1]) / progress if pd.notna(volume.iloc[-1]) else np.nan
    avg_volume_20 = float(volume.rolling(20, min_periods=10).mean().iloc[-1])
    projected_rvol_20 = projected_volume / avg_volume_20 if avg_volume_20 and pd.notna(avg_volume_20) else np.nan

    return {
        "Symbol": symbol,
        "Close": latest_close,
        "EMA10": latest_ema10,
        "EMA20": _latest(ema20),
        "EMA50": _latest(ema50),
        "Dist_From_EMA10_Pct": ((latest_close - latest_ema10) / latest_ema10 * 100) if pd.notna(latest_ema10) and latest_ema10 != 0 else np.nan,
        "ADX14": _latest(adx14),
        "RSI14": _latest(rsi14),
        "Mansfield_RS": _latest(mrs),
        "RS_Momentum_10D": _latest(rsm),
        "Volume_Ratio_20D": _latest(vol_ratio_20),
        "Volume_Ratio_50D": _latest(vol_ratio_50),
        "RVol_20D_Pct": _latest(vol_ratio_20) * 100 if pd.notna(_latest(vol_ratio_20)) else np.nan,
        "Projected_RVol_20D": projected_rvol_20,
        "Projected_RVol_20D_Pct": projected_rvol_20 * 100 if pd.notna(projected_rvol_20) else np.nan,
        "Close_Location_Value": clv,
        "ATR14": latest_atr,
        "Day_Range_ATR": day_range_atr,
        "Resistance_20D": resistance_20d,
        "Resistance_50D": resistance_50d,
        "Dist_To_20D_High_Pct": dist_to_20d_high,
        "Dist_To_50D_High_Pct": dist_to_50d_high,
        "Breakout_Hold_20D": bool(pd.notna(resistance_20d) and latest_close > resistance_20d * 1.005),
        "Breakout_Hold_50D": bool(pd.notna(resistance_50d) and latest_close > resistance_50d * 1.005),
        "Avg_Traded_Value_20D_Cr": avg_traded_value_cr,
        "TradingView": tradingview_link(symbol),
    }


def _download_and_compute(symbols: list[str], benchmark_symbol: str, period: str, interval: str, compute_fn, min_history: int = 180, **kwargs):
    benchmark_close = download_benchmark(benchmark_symbol, period=period, interval=interval)
    if benchmark_close.empty:
        raise RuntimeError(f"Could not download benchmark data for {benchmark_symbol}")

    data = download_universe(symbols, period=period, interval=interval)
    rows = []
    errors = []

    for symbol in symbols:
        df = data.get(symbol)
        if df is None or df.empty:
            errors.append({"Symbol": symbol, "Error": "No OHLCV data"})
            continue
        if len(df) < min_history:
            errors.append({"Symbol": symbol, "Error": f"Insufficient history: {len(df)} bars"})
            continue
        try:
            rows.append(compute_fn(symbol, df, benchmark_close, **kwargs))
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})

    return pd.DataFrame(rows), pd.DataFrame(errors)


def run_mode_b_screener(
    symbols: list[str],
    benchmark_symbol: str = "^NSEI",
    period: str = "2y",
    interval: str = "1d",
    price_vs_150dma_min: float = 0.95,
    price_vs_150dma_max: float = 1.15,
    min_base_weeks: int = 4,
    min_price: float = 20.0,
    min_avg_traded_value_cr: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, errors = _download_and_compute(symbols, benchmark_symbol, period, interval, compute_mode_b_latest_factors, min_history=180)
    if results.empty:
        return pd.DataFrame(), errors

    results = apply_mode_b_gates(
        results,
        price_vs_150dma_min=price_vs_150dma_min,
        price_vs_150dma_max=price_vs_150dma_max,
        min_base_weeks=min_base_weeks,
        min_price=min_price,
        min_avg_traded_value_cr=min_avg_traded_value_cr,
    )
    results = compute_mode_b_scores(results)
    results["Setup_Status"] = results.apply(classify_setup, axis=1)
    results["Setup_Reason"] = results.apply(setup_reason, axis=1)
    results = results.sort_values(["Pass_Gates", "ModeB_Technical_Score"], ascending=[False, False]).reset_index(drop=True)
    return results, errors


def run_mode_c_eod_screener(
    symbols: list[str],
    benchmark_symbol: str = "^NSEI",
    period: str = "2y",
    interval: str = "1d",
    min_price: float = 20.0,
    min_avg_traded_value_cr: float = 2.0,
    adx_min: float = 25.0,
    adx_max: float = 45.0,
    rsi_min: float = 55.0,
    rsi_max: float = 72.0,
    max_dist_ema10_pct: float = 6.0,
    min_volume_ratio: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, errors = _download_and_compute(symbols, benchmark_symbol, period, interval, compute_mode_c_latest_factors, min_history=180)
    if results.empty:
        return pd.DataFrame(), errors
    results = apply_mode_c_eod_gates(
        results,
        min_price=min_price,
        min_avg_traded_value_cr=min_avg_traded_value_cr,
        adx_min=adx_min,
        adx_max=adx_max,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        max_dist_ema10_pct=max_dist_ema10_pct,
        min_volume_ratio=min_volume_ratio,
    )
    results = compute_mode_c_scores(results, near_close=False)
    results["Setup_Status"] = results.apply(lambda r: classify_mode_c(r, near_close=False), axis=1)
    results["Setup_Reason"] = results.apply(mode_c_reason, axis=1)
    return results.sort_values(["Pass_Gates", "ModeC_Technical_Score"], ascending=[False, False]).reset_index(drop=True), errors


def run_mode_c_near_close_screener(
    symbols: list[str],
    benchmark_symbol: str = "^NSEI",
    period: str = "2y",
    interval: str = "1d",
    min_price: float = 20.0,
    min_avg_traded_value_cr: float = 2.0,
    adx_min: float = 25.0,
    adx_max: float = 45.0,
    rsi_min: float = 55.0,
    rsi_max: float = 70.0,
    max_dist_ema10_pct: float = 6.0,
    projected_rvol_min: float = 1.5,
    close_location_min: float = 70.0,
    day_range_atr_max: float = 2.0,
    require_breakout_hold: bool = False,
    friday_strict: bool = False,
    trading_day_progress_pct: float = 92.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Friday strict tightens defaults regardless of UI values unless user sets stricter ones.
    if friday_strict:
        projected_rvol_min = max(projected_rvol_min, 2.0)
        close_location_min = max(close_location_min, 80.0)
        max_dist_ema10_pct = min(max_dist_ema10_pct, 5.0)
        day_range_atr_max = min(day_range_atr_max, 1.8)
        require_breakout_hold = True
        min_avg_traded_value_cr = max(min_avg_traded_value_cr, 10.0)

    results, errors = _download_and_compute(
        symbols,
        benchmark_symbol,
        period,
        interval,
        compute_mode_c_latest_factors,
        min_history=180,
        trading_day_progress_pct=trading_day_progress_pct,
    )
    if results.empty:
        return pd.DataFrame(), errors
    results = apply_mode_c_near_close_gates(
        results,
        min_price=min_price,
        min_avg_traded_value_cr=min_avg_traded_value_cr,
        adx_min=adx_min,
        adx_max=adx_max,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        max_dist_ema10_pct=max_dist_ema10_pct,
        projected_rvol_min=projected_rvol_min,
        close_location_min=close_location_min,
        day_range_atr_max=day_range_atr_max,
        require_breakout_hold=require_breakout_hold,
    )
    results = compute_mode_c_scores(results, near_close=True)
    results["Setup_Status"] = results.apply(lambda r: classify_mode_c(r, near_close=True, friday_strict=friday_strict), axis=1)
    results["Setup_Reason"] = results.apply(mode_c_reason, axis=1)
    return results.sort_values(["Pass_Gates", "ModeC_Technical_Score"], ascending=[False, False]).reset_index(drop=True), errors

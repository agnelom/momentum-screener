from __future__ import annotations

import numpy as np
import pandas as pd

from .base_detection import detect_recent_base, distance_to_resistance
from .data_loader import download_benchmark, download_universe
from .gates import apply_mode_b_gates
from .indicators import (
    bollinger_band_width_percentile,
    composite_rsi,
    roc,
    sma,
    volume_ratio,
)
from .relative_strength import mansfield_rs, rs_momentum
from .scoring import classify_setup, compute_mode_b_scores
from .utils import tradingview_link


def compute_latest_factors(symbol: str, df: pd.DataFrame, benchmark_close: pd.Series) -> dict:
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

    latest = {
        "Symbol": symbol,
        "Close": latest_close,
        "Price_vs_150DMA": float(price_vs_150.iloc[-1]) if pd.notna(price_vs_150.iloc[-1]) else np.nan,
        "SMA150": float(sma150.iloc[-1]) if pd.notna(sma150.iloc[-1]) else np.nan,
        "SMA150_Slope_20D_Pct": float(sma150_slope.iloc[-1]) if pd.notna(sma150_slope.iloc[-1]) else np.nan,
        "Mansfield_RS": float(mrs.iloc[-1]) if pd.notna(mrs.iloc[-1]) else np.nan,
        "RS_Momentum_10D": float(rsm.iloc[-1]) if pd.notna(rsm.iloc[-1]) else np.nan,
        "Volume_Ratio": float(vol_ratio.iloc[-1]) if pd.notna(vol_ratio.iloc[-1]) else np.nan,
        "BBW": float(bbw.iloc[-1]) if pd.notna(bbw.iloc[-1]) else np.nan,
        "BBW_Pctl": float(bbw_pctl.iloc[-1]) if pd.notna(bbw_pctl.iloc[-1]) else np.nan,
        "ROC_20D_Pct": float(roc20.iloc[-1]) if pd.notna(roc20.iloc[-1]) else np.nan,
        "Composite_RSI": float(crsi.iloc[-1]) if pd.notna(crsi.iloc[-1]) else np.nan,
        "Base_Length_Weeks": base["base_length_weeks"],
        "Base_High": base["base_high"],
        "Base_Low": base["base_low"],
        "Resistance": resistance,
        "Dist_To_Resistance_Pct": dist_res,
        "Close_Tightness_Pct": base["close_tightness_pct"],
        "Avg_Traded_Value_20D_Cr": avg_traded_value_cr,
        "TradingView": tradingview_link(symbol),
    }
    return latest


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
    """
    Returns:
      results_df: scored and gated results
      errors_df: symbols that could not be processed
    """
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

        if len(df) < 180:
            errors.append({"Symbol": symbol, "Error": f"Insufficient history: {len(df)} bars"})
            continue

        try:
            rows.append(compute_latest_factors(symbol, df, benchmark_close))
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})

    if not rows:
        return pd.DataFrame(), pd.DataFrame(errors)

    results = pd.DataFrame(rows)

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

    results = results.sort_values(
        ["Pass_Gates", "ModeB_Technical_Score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Friendly rounding
    round_cols = [
        "Close", "Price_vs_150DMA", "SMA150", "SMA150_Slope_20D_Pct",
        "Mansfield_RS", "RS_Momentum_10D", "Volume_Ratio", "BBW_Pctl",
        "ROC_20D_Pct", "Composite_RSI", "Resistance", "Dist_To_Resistance_Pct",
        "Close_Tightness_Pct", "Avg_Traded_Value_20D_Cr", "ModeB_Technical_Score",
    ]
    for c in round_cols:
        if c in results.columns:
            results[c] = pd.to_numeric(results[c], errors="coerce").round(2)

    return results, pd.DataFrame(errors)

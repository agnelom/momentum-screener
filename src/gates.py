from __future__ import annotations

import pandas as pd


def _add_fail_reasons(out: pd.DataFrame, gate_cols: list[str]) -> pd.DataFrame:
    def fail_reason(row):
        fails = [c.replace("Gate_", "") for c in gate_cols if not bool(row.get(c))]
        return ", ".join(fails)

    out["Pass_Gates"] = out[gate_cols].all(axis=1)
    out["Fail_Reasons"] = out.apply(fail_reason, axis=1)
    return out


def apply_mode_b_gates(
    df: pd.DataFrame,
    price_vs_150dma_min: float = 0.95,
    price_vs_150dma_max: float = 1.15,
    min_base_weeks: int = 4,
    min_price: float = 20.0,
    min_avg_traded_value_cr: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()

    out["Gate_Price_Min"] = out["Close"] >= min_price
    out["Gate_Avg_Traded_Value"] = out["Avg_Traded_Value_20D_Cr"] >= min_avg_traded_value_cr
    out["Gate_Price_vs_150DMA"] = (
        (out["Price_vs_150DMA"] >= price_vs_150dma_min)
        & (out["Price_vs_150DMA"] <= price_vs_150dma_max)
    )
    out["Gate_Base_Length"] = out["Base_Length_Weeks"] >= min_base_weeks

    return _add_fail_reasons(out, ["Gate_Price_Min", "Gate_Avg_Traded_Value", "Gate_Price_vs_150DMA", "Gate_Base_Length"])


def apply_mode_c_eod_gates(
    df: pd.DataFrame,
    min_price: float = 20.0,
    min_avg_traded_value_cr: float = 2.0,
    adx_min: float = 25.0,
    adx_max: float = 45.0,
    rsi_min: float = 55.0,
    rsi_max: float = 72.0,
    max_dist_ema10_pct: float = 6.0,
    min_volume_ratio: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    out["Gate_Price_Min"] = out["Close"] >= min_price
    out["Gate_Avg_Traded_Value"] = out["Avg_Traded_Value_20D_Cr"] >= min_avg_traded_value_cr
    out["Gate_EMA_Stack"] = (out["EMA10"] > out["EMA20"]) & (out["EMA20"] > out["EMA50"]) & (out["Close"] > out["EMA10"])
    out["Gate_ADX"] = (out["ADX14"] >= adx_min) & (out["ADX14"] <= adx_max)
    out["Gate_RSI"] = (out["RSI14"] >= rsi_min) & (out["RSI14"] <= rsi_max)
    out["Gate_RS_Positive"] = out["Mansfield_RS"] > 0
    out["Gate_RS_Momentum"] = out["RS_Momentum_10D"] > 0
    out["Gate_Not_Extended"] = (out["Dist_From_EMA10_Pct"] >= 0) & (out["Dist_From_EMA10_Pct"] <= max_dist_ema10_pct)
    out["Gate_Min_Volume"] = out["Volume_Ratio_20D"] >= min_volume_ratio

    gate_cols = [
        "Gate_Price_Min", "Gate_Avg_Traded_Value", "Gate_EMA_Stack", "Gate_ADX", "Gate_RSI",
        "Gate_RS_Positive", "Gate_RS_Momentum", "Gate_Not_Extended", "Gate_Min_Volume",
    ]
    return _add_fail_reasons(out, gate_cols)


def apply_mode_c_near_close_gates(
    df: pd.DataFrame,
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
) -> pd.DataFrame:
    out = df.copy()
    out["Gate_Price_Min"] = out["Close"] >= min_price
    out["Gate_Avg_Traded_Value"] = out["Avg_Traded_Value_20D_Cr"] >= min_avg_traded_value_cr
    out["Gate_EMA_Stack"] = (out["EMA10"] > out["EMA20"]) & (out["EMA20"] > out["EMA50"]) & (out["Close"] > out["EMA10"])
    out["Gate_ADX"] = (out["ADX14"] >= adx_min) & (out["ADX14"] <= adx_max)
    out["Gate_RSI"] = (out["RSI14"] >= rsi_min) & (out["RSI14"] <= rsi_max)
    out["Gate_RS_Positive"] = out["Mansfield_RS"] > 0
    out["Gate_RS_Momentum"] = out["RS_Momentum_10D"] > 0
    out["Gate_Not_Extended"] = (out["Dist_From_EMA10_Pct"] >= 0) & (out["Dist_From_EMA10_Pct"] <= max_dist_ema10_pct)
    out["Gate_Projected_RVol"] = out["Projected_RVol_20D"] >= projected_rvol_min
    out["Gate_Close_Location"] = out["Close_Location_Value"] >= close_location_min
    out["Gate_Day_Range_ATR"] = out["Day_Range_ATR"] <= day_range_atr_max
    out["Gate_Breakout_Hold"] = True if not require_breakout_hold else out["Breakout_Hold_20D"]

    gate_cols = [
        "Gate_Price_Min", "Gate_Avg_Traded_Value", "Gate_EMA_Stack", "Gate_ADX", "Gate_RSI",
        "Gate_RS_Positive", "Gate_RS_Momentum", "Gate_Not_Extended", "Gate_Projected_RVol",
        "Gate_Close_Location", "Gate_Day_Range_ATR", "Gate_Breakout_Hold",
    ]
    return _add_fail_reasons(out, gate_cols)

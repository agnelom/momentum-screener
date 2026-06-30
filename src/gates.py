from __future__ import annotations

import pandas as pd


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

    gate_cols = [
        "Gate_Price_Min",
        "Gate_Avg_Traded_Value",
        "Gate_Price_vs_150DMA",
        "Gate_Base_Length",
    ]

    out["Pass_Gates"] = out[gate_cols].all(axis=1)

    def fail_reason(row):
        fails = [c.replace("Gate_", "") for c in gate_cols if not bool(row.get(c))]
        return ", ".join(fails)

    out["Fail_Reasons"] = out.apply(fail_reason, axis=1)
    return out

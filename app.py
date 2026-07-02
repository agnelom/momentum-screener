from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_BENCHMARK,
    DEFAULT_MIN_AVG_TRADED_VALUE_CR,
    DEFAULT_MIN_BASE_WEEKS,
    DEFAULT_MIN_PRICE,
    DEFAULT_PRICE_VS_150DMA_MAX,
    DEFAULT_PRICE_VS_150DMA_MIN,
    DEFAULT_TOP_N,
)
from src.data_loader import parse_manual_symbols, parse_universe_csv
from src.screener import run_mode_b_screener, run_mode_c_eod_screener, run_mode_c_near_close_screener


st.set_page_config(page_title="Momentum Screener", layout="wide")

st.title("Momentum Screener")
st.caption("Mode B pre-breakout, Mode C EOD continuation, and Mode C near-close continuation. Screening tool only; manually verify charts and risk.")

MODE_B = "Mode B: Early Stage 2 / Pre-Breakout"
MODE_C_EOD = "Mode C: Momentum Continuation — EOD"
MODE_C_NEAR_CLOSE = "Mode C: Near-Close Momentum Continuation"

with st.sidebar:
    st.header("Universe")
    uploaded = st.file_uploader("Upload NSE universe CSV", type=["csv"])
    manual_symbols = st.text_area(
        "Or paste NSE symbols",
        value="",
        placeholder="DATAPATTNS, COCHINSHIP, TITAN, BSE",
        height=100,
    )

    st.header("Mode")
    mode = st.selectbox("Screening mode", [MODE_B, MODE_C_EOD, MODE_C_NEAR_CLOSE], index=0)

    st.header("Data")
    benchmark = st.text_input("Benchmark Yahoo symbol", value=DEFAULT_BENCHMARK, help="Default is ^NSEI. You can try ^CNX500 if Yahoo supports it.")
    period = st.selectbox("Lookback period", ["1y", "2y", "3y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d"], index=0)
    top_n = st.number_input("Show top N", min_value=10, max_value=500, value=int(DEFAULT_TOP_N), step=10)

    st.header("Common gates")
    min_price = st.number_input("Minimum price ₹", min_value=0.0, value=float(DEFAULT_MIN_PRICE), step=5.0)
    min_atv = st.number_input("Minimum avg traded value ₹ Cr", min_value=0.0, value=float(DEFAULT_MIN_AVG_TRADED_VALUE_CR), step=0.5)

    mode_b_params = {}
    mode_c_params = {}
    near_close_params = {}

    if mode == MODE_B:
        st.header("Mode B gates")
        mode_b_params["price_vs_150dma_min"] = st.number_input("Min Close / 150-DMA", min_value=0.50, max_value=2.0, value=float(DEFAULT_PRICE_VS_150DMA_MIN), step=0.01)
        mode_b_params["price_vs_150dma_max"] = st.number_input("Max Close / 150-DMA", min_value=0.50, max_value=3.0, value=float(DEFAULT_PRICE_VS_150DMA_MAX), step=0.01)
        mode_b_params["min_base_weeks"] = st.number_input("Minimum base length weeks", min_value=1, max_value=30, value=int(DEFAULT_MIN_BASE_WEEKS), step=1)

    if mode in [MODE_C_EOD, MODE_C_NEAR_CLOSE]:
        st.header("Mode C trend gates")
        c1, c2 = st.columns(2)
        with c1:
            mode_c_params["adx_min"] = st.number_input("ADX min", min_value=0.0, max_value=80.0, value=25.0, step=1.0)
            mode_c_params["rsi_min"] = st.number_input("RSI min", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
            mode_c_params["max_dist_ema10_pct"] = st.number_input("Max distance from EMA10 %", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
        with c2:
            mode_c_params["adx_max"] = st.number_input("ADX max", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            mode_c_params["rsi_max"] = st.number_input("RSI max", min_value=0.0, max_value=100.0, value=72.0 if mode == MODE_C_EOD else 70.0, step=1.0)
            if mode == MODE_C_EOD:
                mode_c_params["min_volume_ratio"] = st.number_input("Minimum day volume / 20D avg", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if mode == MODE_C_NEAR_CLOSE:
        st.header("Near-close settings")
        friday_strict = st.toggle("Friday Strict Mode", value=False)
        near_close_params["friday_strict"] = friday_strict
        near_close_params["trading_day_progress_pct"] = st.slider(
            "Trading day completed %",
            min_value=50,
            max_value=100,
            value=92,
            step=1,
            help="Used to project full-day volume from current intraday volume. Last 30 minutes of NSE is roughly 92%+ complete.",
        )
        near_close_params["projected_rvol_min"] = st.number_input("Projected RVol minimum", min_value=0.0, max_value=10.0, value=2.0 if friday_strict else 1.5, step=0.1)
        near_close_params["close_location_min"] = st.number_input("Close location minimum %", min_value=0.0, max_value=100.0, value=80.0 if friday_strict else 70.0, step=5.0)
        near_close_params["day_range_atr_max"] = st.number_input("Max day range / ATR14", min_value=0.1, max_value=5.0, value=1.8 if friday_strict else 2.0, step=0.1)
        near_close_params["require_breakout_hold"] = st.toggle("Require 20D breakout hold", value=True if friday_strict else False)
        if friday_strict:
            st.info("Friday Strict Mode automatically enforces: projected RVol >= 2.0, close location >= 80%, EMA10 distance <= 5%, day range <= 1.8 ATR, breakout hold required, and avg traded value >= ₹10 Cr.")

    run_btn = st.button("Run screener", type="primary")


def get_symbols() -> list[str]:
    symbols = []
    if uploaded is not None:
        symbols.extend(parse_universe_csv(uploaded))
    symbols.extend(parse_manual_symbols(manual_symbols))
    return sorted(set(symbols))


def show_results(results: pd.DataFrame, errors: pd.DataFrame, score_col: str, display_cols: list[str], file_name: str):
    if results.empty:
        st.error("No valid results generated.")
        if not errors.empty:
            st.dataframe(errors, use_container_width=True, hide_index=True)
        return

    passing = results[results["Pass_Gates"] == True].copy()
    rejected = results[results["Pass_Gates"] == False].copy()

    st.subheader("Status summary")
    status_counts = results["Setup_Status"].value_counts().reset_index()
    status_counts.columns = ["Setup_Status", "Count"]
    st.dataframe(status_counts, use_container_width=True, hide_index=True)

    st.subheader("Top candidates")
    if passing.empty:
        st.warning("No stocks passed the current gates. Try loosening gates or reviewing rejected reasons.")
    else:
        top = passing.head(int(top_n))
        available_cols = [c for c in display_cols if c in top.columns]
        st.dataframe(
            top[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "TradingView": st.column_config.LinkColumn("TradingView"),
                score_col: st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results CSV", data=csv, file_name=file_name, mime="text/csv")

    with st.expander("Rejected stocks and reasons"):
        cols = ["Symbol", "Close", score_col, "Fail_Reasons", "TradingView"]
        cols = [c for c in cols if c in rejected.columns]
        st.dataframe(rejected[cols], use_container_width=True, hide_index=True, column_config={"TradingView": st.column_config.LinkColumn("TradingView")})

    if not errors.empty:
        with st.expander("Download / processing errors"):
            st.dataframe(errors, use_container_width=True, hide_index=True)


symbols = get_symbols()
if symbols:
    st.info(f"Universe loaded: {len(symbols)} symbols")
else:
    st.warning("Upload a CSV or paste NSE symbols to start.")

if run_btn:
    if not symbols:
        st.error("No symbols found. Upload a CSV or paste symbols first.")
        st.stop()

    with st.spinner(f"Running {mode}..."):
        try:
            if mode == MODE_B:
                results, errors = run_mode_b_screener(
                    symbols=symbols,
                    benchmark_symbol=benchmark,
                    period=period,
                    interval=interval,
                    min_price=float(min_price),
                    min_avg_traded_value_cr=float(min_atv),
                    **mode_b_params,
                )
                display_cols = [
                    "Symbol", "Setup_Status", "Setup_Reason", "ModeB_Technical_Score", "Close",
                    "Price_vs_150DMA", "SMA150_Slope_20D_Pct", "Mansfield_RS", "RS_Momentum_10D",
                    "BBW_Pctl", "Base_Length_Weeks", "Base_Range_Pct", "Higher_Low_Count",
                    "Prior_Drop_Pct", "Dist_To_Resistance_Pct", "Volume_Ratio", "Composite_RSI",
                    "Avg_Traded_Value_20D_Cr", "TradingView",
                ]
                show_results(results, errors, "ModeB_Technical_Score", display_cols, "momentum_screener_results.csv")

            elif mode == MODE_C_EOD:
                results, errors = run_mode_c_eod_screener(
                    symbols=symbols,
                    benchmark_symbol=benchmark,
                    period=period,
                    interval=interval,
                    min_price=float(min_price),
                    min_avg_traded_value_cr=float(min_atv),
                    **mode_c_params,
                )
                display_cols = [
                    "Symbol", "Setup_Status", "Setup_Reason", "ModeC_Technical_Score", "Close",
                    "EMA10", "EMA20", "EMA50", "Dist_From_EMA10_Pct", "ADX14", "RSI14",
                    "Mansfield_RS", "RS_Momentum_10D", "Volume_Ratio_20D", "RVol_20D_Pct",
                    "Close_Location_Value", "Day_Range_ATR", "Dist_To_20D_High_Pct", "Breakout_Hold_20D",
                    "Avg_Traded_Value_20D_Cr", "TradingView",
                ]
                show_results(results, errors, "ModeC_Technical_Score", display_cols, "momentum_continuation_eod_results.csv")

            else:
                results, errors = run_mode_c_near_close_screener(
                    symbols=symbols,
                    benchmark_symbol=benchmark,
                    period=period,
                    interval=interval,
                    min_price=float(min_price),
                    min_avg_traded_value_cr=float(min_atv),
                    **mode_c_params,
                    **near_close_params,
                )
                display_cols = [
                    "Symbol", "Setup_Status", "Setup_Reason", "ModeC_Technical_Score", "Close",
                    "EMA10", "EMA20", "EMA50", "Dist_From_EMA10_Pct", "ADX14", "RSI14",
                    "Mansfield_RS", "RS_Momentum_10D", "Volume_Ratio_20D", "Projected_RVol_20D",
                    "Projected_RVol_20D_Pct", "Close_Location_Value", "Day_Range_ATR", "Dist_To_20D_High_Pct",
                    "Breakout_Hold_20D", "Avg_Traded_Value_20D_Cr", "TradingView",
                ]
                show_results(results, errors, "ModeC_Technical_Score", display_cols, "near_close_momentum_results.csv")
        except Exception as exc:
            st.error(f"Screener failed: {exc}")
            st.stop()
else:
    st.markdown(
        """
        ### Available modes

        **Mode B: Early Stage 2 / Pre-Breakout**  
        Finds stocks near transition from base to breakout. Best for EOD/weekend research.

        **Mode C: Momentum Continuation — EOD**  
        Finds stocks already in confirmed momentum using EMA stack, ADX, RSI, RS, volume and extension control.

        **Mode C: Near-Close Momentum Continuation**  
        Designed for the last 30 minutes of market hours. Adds projected RVol, close-location value, breakout hold, day-range/ATR and optional Friday Strict Mode.

        ### Practical usage

        For regular EOD screening, use **Mode C: Momentum Continuation — EOD**.  
        For last-30-minute scans, use **Mode C: Near-Close Momentum Continuation**.  
        On Friday, switch **Friday Strict Mode** on and reduce position size because of weekend gap risk.
        """
    )

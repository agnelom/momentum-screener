from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_BENCHMARK,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK_PERIOD,
    DEFAULT_MIN_AVG_TRADED_VALUE_CR,
    DEFAULT_MIN_BASE_WEEKS,
    DEFAULT_MIN_PRICE,
    DEFAULT_PRICE_VS_150DMA_MAX,
    DEFAULT_PRICE_VS_150DMA_MIN,
    DEFAULT_TOP_N,
)
from src.data_loader import parse_manual_symbols, parse_universe_csv
from src.screener import run_mode_b_screener


st.set_page_config(page_title="Momentum Screener v0.1", layout="wide")

st.title("Momentum Screener v0.1 — Mode B Technical MVP")
st.caption("Early Stage 2 / pre-breakout screener for NSE stocks. Use this for watchlist generation, not direct buy/sell decisions.")

with st.sidebar:
    st.header("Universe")

    uploaded = st.file_uploader("Upload NSE universe CSV", type=["csv"])
    manual_symbols = st.text_area(
        "Or paste NSE symbols",
        value="",
        placeholder="DATAPATTNS, COCHINSHIP, TITAN, BSE",
        height=100,
    )

    st.header("Data")
    benchmark = st.text_input("Benchmark Yahoo symbol", value=DEFAULT_BENCHMARK, help="Default is ^NSEI. Try ^CNX500 if Yahoo supports it.")
    period = st.selectbox("Lookback period", ["1y", "2y", "3y", "5y"], index=1)
    interval = st.selectbox("Interval", ["1d"], index=0)

    st.header("Mode B Gates")
    min_price = st.number_input("Minimum price ₹", min_value=0.0, value=float(DEFAULT_MIN_PRICE), step=5.0)
    min_atv = st.number_input("Minimum avg traded value ₹ Cr", min_value=0.0, value=float(DEFAULT_MIN_AVG_TRADED_VALUE_CR), step=0.5)
    pv_min = st.number_input("Min Close / 150-DMA", min_value=0.50, max_value=2.0, value=float(DEFAULT_PRICE_VS_150DMA_MIN), step=0.01)
    pv_max = st.number_input("Max Close / 150-DMA", min_value=0.50, max_value=3.0, value=float(DEFAULT_PRICE_VS_150DMA_MAX), step=0.01)
    min_base_weeks = st.number_input("Minimum base length weeks", min_value=1, max_value=30, value=int(DEFAULT_MIN_BASE_WEEKS), step=1)
    top_n = st.number_input("Show top N", min_value=10, max_value=500, value=int(DEFAULT_TOP_N), step=10)

    run_btn = st.button("Run screener", type="primary")


def get_symbols():
    symbols = []
    if uploaded is not None:
        symbols.extend(parse_universe_csv(uploaded))
    symbols.extend(parse_manual_symbols(manual_symbols))
    return sorted(set(symbols))


symbols = get_symbols()

if symbols:
    st.info(f"Universe loaded: {len(symbols)} symbols")
else:
    st.warning("Upload a CSV or paste NSE symbols to start.")

if run_btn:
    if not symbols:
        st.error("No symbols found. Upload a CSV or paste symbols first.")
        st.stop()

    with st.spinner("Running Mode B technical screener..."):
        try:
            results, errors = run_mode_b_screener(
                symbols=symbols,
                benchmark_symbol=benchmark,
                period=period,
                interval=interval,
                price_vs_150dma_min=pv_min,
                price_vs_150dma_max=pv_max,
                min_base_weeks=int(min_base_weeks),
                min_price=float(min_price),
                min_avg_traded_value_cr=float(min_atv),
            )
        except Exception as exc:
            st.error(f"Screener failed: {exc}")
            st.stop()

    if results.empty:
        st.error("No valid results generated.")
        if not errors.empty:
            st.dataframe(errors, use_container_width=True)
        st.stop()

    passing = results[results["Pass_Gates"] == True].copy()
    rejected = results[results["Pass_Gates"] == False].copy()

    st.subheader("Top Mode B Candidates")
    st.caption("Sorted by Mode B Technical Score. Review charts manually before taking action.")

    display_cols = [
        "Symbol",
        "Setup_Status",
        "ModeB_Technical_Score",
        "Close",
        "Price_vs_150DMA",
        "SMA150_Slope_20D_Pct",
        "Mansfield_RS",
        "RS_Momentum_10D",
        "BBW_Pctl",
        "Base_Length_Weeks",
        "Dist_To_Resistance_Pct",
        "Volume_Ratio",
        "Composite_RSI",
        "Avg_Traded_Value_20D_Cr",
        "TradingView",
    ]

    top = passing.head(int(top_n))
    st.dataframe(
        top[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "TradingView": st.column_config.LinkColumn("TradingView"),
            "ModeB_Technical_Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full results CSV",
        data=csv,
        file_name="momentum_screener_v01_results.csv",
        mime="text/csv",
    )

    with st.expander("Rejected stocks and reasons"):
        st.dataframe(
            rejected[["Symbol", "Close", "ModeB_Technical_Score", "Fail_Reasons", "TradingView"]],
            use_container_width=True,
            hide_index=True,
            column_config={"TradingView": st.column_config.LinkColumn("TradingView")},
        )

    if not errors.empty:
        with st.expander("Download / processing errors"):
            st.dataframe(errors, use_container_width=True, hide_index=True)

else:
    st.markdown(
        """
        ### How to use

        1. Upload a CSV with a `Symbol` column, or paste symbols manually.
        2. Keep the benchmark as `^NSEI` for the first run.
        3. Start with loose gates.
        4. Run the screener.
        5. Review the top candidates on TradingView.

        ### Suggested first test universe

        Paste 20–50 symbols first to validate the output before running a large universe.
        """
    )

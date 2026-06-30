# Momentum Screener v0.1 — Mode B Technical MVP

This is the first working version of the Mode B / Early Stage 2 momentum screener.

## What it does

- Accepts an uploaded NSE universe CSV or lets you paste symbols manually.
- Downloads daily OHLCV data using Yahoo Finance via `yfinance`.
- Computes Mode B technical factors:
  - Price vs 150-DMA
  - 150-DMA slope
  - Mansfield Relative Strength
  - RS Momentum
  - Volume Ratio
  - Bollinger Band Width Percentile
  - ROC 20
  - Composite RSI
  - Base length
  - Resistance
  - Distance to resistance
  - Average traded value
- Applies configurable Mode B gates.
- Produces a ranked shortlist with setup status and TradingView links.
- Allows CSV download from the Streamlit UI.

## Install

```bash
cd momentum_screener_v01
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## CSV format

Upload a CSV with one of these columns:

```text
Symbol
SYMBOL
symbol
Ticker
ticker
```

Symbols can be like:

```text
TITAN
BSE
COCHINSHIP
DATAPATTNS
```

The app will automatically append `.NS` for Yahoo Finance.

## Important notes

This is a screening tool, not a buy/sell system. Use it to generate a ranked watchlist and then manually review charts, fundamentals, liquidity, results, and news.

Version 0.1 intentionally excludes fundamentals. Fundamental factors such as revenue growth, EPS acceleration, ROCE, promoter pledge, debt/equity, and Piotroski F-Score should be added in Version 0.2/0.3 after the technical engine is validated.

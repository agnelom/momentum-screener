# Momentum Screener — Mode B Technical

This is a Streamlit-based early Stage 2 / pre-breakout momentum screener for NSE stocks.

The project intentionally starts with technical factors first so the chart logic can be validated before adding fundamentals.

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
  - Base range
  - Higher-low count
  - Prior drop into base
  - Resistance
  - Distance to resistance
  - Average traded value
- Applies configurable Mode B gates.
- Produces descriptive setup labels and a ranked shortlist with TradingView links.
- Allows CSV download from the Streamlit UI.

## Key improvement in this build

The earlier `WATCH` label was too broad. This build splits it into more useful statuses:

- `SETUP_READY`
- `TRIGGERED_BREAKOUT`
- `WATCH_WEAK_RS`
- `WATCH_WEAK_RS_TOO_FAR`
- `WATCH_TOO_FAR_FROM_RESISTANCE`
- `WATCH_NEAR_RESISTANCE_NOT_TIGHT`
- `WATCH_NEEDS_HIGHER_LOW`
- `WATCH_POST_BREAKDOWN`
- `WATCH_REVIEW`
- `REJECTED`

The output also includes `Setup_Reason`, so you can quickly see why a stock received its label.

## Install

```bash
cd momentum_screener
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

Fundamental factors such as revenue growth, EPS acceleration, ROCE, promoter pledge, debt/equity, and Piotroski F-Score are intentionally excluded until the technical engine is validated.

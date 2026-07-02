# Momentum Screener

Streamlit-based NSE momentum screener with three modes:

1. **Mode B: Early Stage 2 / Pre-Breakout**
2. **Mode C: Momentum Continuation — EOD**
3. **Mode C: Near-Close Momentum Continuation** with **Friday Strict Mode ON/OFF**

The project is designed as a screening and watchlist-generation tool, not a buy/sell system. Always manually review charts, liquidity, market context, results/news risk, and your stop loss.

## What changed in this build

This build adds two new continuation modes:

### Mode C: Momentum Continuation — EOD

Use after the full daily candle is complete. It looks for stocks that are already in confirmed momentum and may continue another 10%–12%.

Core checks:

- EMA10 > EMA20 > EMA50
- Close above EMA10
- ADX(14) between configurable min/max, default 25–45
- RSI(14) in configurable zone, default 55–72
- Mansfield RS positive versus selected benchmark
- RS momentum positive
- Volume ratio / RVol
- Distance from EMA10 to avoid extended entries
- Distance to 20D/50D high and breakout-hold status

### Mode C: Near-Close Momentum Continuation

Use during the final 30 minutes of the market. It adds near-close entry-quality checks:

- Projected RVol based on trading day completion percentage
- Close Location Value: where current price is within the day's high-low range
- Day Range / ATR14 to avoid exhaustion candles
- Optional 20D breakout hold requirement
- Friday Strict Mode ON/OFF

### Friday Strict Mode

When ON, the app automatically enforces stricter rules:

- Projected RVol >= 2.0x
- Close Location Value >= 80%
- Distance from EMA10 <= 5%
- Day Range <= 1.8 × ATR14
- 20D breakout hold required
- Minimum average traded value >= ₹10 Cr

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

The app automatically appends `.NS` for Yahoo Finance.

## Important data note

The app uses Yahoo Finance via `yfinance`. During market hours, daily candles and volume can be delayed or incomplete depending on Yahoo availability. The Near-Close mode projects volume from the currently available volume, but you should verify critical candidates on TradingView/NSE before entering.

## Output status examples

### Mode B

- `SETUP_READY`
- `TRIGGERED_BREAKOUT`
- `WATCH_WEAK_RS`
- `WATCH_TOO_FAR_FROM_RESISTANCE`
- `WATCH_POST_BREAKDOWN`
- `REJECTED`

### Mode C EOD

- `MOMENTUM_BREAKOUT_READY`
- `MOMENTUM_CONTINUATION_READY`
- `MOMENTUM_WATCHLIST`
- `EXTENDED_AVOID`
- `REJECTED`

### Mode C Near-Close

- `NEAR_CLOSE_BREAKOUT_READY`
- `NEAR_CLOSE_ENTRY_READY`
- `FRIDAY_STRONG_CLOSE`
- `FRIDAY_ENTRY_READY`
- `FADING_INTO_CLOSE`
- `EXTENDED_AVOID`
- `REJECTED`


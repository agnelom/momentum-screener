from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from .utils import clean_symbol, to_yahoo_symbol


def parse_universe_csv(uploaded_file) -> list[str]:
    """Read uploaded CSV and return cleaned NSE symbols."""
    df = pd.read_csv(uploaded_file)
    candidates = ["Symbol", "SYMBOL", "symbol", "Ticker", "ticker", "NSE Symbol"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"CSV must contain one of these columns: {candidates}")
    symbols = [clean_symbol(x) for x in df[col].dropna().tolist()]
    return sorted(set([s for s in symbols if s]))


def parse_manual_symbols(text: str) -> list[str]:
    """Parse comma/newline separated symbols."""
    if not text:
        return []
    raw = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            raw.append(part)
    symbols = [clean_symbol(x) for x in raw]
    return sorted(set([s for s in symbols if s]))


def download_ohlcv(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download single-stock OHLCV data from Yahoo Finance."""
    yf_symbol = to_yahoo_symbol(symbol)
    if not yf_symbol:
        return pd.DataFrame()

    df = yf.download(
        yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["Close"])
    return df


def download_benchmark(benchmark_symbol: str, period: str = "2y", interval: str = "1d") -> pd.Series:
    """Download benchmark close series from Yahoo Finance."""
    df = yf.download(
        benchmark_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)
    close.name = "Benchmark_Close"
    return close.dropna()


def download_universe(symbols: Iterable[str], period: str = "2y", interval: str = "1d", sleep_sec: float = 0.05) -> dict[str, pd.DataFrame]:
    """Download data symbol-by-symbol for better error handling."""
    data = {}
    for symbol in symbols:
        df = download_ohlcv(symbol, period=period, interval=interval)
        if not df.empty:
            data[clean_symbol(symbol)] = df
        time.sleep(sleep_sec)
    return data

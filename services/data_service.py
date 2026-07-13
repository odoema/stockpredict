"""
Data service: downloads OHLCV data from Yahoo Finance for any valid ticker
and caches it on disk so repeated dashboard interactions don't re-hit the
network every time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class TickerNotFoundError(Exception):
    """Raised when Yahoo Finance returns no data for a requested symbol."""


@dataclass
class DataRequest:
    ticker: str
    start: str  # ISO date "YYYY-MM-DD"
    end: str    # ISO date "YYYY-MM-DD"


class DataService:
    """Fetches and caches historical OHLCV data."""

    def __init__(self, cache_dir: Path, cache_ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = cache_ttl_seconds

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        safe_ticker = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
        return self.cache_dir / f"{safe_ticker}_{start}_{end}.parquet"

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl_seconds:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt cache should not crash the app
            logger.warning("Failed to read cache file %s, re-downloading", path)
            return None

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame indexed by date for `ticker`.

        Raises:
            TickerNotFoundError: if Yahoo Finance has no data for the symbol
                or date range.
        """
        ticker = ticker.strip().upper()
        cache_path = self._cache_path(ticker, start, end)

        cached = self._read_cache(cache_path)
        if cached is not None and not cached.empty:
            return cached

        logger.info("Downloading %s from %s to %s", ticker, start, end)
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )

        if df is None or df.empty:
            raise TickerNotFoundError(
                f"No data returned for ticker '{ticker}'. Check the symbol is "
                f"valid on Yahoo Finance and the date range contains trading days."
            )

        df = df.rename(columns=str.title)
        expected_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise TickerNotFoundError(
                f"Data for '{ticker}' is missing expected columns: {sorted(missing)}"
            )

        df.index.name = "Date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        try:
            df.to_parquet(cache_path)
        except Exception:  # noqa: BLE001 - caching is best-effort
            logger.warning("Could not write cache file %s", cache_path)

        return df

    def get_info(self, ticker: str) -> dict:
        """Return basic descriptive info for a ticker (name, currency, etc.)."""
        try:
            info = yf.Ticker(ticker).fast_info
            return {
                "ticker": ticker.upper(),
                "currency": getattr(info, "currency", None),
                "last_price": getattr(info, "last_price", None),
                "market_cap": getattr(info, "market_cap", None),
                "exchange": getattr(info, "exchange", None),
            }
        except Exception:  # noqa: BLE001
            return {"ticker": ticker.upper()}

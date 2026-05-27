"""Data fetching: Yahoo Finance search API + yfinance."""

import requests
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List, Any

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

PERIOD_INTERVAL_MAP: Dict[str, tuple] = {
    "1D":  ("1d",  "5m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1d"),   # daily: ~504 candles, SMA200 computable
    "5Y":  ("5y",  "1wk"),  # weekly: ~260 candles, SMA200 computable (200wk ≈ 4yr)
    "MAX": ("max", "1wk"),  # weekly: more candles than monthly, SMA200 more likely valid
}


def search_tickers(query: str) -> List[Dict]:
    """Search tickers via Yahoo Finance search API."""
    if not query or len(query.strip()) < 1:
        return []
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query.strip(),
            "quotesCount": 10,
            "newsCount": 0,
            "listsCount": 0,
            "enableFuzzyQuery": True,
            "enableEnhancedTrivialQuery": True,
        }
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=6)
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
        allowed = {"EQUITY", "ETF", "CRYPTOCURRENCY", "CURRENCY", "FUTURE", "INDEX", "MUTUALFUND"}
        results = []
        for q in quotes:
            qtype = q.get("quoteType", "")
            if qtype in allowed:
                results.append({
                    "symbol":   q.get("symbol", ""),
                    "name":     q.get("longname") or q.get("shortname") or q.get("symbol", ""),
                    "exchange": q.get("exchange", ""),
                    "type":     qtype,
                })
        return results
    except Exception:
        return []


def get_price_data(ticker: str, period_key: str = "3M") -> Optional[pd.DataFrame]:
    """Fetch OHLCV history for a ticker."""
    period, interval = PERIOD_INTERVAL_MAP.get(period_key, ("3mo", "1d"))
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def get_fundamentals(ticker: str) -> Dict:
    """Return key fundamental fields from yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":        info.get("longName") or info.get("shortName", ticker),
            "sector":      info.get("sector"),
            "industry":    info.get("industry"),
            "country":     info.get("country"),
            "currency":    info.get("currency", "USD"),
            "market_cap":  info.get("marketCap"),
            "pe_ratio":    info.get("trailingPE"),
            "forward_pe":  info.get("forwardPE"),
            "eps":         info.get("trailingEps"),
            "peg_ratio":   info.get("pegRatio"),
            "beta":        info.get("beta"),
            "52w_high":    info.get("fiftyTwoWeekHigh"),
            "52w_low":     info.get("fiftyTwoWeekLow"),
            "avg_volume":  info.get("averageVolume"),
            "description": info.get("longBusinessSummary", ""),
            "website":     info.get("website", ""),
            "employees":   info.get("fullTimeEmployees"),
            "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct":  info.get("regularMarketChangePercent"),
        }
    except Exception:
        return {}


def get_dividends(ticker: str) -> Dict:
    """Return dividend summary + payment history."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.dividends
        ex_date = info.get("exDividendDate")
        if ex_date:
            try:
                ex_date = pd.to_datetime(ex_date, unit="s").strftime("%Y-%m-%d")
            except Exception:
                ex_date = str(ex_date)
        return {
            "yield":        info.get("dividendYield"),
            "rate":         info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),
            "ex_date":      ex_date,
            "history":      hist.tail(24) if not hist.empty else pd.Series(dtype=float),
        }
    except Exception:
        return {}


def get_financials(ticker: str) -> Dict:
    """Return income statement and balance sheet."""
    try:
        t = yf.Ticker(ticker)
        return {
            "income":  t.financials,
            "balance": t.balance_sheet,
        }
    except Exception:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame()}


def get_analyst_data(ticker: str) -> Dict:
    """Return analyst price targets + upgrades/downgrades history."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist_df = pd.DataFrame()
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                hist_df = ud.sort_index(ascending=False).head(25)
        except Exception:
            pass
        return {
            "target_mean":    info.get("targetMeanPrice"),
            "target_high":    info.get("targetHighPrice"),
            "target_low":     info.get("targetLowPrice"),
            "recommendation": (info.get("recommendationKey") or "n/a").upper(),
            "num_analysts":   info.get("numberOfAnalystOpinions"),
            "history":        hist_df,
        }
    except Exception:
        return {}

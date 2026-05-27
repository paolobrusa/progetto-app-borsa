"""Technical indicators and support/resistance detection — pure pandas/numpy (no external TA lib)."""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List


# ── Indicators ──────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_g = gain.ewm(com=n - 1, min_periods=n).mean()
    avg_l = loss.ewm(com=n - 1, min_periods=n).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    ema_f = _ema(s, fast)
    ema_s = _ema(s, slow)
    macd = ema_f - ema_s
    signal = _ema(macd, sig)
    return macd, signal, macd - signal

def _bollinger(s: pd.Series, n=20, k=2):
    ma  = _sma(s, n)
    std = s.rolling(n, min_periods=n).std()
    return ma + k * std, ma, ma - k * std

def _stochastic(df: pd.DataFrame, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d

def _atr(df: pd.DataFrame, n=14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, min_periods=n).mean()

def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0)
    return (sign * df["Volume"]).cumsum()


# ── Support / Resistance ─────────────────────────────────────────────────────

def find_support_resistance(
    df: pd.DataFrame,
    order: int = None,
    tolerance: float = 0.015,
    min_touches: int = 2,
    max_levels: int = 8,
) -> Dict[str, List[float]]:
    """Detect S/R via fractal extremes + level clustering."""
    n = len(df)
    if n < 20:
        return {"resistance": [], "support": []}

    # Fixed order: if not provided use a modest constant so different data lengths
    # (1Y daily vs 2Y weekly) don't produce completely different S/R levels.
    order = order or 10
    highs = df["High"].values
    lows  = df["Low"].values

    try:
        res_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        sup_idx = argrelextrema(lows,  np.less_equal,    order=order)[0]
    except Exception:
        return {"resistance": [], "support": []}

    def cluster(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        groups: List[List[float]] = [[levels[0]]]
        for lvl in levels[1:]:
            ref = np.mean(groups[-1])
            if abs(lvl - ref) / ref < tolerance:
                groups[-1].append(lvl)
            else:
                groups.append([lvl])
        return [float(np.mean(g)) for g in groups if len(g) >= min_touches]

    current = float(df["Close"].iloc[-1])

    def nearby(levels: List[float]) -> List[float]:
        return sorted(levels, key=lambda x: abs(x - current))[:max_levels]

    return {
        "resistance": nearby(cluster([float(highs[i]) for i in res_idx])),
        "support":    nearby(cluster([float(lows[i])  for i in sup_idx])),
    }


# ── Enrich dataframe ─────────────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to the OHLCV dataframe."""
    c = df["Close"]

    # Moving averages
    df["SMA20"]  = _sma(c, 20)
    df["SMA50"]  = _sma(c, 50)
    df["SMA200"] = _sma(c, 200)
    df["EMA20"]  = _ema(c, 20)
    df["EMA9"]   = _ema(c, 9)

    # Bollinger Bands (20, 2σ)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = _bollinger(c, 20, 2)

    # Momentum
    df["RSI"] = _rsi(c, 14)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = _macd(c)

    # Stochastic %K/%D
    df["STOCH_K"], df["STOCH_D"] = _stochastic(df, 14, 3)

    # Volatility
    df["ATR"] = _atr(df, 14)

    # Volume trend
    df["OBV"] = _obv(df)

    return df

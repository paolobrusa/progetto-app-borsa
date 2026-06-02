"""Technical indicators and support/resistance detection — pandas-ta."""

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema
from typing import Dict, List


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
    """Add all technical indicator columns to the OHLCV dataframe using pandas-ta."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    df["SMA20"]  = ta.sma(c, length=20)
    df["SMA50"]  = ta.sma(c, length=50)
    df["SMA200"] = ta.sma(c, length=200)
    df["EMA20"]  = ta.ema(c, length=20)
    df["EMA9"]   = ta.ema(c, length=9)

    # Bollinger Bands (20, 2σ)
    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        df["BB_upper"] = bb["BBU_20_2.0_2.0"]
        df["BB_mid"]   = bb["BBM_20_2.0_2.0"]
        df["BB_lower"] = bb["BBL_20_2.0_2.0"]

    # RSI
    df["RSI"] = ta.rsi(c, length=14)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"]      = macd["MACD_12_26_9"]
        df["MACD_sig"]  = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]

    # Stochastic %K/%D
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df["STOCH_K"] = stoch["STOCHk_14_3_3"]
        df["STOCH_D"] = stoch["STOCHd_14_3_3"]

    # ATR
    df["ATR"] = ta.atr(h, l, c, length=14)

    # OBV
    df["OBV"] = ta.obv(c, v)

    # ADX — trend strength + directional indicators
    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        df["ADX"]      = adx.get("ADX_14")
        df["DI_plus"]  = adx.get("DMP_14")
        df["DI_minus"] = adx.get("DMN_14")

    # CMF — Chaikin Money Flow (institutional buying pressure)
    cmf = ta.cmf(h, l, c, v, length=20)
    if cmf is not None:
        df["CMF"] = cmf

    return df

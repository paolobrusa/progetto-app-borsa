"""Rule-based analysis engine: reads indicators + fundamentals, returns structured verdict."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ── Signal helpers ────────────────────────────────────────────────────────────
# Each signal: (label, sentiment, detail)
# sentiment: "bullish" | "mild_bullish" | "neutral" | "mild_bearish" | "bearish"

def _tech_signals(df: pd.DataFrame, sr: dict, close: float) -> Dict[str, Tuple]:
    sigs = {}

    # RSI
    if "RSI" in df:
        rsi = df["RSI"].dropna()
        if not rsi.empty:
            v = float(rsi.iloc[-1])
            if v < 25:
                sigs["RSI"] = ("IPERVENDUTO", "bullish", f"RSI {v:.1f} — zona di ipervenduto, alta probabilità di rimbalzo")
            elif v < 35:
                sigs["RSI"] = ("DEBOLEZZA ESTREMA", "mild_bullish", f"RSI {v:.1f} — sotto 35, pressione di vendita in esaurimento")
            elif v > 75:
                sigs["RSI"] = ("IPERCOMPRATO", "bearish", f"RSI {v:.1f} — zona di ipercomprato, rischio di correzione")
            elif v > 60:
                sigs["RSI"] = ("FORZA", "mild_bullish", f"RSI {v:.1f} — momentum positivo")
            elif v < 45:
                sigs["RSI"] = ("DEBOLEZZA", "mild_bearish", f"RSI {v:.1f} — momentum debole")
            else:
                sigs["RSI"] = ("NEUTRALE", "neutral", f"RSI {v:.1f} — zona neutra")

    # MACD
    if "MACD" in df and "MACD_sig" in df and "MACD_hist" in df:
        macd_v = df["MACD"].dropna()
        sig_v  = df["MACD_sig"].dropna()
        hist_v = df["MACD_hist"].dropna()
        if not macd_v.empty:
            m = float(macd_v.iloc[-1])
            s = float(sig_v.iloc[-1])
            h = float(hist_v.iloc[-1])
            prev_h = float(hist_v.iloc[-2]) if len(hist_v) >= 2 else h
            if m > s and h > 0:
                cross = "incrocio rialzista recente — " if prev_h <= 0 else ""
                sigs["MACD"] = ("RIALZISTA", "bullish", f"MACD sopra signal — {cross}momentum positivo")
            elif m < s and h < 0:
                cross = "incrocio ribassista recente — " if prev_h >= 0 else ""
                sigs["MACD"] = ("RIBASSISTA", "bearish", f"MACD sotto signal — {cross}momentum negativo")
            else:
                sigs["MACD"] = ("NEUTRALE", "neutral", "MACD in zona di transizione")

    # SMA 200 (trend primario)
    if "SMA200" in df:
        sma200 = df["SMA200"].dropna()
        if not sma200.empty:
            v = float(sma200.iloc[-1])
            pct = (close - v) / v * 100
            if close > v:
                sigs["SMA 200"] = ("SOPRA MEDIA", "bullish", f"Prezzo {pct:+.1f}% sopra SMA200 ({v:.2f}) — trend primario rialzista")
            else:
                sigs["SMA 200"] = ("SOTTO MEDIA", "bearish", f"Prezzo {pct:+.1f}% sotto SMA200 ({v:.2f}) — trend primario ribassista")

    # Golden / Death Cross (SMA50 vs SMA200)
    if "SMA50" in df and "SMA200" in df:
        s50 = df["SMA50"].dropna()
        s200 = df["SMA200"].dropna()
        if not s50.empty and not s200.empty:
            v50  = float(s50.iloc[-1])
            v200 = float(s200.iloc[-1])
            if v50 > v200:
                sigs["Golden/Death Cross"] = ("GOLDEN CROSS", "bullish", f"SMA50 ({v50:.2f}) sopra SMA200 ({v200:.2f}) — segnale rialzista di medio periodo")
            else:
                sigs["Golden/Death Cross"] = ("DEATH CROSS", "bearish", f"SMA50 ({v50:.2f}) sotto SMA200 ({v200:.2f}) — segnale ribassista di medio periodo")

    # Bollinger Bands
    if "BB_upper" in df and "BB_lower" in df:
        bbu = df["BB_upper"].dropna()
        bbl = df["BB_lower"].dropna()
        bbm = df["BB_mid"].dropna()
        if not bbu.empty:
            u = float(bbu.iloc[-1])
            l = float(bbl.iloc[-1])
            rng = u - l
            if rng > 0:
                pos = (close - l) / rng
                width_pct = rng / float(bbm.iloc[-1]) * 100 if not bbm.empty else 0
                squeeze = " (bande strette — breakout imminente)" if width_pct < 5 else ""
                if pos < 0.15:
                    sigs["Bollinger"] = ("SOTTO BANDA INF.", "bullish", f"Prezzo vicino alla banda inferiore ({l:.2f}){squeeze} — possibile rimbalzo")
                elif pos > 0.85:
                    sigs["Bollinger"] = ("SOPRA BANDA SUP.", "bearish", f"Prezzo vicino alla banda superiore ({u:.2f}){squeeze} — possibile ritracciamento")
                elif 0.4 < pos < 0.6:
                    sigs["Bollinger"] = ("CENTRO BANDE", "neutral", f"Prezzo nella zona centrale{squeeze}")
                elif pos < 0.4:
                    sigs["Bollinger"] = ("LIEVE DEBOLEZZA", "mild_bearish", f"Prezzo nella metà inferiore delle bande{squeeze}")
                else:
                    sigs["Bollinger"] = ("LIEVE FORZA", "mild_bullish", f"Prezzo nella metà superiore delle bande{squeeze}")

    # Stochastic
    if "STOCH_K" in df and "STOCH_D" in df:
        sk = df["STOCH_K"].dropna()
        sd = df["STOCH_D"].dropna()
        if not sk.empty:
            kv = float(sk.iloc[-1])
            dv = float(sd.iloc[-1]) if not sd.empty else kv
            if kv < 20 and kv > dv:
                sigs["Stocastico"] = ("IPERVENDUTO ↑", "bullish", f"%K={kv:.1f} in zona ipervenduto con incrocio rialzista")
            elif kv < 20:
                sigs["Stocastico"] = ("IPERVENDUTO", "mild_bullish", f"%K={kv:.1f} — zona di ipervenduto")
            elif kv > 80 and kv < dv:
                sigs["Stocastico"] = ("IPERCOMPRATO ↓", "bearish", f"%K={kv:.1f} in zona ipercomprato con incrocio ribassista")
            elif kv > 80:
                sigs["Stocastico"] = ("IPERCOMPRATO", "mild_bearish", f"%K={kv:.1f} — zona di ipercomprato")
            else:
                sigs["Stocastico"] = ("NEUTRALE", "neutral", f"%K={kv:.1f}")

    # ATR (volatilità)
    if "ATR" in df:
        atr = df["ATR"].dropna()
        if not atr.empty and close > 0:
            atr_pct = float(atr.iloc[-1]) / close * 100
            if atr_pct > 5:
                sigs["Volatilità (ATR)"] = ("ALTA", "mild_bearish", f"ATR = {atr_pct:.1f}% del prezzo — alta volatilità, rischio elevato")
            elif atr_pct < 1.5:
                sigs["Volatilità (ATR)"] = ("BASSA", "mild_bullish", f"ATR = {atr_pct:.1f}% del prezzo — bassa volatilità, mercato tranquillo")
            else:
                sigs["Volatilità (ATR)"] = ("NORMALE", "neutral", f"ATR = {atr_pct:.1f}% del prezzo — volatilità nella norma")

    # OBV trend
    if "OBV" in df:
        obv = df["OBV"].dropna()
        if len(obv) >= 20:
            obv_start = float(obv.iloc[-20])
            obv_end   = float(obv.iloc[-1])
            if obv_end > obv_start * 1.05:
                sigs["OBV (volume)"] = ("ACCUMULAZIONE", "bullish", "OBV in crescita — il volume supporta il trend rialzista")
            elif obv_end < obv_start * 0.95:
                sigs["OBV (volume)"] = ("DISTRIBUZIONE", "bearish", "OBV in calo — il volume segnala uscita degli acquirenti")
            else:
                sigs["OBV (volume)"] = ("NEUTRALE", "neutral", "OBV stabile — nessuna pressione di volume direzionale")

    # CMF — Chaikin Money Flow
    if "CMF" in df:
        cmf_s = df["CMF"].dropna()
        if not cmf_s.empty:
            cmf_v = float(cmf_s.iloc[-1])
            if cmf_v > 0.15:
                sigs["CMF (flusso)"] = ("FORTE ACCUMULO", "bullish", f"CMF={cmf_v:.3f} — forte pressione d'acquisto istituzionale")
            elif cmf_v > 0.05:
                sigs["CMF (flusso)"] = ("ACCUMULO", "mild_bullish", f"CMF={cmf_v:.3f} — pressione di acquisto presente")
            elif cmf_v < -0.15:
                sigs["CMF (flusso)"] = ("FORTE DISTRIBUZIONE", "bearish", f"CMF={cmf_v:.3f} — forte pressione di vendita istituzionale")
            elif cmf_v < -0.05:
                sigs["CMF (flusso)"] = ("DISTRIBUZIONE", "mild_bearish", f"CMF={cmf_v:.3f} — pressione di vendita presente")
            else:
                sigs["CMF (flusso)"] = ("NEUTRALE", "neutral", f"CMF={cmf_v:.3f} — flusso di denaro bilanciato")

    # ADX — trend strength
    if "ADX" in df and "DI_plus" in df and "DI_minus" in df:
        adx_s = df["ADX"].dropna()
        dip_s = df["DI_plus"].dropna()
        dim_s = df["DI_minus"].dropna()
        if not adx_s.empty and not dip_s.empty and not dim_s.empty:
            adx_v = float(adx_s.iloc[-1])
            dip_v = float(dip_s.iloc[-1])
            dim_v = float(dim_s.iloc[-1])
            if adx_v > 25 and dip_v > dim_v:
                sigs["ADX (trend)"] = ("TREND FORTE ↑", "bullish", f"ADX={adx_v:.1f} DI+={dip_v:.1f}>DI-={dim_v:.1f} — trend rialzista solido")
            elif adx_v > 25 and dim_v > dip_v:
                sigs["ADX (trend)"] = ("TREND FORTE ↓", "bearish", f"ADX={adx_v:.1f} DI-={dim_v:.1f}>DI+={dip_v:.1f} — trend ribassista solido")
            elif adx_v > 18:
                direction = "mild_bullish" if dip_v > dim_v else "mild_bearish"
                sigs["ADX (trend)"] = ("TREND MODERATO", direction, f"ADX={adx_v:.1f} — trend presente, direzione {'rialzista' if dip_v > dim_v else 'ribassista'}")
            else:
                sigs["ADX (trend)"] = ("MERCATO LATERALE", "neutral", f"ADX={adx_v:.1f} — nessun trend definito, indicatori oscillatori meno affidabili")

    # EMA trend stacking (EMA9 > EMA20 > SMA50 = perfetto allineamento rialzista)
    if "EMA9" in df and "EMA20" in df and "SMA50" in df:
        e9_s  = df["EMA9"].dropna()
        e20_s = df["EMA20"].dropna()
        s50_s = df["SMA50"].dropna()
        if not e9_s.empty and not e20_s.empty and not s50_s.empty:
            v9  = float(e9_s.iloc[-1])
            v20 = float(e20_s.iloc[-1])
            v50 = float(s50_s.iloc[-1])
            if close > v9 > v20 > v50:
                sigs["Trend EMA"] = ("ALLINEAMENTO ↑", "bullish", f"Prezzo>{v9:.2f}>{v20:.2f}>{v50:.2f} — allineamento rialzista perfetto")
            elif close > v20 and close > v50:
                sigs["Trend EMA"] = ("TREND POSITIVO", "mild_bullish", f"Prezzo sopra EMA20({v20:.2f}) e SMA50({v50:.2f})")
            elif close < v9 < v20 < v50:
                sigs["Trend EMA"] = ("ALLINEAMENTO ↓", "bearish", f"Prezzo<{v9:.2f}<{v20:.2f}<{v50:.2f} — allineamento ribassista")
            elif close < v20 and close < v50:
                sigs["Trend EMA"] = ("TREND NEGATIVO", "mild_bearish", f"Prezzo sotto EMA20({v20:.2f}) e SMA50({v50:.2f})")
            else:
                sigs["Trend EMA"] = ("MISTO", "neutral", f"EMA non allineate — mercato indeciso ({v9:.2f}/{v20:.2f}/{v50:.2f})")

    # Support / Resistance proximity
    if sr:
        resis = [r for r in sr.get("resistance", []) if r > close]
        supps  = [s for s in sr.get("support", [])    if s < close]
        near_r = min(resis, key=lambda x: x - close, default=None)
        near_s = max(supps, key=lambda x: x,          default=None)
        if near_r and near_s:
            d_r = (near_r - close) / close * 100
            d_s = (close - near_s) / close * 100
            if d_r < 2:
                sigs["Sup/Res"] = ("RESISTENZA VICINA", "bearish", f"Resistenza a {near_r:.2f} (+{d_r:.1f}%) — possibile ostacolo rialzista")
            elif d_s < 2:
                sigs["Sup/Res"] = ("SUPPORTO VICINO", "bullish", f"Supporto a {near_s:.2f} (-{d_s:.1f}%) — possibile zona di rimbalzo")
            else:
                sigs["Sup/Res"] = ("IN AREA LIBERA", "neutral",
                    f"Supporto: {near_s:.2f} (-{d_s:.1f}%)  |  Resistenza: {near_r:.2f} (+{d_r:.1f}%)")

    return sigs


def _fund_signals(fundamentals: dict, analysts: dict, close: float) -> Dict[str, Tuple]:
    sigs = {}

    # Posizione nel range 52 settimane (momentum di prezzo)
    w52h = fundamentals.get("52w_high")
    w52l = fundamentals.get("52w_low")
    if w52h and w52l and close and (w52h - w52l) > 0:
        pos_52 = (close - w52l) / (w52h - w52l) * 100
        if pos_52 >= 80:
            sigs["Posizione 52W"] = ("VICINO AI MASSIMI", "bullish", f"Prezzo all'{pos_52:.0f}% del range annuale — forte momentum")
        elif pos_52 >= 55:
            sigs["Posizione 52W"] = ("ZONA ALTA", "mild_bullish", f"Prezzo al {pos_52:.0f}% del range annuale — trend positivo")
        elif pos_52 <= 20:
            sigs["Posizione 52W"] = ("ZONA MINIMI", "mild_bearish", f"Prezzo al {pos_52:.0f}% del range annuale — debolezza strutturale")
        else:
            sigs["Posizione 52W"] = ("ZONA MEDIA", "neutral", f"Prezzo al {pos_52:.0f}% del range annuale")

    # P/E — soglie aggiornate: S&P500 medio attuale = 27, storico 20 anni = 23
    pe = fundamentals.get("pe_ratio")
    if pe and pe > 0:
        if pe < 13:
            sigs["P/E Ratio"] = ("MOLTO ECONOMICO", "bullish", f"P/E = {pe:.1f} — titolo sottovalutato rispetto al mercato")
        elif pe < 22:
            sigs["P/E Ratio"] = ("FAIR VALUE", "mild_bullish", f"P/E = {pe:.1f} — valutazione sotto la media di mercato (S&P ~27)")
        elif pe < 35:
            sigs["P/E Ratio"] = ("NELLA NORMA", "neutral", f"P/E = {pe:.1f} — in linea con la media di mercato attuale")
        elif pe < 50:
            sigs["P/E Ratio"] = ("ELEVATO", "mild_bearish", f"P/E = {pe:.1f} — sopra la media, richiede crescita sostenuta")
        else:
            sigs["P/E Ratio"] = ("MOLTO ELEVATO", "bearish", f"P/E = {pe:.1f} — valutazione da titolo growth aggressivo")

    # Forward P/E — media S&P500 attuale ~22-24, storica ~17-18
    fpe = fundamentals.get("forward_pe")
    if fpe and fpe > 0:
        if fpe < 15:
            sigs["P/E Forward"] = ("MOLTO ECONOMICO", "bullish", f"Forward P/E = {fpe:.1f} — utili futuri molto convenienti vs mercato")
        elif fpe < 22:
            sigs["P/E Forward"] = ("ECONOMICO", "mild_bullish", f"Forward P/E = {fpe:.1f} — sotto la media di mercato (~22-24)")
        elif fpe < 30:
            sigs["P/E Forward"] = ("EQUO", "neutral", f"Forward P/E = {fpe:.1f} — in linea con la media di mercato")
        else:
            sigs["P/E Forward"] = ("CARO", "mild_bearish", f"Forward P/E = {fpe:.1f} — crescita futura già prezzata in anticipo")

    # PEG — soglie più realistiche: PEG < 1 = ottimo, 1–2 = normale per growth
    peg = fundamentals.get("peg_ratio")
    if peg and peg > 0:
        if peg < 1.0:
            sigs["PEG Ratio"] = ("SOTTOVALUTATO", "bullish", f"PEG = {peg:.2f} — crescita non ancora prezzata, opportunità")
        elif peg < 2.0:
            sigs["PEG Ratio"] = ("EQUO", "neutral", f"PEG = {peg:.2f} — crescita correttamente prezzata")
        elif peg < 3.0:
            sigs["PEG Ratio"] = ("SOPRAVVALUTATO", "mild_bearish", f"PEG = {peg:.2f} — crescita eccessivamente prezzata")
        else:
            sigs["PEG Ratio"] = ("MOLTO CARO", "bearish", f"PEG = {peg:.2f} — valutazione molto aggressiva")

    # Beta — indicatore di rischio, non direzionale (non deve abbassare il punteggio buy/sell)
    beta = fundamentals.get("beta")
    if beta:
        if beta < 0.6:
            sigs["Beta / Rischio"] = ("MOLTO DIFENSIVO", "mild_bullish", f"Beta = {beta:.2f} — titolo difensivo, bassa correlazione col mercato")
        elif beta < 1.2:
            sigs["Beta / Rischio"] = ("MODERATO", "neutral", f"Beta = {beta:.2f} — volatilità in linea con il mercato")
        elif beta < 2.0:
            sigs["Beta / Rischio"] = ("AGGRESSIVO", "neutral", f"Beta = {beta:.2f} — più volatile del mercato (normale per titoli growth/tech)")
        else:
            sigs["Beta / Rischio"] = ("MOLTO VOLATILE", "mild_bearish", f"Beta = {beta:.2f} — elevata volatilità, gestione rischio essenziale")

    # Analyst consensus + target upside
    if analysts:
        rec = (analysts.get("recommendation") or "N/A").upper()
        target = analysts.get("target_mean")
        n_an   = analysts.get("num_analysts", 0)
        label_rec = rec.replace("_", " ")
        detail_extra = f" (su {n_an} analisti)" if n_an else ""
        if target and close:
            upside = (target - close) / close * 100
            detail_extra += f" — Target medio: {target:.2f} ({upside:+.1f}%)"

        if rec in ("BUY", "STRONG_BUY", "OUTPERFORM", "OVERWEIGHT"):
            sigs["Consenso Analisti"] = ("ACQUISTO", "bullish", f"{label_rec}{detail_extra}")
        elif rec in ("SELL", "STRONG_SELL", "UNDERPERFORM", "UNDERWEIGHT"):
            sigs["Consenso Analisti"] = ("VENDITA", "bearish", f"{label_rec}{detail_extra}")
        elif rec in ("HOLD", "NEUTRAL", "MARKET_PERFORM", "EQUAL_WEIGHT"):
            sigs["Consenso Analisti"] = ("NEUTRALE", "neutral", f"{label_rec}{detail_extra}")

        # Upside/downside
        if target and close:
            upside = (target - close) / close * 100
            if upside > 20:
                sigs["Upside vs Target"] = ("ALTO POTENZIALE", "bullish", f"Upside verso target: +{upside:.1f}%")
            elif upside > 5:
                sigs["Upside vs Target"] = ("POTENZIALE MODERATO", "mild_bullish", f"Upside verso target: +{upside:.1f}%")
            elif upside < -10:
                sigs["Upside vs Target"] = ("DOWNSIDE", "bearish", f"Downside verso target: {upside:.1f}%")
            else:
                sigs["Upside vs Target"] = ("LIMITATO", "neutral", f"Target vicino al prezzo attuale: {upside:+.1f}%")

    return sigs


# Per-signal weight multipliers per horizon (missing key = 1.0)
_HORIZON_W: Dict[str, Dict[str, float]] = {
    "breve": {
        # Short-term signals amplified
        "RSI": 2.5, "Stocastico": 2.5, "Bollinger": 2.0,
        "MACD": 1.8, "OBV (volume)": 1.5, "Sup/Res": 2.5, "Volatilità (ATR)": 1.0,
        "ADX (trend)": 2.0, "Trend EMA": 1.5, "CMF (flusso)": 1.5,
        # Long-term signals muted
        "SMA 200": 0.3, "Golden/Death Cross": 0.2,
        # Fundamentals quasi irrilevanti sul breve
        "P/E Ratio": 0.1, "P/E Forward": 0.1, "PEG Ratio": 0.1,
        "Beta / Rischio": 0.1, "Consenso Analisti": 0.3, "Upside vs Target": 0.1,
        "Posizione 52W": 0.8,
    },
    "medio": {
        # Bilanciato con un po' di enfasi sui segnali di trend
        "MACD": 1.3, "SMA 200": 1.3, "Golden/Death Cross": 1.3,
        "ADX (trend)": 1.4, "Trend EMA": 1.3, "CMF (flusso)": 1.2,
        "RSI": 1.1, "OBV (volume)": 1.2,
        # Fondamentali a peso ridotto nel medio
        "P/E Ratio": 0.7, "P/E Forward": 0.8, "PEG Ratio": 0.7,
        "Beta / Rischio": 0.3, "Consenso Analisti": 1.5, "Upside vs Target": 1.3,
        "Posizione 52W": 1.0,
    },
    "lungo": {
        # Short-term signals molto ridotti
        "RSI": 0.2, "Stocastico": 0.1, "Bollinger": 0.3,
        "Sup/Res": 0.2, "Volatilità (ATR)": 0.1,
        # Trend signals
        "MACD": 0.7, "OBV (volume)": 0.8,
        "ADX (trend)": 1.5, "Trend EMA": 1.2, "CMF (flusso)": 1.0,
        # Long-term signals amplified
        "SMA 200": 2.0, "Golden/Death Cross": 2.0,
        # Fondamentali importanti ma senza esagerare
        "P/E Ratio": 1.2, "P/E Forward": 1.0, "PEG Ratio": 1.5,
        "Beta / Rischio": 0.3, "Consenso Analisti": 2.0, "Upside vs Target": 1.8,
        "Posizione 52W": 1.2,
    },
}

_HORIZON_LABEL: Dict[str, str] = {
    "breve": "Breve termine  (1–4 settimane)",
    "medio": "Medio termine  (1–3 mesi)",
    "lungo": "Lungo termine  (6–18 mesi)",
}

_SENTIMENT_W = {
    "bullish": +18, "mild_bullish": +9,
    "neutral": 0,
    "mild_bearish": -9, "bearish": -18,
}


def _score_from_signals(tech: dict, fund: dict, horizon: str) -> int:
    hw = _HORIZON_W.get(horizon, {})

    def weighted(signals: dict, base_mult: float) -> float:
        total = 0.0
        for name, tup in signals.items():
            sentiment_pts = _SENTIMENT_W.get(tup[1], 0)
            mult = hw.get(name, 1.0) * base_mult
            total += sentiment_pts * mult
        return total

    score = weighted(tech, 1.0) + weighted(fund, 0.7)
    return int(max(-100, min(100, score)))


def _strategies(score: int, df: pd.DataFrame, sr: dict, close: float, horizon: str) -> List[dict]:
    atr_series = df["ATR"].dropna() if "ATR" in df else pd.Series(dtype=float)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else close * 0.02

    resis = sorted([r for r in sr.get("resistance", []) if r > close]) if sr else []
    supps = sorted([s for s in sr.get("support", []) if s < close], reverse=True) if sr else []
    near_r = resis[0] if resis else close * 1.05
    near_s = supps[0] if supps else close * 0.95

    strats: List[dict] = []

    # ── BREVE (1-4 settimane) ────────────────────────────────────────────────
    if horizon == "breve":
        if score >= 45:
            sl  = max(near_s, close - 1.5 * atr)
            tp1 = close + 1.5 * atr
            tp2 = near_r
            strats += [
                {"nome": "Swing Trade Rialzista",
                 "rischio": "MEDIO-ALTO",
                 "desc": (f"Entrata rapida a mercato. Stop loss stretto: <b>{sl:.2f}</b> "
                          f"(-{(close-sl)/close*100:.1f}%). "
                          f"Primo target: <b>{tp1:.2f}</b> (+{(tp1-close)/close*100:.1f}%), "
                          f"secondo target: <b>{tp2:.2f}</b>. "
                          f"Orizzonte: 1–2 settimane. Monitorare ogni giorno.")},
                {"nome": "Breakout su Resistenza",
                 "rischio": "MEDIO",
                 "desc": (f"Attendere la rottura confermata sopra <b>{near_r:.2f}</b> "
                          f"con volume elevato. Entrare solo al breakout, stop a {near_r * 0.99:.2f}. "
                          f"Strategia ideale per catturare impulsi di breve durata (3–10 giorni).")},
            ]
        elif score >= 20:
            sl = max(near_s, close - atr)
            strats += [
                {"nome": "Swing Trade Moderato",
                 "rischio": "MEDIO",
                 "desc": (f"Entrata con il 50% della posizione. "
                          f"Stop loss: <b>{sl:.2f}</b> (-{(close-sl)/close*100:.1f}%). "
                          f"Target: <b>{near_r:.2f}</b>. "
                          f"Orizzonte: 1–3 settimane. Evitare di mantenere oltre.")},
            ]
        elif score >= -20:
            strats += [
                {"nome": "Flat — Nessuna Posizione",
                 "rischio": "BASSO",
                 "desc": (f"Segnali di breve termine contrastanti. "
                          f"Stare fuori dal mercato. Rientrare solo su breakout di "
                          f"<b>{near_r:.2f}</b> o rimbalzo confermato da <b>{near_s:.2f}</b>. "
                          f"Orizzonte di attesa: pochi giorni.")},
            ]
        else:
            sl = min(near_r, close + atr)
            strats += [
                {"nome": "Evitare Posizioni Long",
                 "rischio": "ALTO se si compra",
                 "desc": (f"Segnali tecnici di breve negativi. "
                          f"Chi è già in posizione abbassi lo stop loss a <b>{near_s:.2f}</b>. "
                          f"Attendere stabilizzazione prima di rientrare.")},
            ]

    # ── MEDIO (1-3 mesi) ─────────────────────────────────────────────────────
    elif horizon == "medio":
        if score >= 45:
            sl  = max(near_s, close - 2 * atr)
            tp1 = near_r
            tp2 = close + 4 * atr
            strats += [
                {"nome": "Posizione Long — Momentum",
                 "rischio": "MEDIO-ALTO",
                 "desc": (f"Entrata a mercato. Stop loss: <b>{sl:.2f}</b> "
                          f"(-{(close-sl)/close*100:.1f}%). "
                          f"Target 1: <b>{tp1:.2f}</b> (+{(tp1-close)/close*100:.1f}%), "
                          f"Target 2: <b>{tp2:.2f}</b> (+{(tp2-close)/close*100:.1f}%). "
                          f"Orizzonte: 1–3 mesi. Rivalutare mensalmente.")},
                {"nome": "DCA Parziale",
                 "rischio": "BASSO-MEDIO",
                 "desc": (f"Acquisto in 3 tranche mensili: 40% subito, "
                          f"30% verso {close*0.97:.2f}, 30% su conferma sopra {near_r:.2f}. "
                          f"Riduce il rischio da timing su un orizzonte trimestrale.")},
            ]
        elif score >= 20:
            sl  = max(near_s, close - 1.5 * atr)
            strats += [
                {"nome": "Long Moderato",
                 "rischio": "MEDIO",
                 "desc": (f"Entrata con 50–60% della posizione. "
                          f"Stop loss: <b>{sl:.2f}</b>. Target: <b>{near_r:.2f}</b> "
                          f"(+{(near_r-close)/close*100:.1f}%). "
                          f"Aumentare solo su conferma del trend. Orizzonte: 4–8 settimane.")},
                {"nome": "DCA Mensile",
                 "rischio": "BASSO",
                 "desc": ("Acquisti mensili fissi indipendentemente dal prezzo. "
                          "Abbassa il prezzo medio nel tempo. "
                          "Adatto a chi vuole costruire una posizione gradualmente "
                          "nel corso del trimestre.")},
            ]
        elif score >= -20:
            strats += [
                {"nome": "Attendere Conferma Direzionale",
                 "rischio": "BASSO",
                 "desc": (f"Segnali misti: attendere rottura di <b>{near_r:.2f}</b> (long) "
                          f"o sfondamento di <b>{near_s:.2f}</b> (short). "
                          f"Su un orizzonte mensile è meglio pazientare 1–2 settimane "
                          f"prima di costruire una posizione.")},
                {"nome": "Swing Trading in Range",
                 "rischio": "MEDIO",
                 "desc": (f"Comprare a supporto <b>{near_s:.2f}</b>, stop a {near_s*0.98:.2f}, "
                          f"vendere a resistenza <b>{near_r:.2f}</b>. "
                          f"Valido finché il range regge (orizzonte 2–6 settimane).")},
            ]
        elif score >= -45:
            strats += [
                {"nome": "Ridurre Esposizione",
                 "rischio": "ALTO se si mantiene",
                 "desc": (f"Alleggerire o chiudere. Stop loss a <b>{near_s:.2f}</b>. "
                          f"Evitare nuovi acquisti per almeno 4–6 settimane.")},
                {"nome": "Hedging di Portafoglio",
                 "rischio": "MEDIO",
                 "desc": ("Valutare opzioni put o ETF short sul settore "
                          "per proteggere l'esposizione nel corso del trimestre.")},
            ]
        else:
            strats += [
                {"nome": "Uscita dalla Posizione",
                 "rischio": "MOLTO ALTO se si rimane",
                 "desc": ("Liquidare le posizioni long. "
                          "Non aprire nuovi acquisti. "
                          "Rientrare solo dopo inversione confermata del trend primario.")},
            ]

    # ── LUNGO (6-18 mesi) ────────────────────────────────────────────────────
    else:
        if score >= 45:
            strats += [
                {"nome": "Buy & Hold",
                 "rischio": "BASSO-MEDIO",
                 "desc": (f"Acquisto della posizione completa e mantenimento per 6–18 mesi. "
                          f"Ignorare le oscillazioni di breve termine. "
                          f"Stop loss ampio o assente (approccio fondamentale). "
                          f"Rivalutare trimestralmente i dati di bilancio.")},
                {"nome": "DCA su 12 mesi",
                 "rischio": "BASSO",
                 "desc": ("Acquisti mensili costanti per 12 mesi indipendentemente dal prezzo. "
                          "Strategia ottimale per chi investe con ottica annuale: "
                          "azzera il problema del timing e sfrutta la volatilità a proprio favore.")},
            ]
        elif score >= 20:
            strats += [
                {"nome": "Accumulo Progressivo",
                 "rischio": "BASSO-MEDIO",
                 "desc": (f"Costruire la posizione gradualmente in 6 mesi. "
                          f"I fondamentali supportano l'investimento ma aspettare "
                          f"conferma tecnica prima di aumentare l'esposizione. "
                          f"Orizzonte di ritorno: 9–18 mesi.")},
                {"nome": "DCA Trimestrale",
                 "rischio": "BASSO",
                 "desc": ("Acquisti ogni trimestre per 12–18 mesi. "
                          "Ideale per costruire una posizione di lungo periodo "
                          "su un titolo fondamentalmente solido con valutazione equa.")},
            ]
        elif score >= -20:
            strats += [
                {"nome": "Attendere — Fondamentali Incerti",
                 "rischio": "BASSO",
                 "desc": (f"Su un orizzonte lungo i segnali fondamentali non sono abbastanza chiari. "
                          f"Attendere il prossimo trimestrale o un miglioramento del quadro macro "
                          f"prima di costruire una posizione di lungo periodo.")},
            ]
        else:
            strats += [
                {"nome": "Evitare — Fondamentali Deboli",
                 "rischio": "ALTO",
                 "desc": ("I segnali fondamentali e tecnici di lungo periodo sono negativi. "
                          "Non adatto a un investimento di lungo termine in questa fase. "
                          "Valutare alternative nello stesso settore con migliori fondamentali.")},
            ]

    return strats


# ── Public API ────────────────────────────────────────────────────────────────

def generate_analysis(
    df: pd.DataFrame,
    sr: dict,
    fundamentals: dict,
    analysts: dict,
    horizon: str = "medio",
) -> dict:
    close = float(df["Close"].iloc[-1])

    tech   = _tech_signals(df, sr, close)
    fund   = _fund_signals(fundamentals, analysts, close)
    score  = _score_from_signals(tech, fund, horizon)
    strats = _strategies(score, df, sr, close, horizon)

    if score >= 45:
        rec = ("FORTE ACQUISTO", "#00e676")
    elif score >= 18:
        rec = ("ACQUISTO",       "#26c6a1")
    elif score >= -18:
        rec = ("NEUTRALE",       "#FFB300")
    elif score >= -45:
        rec = ("VENDITA",        "#f44336")
    else:
        rec = ("FORTE VENDITA",  "#b71c1c")

    return {
        "score":       score,
        "rec_label":   rec[0],
        "rec_color":   rec[1],
        "tech":        tech,
        "fund":        fund,
        "strategies":  strats,
        "close":       close,
        "horizon":     horizon,
    }


def build_analysis_html(result: dict, ticker: str, period: str = "3M") -> str:
    """Render the analysis dict as a styled HTML page for QTextBrowser."""
    score      = result["score"]
    rec_label  = result["rec_label"]
    rec_color  = result["rec_color"]
    tech       = result["tech"]
    fund       = result["fund"]
    strategies = result["strategies"]
    close      = result["close"]
    horizon    = result.get("horizon", "medio")
    hor_label  = _HORIZON_LABEL.get(horizon, horizon)

    # Warning shown when the selected period uses non-daily candles
    _interval_warn = {
        "2Y": ("2Y usa candele settimanali", "1Y"),
        "5Y": ("5Y usa candele settimanali", "1Y"),
        "MAX": ("MAX usa candele mensili", "1Y"),
    }
    warn_period, suggest = _interval_warn.get(period, (None, None))
    period_warning = (
        f'<div style="background:#2a1a00;border:1px solid #ffa72655;border-radius:6px;'
        f'padding:8px 12px;margin-bottom:12px;font-size:11px;color:#ffa726;">'
        f'Attenzione: il periodo <b>{period}</b> ({warn_period}) applica gli indicatori '
        f'su candele non giornaliere — RSI, MACD e medie mobili misurano settimane/mesi, '
        f'non giorni. Per un\'analisi di lungo termine affidabile usa <b>{suggest}</b>.'
        f'</div>'
    ) if warn_period else ""

    bar_fill   = max(0, min(100, (score + 100) // 2))
    bar_color  = rec_color

    SENT_COLORS = {
        "bullish":      ("#00e676", "#002a12"),
        "mild_bullish": ("#80cbc4", "#012a25"),
        "neutral":      ("#78909c", "#0d1a1f"),
        "mild_bearish": ("#ffa726", "#2a1600"),
        "bearish":      ("#f44336", "#2a0000"),
    }

    def badge(label: str, sentiment: str) -> str:
        fg, bg = SENT_COLORS.get(sentiment, ("#fff", "#333"))
        return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}66;'
                f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">'
                f'{label}</span>')

    def signal_row(name: str, tup: tuple) -> str:
        label, sentiment, detail = tup
        return (f'<tr>'
                f'<td style="padding:5px 10px;color:#a0a0c0;font-size:11px;white-space:nowrap;">{name}</td>'
                f'<td style="padding:5px 10px;">{badge(label, sentiment)}</td>'
                f'<td style="padding:5px 10px;color:#b0b0d0;font-size:11px;">{detail}</td>'
                f'</tr>')

    tech_rows = "".join(signal_row(k, v) for k, v in tech.items())
    fund_rows = "".join(signal_row(k, v) for k, v in fund.items())

    def strat_card(s: dict) -> str:
        risk_colors = {
            "BASSO": "#26c6a1", "BASSO-MEDIO": "#80cbc4",
            "MEDIO": "#FFB300", "MEDIO-ALTO": "#ffa726",
            "ALTO se si mantiene": "#f44336",
            "MOLTO ALTO se si rimane": "#b71c1c",
        }
        rc = risk_colors.get(s["rischio"], "#999")
        return (f'<div style="background:#16162a;border:1px solid #2a2a50;border-radius:8px;'
                f'padding:14px 18px;margin-bottom:10px;">'
                f'<div style="font-size:14px;font-weight:700;color:#e0e0ff;margin-bottom:6px;">{s["nome"]}</div>'
                f'<div style="font-size:12px;color:#c0c0e0;line-height:1.6;margin-bottom:8px;">{s["desc"]}</div>'
                f'<div style="font-size:10px;">Livello di rischio: '
                f'<b style="color:{rc};">{s["rischio"]}</b></div>'
                f'</div>')

    strat_cards = "".join(strat_card(s) for s in strategies)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background:#0f0f1a; color:#d0d0e8;
          font-family:'Segoe UI',Arial,sans-serif; font-size:13px;
          padding:16px 20px; }}
  h2 {{ font-size:13px; color:#8080b0; font-weight:600;
        text-transform:uppercase; letter-spacing:1px;
        margin:20px 0 10px; border-bottom:1px solid #1e1e38; padding-bottom:6px; }}
  table {{ border-collapse:collapse; width:100%; }}
  tr:hover td {{ background:#1a1a2e; }}
  .score-bar-outer {{ background:#1a1a2e; border-radius:6px; height:10px;
                      width:100%; margin:8px 0; border:1px solid #2a2a50; }}
  .score-bar-inner {{ height:100%; border-radius:6px; background:{bar_color}; width:{bar_fill}%; }}
</style>
</head>
<body>

<!-- Header -->
<div style="background:#12122a;border:1px solid #2a2a50;border-radius:10px;
            padding:20px 24px;margin-bottom:16px;text-align:center;">
  <div style="font-size:13px;color:#8080b0;margin-bottom:8px;">
    Analisi — <b style="color:#d0d0e8;">{ticker}</b>
    &nbsp;|&nbsp; Prezzo: <b>{close:.2f}</b>
  </div>
  <div style="font-size:12px;color:#6060a0;margin-bottom:6px;">
    Orizzonte: <b style="color:#a0a0d0;">{hor_label}</b>
  </div>
  <div style="font-size:32px;font-weight:800;color:{rec_color};
              letter-spacing:2px;margin-bottom:12px;">{rec_label}</div>
  <div style="font-size:12px;color:#8080b0;margin-bottom:6px;">
    Score tecnico + fondamentale: <b style="color:{rec_color};">{score:+d} / 100</b>
  </div>
  <div class="score-bar-outer">
    <div class="score-bar-inner"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;color:#505070;margin-top:3px;">
    <span>FORTE VENDITA</span><span>NEUTRALE</span><span>FORTE ACQUISTO</span>
  </div>
</div>

{period_warning}

<!-- Technical signals -->
<h2>📊 Segnali Tecnici</h2>
<table>{tech_rows}</table>

<!-- Fundamental signals -->
<h2>📈 Segnali Fondamentali</h2>
<table>{fund_rows}</table>

<!-- Strategies -->
<h2>💡 Strategie Consigliate</h2>
{strat_cards}

<div style="font-size:10px;color:#404060;margin-top:14px;padding-top:8px;
            border-top:1px solid #1e1e38;">
  ⚠️ Questo parere è generato automaticamente da indicatori tecnici e fondamentali.
  Non costituisce consulenza finanziaria. Investi sempre con consapevolezza del rischio.
</div>

</body>
</html>"""

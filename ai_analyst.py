"""Google Gemini AI analyst — financial analysis with real-time web search grounding."""

from __future__ import annotations

import os
import re
import datetime
from typing import Optional
import pandas as pd


# ── API key management ────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Load .env manually without requiring python-dotenv."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v


def get_api_key() -> Optional[str]:
    """Return Gemini API key from environment (loads from .env if present)."""
    _load_dotenv()
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def save_api_key(key: str) -> None:
    """Persist key to .env file next to this script and set in current env."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    lines: list[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    lines.append(f"GEMINI_API_KEY={key}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"GEMINI_API_KEY={key}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.environ["GEMINI_API_KEY"] = key


# ── Format helpers ────────────────────────────────────────────────────────────

def _fv(v, decimals: int = 2) -> str:
    """Format numeric value; N/A for None/NaN."""
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _fmc(v) -> str:
    """Format market cap as human-readable shorthand."""
    if v is None:
        return "N/A"
    if v >= 1e12:
        return f"{v / 1e12:.2f} T"
    if v >= 1e9:
        return f"{v / 1e9:.2f} B"
    if v >= 1e6:
        return f"{v / 1e6:.2f} M"
    return str(v)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    ticker: str,
    fundamentals: dict,
    df: pd.DataFrame,
    sr: dict,
    analysts: dict,
    period: str,
) -> str:
    """
    Build the optimised Gemini prompt for comprehensive stock analysis.

    Includes all available technical indicators, fundamental metrics, S/R levels
    and instructs the model to search for current news before analysing.
    """
    close    = float(df["Close"].iloc[-1])
    currency = fundamentals.get("currency", "USD")

    def _last(col: str) -> Optional[float]:
        if col in df.columns:
            s = df[col].dropna()
            return float(s.iloc[-1]) if not s.empty else None
        return None

    rsi       = _last("RSI")
    macd      = _last("MACD")
    macd_sig  = _last("MACD_sig")
    macd_hist = _last("MACD_hist")
    sma20     = _last("SMA20")
    sma50     = _last("SMA50")
    sma200    = _last("SMA200")
    atr       = _last("ATR")
    stoch_k   = _last("STOCH_K")
    stoch_d   = _last("STOCH_D")
    bb_upper  = _last("BB_upper")
    bb_lower  = _last("BB_lower")
    bb_mid    = _last("BB_mid")

    # OBV trend (last 20 bars)
    obv_trend = "N/A"
    if "OBV" in df.columns:
        obv = df["OBV"].dropna()
        if len(obv) >= 20:
            base = float(obv.iloc[-20])
            if base != 0:
                ratio = float(obv.iloc[-1]) / base
                obv_trend = (
                    "in crescita — accumulazione" if ratio > 1.05
                    else "in calo — distribuzione" if ratio < 0.95
                    else "stabile — neutrale"
                )

    def _ab(ma_val: Optional[float]) -> str:
        if ma_val is None:
            return "N/A"
        diff = (close - ma_val) / ma_val * 100
        return f"{'sopra' if close > ma_val else 'sotto'} @ {ma_val:.2f} ({diff:+.1f}%)"

    # Bollinger position
    bb_pos = "N/A"
    if bb_upper and bb_lower and bb_mid and (bb_upper - bb_lower) > 0:
        pos       = (close - bb_lower) / (bb_upper - bb_lower)
        width_pct = (bb_upper - bb_lower) / bb_mid * 100
        bb_pos    = f"{pos * 100:.0f}% della banda | larghezza: {width_pct:.1f}%"

    atr_pct = f"{atr / close * 100:.2f}%" if atr and close else "N/A"

    # S/R levels nearest to price
    resistances = sorted([r for r in sr.get("resistance", []) if r > close])[:4]
    supports    = sorted([s for s in sr.get("support",    []) if s < close], reverse=True)[:4]

    # Short-term performance (last 5 bars)
    perf_5 = "N/A"
    if len(df) >= 6:
        p5    = float(df["Close"].iloc[-6])
        perf_5 = f"{(close - p5) / p5 * 100:+.1f}%"

    # Volume ratio vs 20-bar average
    vol_info = "N/A"
    if "Volume" in df.columns:
        vl = float(df["Volume"].iloc[-1])
        va = float(df["Volume"].tail(20).mean())
        if va > 0:
            vol_info = f"{vl / va:.2f}× media 20 barre"

    # Golden / Death Cross
    cross = "N/A"
    if sma50 and sma200:
        cross = (
            "✅ Golden Cross (SMA50 > SMA200 — trend rialzista di lungo)"
            if sma50 > sma200
            else "⚠️ Death Cross (SMA50 < SMA200 — trend ribassista di lungo)"
        )

    # Analyst upside
    t_mean = analysts.get("target_mean")
    upside = f"{(t_mean - close) / close * 100:+.1f}%" if t_mean and close else "N/A"

    today = datetime.date.today().strftime("%d %B %Y")

    # ── RSI / MACD / Stoch labels ─────────────────────────────────────────────
    rsi_lbl = (
        "⚠️ Ipervenduto (<30)"  if rsi and rsi < 30
        else "🔴 Ipercomprato (>70)"  if rsi and rsi > 70
        else "🟡 Neutrale (30–70)"    if rsi
        else "N/A"
    )
    macd_lbl = (
        "🟢 Rialzista (MACD > Signal)" if macd and macd_sig and macd > macd_sig
        else "🔴 Ribassista (MACD < Signal)" if macd and macd_sig
        else "N/A"
    )
    stoch_lbl = (
        "⚠️ Ipervenduto (<20)" if stoch_k and stoch_k < 20
        else "🔴 Ipercomprato (>80)" if stoch_k and stoch_k > 80
        else "🟡 Neutrale"     if stoch_k
        else "N/A"
    )

    prompt = f"""Sei un analista finanziario senior con 20 anni di esperienza in equity research, analisi tecnica quantitativa e gestione di portafogli istituzionali. Oggi è il **{today}**.

---

## TITOLO DA ANALIZZARE: **{ticker}** — {fundamentals.get('name', ticker)}

### SNAPSHOT DI MERCATO
- Prezzo attuale: **{close:.2f} {currency}**
- Performance {period} (ultimi 5 periodi): {perf_5}
- Volume ultimo periodo vs media 20: {vol_info}
- Settore: {fundamentals.get('sector', 'N/A')} | Industria: {fundamentals.get('industry', 'N/A')}
- Paese: {fundamentals.get('country', 'N/A')}

---

### METRICHE FONDAMENTALI
| Metrica | Valore |
|---------|--------|
| Market Cap | {_fmc(fundamentals.get('market_cap'))} |
| P/E trailing | {_fv(fundamentals.get('pe_ratio'))} |
| P/E forward | {_fv(fundamentals.get('forward_pe'))} |
| EPS | {_fv(fundamentals.get('eps'))} {currency} |
| PEG Ratio | {_fv(fundamentals.get('peg_ratio'))} |
| Beta | {_fv(fundamentals.get('beta'))} |
| 52W High / Low | {_fv(fundamentals.get('52w_high'))} / {_fv(fundamentals.get('52w_low'))} |
| Dipendenti | {fundamentals.get('employees', 'N/A')} |

### CONSENSO ANALISTI
- Rating: **{analysts.get('recommendation', 'N/A')}** ({analysts.get('num_analysts', '?')} analisti)
- Target medio: {_fv(t_mean)} {currency} — upside implicito: **{upside}**
- Range target: {_fv(analysts.get('target_low'))} – {_fv(analysts.get('target_high'))} {currency}

---

### INDICATORI TECNICI (timeframe: {period})
| Indicatore | Valore | Segnale |
|-----------|--------|---------|
| RSI (14) | {_fv(rsi)} | {rsi_lbl} |
| MACD | {_fv(macd)} / Signal: {_fv(macd_sig)} / Hist: {_fv(macd_hist)} | {macd_lbl} |
| Bollinger Bands | {bb_pos} | — |
| SMA 20 | Prezzo {_ab(sma20)} | — |
| SMA 50 | Prezzo {_ab(sma50)} | — |
| SMA 200 | Prezzo {_ab(sma200)} | — |
| Golden/Death Cross | {cross} | — |
| Stocastico %K / %D | {_fv(stoch_k)} / {_fv(stoch_d)} | {stoch_lbl} |
| ATR (14) | {_fv(atr)} ({atr_pct} del prezzo) | Misura di volatilità |
| OBV | {obv_trend} | Pressione volume |

### LIVELLI TECNICI CHIAVE
- **Resistenze sopra il prezzo:** {', '.join([f'{r:.2f}' for r in resistances]) or 'Non identificate'}
- **Supporti sotto il prezzo:** {', '.join([f'{s:.2f}' for s in supports]) or 'Non identificati'}

---

## LA TUA ANALISI

**PASSO 1 — Ricerca contestuale:** prima di rispondere, usa Google Search per trovare le notizie più recenti su **{ticker}** ({fundamentals.get('name', ticker)}): risultati trimestrali, upgrade/downgrade analisti, lanci di prodotti, acquisizioni, dati macro che impattano il settore, sentiment generale — tutto ciò che è accaduto negli ultimi 30–60 giorni.

**PASSO 2 — Analisi completa:** sulla base dei dati forniti + notizie trovate, scrivi un'analisi di investimento in **italiano** con queste sezioni:

### 1. 📰 Contesto Attuale e Notizie Recenti
Cosa sta succedendo con questo titolo *adesso*? Earnings, guidance, notizie chiave, catalizzatori imminenti. Contestualizza rispetto al settore e al mercato.

### 2. 📊 Analisi Tecnica
Interpreta gli indicatori sopra in modo critico. Qual è il trend dominante (daily/weekly)? Quali livelli di S/R sono più rilevanti in questo momento? Gli indicatori convergono o divergono? Segnali di continuazione o potenziale inversione?

### 3. 💼 Valutazione e Fondamentali
La valutazione è attraente rispetto ai peer del settore? Commenta P/E, PEG, EPS growth, qualità del business, redditività, solidità del bilancio. Vale il prezzo attuale fondamentalmente?

### 4. ⚡ Breve Termine (1–4 settimane)
Scenario operativo: livelli precisi di entrata, stop loss e take profit. Evento chiave più vicino da monitorare. Rapporto rischio/rendimento specifico (es. 1:2.5).

### 5. 📅 Medio Termine (1–3 mesi)
Direzione attesa con i catalizzatori principali. Presenta 3 scenari (base / rialzista / ribassista) con prezzi target espliciti e probabilità stimate.

### 6. 🎯 Lungo Termine (6–18 mesi)
Tesi d'investimento di lungo periodo. Target di prezzo. La storia fondamentale è intatta? Vale costruire una posizione core?

### 7. 💡 Strategie Operative
Fornisci **2–3 strategie concrete** e diverse tra loro. Per ciascuna specifica:
- Nome e tipo di strategia
- Livello di entrata preciso
- Stop loss
- Target 1 e Target 2
- Sizing consigliato (% del portafoglio)
- Orizzonte temporale
- Condizioni di invalidazione

### 8. ⚠️ Rischi Principali
Elenca i **top 4 rischi specifici** per questo titolo in questo momento (non rischi generici da manuale). Quantifica dove possibile.

### 9. ✅ Verdetto Finale
Raccomandazione chiara:
**FORTE ACQUISTO** / **ACQUISTO** / **NEUTRALE** / **VENDITA** / **FORTE VENDITA**
Convinzione: ★★★★★ (1–5 stelle)
Sintesi della tesi in 3 righe max. Il verdetto finale deve essere sul lungo termine ovvero 6-12 mesi e sul medio termine, quindi devi farmi due verdetti.

---
*Sii specifico e usa prezzi e percentuali precisi. Ogni strategia deve essere immediatamente implementabile da un trader retail.*
"""
    return prompt


# ── Gemini API call ───────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call Gemini 2.0 Flash with Google Search grounding.

    Returns the full response text (markdown).
    Raises ImportError if google-genai is not installed,
    RuntimeError on API errors.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as e:
        raise ImportError(
            "La libreria google-genai non è installata.\n"
            "Esegui: pip install google-genai"
        ) from e

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.35,          # low temp for factual financial analysis
                max_output_tokens=4096,
            ),
        )
        return response.text
    except Exception as e:
        raise RuntimeError(str(e)) from e


# ── Response rendering ────────────────────────────────────────────────────────

def _markdown_to_body(text: str) -> str:
    """Convert markdown to an HTML fragment (no page wrapper)."""
    try:
        import markdown as md  # type: ignore
        return md.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        return _fallback_md_to_html(text)


_PAGE_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f1a; color: #d0d0e8;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px; line-height: 1.75; padding: 22px 28px;
  }
  h1 { font-size: 18px; color: #c8c8ff; margin: 24px 0 10px;
       border-bottom: 1px solid #2e2e50; padding-bottom: 7px; }
  h2 { font-size: 15px; color: #a0a8d8; margin: 20px 0 8px;
       border-bottom: 1px solid #1e1e38; padding-bottom: 5px; }
  h3 { font-size: 13px; color: #9090c8; margin: 16px 0 6px; }
  p  { margin: 8px 0; }
  strong { color: #e8e8ff; }
  em     { color: #90c8ff; font-style: normal; }
  a  { color: #4090ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  ul, ol { padding-left: 22px; margin: 8px 0; }
  li { margin: 4px 0; }
  hr { border: none; border-top: 1px solid #1e1e38; margin: 18px 0; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 12px; }
  th, td { border: 1px solid #2a2a50; padding: 7px 14px; text-align: left; }
  th { background: #1a1a30; color: #9090c8; font-size: 11px; text-transform: uppercase; }
  tr:nth-child(even) td { background: #121220; }
  tr:hover td { background: #1a1a2e; }
  code {
    background: #1a1a30; color: #80d8a0;
    padding: 1px 6px; border-radius: 3px;
    font-size: 11px; font-family: 'Consolas', monospace;
  }
  pre {
    background: #131328; border: 1px solid #2a2a50; border-radius: 6px;
    padding: 14px; overflow-x: auto; margin: 12px 0;
  }
  pre code { background: none; padding: 0; }
  blockquote {
    border-left: 3px solid #4090ff; margin: 12px 0; padding: 8px 16px;
    background: #131328; color: #b0b8d8; border-radius: 0 6px 6px 0;
  }
  /* Conversation items */
  .user-turn {
    background: #12122a; border: 1px solid #2a3060;
    border-radius: 8px; padding: 10px 16px; margin: 28px 0 14px;
  }
  .user-label {
    font-size: 10px; color: #4060a0; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600; display: block; margin-bottom: 5px;
  }
  .user-text { color: #a0b0d8; font-size: 13px; margin: 0; }
  .turn-sep  { border: none; border-top: 2px solid #1e1e38; margin: 30px 0; }
  .loading   { color: #505080; font-style: italic; font-size: 12px; padding: 14px 0; }
  .disclaimer {
    font-size: 10px; color: #404060; margin-top: 24px;
    padding-top: 10px; border-top: 1px solid #1e1e38; line-height: 1.5;
  }
"""

_DISCLAIMER = (
    "Analisi generata da Google Gemini AI con ricerca web in tempo reale. "
    "Non costituisce consulenza finanziaria professionale. "
    "I mercati finanziari comportano rischi significativi. "
    "Consulta sempre un consulente finanziario qualificato prima di investire."
)


def render_conversation_html(display_items: list) -> str:
    """
    Render a list of conversation items to a full dark-theme HTML page.

    display_items: list of dicts with keys:
      role  — "model" | "user" | "loading" | "error"
      text  — raw markdown (model) or plain text (user/error) or "" (loading)
    """
    parts: list[str] = []
    for i, item in enumerate(display_items):
        role = item["role"]
        text = item.get("text", "")

        if role == "model":
            sep = '<hr class="turn-sep">' if i > 0 else ""
            parts.append(f'{sep}<div class="ai-turn">{_markdown_to_body(text)}</div>')

        elif role == "user":
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(
                f'<div class="user-turn">'
                f'<span class="user-label">La tua domanda</span>'
                f'<p class="user-text">{safe}</p>'
                f'</div>'
            )

        elif role == "loading":
            parts.append('<p class="loading">Elaborazione risposta in corso…</p>')

        elif role == "error":
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(
                f'<div style="background:#1a0000;border:1px solid #f4433655;'
                f'border-radius:6px;padding:12px 16px;margin:16px 0;'
                f'font-size:12px;color:#f09090;">'
                f'Errore: {safe}</div>'
            )

    content = "\n".join(parts)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_PAGE_CSS}</style>
</head>
<body>
{content}
<div class="disclaimer">{_DISCLAIMER}</div>
</body>
</html>"""


def response_to_html(markdown_text: str) -> str:
    """Convert a single Gemini response to a styled dark-theme HTML page (backward compat)."""
    return render_conversation_html([{"role": "model", "text": markdown_text}])


def build_followup_context(ticker: str, fundamentals: dict, period: str) -> str:
    """
    Build a compact (low-token) opener for follow-up conversations.
    Replaces the full initial prompt so subsequent turns are token-efficient.
    """
    name     = fundamentals.get("name", ticker)
    price    = fundamentals.get("price") or fundamentals.get("52w_high", "")
    currency = fundamentals.get("currency", "USD")
    sector   = fundamentals.get("sector", "N/A")
    return (
        f"Sei un analista finanziario esperto. Hai appena completato un'analisi "
        f"completa del titolo {ticker} ({name}), "
        f"prezzo corrente {price} {currency}, settore {sector}, "
        f"periodo di riferimento {period}. "
        f"Di seguito trovi la tua analisi precedente come contesto."
    )


def call_gemini_followup(full_history: list, api_key: str) -> str:
    """
    Continue a conversation using the compact history.

    full_history: list of {"role": "user"|"model", "text": str}
                  The last item must be role="user" (the new question).
    Token-efficient: caller should pass compact_context + first_response + question,
    NOT the full initial prompt.
    """
    try:
        from google import genai   # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as e:
        raise ImportError("Esegui: pip install google-genai") from e

    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["text"])],
        )
        for msg in full_history
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.35,
                max_output_tokens=2048,   # follow-ups shorter than initial
            ),
        )
        return response.text
    except Exception as e:
        raise RuntimeError(str(e)) from e




def _fallback_md_to_html(text: str) -> str:
    """Minimal markdown → HTML fallback (no external library needed)."""
    lines: list[str] = []
    in_table = False
    in_pre   = False
    table_header_done = False

    for raw in text.split("\n"):
        line = raw

        # Fenced code blocks
        if line.strip().startswith("```"):
            if not in_pre:
                lines.append("<pre><code>")
                in_pre = True
            else:
                lines.append("</code></pre>")
                in_pre = False
            continue
        if in_pre:
            lines.append(raw.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Table rows
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip separator rows like |---|---|
            if all(re.match(r"^[-: ]+$", c) for c in cells):
                if not table_header_done:
                    # First separator → close thead, open tbody
                    lines.append("</tr></thead><tbody>")
                    table_header_done = True
                continue
            if not in_table:
                lines.append('<table><thead><tr>')
                in_table = True
                table_header_done = False
                tag = "th"
            else:
                tag = "td"
            lines.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        else:
            if in_table:
                lines.append("</tbody></table>")
                in_table = False
                table_header_done = False

        # Headings
        if line.startswith("### "):
            line = f"<h3>{_inline(line[4:])}</h3>"
        elif line.startswith("## "):
            line = f"<h2>{_inline(line[3:])}</h2>"
        elif line.startswith("# "):
            line = f"<h1>{_inline(line[2:])}</h1>"
        elif line.strip() in ("---", "***", "___"):
            line = "<hr>"
        elif line.startswith("- ") or line.startswith("* "):
            line = f"<li>{_inline(line[2:])}</li>"
        elif line.strip():
            line = f"<p>{_inline(line)}</p>"
        else:
            line = "<br>"

        lines.append(line)

    if in_table:
        lines.append("</tbody></table>")

    # Wrap consecutive <li> in <ul>
    result = "\n".join(lines)
    result = re.sub(r"(<li>.*?</li>\n?)+", lambda m: "<ul>" + m.group(0) + "</ul>", result, flags=re.S)
    return result


def _inline(text: str) -> str:
    """Convert inline markdown (bold, italic, code) to HTML."""
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>",          text)
    text = re.sub(r"\*(.+?)\*",         r"<em>\1</em>",                  text)
    text = re.sub(r"`(.+?)`",           r"<code>\1</code>",              text)
    return text

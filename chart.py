"""Plotly chart builder — returns a self-contained HTML string."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Set

# ── Colour palette ───────────────────────────────────────────────────────────
BG          = "#0f0f1a"
PAPER       = "#0f0f1a"
GRID        = "#1e1e30"
TEXT        = "#d0d0e8"
UP          = "#26c6a1"
DOWN        = "#f44336"
VOL_UP      = "rgba(38,198,161,0.45)"
VOL_DOWN    = "rgba(244,67,54,0.45)"
SMA20_C     = "#FFB300"
SMA50_C     = "#AB47BC"
SMA200_C    = "#42A5F5"
EMA20_C     = "#FF7043"
EMA9_C      = "#80CBC4"
BB_C        = "rgba(100,180,255,0.65)"
BB_FILL     = "rgba(100,180,255,0.05)"
RSI_C       = "#CE93D8"
MACD_C      = "#64B5F6"
SIG_C       = "#FF8A65"
STOCH_K_C   = "#A5D6A7"
STOCH_D_C   = "#EF9A9A"
ATR_C       = "#80DEEA"
OBV_C       = "#FFCC02"
RES_C       = "rgba(244,67,54,0.65)"
SUP_C       = "rgba(38,198,161,0.65)"
RES_FONT    = "#f44336"
SUP_FONT    = "#26c6a1"


def _axis_style() -> dict:
    return dict(
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=GRID,
        tickfont=dict(color=TEXT, size=9),
        title_font=dict(color=TEXT, size=9),
    )


def build_chart(
    df: pd.DataFrame,
    sr_levels: Dict[str, List[float]],
    indicators: Set[str],
    ticker: str = "",
) -> str:
    """Return a full-HTML Plotly chart string."""

    show_rsi   = "RSI"  in indicators
    show_macd  = "MACD" in indicators
    show_stoch = "STOCH" in indicators
    show_atr   = "ATR"  in indicators
    show_obv   = "OBV"  in indicators

    # Build row layout dynamically
    row_cfg: List[Dict] = [{"name": "price", "height": 0.52}]
    row_cfg.append({"name": "volume", "height": 0.10})
    if show_rsi:
        row_cfg.append({"name": "rsi", "height": 0.13})
    if show_macd:
        row_cfg.append({"name": "macd", "height": 0.13})
    if show_stoch:
        row_cfg.append({"name": "stoch", "height": 0.12})
    if show_atr:
        row_cfg.append({"name": "atr", "height": 0.10})
    if show_obv:
        row_cfg.append({"name": "obv", "height": 0.10})

    total_h = sum(r["height"] for r in row_cfg)
    heights = [r["height"] / total_h for r in row_cfg]
    names   = [r["name"] for r in row_cfg]
    n_rows  = len(row_cfg)

    row_of = {name: i + 1 for i, name in enumerate(names)}

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=heights,
    )

    x = df.index

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
        showlegend=False,
        whiskerwidth=0.4,
    ), row=1, col=1)

    # ── Moving averages ───────────────────────────────────────────────────────
    if "MA" in indicators:
        for col, color, label, width in [
            ("EMA9",   EMA9_C,  "EMA 9",   1.2),
            ("EMA20",  EMA20_C, "EMA 20",  1.4),
            ("SMA20",  SMA20_C, "SMA 20",  1.4),
            ("SMA50",  SMA50_C, "SMA 50",  1.5),
            ("SMA200", SMA200_C,"SMA 200", 2.0),
        ]:
            s = df[col].dropna() if col in df else pd.Series(dtype=float)
            if not s.empty:
                fig.add_trace(go.Scatter(
                    x=s.index, y=s, name=label,
                    line=dict(color=color, width=width), opacity=0.9,
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if "BB" in indicators and "BB_upper" in df:
        bb_u = df["BB_upper"].dropna()
        bb_l = df["BB_lower"].dropna()
        bb_m = df["BB_mid"].dropna()
        if not bb_u.empty:
            fig.add_trace(go.Scatter(
                x=bb_u.index, y=bb_u, name="BB Upper",
                line=dict(color=BB_C, width=1, dash="dot"), showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bb_l.index, y=bb_l, name="BB Lower",
                line=dict(color=BB_C, width=1, dash="dot"),
                fill="tonexty", fillcolor=BB_FILL, showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bb_m.index, y=bb_m, name="BB Mid",
                line=dict(color=BB_C, width=0.8), opacity=0.4, showlegend=False,
            ), row=1, col=1)

    # ── Support & Resistance ──────────────────────────────────────────────────
    if "SR" in indicators and sr_levels:
        current = float(df["Close"].iloc[-1])
        x0, x1 = df.index[0], df.index[-1]
        for level in sr_levels.get("resistance", []):
            if abs(level - current) / current < 0.6:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                              line=dict(color=RES_C, width=1, dash="dash"), row=1, col=1)
                fig.add_annotation(x=x1, y=level, text=f" R {level:.2f}",
                                   font=dict(size=8, color=RES_FONT), showarrow=False,
                                   xanchor="left", bgcolor="rgba(15,15,26,0.75)",
                                   row=1, col=1)
        for level in sr_levels.get("support", []):
            if abs(level - current) / current < 0.6:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                              line=dict(color=SUP_C, width=1, dash="dash"), row=1, col=1)
                fig.add_annotation(x=x1, y=level, text=f" S {level:.2f}",
                                   font=dict(size=8, color=SUP_FONT), showarrow=False,
                                   xanchor="left", bgcolor="rgba(15,15,26,0.75)",
                                   row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_colors = [VOL_UP if c >= o else VOL_DOWN
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=x, y=df["Volume"], name="Volume",
        marker_color=vol_colors, showlegend=False,
    ), row=row_of["volume"], col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    if show_rsi and "RSI" in df:
        rsi = df["RSI"].dropna()
        fig.add_trace(go.Scatter(
            x=rsi.index, y=rsi, name="RSI 14",
            line=dict(color=RSI_C, width=1.4), showlegend=False,
        ), row=row_of["rsi"], col=1)
        for lvl, clr in [(70, "rgba(244,67,54,0.45)"), (30, "rgba(38,198,161,0.45)"), (50, GRID)]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          row=row_of["rsi"], col=1)
        fig.update_yaxes(range=[0, 100], row=row_of["rsi"], col=1,
                         tickvals=[30, 50, 70], title_text="RSI", **_axis_style())

    # ── MACD ──────────────────────────────────────────────────────────────────
    if show_macd and "MACD" in df:
        macd = df["MACD"].dropna()
        sig  = df["MACD_sig"].dropna()
        hist = df["MACD_hist"].dropna()
        h_colors = ["rgba(38,198,161,0.7)" if v >= 0 else "rgba(244,67,54,0.7)"
                    for v in hist]
        fig.add_trace(go.Bar(x=hist.index, y=hist, name="MACD Hist",
                             marker_color=h_colors, showlegend=False),
                      row=row_of["macd"], col=1)
        fig.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD",
                                 line=dict(color=MACD_C, width=1.3), showlegend=False),
                      row=row_of["macd"], col=1)
        fig.add_trace(go.Scatter(x=sig.index, y=sig, name="Signal",
                                 line=dict(color=SIG_C, width=1.3), showlegend=False),
                      row=row_of["macd"], col=1)
        fig.update_yaxes(title_text="MACD", row=row_of["macd"], col=1, **_axis_style())

    # ── Stochastic ────────────────────────────────────────────────────────────
    if show_stoch and "STOCH_K" in df:
        sk = df["STOCH_K"].dropna()
        sd = df["STOCH_D"].dropna()
        fig.add_trace(go.Scatter(x=sk.index, y=sk, name="%K",
                                 line=dict(color=STOCH_K_C, width=1.3), showlegend=False),
                      row=row_of["stoch"], col=1)
        fig.add_trace(go.Scatter(x=sd.index, y=sd, name="%D",
                                 line=dict(color=STOCH_D_C, width=1.3), showlegend=False),
                      row=row_of["stoch"], col=1)
        for lvl, clr in [(80, "rgba(244,67,54,0.45)"), (20, "rgba(38,198,161,0.45)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          row=row_of["stoch"], col=1)
        fig.update_yaxes(range=[0, 100], title_text="Stoch", row=row_of["stoch"], col=1,
                         **_axis_style())

    # ── ATR ───────────────────────────────────────────────────────────────────
    if show_atr and "ATR" in df:
        atr = df["ATR"].dropna()
        fig.add_trace(go.Scatter(x=atr.index, y=atr, name="ATR 14",
                                 line=dict(color=ATR_C, width=1.3), fill="tozeroy",
                                 fillcolor="rgba(128,222,234,0.08)", showlegend=False),
                      row=row_of["atr"], col=1)
        fig.update_yaxes(title_text="ATR", row=row_of["atr"], col=1, **_axis_style())

    # ── OBV ───────────────────────────────────────────────────────────────────
    if show_obv and "OBV" in df:
        obv = df["OBV"].dropna()
        fig.add_trace(go.Scatter(x=obv.index, y=obv, name="OBV",
                                 line=dict(color=OBV_C, width=1.2), showlegend=False),
                      row=row_of["obv"], col=1)
        fig.update_yaxes(title_text="OBV", row=row_of["obv"], col=1, **_axis_style())

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="Segoe UI, Arial", color=TEXT, size=10),
        margin=dict(l=8, r=90, t=28, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a2e", font=dict(color=TEXT, size=10)),
        legend=dict(
            bgcolor="rgba(15,15,26,0.85)", bordercolor=GRID, borderwidth=1,
            font=dict(size=9), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )

    # Apply common axis style to every axis
    for i in range(1, n_rows + 1):
        suf = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{suf}": dict(showgrid=True, gridcolor=GRID, gridwidth=0.5,
                                zeroline=False, showline=True, linecolor=GRID,
                                tickfont=dict(color=TEXT, size=9)),
            f"yaxis{suf}": dict(showgrid=True, gridcolor=GRID, gridwidth=0.5,
                                zeroline=False, showline=True, linecolor=GRID,
                                tickfont=dict(color=TEXT, size=9), side="right"),
        })

    return fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": f"chart_{ticker}"},
        },
    )

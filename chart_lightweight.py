"""TradingView Lightweight Charts builder — returns a self-contained HTML string."""

import pandas as pd
import json
from typing import Dict, List, Set

# ── Color palette (dark theme matching Trade Republic style) ────────────────
BG = "#0f0f1a"
UP_COLOR = "#26c6a1"
DOWN_COLOR = "#f44336"
GRID_COLOR = "#1e1e30"
TEXT_COLOR = "#d0d0e8"

# Indicator colors
SMA20_COLOR = "#FFB300"
SMA50_COLOR = "#AB47BC"
SMA200_COLOR = "#42A5F5"
EMA9_COLOR = "#80CBC4"
EMA20_COLOR = "#FF7043"
BB_UPPER_COLOR = "#64B5F6"
BB_LOWER_COLOR = "#64B5F6"
BB_MID_COLOR = "#8090c0"
RSI_COLOR = "#CE93D8"
MACD_COLOR = "#64B5F6"
SIGNAL_COLOR = "#FF8A65"
HISTOGRAM_UP_COLOR = "#26c6a140"
HISTOGRAM_DOWN_COLOR = "#f4433640"
STOCH_K_COLOR = "#A5D6A7"
STOCH_D_COLOR = "#EF9A9A"
ATR_COLOR = "#80DEEA"
OBV_COLOR = "#FFCC02"
RESISTANCE_COLOR = "#f44336"
SUPPORT_COLOR = "#26c6a1"


def _prepare_candlestick_data(df: pd.DataFrame) -> str:
    """Convert DataFrame to candlestick JSON format."""
    data = []
    for idx, row in df.iterrows():
        data.append({
            "time": int(idx.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })
    return json.dumps(data)


def _prepare_volume_data(df: pd.DataFrame) -> str:
    """Convert DataFrame to volume JSON format with colors."""
    data = []
    for idx, row in df.iterrows():
        color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
        data.append({
            "time": int(idx.timestamp()),
            "value": float(row["Volume"]),
            "color": color + "80",  # Add transparency
        })
    return json.dumps(data)


def _prepare_line_data(series: pd.Series) -> str:
    """Convert Series to line JSON format."""
    data = []
    for idx, value in series.dropna().items():
        data.append({
            "time": int(idx.timestamp()),
            "value": float(value),
        })
    return json.dumps(data)


def _prepare_histogram_data(series: pd.Series) -> str:
    """Convert Series to histogram JSON format with colors based on positive/negative."""
    data = []
    for idx, value in series.dropna().items():
        color = HISTOGRAM_UP_COLOR if value >= 0 else HISTOGRAM_DOWN_COLOR
        data.append({
            "time": int(idx.timestamp()),
            "value": float(value),
            "color": color,
        })
    return json.dumps(data)


def build_chart(
    df: pd.DataFrame,
    sr_levels: Dict[str, List[float]],
    indicators: Set[str],
    ticker: str = "",
) -> str:
    """
    Build a TradingView Lightweight Charts HTML with selected indicators.

    Returns a self-contained HTML string that can be loaded in QWebEngineView.
    """

    # Prepare main candlestick data
    candle_data = _prepare_candlestick_data(df)
    volume_data = _prepare_volume_data(df)

    # Determine which indicators to show
    show_ma = "MA" in indicators
    show_bb = "BB" in indicators
    show_sr = "SR" in indicators
    show_rsi = "RSI" in indicators
    show_macd = "MACD" in indicators
    show_stoch = "STOCH" in indicators
    show_atr = "ATR" in indicators
    show_obv = "OBV" in indicators

    # Prepare indicator data
    indicator_series = {}

    if show_ma:
        for col in ["SMA20", "SMA50", "SMA200", "EMA9", "EMA20"]:
            if col in df.columns:
                indicator_series[col] = _prepare_line_data(df[col])

    if show_bb:
        for col in ["BB_upper", "BB_lower", "BB_mid"]:
            if col in df.columns:
                indicator_series[col] = _prepare_line_data(df[col])

    if show_rsi and "RSI" in df.columns:
        indicator_series["RSI"] = _prepare_line_data(df["RSI"])

    if show_macd:
        if "MACD" in df.columns:
            indicator_series["MACD"] = _prepare_line_data(df["MACD"])
        if "MACD_sig" in df.columns:
            indicator_series["MACD_sig"] = _prepare_line_data(df["MACD_sig"])
        if "MACD_hist" in df.columns:
            indicator_series["MACD_hist"] = _prepare_histogram_data(df["MACD_hist"])

    if show_stoch:
        if "STOCH_K" in df.columns:
            indicator_series["STOCH_K"] = _prepare_line_data(df["STOCH_K"])
        if "STOCH_D" in df.columns:
            indicator_series["STOCH_D"] = _prepare_line_data(df["STOCH_D"])

    if show_atr and "ATR" in df.columns:
        indicator_series["ATR"] = _prepare_line_data(df["ATR"])

    if show_obv and "OBV" in df.columns:
        indicator_series["OBV"] = _prepare_line_data(df["OBV"])

    # Prepare support/resistance levels
    current_price = float(df["Close"].iloc[-1])
    resistances = [r for r in sr_levels.get("resistance", []) if abs(r - current_price) / current_price < 0.6]
    supports = [s for s in sr_levels.get("support", []) if abs(s - current_price) / current_price < 0.6]

    # Build the HTML with embedded JavaScript
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Chart</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: {BG};
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        #mainChart {{
            flex: 1;
            position: relative;
        }}
        #indicators {{
            display: flex;
            flex-direction: column;
        }}
        .indicator-chart {{
            position: relative;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="mainChart"></div>
        <div id="indicators">
            {f'<div id="rsiChart" class="indicator-chart"></div>' if show_rsi else ''}
            {f'<div id="macdChart" class="indicator-chart"></div>' if show_macd else ''}
            {f'<div id="stochChart" class="indicator-chart"></div>' if show_stoch else ''}
            {f'<div id="atrChart" class="indicator-chart"></div>' if show_atr else ''}
            {f'<div id="obvChart" class="indicator-chart"></div>' if show_obv else ''}
        </div>
    </div>

    <script>
        // Chart options
        const chartOptions = {{
            layout: {{
                background: {{ color: '{BG}' }},
                textColor: '{TEXT_COLOR}',
            }},
            grid: {{
                vertLines: {{ color: '{GRID_COLOR}' }},
                horzLines: {{ color: '{GRID_COLOR}' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            timeScale: {{
                timeVisible: true,
                secondsVisible: false,
                borderColor: '{GRID_COLOR}',
            }},
            rightPriceScale: {{
                borderColor: '{GRID_COLOR}',
            }},
        }};

        // Calculate heights
        const indicatorCount = {sum([show_rsi, show_macd, show_stoch, show_atr, show_obv])};
        const mainHeight = indicatorCount === 0 ? window.innerHeight :
                          window.innerHeight * 0.70;  // 70% per il grafico principale
        const indicatorHeight = indicatorCount > 0 ?
                               Math.max(100, (window.innerHeight - mainHeight) / indicatorCount) : 0;  // Minimo 100px per indicatore

        // Main chart
        const mainContainer = document.getElementById('mainChart');
        mainContainer.style.height = mainHeight + 'px';
        const mainChart = LightweightCharts.createChart(mainContainer, {{
            ...chartOptions,
            width: mainContainer.clientWidth,
            height: mainHeight,
        }});

        // Candlestick series
        const candlestickSeries = mainChart.addCandlestickSeries({{
            upColor: '{UP_COLOR}',
            downColor: '{DOWN_COLOR}',
            borderUpColor: '{UP_COLOR}',
            borderDownColor: '{DOWN_COLOR}',
            wickUpColor: '{UP_COLOR}',
            wickDownColor: '{DOWN_COLOR}',
        }});
        candlestickSeries.setData({candle_data});

        // Volume series
        const volumeSeries = mainChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: 'volume',
        }});
        volumeSeries.setData({volume_data});
        mainChart.priceScale('volume').applyOptions({{
            scaleMargins: {{ top: 0.8, bottom: 0 }},
        }});
"""

    # Add moving averages
    if show_ma:
        if "SMA20" in indicator_series:
            html += f"""
        const sma20Series = mainChart.addLineSeries({{
            color: '{SMA20_COLOR}',
            lineWidth: 1.5,
            title: 'SMA 20',
        }});
        sma20Series.setData({indicator_series["SMA20"]});
"""
        if "SMA50" in indicator_series:
            html += f"""
        const sma50Series = mainChart.addLineSeries({{
            color: '{SMA50_COLOR}',
            lineWidth: 1.5,
            title: 'SMA 50',
        }});
        sma50Series.setData({indicator_series["SMA50"]});
"""
        if "SMA200" in indicator_series:
            html += f"""
        const sma200Series = mainChart.addLineSeries({{
            color: '{SMA200_COLOR}',
            lineWidth: 2,
            title: 'SMA 200',
        }});
        sma200Series.setData({indicator_series["SMA200"]});
"""
        if "EMA9" in indicator_series:
            html += f"""
        const ema9Series = mainChart.addLineSeries({{
            color: '{EMA9_COLOR}',
            lineWidth: 1.2,
            title: 'EMA 9',
        }});
        ema9Series.setData({indicator_series["EMA9"]});
"""
        if "EMA20" in indicator_series:
            html += f"""
        const ema20Series = mainChart.addLineSeries({{
            color: '{EMA20_COLOR}',
            lineWidth: 1.4,
            title: 'EMA 20',
        }});
        ema20Series.setData({indicator_series["EMA20"]});
"""

    # Add Bollinger Bands
    if show_bb:
        if "BB_upper" in indicator_series:
            html += f"""
        const bbUpperSeries = mainChart.addLineSeries({{
            color: '{BB_UPPER_COLOR}',
            lineWidth: 1,
            lineStyle: 2, // dashed
            title: 'BB Upper',
        }});
        bbUpperSeries.setData({indicator_series["BB_upper"]});
"""
        if "BB_lower" in indicator_series:
            html += f"""
        const bbLowerSeries = mainChart.addLineSeries({{
            color: '{BB_LOWER_COLOR}',
            lineWidth: 1,
            lineStyle: 2, // dashed
            title: 'BB Lower',
        }});
        bbLowerSeries.setData({indicator_series["BB_lower"]});
"""
        if "BB_mid" in indicator_series:
            html += f"""
        const bbMidSeries = mainChart.addLineSeries({{
            color: '{BB_MID_COLOR}',
            lineWidth: 0.8,
            title: 'BB Mid',
        }});
        bbMidSeries.setData({indicator_series["BB_mid"]});
"""

    # Add support/resistance levels
    if show_sr:
        for resistance in resistances[:5]:  # Limit to 5 most relevant
            html += f"""
        candlestickSeries.createPriceLine({{
            price: {resistance},
            color: '{RESISTANCE_COLOR}',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'R {resistance:.2f}',
        }});
"""
        for support in supports[:5]:  # Limit to 5 most relevant
            html += f"""
        candlestickSeries.createPriceLine({{
            price: {support},
            color: '{SUPPORT_COLOR}',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'S {support:.2f}',
        }});
"""

    # RSI Chart
    if show_rsi and "RSI" in indicator_series:
        html += f"""
        // RSI Chart
        const rsiContainer = document.getElementById('rsiChart');
        rsiContainer.style.height = indicatorHeight + 'px';
        const rsiChart = LightweightCharts.createChart(rsiContainer, {{
            ...chartOptions,
            width: rsiContainer.clientWidth,
            height: indicatorHeight,
        }});
        rsiChart.timeScale().applyOptions({{
            visible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
        }});

        const rsiSeries = rsiChart.addLineSeries({{
            color: '{RSI_COLOR}',
            lineWidth: 1.5,
        }});
        rsiSeries.setData({indicator_series["RSI"]});

        // Add reference lines
        rsiSeries.createPriceLine({{
            price: 70,
            color: '{DOWN_COLOR}60',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});
        rsiSeries.createPriceLine({{
            price: 30,
            color: '{UP_COLOR}60',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});
        rsiSeries.createPriceLine({{
            price: 50,
            color: '{GRID_COLOR}',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});

        rsiChart.priceScale().applyOptions({{
            scaleMargins: {{ top: 0.1, bottom: 0.1 }},
            autoScale: true,
        }});

        // Sync with main chart
        mainChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
            if (timeRange) {{
                rsiChart.timeScale().setVisibleRange(timeRange);
            }}
        }});
"""

    # MACD Chart
    if show_macd and ("MACD" in indicator_series or "MACD_hist" in indicator_series):
        html += f"""
        // MACD Chart
        const macdContainer = document.getElementById('macdChart');
        macdContainer.style.height = indicatorHeight + 'px';
        const macdChart = LightweightCharts.createChart(macdContainer, {{
            ...chartOptions,
            width: macdContainer.clientWidth,
            height: indicatorHeight,
        }});
        macdChart.timeScale().applyOptions({{
            visible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
        }});
"""
        if "MACD_hist" in indicator_series:
            html += f"""
        const macdHistSeries = macdChart.addHistogramSeries();
        macdHistSeries.setData({indicator_series["MACD_hist"]});
"""
        if "MACD" in indicator_series:
            html += f"""
        const macdLineSeries = macdChart.addLineSeries({{
            color: '{MACD_COLOR}',
            lineWidth: 1.3,
        }});
        macdLineSeries.setData({indicator_series["MACD"]});
"""
        if "MACD_sig" in indicator_series:
            html += f"""
        const macdSigSeries = macdChart.addLineSeries({{
            color: '{SIGNAL_COLOR}',
            lineWidth: 1.3,
        }});
        macdSigSeries.setData({indicator_series["MACD_sig"]});
"""
        # Add zero line to the first series added
        first_series = "macdHistSeries" if "MACD_hist" in indicator_series else ("macdLineSeries" if "MACD" in indicator_series else None)
        if first_series:
            html += f"""
        {first_series}.createPriceLine({{
            price: 0,
            color: '{GRID_COLOR}',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});

        macdChart.priceScale().applyOptions({{
            autoScale: true,
        }});

        mainChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
            if (timeRange) {{
                macdChart.timeScale().setVisibleRange(timeRange);
            }}
        }});
"""

    # Stochastic Chart
    if show_stoch and ("STOCH_K" in indicator_series or "STOCH_D" in indicator_series):
        html += f"""
        // Stochastic Chart
        const stochContainer = document.getElementById('stochChart');
        stochContainer.style.height = indicatorHeight + 'px';
        const stochChart = LightweightCharts.createChart(stochContainer, {{
            ...chartOptions,
            width: stochContainer.clientWidth,
            height: indicatorHeight,
        }});
        stochChart.timeScale().applyOptions({{
            visible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
        }});
"""
        if "STOCH_K" in indicator_series:
            html += f"""
        const stochKSeries = stochChart.addLineSeries({{
            color: '{STOCH_K_COLOR}',
            lineWidth: 1.3,
        }});
        stochKSeries.setData({indicator_series["STOCH_K"]});
"""
        if "STOCH_D" in indicator_series:
            html += f"""
        const stochDSeries = stochChart.addLineSeries({{
            color: '{STOCH_D_COLOR}',
            lineWidth: 1.3,
        }});
        stochDSeries.setData({indicator_series["STOCH_D"]});
"""
        # Add reference lines to the first series
        first_stoch_series = "stochKSeries" if "STOCH_K" in indicator_series else "stochDSeries"
        html += f"""
        {first_stoch_series}.createPriceLine({{
            price: 80,
            color: '{DOWN_COLOR}60',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});
        {first_stoch_series}.createPriceLine({{
            price: 20,
            color: '{UP_COLOR}60',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: false,
        }});

        stochChart.priceScale().applyOptions({{
            scaleMargins: {{ top: 0.1, bottom: 0.1 }},
            autoScale: true,
        }});

        mainChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
            if (timeRange) {{
                stochChart.timeScale().setVisibleRange(timeRange);
            }}
        }});
"""

    # ATR Chart
    if show_atr and "ATR" in indicator_series:
        html += f"""
        // ATR Chart
        const atrContainer = document.getElementById('atrChart');
        atrContainer.style.height = indicatorHeight + 'px';
        const atrChart = LightweightCharts.createChart(atrContainer, {{
            ...chartOptions,
            width: atrContainer.clientWidth,
            height: indicatorHeight,
        }});
        atrChart.timeScale().applyOptions({{
            visible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
        }});

        const atrSeries = atrChart.addLineSeries({{
            color: '{ATR_COLOR}',
            lineWidth: 1.3,
        }});
        atrSeries.setData({indicator_series["ATR"]});

        atrChart.priceScale().applyOptions({{
            autoScale: true,
        }});

        mainChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
            if (timeRange) {{
                atrChart.timeScale().setVisibleRange(timeRange);
            }}
        }});
"""

    # OBV Chart
    if show_obv and "OBV" in indicator_series:
        html += f"""
        // OBV Chart
        const obvContainer = document.getElementById('obvChart');
        obvContainer.style.height = indicatorHeight + 'px';
        const obvChart = LightweightCharts.createChart(obvContainer, {{
            ...chartOptions,
            width: obvContainer.clientWidth,
            height: indicatorHeight,
        }});
        obvChart.timeScale().applyOptions({{
            visible: true,
            fixLeftEdge: true,
            fixRightEdge: true,
        }});

        const obvSeries = obvChart.addLineSeries({{
            color: '{OBV_COLOR}',
            lineWidth: 1.2,
        }});
        obvSeries.setData({indicator_series["OBV"]});

        obvChart.priceScale().applyOptions({{
            autoScale: true,
        }});

        mainChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
            if (timeRange) {{
                obvChart.timeScale().setVisibleRange(timeRange);
            }}
        }});
"""

    # Add resize handler and fit content
    html += """
        // Handle window resize
        window.addEventListener('resize', () => {
            const newMainHeight = indicatorCount === 0 ? window.innerHeight :
                                 window.innerHeight * 0.70;  // 70% per il grafico principale
            const newIndicatorHeight = indicatorCount > 0 ?
                                      Math.max(100, (window.innerHeight - newMainHeight) / indicatorCount) : 0;  // Minimo 100px

            mainChart.applyOptions({
                width: mainContainer.clientWidth,
                height: newMainHeight,
            });
"""

    if show_rsi:
        html += """
            rsiChart.applyOptions({
                width: rsiContainer.clientWidth,
                height: newIndicatorHeight,
            });
"""
    if show_macd:
        html += """
            macdChart.applyOptions({
                width: macdContainer.clientWidth,
                height: newIndicatorHeight,
            });
"""
    if show_stoch:
        html += """
            stochChart.applyOptions({
                width: stochContainer.clientWidth,
                height: newIndicatorHeight,
            });
"""
    if show_atr:
        html += """
            atrChart.applyOptions({
                width: atrContainer.clientWidth,
                height: newIndicatorHeight,
            });
"""
    if show_obv:
        html += """
            obvChart.applyOptions({
                width: obvContainer.clientWidth,
                height: newIndicatorHeight,
            });
"""

    html += """
        });

        // Fit content to visible range after a short delay to ensure charts are rendered
        setTimeout(() => {
            mainChart.timeScale().fitContent();
"""

    # Add fitContent for all indicator charts
    if show_rsi:
        html += """
            rsiChart.timeScale().fitContent();
"""
    if show_macd:
        html += """
            macdChart.timeScale().fitContent();
"""
    if show_stoch:
        html += """
            stochChart.timeScale().fitContent();
"""
    if show_atr:
        html += """
            atrChart.timeScale().fitContent();
"""
    if show_obv:
        html += """
            obvChart.timeScale().fitContent();
"""

    html += """
        }, 100);
    </script>
</body>
</html>
"""

    return html

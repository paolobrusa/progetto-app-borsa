"""Main application window."""

import os
import tempfile

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QCheckBox, QListWidget, QListWidgetItem,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QFrame,
    QHeaderView, QSizePolicy, QScrollArea, QMessageBox,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QFont, QColor

import pandas as pd

from data import (
    search_tickers, get_price_data, get_fundamentals,
    get_dividends, get_financials, get_analyst_data,
)
from analysis import enrich_dataframe, find_support_resistance
from chart_lightweight import build_chart
from analysis_engine import generate_analysis, build_analysis_html
from ai_analyst import (
    get_api_key, save_api_key,
    build_prompt, call_gemini, response_to_html,
    render_conversation_html, call_gemini_followup, build_followup_context,
)


# ── Background workers ───────────────────────────────────────────────────────

class SearchWorker(QThread):
    results_ready = pyqtSignal(list)
    def __init__(self, query): super().__init__(); self.query = query
    def run(self): self.results_ready.emit(search_tickers(self.query))


class DataWorker(QThread):
    data_ready = pyqtSignal(object, object, dict)
    error      = pyqtSignal(str)
    def __init__(self, ticker, period):
        super().__init__()
        self.ticker = ticker
        self.period = period
    def run(self):
        df = get_price_data(self.ticker, self.period)
        if df is None or df.empty:
            self.error.emit(f"Nessun dato disponibile per {self.ticker}")
            return
        df  = enrich_dataframe(df)
        sr  = find_support_resistance(df)
        fun = get_fundamentals(self.ticker)
        self.data_ready.emit(df, sr, fun)


class FundWorker(QThread):
    ready = pyqtSignal(dict, dict, dict)
    def __init__(self, ticker): super().__init__(); self.ticker = ticker
    def run(self):
        self.ready.emit(
            get_dividends(self.ticker),
            get_financials(self.ticker),
            get_analyst_data(self.ticker),
        )


class AIWorker(QThread):
    """Background thread for both initial analysis and follow-up questions."""
    # (raw_text, followup_question)  — question is "" for the initial analysis
    finished = pyqtSignal(str, str)
    error    = pyqtSignal(str)

    def __init__(self, ticker, fundamentals, df, sr, analysts, period, api_key,
                 followup_question: str = "", api_history: list | None = None):
        super().__init__()
        self.ticker            = ticker
        self.fundamentals      = fundamentals
        self.df                = df
        self.sr                = sr
        self.analysts          = analysts
        self.period            = period
        self.api_key           = api_key
        self.followup_question = followup_question
        self.api_history       = api_history or []

    def run(self):
        try:
            if self.followup_question:
                # Follow-up: compact history + new question (token-efficient)
                history = self.api_history + [
                    {"role": "user", "text": self.followup_question}
                ]
                raw_text = call_gemini_followup(history, self.api_key)
                self.finished.emit(raw_text, self.followup_question)
            else:
                # Initial analysis: full prompt
                prompt   = build_prompt(
                    self.ticker, self.fundamentals,
                    self.df, self.sr, self.analysts, self.period,
                )
                raw_text = call_gemini(prompt, self.api_key)
                self.finished.emit(raw_text, "")
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ──────────────────────────────────────────────────────────────

_PERIODS = ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"]

_AI_BTN_LABEL   = "Genera Analisi AI"
_AI_BTN_LOADING = "Analisi in corso…"

_INDICATORS = [
    ("MA",    "MA/EMA"),
    ("BB",    "Bollinger"),
    ("SR",    "Sup/Res"),
    ("RSI",   "RSI"),
    ("MACD",  "MACD"),
    ("STOCH", "Stocastico"),
    ("ATR",   "ATR"),
    ("OBV",   "OBV"),
]

_WELCOME_HTML = """<!DOCTYPE html>
<html><body style="margin:0;background:#0f0f1a;display:flex;align-items:center;
justify-content:center;height:100vh;flex-direction:column;
font-family:'Segoe UI',Arial;color:#8888aa;">
<div style="font-size:52px;margin-bottom:18px;">&#128200;</div>
<div style="font-size:24px;color:#d0d0e8;margin-bottom:10px;font-weight:600;">Stock Analyzer Pro</div>
<div style="font-size:14px;">Cerca un ticker o il nome di un&apos;azienda per iniziare l&apos;analisi</div>
<div style="font-size:12px;margin-top:10px;opacity:0.55;">
  Esempi&nbsp;&nbsp;|&nbsp;&nbsp;NVDA &nbsp;&#8226;&nbsp; AAPL &nbsp;&#8226;&nbsp; BTC-USD &nbsp;&#8226;&nbsp; ENI.MI &nbsp;&#8226;&nbsp; EURUSD=X &nbsp;&#8226;&nbsp; SPY
</div>
</body></html>"""

_AI_WELCOME_HTML = """<!DOCTYPE html>
<html><body style="margin:0;background:#0f0f1a;display:flex;align-items:center;
justify-content:center;height:100%;flex-direction:column;
font-family:'Segoe UI',Arial;color:#6060a0;text-align:center;padding:30px;">
<div style="font-size:48px;margin-bottom:16px;">🤖</div>
<div style="font-size:18px;color:#9090c8;margin-bottom:10px;font-weight:600;">AI Analyst — Google Gemini</div>
<div style="font-size:13px;color:#5050808;max-width:460px;line-height:1.7;">
  Carica un ticker, inserisci la tua <b style="color:#7080b0;">Gemini API key</b>
  (gratuita su <a href="#" style="color:#4090ff;">aistudio.google.com</a>)
  e premi <b style="color:#80c8ff;">Genera Analisi AI</b> per ottenere un parere
  completo con notizie in tempo reale, analisi tecnica/fondamentale
  e strategie operative per breve, medio e lungo termine.
</div>
<div style="font-size:11px;margin-top:20px;opacity:0.45;">
  Powered by Gemini 2.0 Flash &nbsp;·&nbsp; Google Search Grounding &nbsp;·&nbsp; ~1500 analisi/giorno gratuite
</div>
</body></html>"""

_AI_LOADING_HTML = """<!DOCTYPE html>
<html><body style="margin:0;background:#0f0f1a;display:flex;align-items:center;
justify-content:center;height:100%;flex-direction:column;
font-family:'Segoe UI',Arial;color:#6868a0;text-align:center;">
<div style="font-size:40px;margin-bottom:20px;animation:spin 1.5s linear infinite;">⚙️</div>
<div style="font-size:16px;color:#a0a8d8;margin-bottom:8px;">Analisi in corso…</div>
<div style="font-size:12px;color:#505080;">Gemini sta cercando notizie aggiornate e analizzando i dati.</div>
<div style="font-size:11px;margin-top:6px;opacity:0.5;">Può richiedere 15–30 secondi.</div>
<style>@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}</style>
</body></html>"""


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet("color:#1e1e30;")
    return f


def _fmt(val, suffix="", pct=False, billions=False, decimals=2) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "–"
    if pct:
        return f"{val * 100:.2f}%"
    if billions:
        if abs(val) >= 1e12: return f"{val/1e12:.2f} T"
        if abs(val) >= 1e9:  return f"{val/1e9:.2f} B"
        if abs(val) >= 1e6:  return f"{val/1e6:.2f} M"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return f"{val}{suffix}"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.current_ticker: str | None = None
        self.current_period: str = "3M"
        self.current_df: pd.DataFrame | None = None
        self.current_sr: dict | None = None
        self.current_fundamentals: dict = {}
        self.current_analysts: dict = {}
        self._chart_tmp: str | None = None

        self._search_worker: SearchWorker | None = None
        self._data_worker:   DataWorker   | None = None
        self._fund_worker:   FundWorker   | None = None
        self._ai_worker:     AIWorker     | None = None

        # AI conversation state
        # _ai_api_history: compact history sent to Gemini on follow-ups
        # _ai_display:     items shown in the view  (role, text)
        self._ai_api_history: list = []
        self._ai_display:     list = []

        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._do_search)

        self.setWindowTitle("Stock Analyzer Pro")
        self.setMinimumSize(1100, 720)
        self.resize(1450, 920)

        self._build_ui()
        self._apply_styles()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 8, 10, 6)
        lay.setSpacing(5)

        # Top bar
        lay.addWidget(self._build_topbar())

        # Search dropdown (hidden until results arrive)
        self.search_list = QListWidget()
        self.search_list.setMaximumHeight(220)
        self.search_list.setHidden(True)
        self.search_list.itemClicked.connect(self._on_result_clicked)
        lay.addWidget(self.search_list)

        # Controls (period + indicators)
        lay.addWidget(self._build_controls())

        # Chart + fundamental splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(320)
        # Allow local files to load remote resources (needed for any CDN assets)
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        self.web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.web_view.setHtml(_WELCOME_HTML)
        splitter.addWidget(self.web_view)

        splitter.addWidget(self._build_fund_panel())
        splitter.setSizes([620, 210])
        lay.addWidget(splitter)

        # Status bar
        self.status_lbl = QLabel("Pronto. Cerca un ticker per iniziare.")
        self.status_lbl.setStyleSheet("color:#6060a0;font-size:10px;padding:1px 2px;")
        lay.addWidget(self.status_lbl)

    def _build_topbar(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(48)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Cerca ticker o azienda...  (es. nvidia, BTC-USD, ENI.MI, EURUSD=X, SPY)"
        )
        self.search_input.setFixedHeight(42)
        self.search_input.textChanged.connect(self._on_text_changed)
        self.search_input.returnPressed.connect(self._do_search)

        self.ticker_lbl = QLabel("–")
        self.ticker_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.ticker_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.ticker_lbl.setMinimumWidth(220)

        self.price_lbl = QLabel("")
        self.price_lbl.setFont(QFont("Segoe UI", 13))
        self.price_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        h.addWidget(self.search_input, 4)
        h.addWidget(self.ticker_lbl, 3)
        h.addWidget(self.price_lbl, 2)
        h.addStretch()
        return w

    def _build_controls(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(40)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        h.addWidget(QLabel("Periodo:"))
        self.period_btns: dict[str, QPushButton] = {}
        for p in _PERIODS:
            btn = QPushButton(p)
            btn.setFixedSize(40, 28)
            btn.setCheckable(True)
            btn.setChecked(p == self.current_period)
            btn.clicked.connect(lambda _, pp=p: self._on_period(pp))
            self.period_btns[p] = btn
            h.addWidget(btn)

        h.addWidget(_sep())
        h.addWidget(QLabel("Indicatori:"))

        self.ind_checks: dict[str, QCheckBox] = {}
        default_on = {"MA", "BB", "SR", "RSI", "MACD"}
        for key, label in _INDICATORS:
            chk = QCheckBox(label)
            chk.setChecked(key in default_on)
            chk.stateChanged.connect(self._on_indicator_changed)
            self.ind_checks[key] = chk
            h.addWidget(chk)

        h.addStretch()
        return w

    # ── Fundamental panel ────────────────────────────────────────────────────

    def _build_fund_panel(self) -> QTabWidget:
        self.fund_tabs = QTabWidget()
        self.fund_tabs.setMinimumHeight(160)
        self.fund_tabs.addTab(self._build_overview_tab(),   "Panoramica")
        self.fund_tabs.addTab(self._build_financials_tab(), "Bilancio")
        self.fund_tabs.addTab(self._build_dividends_tab(),  "Dividendi")
        self.fund_tabs.addTab(self._build_analysts_tab(),   "Analisti")
        self.fund_tabs.addTab(self._build_analysis_tab(),   "Analisi e Parere")
        self.fund_tabs.addTab(self._build_ai_tab(),         "AI Analyst")
        return self.fund_tabs

    def _build_overview_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(0)
        self.ov_vals: dict[str, QLabel] = {}
        fields = [
            ("Settore",     "sector"),   ("Industria",  "industry"),
            ("Paese",       "country"),  ("Market Cap", "market_cap"),
            ("P/E trail.",  "pe_ratio"), ("P/E fwd.",   "forward_pe"),
            ("EPS",         "eps"),      ("PEG",        "peg_ratio"),
            ("Beta",        "beta"),     ("52W High",   "52w_high"),
            ("52W Low",     "52w_low"),  ("Dipendenti", "employees"),
        ]
        for i, (lbl, key) in enumerate(fields):
            col = QWidget()
            cv  = QVBoxLayout(col)
            cv.setContentsMargins(10, 4, 10, 4)
            cv.setSpacing(1)
            lb = QLabel(lbl)
            lb.setStyleSheet("color:#6868a0;font-size:9px;")
            vl = QLabel("–")
            vl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            vl.setWordWrap(True)
            self.ov_vals[key] = vl
            cv.addWidget(lb)
            cv.addWidget(vl)
            h.addWidget(col)
            if i < len(fields) - 1:
                h.addWidget(_sep())
        h.addStretch()
        return w

    def _build_financials_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.fin_table = _make_table(["Voce", "Anno 0", "Anno -1", "Anno -2", "Anno -3"])
        v.addWidget(self.fin_table)
        return w

    def _build_dividends_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(16)
        self.div_vals: dict[str, QLabel] = {}
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        for lbl, key in [("Rendimento (%)", "yield"), ("Dividendo/Anno", "rate"),
                          ("Payout Ratio", "payout_ratio"), ("Ex-Date", "ex_date")]:
            l = QLabel(lbl); l.setStyleSheet("color:#6868a0;font-size:9px;")
            v = QLabel("–"); v.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            self.div_vals[key] = v
            lv.addWidget(l); lv.addWidget(v)
        lv.addStretch()
        h.addWidget(left)
        self.div_table = _make_table(["Data", "Importo"])
        h.addWidget(self.div_table)
        return w

    def _build_analysts_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(16)
        self.an_vals: dict[str, QLabel] = {}
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        for lbl, key in [("Consenso", "recommendation"), ("Target Medio", "target_mean"),
                          ("Target Max", "target_high"), ("Target Min", "target_low"),
                          ("N° Analisti", "num_analysts")]:
            l = QLabel(lbl); l.setStyleSheet("color:#6868a0;font-size:9px;")
            v = QLabel("–"); v.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            self.an_vals[key] = v
            lv.addWidget(l); lv.addWidget(v)
        lv.addStretch()
        h.addWidget(left)
        self.an_table = _make_table(["Data", "Società", "Rating", "Azione"])
        h.addWidget(self.an_table)
        return w

    def _build_analysis_tab(self) -> QWidget:
        from PyQt6.QtWidgets import QTextBrowser
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.analysis_browser = QTextBrowser()
        self.analysis_browser.setOpenExternalLinks(False)
        self.analysis_browser.setStyleSheet(
            "QTextBrowser { background:#0f0f1a; border:none; color:#d0d0e8; }"
        )
        self.analysis_browser.setHtml(
            "<body style='background:#0f0f1a;color:#505070;font-family:Segoe UI;padding:30px;'>"
            "Carica un ticker per vedere l'analisi completa.</body>"
        )
        v.addWidget(self.analysis_browser)
        return w

    def _build_ai_tab(self) -> QWidget:
        """Tab with Gemini AI analysis: API key input + generate button + WebView output."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(6)

        # ── Top bar ──────────────────────────────────────────────────────────
        bar = QWidget()
        bar.setFixedHeight(38)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        # Generate button
        self.ai_btn = QPushButton(_AI_BTN_LABEL)
        self.ai_btn.setFixedHeight(32)
        self.ai_btn.setMinimumWidth(180)
        self.ai_btn.setStyleSheet(
            "QPushButton { background:#1a3060; border:1px solid #4090ff; border-radius:5px;"
            " color:#80c8ff; font-size:12px; font-weight:600; }"
            "QPushButton:hover { background:#203a70; border-color:#60b0ff; }"
            "QPushButton:disabled { background:#111128; border-color:#2a2a50; color:#404060; }"
        )
        self.ai_btn.clicked.connect(self._request_ai_analysis)
        h.addWidget(self.ai_btn)

        h.addWidget(_sep())

        # API key label + input
        lbl_key = QLabel("Gemini API key:")
        lbl_key.setStyleSheet("color:#6868a0; font-size:10px;")
        lbl_key.setFixedWidth(90)
        h.addWidget(lbl_key)

        self.ai_key_input = QLineEdit()
        self.ai_key_input.setPlaceholderText("Inserisci la tua API key per Gemini (da aistudio.google.com)")
        self.ai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key_input.setFixedHeight(28)
        self.ai_key_input.setMinimumWidth(260)
        # Pre-fill if already saved
        saved_key = get_api_key()
        if saved_key:
            self.ai_key_input.setText(saved_key)
        h.addWidget(self.ai_key_input, 1)

        # Toggle show/hide password
        self.ai_key_eye = QPushButton("👁")
        self.ai_key_eye.setFixedSize(28, 28)
        self.ai_key_eye.setToolTip("Mostra / Nascondi chiave")
        self.ai_key_eye.setCheckable(True)
        self.ai_key_eye.toggled.connect(
            lambda on: self.ai_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        h.addWidget(self.ai_key_eye)

        # Save key button
        btn_save = QPushButton("💾 Salva")
        btn_save.setFixedHeight(28)
        btn_save.setToolTip("Salva la chiave nel file .env (non la ricordare più)")
        btn_save.clicked.connect(self._save_ai_key)
        h.addWidget(btn_save)

        h.addStretch()
        v.addWidget(bar)

        # ── Status label ─────────────────────────────────────────────────────
        self.ai_status = QLabel(
            "Inserisci la tua Gemini API key e premi il pulsante dopo aver caricato un ticker."
        )
        self.ai_status.setStyleSheet("color:#5858a0; font-size:10px; padding:0 2px;")
        self.ai_status.setWordWrap(True)
        v.addWidget(self.ai_status)

        # ── WebEngineView for rendered response ───────────────────────────────
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        self.ai_view = QWebEngineView()
        self.ai_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.ai_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.ai_view.setHtml(_AI_WELCOME_HTML)
        v.addWidget(self.ai_view, stretch=1)

        # ── Follow-up input bar ───────────────────────────────────────────────
        fu_bar = QWidget()
        fu_bar.setFixedHeight(38)
        hf = QHBoxLayout(fu_bar)
        hf.setContentsMargins(0, 4, 0, 0)
        hf.setSpacing(6)

        self.ai_followup_input = QLineEdit()
        self.ai_followup_input.setPlaceholderText(
            "Genera prima un'analisi, poi scrivi qui le tue domande di approfondimento…"
        )
        self.ai_followup_input.setFixedHeight(30)
        self.ai_followup_input.setEnabled(False)
        self.ai_followup_input.returnPressed.connect(self._request_ai_followup)
        hf.addWidget(self.ai_followup_input, 1)

        self.ai_followup_btn = QPushButton("Invia")
        self.ai_followup_btn.setFixedSize(64, 30)
        self.ai_followup_btn.setEnabled(False)
        self.ai_followup_btn.clicked.connect(self._request_ai_followup)
        hf.addWidget(self.ai_followup_btn)

        v.addWidget(fu_bar)

        return w

    # ── Styles ───────────────────────────────────────────────────────────────

    def _apply_styles(self):
        self.setStyleSheet("""
            * { font-family: 'Segoe UI', Arial; }
            QMainWindow, QWidget { background: #0f0f1a; color: #d0d0e8; }
            QLineEdit {
                background: #1a1a2e; border: 1px solid #2e2e50;
                border-radius: 7px; padding: 4px 12px;
                color: #d0d0e8; font-size: 13px;
            }
            QLineEdit:focus { border-color: #4090ff; }
            QListWidget {
                background: #14142a; border: 1px solid #2e2e50;
                border-radius: 6px;
            }
            QListWidget::item { padding: 7px 12px; border-bottom: 1px solid #1e1e38; }
            QListWidget::item:hover    { background: #1e2a4a; }
            QListWidget::item:selected { background: #1a3a7a; color: white; }
            QPushButton {
                background: #1a1a2e; border: 1px solid #2e2e50;
                border-radius: 4px; color: #d0d0e8; font-size: 10px;
            }
            QPushButton:hover   { background: #1e2a4e; border-color: #4090ff; }
            QPushButton:checked { background: #1a3a7a; border-color: #4090ff; color: #fff; }
            QCheckBox { color: #b0b0d0; spacing: 5px; font-size: 11px; }
            QCheckBox::indicator {
                width: 13px; height: 13px;
                border: 1px solid #3030558; border-radius: 3px;
                background: #1a1a2e;
            }
            QCheckBox::indicator:checked  { background: #4090ff; border-color: #4090ff; }
            QTabWidget::pane { border: 1px solid #1e1e38; background: #0f0f1a; }
            QTabBar::tab {
                background: #14142a; border: 1px solid #1e1e38;
                border-bottom: none; padding: 4px 14px;
                color: #7070a0; margin-right: 2px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #1a1a30; color: #d0d0e8; border-color: #2e2e50; }
            QTableWidget {
                background: #121220; alternate-background-color: #161628;
                border: none; gridline-color: #1e1e38; font-size: 11px;
            }
            QHeaderView::section {
                background: #1a1a30; color: #9090c0;
                padding: 4px 6px; border: 1px solid #1e1e38; font-size: 10px;
            }
            QScrollBar:vertical {
                background: #12122a; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical { background: #3030508; border-radius: 4px; }
            QLabel { color: #d0d0e8; }
            QSplitter::handle { background: #1e1e38; }
        """)

    # ── Search ───────────────────────────────────────────────────────────────

    def _on_text_changed(self, text: str):
        if len(text.strip()) >= 1:
            self._search_timer.start(380)
        else:
            self.search_list.setHidden(True)

    def _do_search(self):
        q = self.search_input.text().strip()
        if not q:
            return
        self._set_status(f"Cerco '{q}'…")
        if self._search_worker and self._search_worker.isRunning():
            self._search_worker.terminate()
        self._search_worker = SearchWorker(q)
        self._search_worker.results_ready.connect(self._on_search_results)
        self._search_worker.start()

    def _on_search_results(self, results: list):
        self.search_list.clear()
        if not results:
            self._set_status("Nessun risultato trovato.")
            self.search_list.setHidden(True)
            return
        type_icon = {
            "EQUITY": "📈", "ETF": "🗂️", "CRYPTOCURRENCY": "₿",
            "CURRENCY": "💱", "INDEX": "📊", "FUTURE": "⏳",
        }
        for r in results:
            icon = type_icon.get(r["type"], "•")
            text = f"  {icon}  {r['symbol']}   —   {r['name']}   [{r['exchange']}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r["symbol"])
            self.search_list.addItem(item)
        self.search_list.setHidden(False)
        self._set_status(f"{len(results)} risultati trovati.")

    def _on_result_clicked(self, item: QListWidgetItem):
        ticker = item.data(Qt.ItemDataRole.UserRole)
        self.search_list.setHidden(True)
        self.load_ticker(ticker)

    # ── Load ticker ──────────────────────────────────────────────────────────

    def load_ticker(self, ticker: str):
        self.current_ticker = ticker
        self.ticker_lbl.setText(f"⏳ {ticker}")
        self.price_lbl.setText("")
        self._set_status(f"Scarico dati per {ticker}…")
        self._set_controls_enabled(False)

        # Reset AI conversation for the new ticker
        self._ai_api_history = []
        self._ai_display     = []
        self.ai_view.setHtml(_AI_WELCOME_HTML)
        self.ai_followup_input.setEnabled(False)
        self.ai_followup_input.clear()
        self.ai_followup_input.setPlaceholderText(
            "Genera prima un'analisi, poi scrivi qui le tue domande di approfondimento…"
        )
        self.ai_followup_btn.setEnabled(False)
        self.ai_status.setText(
            "Inserisci la tua Gemini API key e premi il pulsante dopo aver caricato un ticker."
        )

        if self._data_worker and self._data_worker.isRunning():
            self._data_worker.terminate()

        self._data_worker = DataWorker(ticker, self.current_period)
        self._data_worker.data_ready.connect(self._on_data_ready)
        self._data_worker.error.connect(self._on_data_error)
        self._data_worker.start()

    def _on_data_ready(self, df: pd.DataFrame, sr: dict, fun: dict):
        self.current_df = df
        self.current_sr = sr
        self._set_controls_enabled(True)

        name = fun.get("name", self.current_ticker)
        cur  = fun.get("currency", "")
        price = fun.get("price")
        chg   = fun.get("change_pct")

        self.ticker_lbl.setText(f"{self.current_ticker} — {name}")
        if price:
            chg_str = ""
            if chg is not None:
                sign = "+" if chg >= 0 else ""
                color = "#26c6a1" if chg >= 0 else "#f44336"
                chg_str = f'  <span style="color:{color};font-size:11px;">{sign}{chg:.2f}%</span>'
            self.price_lbl.setText(f'<b>{cur} {price:,.2f}</b>{chg_str}')
        else:
            self.price_lbl.setText("")
        self.price_lbl.setTextFormat(Qt.TextFormat.RichText)

        n = len(df)
        last = df.index[-1].strftime("%d/%m/%Y %H:%M") if n else "–"
        self._set_status(f"{n} candele  ·  ultimo aggiornamento: {last}  ·  {cur}")

        self.current_fundamentals = fun
        self._fill_overview(fun)
        self._refresh_chart()
        self._refresh_analysis()

        # Fetch extra fundamentals in background
        if self._fund_worker and self._fund_worker.isRunning():
            self._fund_worker.terminate()
        self._fund_worker = FundWorker(self.current_ticker)
        self._fund_worker.ready.connect(self._on_fund_ready)
        self._fund_worker.start()

    def _on_data_error(self, msg: str):
        self._set_controls_enabled(True)
        self.ticker_lbl.setText("Errore")
        self._set_status(f"⚠  {msg}")

    def _on_fund_ready(self, divs: dict, fins: dict, anas: dict):
        self.current_analysts = anas
        self._fill_dividends(divs)
        self._fill_financials(fins)
        self._fill_analysts(anas)
        self._refresh_analysis()   # update with analyst data now available

    # ── Chart ────────────────────────────────────────────────────────────────

    def _refresh_chart(self):
        if self.current_df is None:
            return
        active = {k for k, chk in self.ind_checks.items() if chk.isChecked()}
        html = build_chart(self.current_df, self.current_sr or {}, active, self.current_ticker or "")
        # Write to temp file to avoid setHtml 2 MB limit & enable CDN loads
        if self._chart_tmp:
            try: os.unlink(self._chart_tmp)
            except Exception: pass
        fd, path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        self._chart_tmp = path
        self.web_view.load(QUrl.fromLocalFile(path))

    # ── Fundamental fills ────────────────────────────────────────────────────

    def _fill_overview(self, info: dict):
        self.ov_vals["sector"].setText(str(info.get("sector") or "–"))
        self.ov_vals["industry"].setText(str(info.get("industry") or "–"))
        self.ov_vals["country"].setText(str(info.get("country") or "–"))
        self.ov_vals["market_cap"].setText(_fmt(info.get("market_cap"), billions=True))
        self.ov_vals["pe_ratio"].setText(_fmt(info.get("pe_ratio")))
        self.ov_vals["forward_pe"].setText(_fmt(info.get("forward_pe")))
        self.ov_vals["eps"].setText(_fmt(info.get("eps")))
        self.ov_vals["peg_ratio"].setText(_fmt(info.get("peg_ratio")))
        self.ov_vals["beta"].setText(_fmt(info.get("beta")))
        self.ov_vals["52w_high"].setText(_fmt(info.get("52w_high")))
        self.ov_vals["52w_low"].setText(_fmt(info.get("52w_low")))
        self.ov_vals["employees"].setText(
            f"{info['employees']:,}" if info.get("employees") else "–"
        )

    def _fill_dividends(self, data: dict):
        self.div_vals["yield"].setText(_fmt(data.get("yield"), pct=True))
        self.div_vals["rate"].setText(_fmt(data.get("rate"), decimals=4))
        self.div_vals["payout_ratio"].setText(_fmt(data.get("payout_ratio"), pct=True))
        self.div_vals["ex_date"].setText(str(data.get("ex_date") or "–"))
        hist = data.get("history", pd.Series(dtype=float))
        self.div_table.setRowCount(0)
        if not hist.empty:
            for date, amount in hist.items():
                r = self.div_table.rowCount()
                self.div_table.insertRow(r)
                self.div_table.setItem(r, 0, _cell(str(date)[:10]))
                self.div_table.setItem(r, 1, _cell(f"{amount:.4f}"))

    def _fill_financials(self, data: dict):
        income = data.get("income", pd.DataFrame())
        self.fin_table.setRowCount(0)
        if income is None or income.empty:
            return
        cols = list(income.columns[:4])
        headers = ["Voce"] + [str(c)[:10] for c in cols]
        self.fin_table.setColumnCount(len(headers))
        self.fin_table.setHorizontalHeaderLabels(headers)
        rows_map = [
            ("Total Revenue",    "Fatturato"),
            ("Gross Profit",     "Utile Lordo"),
            ("Operating Income", "EBIT"),
            ("Net Income",       "Utile Netto"),
            ("EBITDA",           "EBITDA"),
            ("Research And Development", "R&D"),
        ]
        for idx_name, display in rows_map:
            if idx_name not in income.index:
                continue
            r = self.fin_table.rowCount()
            self.fin_table.insertRow(r)
            self.fin_table.setItem(r, 0, _cell(display))
            for j, col in enumerate(cols):
                val = income.loc[idx_name, col]
                self.fin_table.setItem(r, j + 1, _cell(_fmt(val, billions=True) if pd.notna(val) else "–"))

    def _fill_analysts(self, data: dict):
        rec = data.get("recommendation", "N/A")
        self.an_vals["recommendation"].setText(rec)
        color = {
            "BUY": "#26c6a1", "STRONG_BUY": "#00e676",
            "SELL": "#f44336", "STRONG_SELL": "#b71c1c",
            "HOLD": "#FFB300", "NEUTRAL": "#FFB300",
        }.get(rec, "#d0d0e8")
        self.an_vals["recommendation"].setStyleSheet(
            f"color:{color};font-size:15px;font-weight:700;"
        )
        self.an_vals["target_mean"].setText(_fmt(data.get("target_mean")))
        self.an_vals["target_high"].setText(_fmt(data.get("target_high")))
        self.an_vals["target_low"].setText(_fmt(data.get("target_low")))
        self.an_vals["num_analysts"].setText(str(data.get("num_analysts") or "–"))

        hist: pd.DataFrame = data.get("history", pd.DataFrame())
        self.an_table.setRowCount(0)
        if not hist.empty:
            for idx, row in hist.iterrows():
                r = self.an_table.rowCount()
                self.an_table.insertRow(r)
                date_str = str(idx)[:10] if not isinstance(idx, int) else "–"
                self.an_table.setItem(r, 0, _cell(date_str))
                self.an_table.setItem(r, 1, _cell(str(row.get("Firm", "–"))))
                self.an_table.setItem(r, 2, _cell(str(row.get("ToGrade", row.get("To Grade", "–")))))
                self.an_table.setItem(r, 3, _cell(str(row.get("Action", "–"))))

    # ── AI Analyst ───────────────────────────────────────────────────────────

    def _save_ai_key(self):
        """Persist the API key typed in the input field."""
        key = self.ai_key_input.text().strip()
        if not key:
            self.ai_status.setText("⚠️  Inserisci una API key prima di salvarla.")
            return
        save_api_key(key)
        self.ai_status.setText("✅  API key salvata nel file .env — non dovrai reinserirla.")

    def _request_ai_analysis(self):
        """Validate state, then launch AIWorker in background."""
        if self.current_df is None or self.current_ticker is None:
            self.ai_status.setText("⚠️  Carica prima un ticker dalla barra di ricerca.")
            return

        api_key = self.ai_key_input.text().strip()
        if not api_key:
            self.ai_status.setText(
                "⚠️  Inserisci la Gemini API key (ottienila gratis su aistudio.google.com)."
            )
            return

        # Stop any running worker
        if self._ai_worker and self._ai_worker.isRunning():
            self._ai_worker.terminate()
            self._ai_worker.wait()

        # Show loading state
        self.ai_btn.setEnabled(False)
        self.ai_btn.setText(_AI_BTN_LOADING)
        self.ai_status.setText(
            f"🔍  Gemini sta ricercando notizie su {self.current_ticker} "
            f"e analizzando i dati — attendere 15–30 secondi…"
        )
        self.ai_view.setHtml(_AI_LOADING_HTML)

        # Switch to AI tab so user sees the spinner
        self.fund_tabs.setCurrentWidget(self.fund_tabs.widget(5))

        # Reset conversation state for fresh analysis
        self._ai_api_history = []
        self._ai_display     = []

        self._ai_worker = AIWorker(
            ticker       = self.current_ticker,
            fundamentals = self.current_fundamentals,
            df           = self.current_df,
            sr           = self.current_sr or {},
            analysts     = self.current_analysts,
            period       = self.current_period,
            api_key      = api_key,
        )
        self._ai_worker.finished.connect(self._on_ai_ready)
        self._ai_worker.error.connect(self._on_ai_error)
        self._ai_worker.start()

    def _on_ai_ready(self, raw_text: str, question: str):
        """
        Unified handler for both initial analysis and follow-up responses.
        question is "" for the initial analysis, non-empty for follow-ups.
        """
        if question:
            # Remove the loading placeholder, add the real model response
            self._ai_display = [i for i in self._ai_display if i["role"] != "loading"]
            self._ai_display.append({"role": "model", "text": raw_text})
            # Append to compact API history
            self._ai_api_history.append({"role": "model", "text": raw_text})
            self.ai_followup_input.clear()
            # Re-enable follow-up controls after response
            self.ai_followup_input.setEnabled(True)
            self.ai_followup_btn.setEnabled(True)
            status = "Risposta ricevuta — puoi continuare a fare domande."
        else:
            # Initial analysis: build display + compact history for future follow-ups
            self._ai_display = [{"role": "model", "text": raw_text}]
            compact_ctx = build_followup_context(
                self.current_ticker or "",
                self.current_fundamentals,
                self.current_period,
            )
            self._ai_api_history = [
                {"role": "user",  "text": compact_ctx},
                {"role": "model", "text": raw_text},
            ]
            # Enable follow-up input now that we have a first response
            self.ai_followup_input.setEnabled(True)
            self.ai_followup_btn.setEnabled(True)
            self.ai_followup_input.setPlaceholderText(
                "Scrivi una domanda di approfondimento e premi Invio…"
            )
            status = (
                f"Analisi completata per {self.current_ticker} "
                f"({self.current_period})  —  Gemini 2.5 Flash-Lite + Google Search"
            )

        html = render_conversation_html(self._ai_display)
        self.ai_view.setHtml(html)
        # Scroll to bottom after render
        self.ai_view.page().runJavaScript(
            "setTimeout(()=>window.scrollTo(0,document.body.scrollHeight),200)"
        )

        self.ai_btn.setEnabled(True)
        self.ai_btn.setText(_AI_BTN_LABEL)
        self.ai_status.setText(status)

    def _on_ai_error(self, msg: str):
        """Show error and re-enable all AI controls."""
        # If a follow-up failed, remove the loading + user turn from display
        self._ai_display = [i for i in self._ai_display
                            if i["role"] not in ("loading",)]
        if self._ai_display and self._ai_display[-1]["role"] == "user":
            self._ai_display.pop()            # remove pending question
        if self._ai_api_history and self._ai_api_history[-1]["role"] == "user":
            self._ai_api_history.pop()        # rollback API history too

        # Append error item and re-render
        self._ai_display.append({"role": "error", "text": msg})
        if self._ai_display:
            self.ai_view.setHtml(render_conversation_html(self._ai_display))
        else:
            error_html = (
                "<html><body style='background:#0f0f1a;color:#f44336;"
                "font-family:Segoe UI;padding:28px;line-height:1.7;'>"
                f"<b>Errore:</b> {msg}<br><br>"
                "<span style='color:#7070a0;font-size:12px;'>"
                "Possibili cause: API key non valida, quota gratuita esaurita, "
                "libreria mancante (pip install google-genai), nessuna connessione."
                "</span></body></html>"
            )
            self.ai_view.setHtml(error_html)

        self.ai_btn.setEnabled(True)
        self.ai_btn.setText(_AI_BTN_LABEL)
        if self._ai_api_history:           # had a previous successful analysis
            self.ai_followup_input.setEnabled(True)
            self.ai_followup_btn.setEnabled(True)
        self.ai_status.setText(f"Errore: {msg[:120]}{'…' if len(msg) > 120 else ''}")

    def _request_ai_followup(self):
        """Submit a follow-up question using the compact conversation history."""
        question = self.ai_followup_input.text().strip()
        if not question:
            return

        api_key = self.ai_key_input.text().strip()
        if not api_key:
            self.ai_status.setText("Inserisci la Gemini API key prima di fare domande.")
            return

        if not self._ai_api_history:
            self.ai_status.setText("Genera prima un'analisi iniziale.")
            return

        # Stop any running worker
        if self._ai_worker and self._ai_worker.isRunning():
            self._ai_worker.terminate()
            self._ai_worker.wait()

        # Immediately show user question + loading placeholder in the view
        self._ai_display.append({"role": "user",    "text": question})
        self._ai_display.append({"role": "loading", "text": ""})
        self.ai_view.setHtml(render_conversation_html(self._ai_display))
        self.ai_view.page().runJavaScript(
            "setTimeout(()=>window.scrollTo(0,document.body.scrollHeight),200)"
        )

        # Disable inputs while waiting
        self.ai_followup_input.setEnabled(False)
        self.ai_followup_btn.setEnabled(False)
        self.ai_btn.setEnabled(False)
        self.ai_status.setText(f'Elaboro: "{question[:80]}{"…" if len(question) > 80 else ""}"')

        # Add question to API history (will be rolled back on error)
        self._ai_api_history.append({"role": "user", "text": question})

        self._ai_worker = AIWorker(
            ticker            = self.current_ticker,
            fundamentals      = self.current_fundamentals,
            df                = self.current_df,
            sr                = self.current_sr or {},
            analysts          = self.current_analysts,
            period            = self.current_period,
            api_key           = api_key,
            followup_question = question,
            api_history       = self._ai_api_history[:-1],  # exclude the just-added question
        )
        self._ai_worker.finished.connect(self._on_ai_ready)
        self._ai_worker.error.connect(self._on_ai_error)
        self._ai_worker.start()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _period_to_horizon(self) -> str:
        # 2Y/5Y/MAX use weekly/monthly candles — indicators measure different
        # time units, so we cap them at "lungo" but the analysis tab will show a warning.
        return {
            "1D": "breve", "5D": "breve", "1M": "breve",
            "3M": "medio", "6M": "medio",
            "1Y": "lungo",
            "2Y": "lungo", "5Y": "lungo", "MAX": "lungo",
        }.get(self.current_period, "medio")

    def _refresh_analysis(self):
        if self.current_df is None or self.current_df.empty:
            return
        try:
            result = generate_analysis(
                self.current_df,
                self.current_sr or {},
                self.current_fundamentals,
                self.current_analysts,
                self._period_to_horizon(),
            )
            html = build_analysis_html(
                result, self.current_ticker or "", self.current_period
            )
            self.analysis_browser.setHtml(html)
        except Exception as e:
            self.analysis_browser.setHtml(
                f"<body style='background:#0f0f1a;color:#f44336;padding:20px;'>"
                f"Errore durante l'analisi: {e}</body>"
            )

    def _on_period(self, period: str):
        self.current_period = period
        for p, btn in self.period_btns.items():
            btn.setChecked(p == period)
        if self.current_ticker:
            self.load_ticker(self.current_ticker)

    def _on_indicator_changed(self):
        self._refresh_chart()

    def _set_controls_enabled(self, on: bool):
        for btn in self.period_btns.values():
            btn.setEnabled(on)
        for chk in self.ind_checks.values():
            chk.setEnabled(on)

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)

    def closeEvent(self, event):
        if self._chart_tmp:
            try: os.unlink(self._chart_tmp)
            except Exception: pass
        super().closeEvent(event)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget()
    t.setColumnCount(len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.verticalHeader().setVisible(False)
    t.setShowGrid(True)
    return t


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item

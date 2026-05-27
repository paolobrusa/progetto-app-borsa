"""Entry point — Stock Analyzer Pro."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt


def _dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    bg   = QColor(15, 15, 26)
    bg2  = QColor(20, 20, 42)
    text = QColor(208, 208, 232)
    acc  = QColor(64, 144, 255)

    p.setColor(QPalette.ColorRole.Window,          bg)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            bg2)
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(22, 22, 40))
    p.setColor(QPalette.ColorRole.ToolTipBase,     bg2)
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          QColor(26, 26, 46))
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Link,            acc)
    p.setColor(QPalette.ColorRole.Highlight,       acc)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(80, 80, 110))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(80, 80, 110))
    app.setPalette(p)


def main():
    # WebEngine must be imported before QApplication is created
    from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

    app = QApplication(sys.argv)
    app.setApplicationName("Stock Analyzer Pro")
    app.setOrganizationName("StockAnalyzer")
    _dark_palette(app)
    app.setFont(QFont("Segoe UI", 10))

    from main_window import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

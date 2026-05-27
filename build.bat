@echo off
echo ============================================================
echo  Stock Analyzer Pro - Build EXE (PyInstaller)
echo ============================================================
echo.

call venv\Scripts\activate.bat 2>nul
python -m pip install pyinstaller >nul

echo Compilazione in corso (puo' richiedere 2-3 minuti)...

pyinstaller ^
    --name "StockAnalyzerPro" ^
    --windowed ^
    --onedir ^
    --noconfirm ^
    --clean ^
    --hidden-import PyQt6.QtWebEngineWidgets ^
    --hidden-import PyQt6.QtWebEngineCore ^
    --hidden-import PyQt6.QtWebEngineQuick ^
    --hidden-import scipy.signal ^
    --hidden-import yfinance ^
    --collect-all PyQt6 ^
    --collect-all PyQt6-WebEngine ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERRORE] Build fallita. Controlla i messaggi sopra.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build completata!
echo  L'eseguibile si trova in:  dist\StockAnalyzerPro\
echo  File da avviare:           StockAnalyzerPro.exe
echo ============================================================
echo.
pause

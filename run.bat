@echo off
echo Avvio Stock Analyzer Pro...
call venv\Scripts\activate.bat 2>nul
python main.py
if errorlevel 1 (
    echo.
    echo [ERRORE] L'app e' uscita con un errore. Hai eseguito setup.bat?
    pause
)

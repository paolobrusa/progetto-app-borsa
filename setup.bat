@echo off
echo ============================================================
echo  Stock Analyzer Pro - Setup
echo ============================================================
echo.

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato. Installa Python 3.11+ da https://python.org
    pause & exit /b 1
)

echo [1/3] Creazione ambiente virtuale...
python -m venv venv
if errorlevel 1 ( echo Errore creazione venv & pause & exit /b 1 )

echo [2/3] Attivazione e installazione dipendenze...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo Errore installazione pacchetti & pause & exit /b 1 )

echo [3/3] Setup completato!
echo.
echo  Per avviare l'app:  esegui run.bat
echo  Per creare l'exe:   esegui build.bat
echo.
pause

@echo off
REM ============================================================
REM  Avvia ROCK-DIC (DIC Landslide Monitor)
REM  Al primo avvio crea l'ambiente Python e installa le librerie.
REM ============================================================
setlocal
set "APPDIR=%~dp0"
set "VENV=%USERPROFILE%\.venvs\rock-dic"

if not exist "%VENV%\Scripts\pythonw.exe" (
    echo Primo avvio: creazione ambiente Python in "%VENV%"...
    py -3.12 -m venv "%VENV%" || python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERRORE: Python non trovato. Installare Python 3.12 da python.org
        pause
        exit /b 1
    )
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV%\Scripts\python.exe" -m pip install -r "%APPDIR%requirements.txt"
    if errorlevel 1 (
        echo ERRORE durante l'installazione delle librerie.
        pause
        exit /b 1
    )
)

cd /d "%APPDIR%"
start "" "%VENV%\Scripts\pythonw.exe" "%APPDIR%main.py"
endlocal

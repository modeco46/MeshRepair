@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto python_missing
)
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe launch.py
if errorlevel 1 goto failed
exit /b 0
:python_missing
echo Install Python 3.12 x64 from https://www.python.org/downloads/windows/
echo Enable Python Launcher during installation. No C++ compiler is needed.
pause
exit /b 1
:failed
echo Startup failed. Copy the error text above.
pause
exit /b 1

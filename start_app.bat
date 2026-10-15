@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts") else (set "PY=")
if defined PY (set "PYEXE=%PY%\python.exe" & set "PYWEXE=%PY%\pythonw.exe") else (set "PYEXE=python" & set "PYWEXE=pythonw")
"%PYEXE%" -c "import streamlit, ruamel.yaml, webview" 2>nul && goto run

echo Installing the app requirements, one time only...
rem A venv made by uv has no pip: use uv for it, or add pip with ensurepip.
"%PYEXE%" -m pip --version >nul 2>nul && goto use_pip
where uv >nul 2>nul && goto use_uv
"%PYEXE%" -m ensurepip --upgrade || goto failed

:use_pip
"%PYEXE%" -m pip install -r requirements.txt -r requirements-app.txt || goto failed
goto run

:use_uv
uv pip install --python "%PYEXE%" -r requirements.txt -r requirements-app.txt || goto failed
goto run

:failed
echo The requirements could not be installed.
pause
exit /b 1

:run
start "" "%PYWEXE%" launcher.py

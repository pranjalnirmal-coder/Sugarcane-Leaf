@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto setup_error
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto setup_error

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_error

".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto launch_error
goto end

:setup_error
echo.
echo Setup failed. Install 64-bit Python 3.12 and check your internet connection, then run this file again.
pause
exit /b 1

:launch_error
echo.
echo The dashboard did not start. Check the error above.
pause
exit /b 1

:end
endlocal

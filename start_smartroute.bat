@echo off
REM One-click launcher: starts SmartRouteAI and opens it in the browser.
cd /d "%~dp0"
if exist "flight\Scripts\python.exe" (set PY=flight\Scripts\python.exe) else (set PY=python)
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:5000"
echo SmartRouteAI running at http://127.0.0.1:5000  (close this window to stop)
%PY% -c "import api; api.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)"
pause

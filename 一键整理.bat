@echo off
chcp 65001 >nul
cd /d "%~dp0"
.venv\Scripts\python.exe main.py --dir "notes"
echo.
echo 任务执行完毕，按任意键关闭窗口...
pause >nul

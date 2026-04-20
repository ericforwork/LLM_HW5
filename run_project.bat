@echo off
echo [INFO] Running project using uv...
uv run first_crew
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Project failed to run. Ensure 'uv' is installed and your data folder is present.
    pause
)
pause

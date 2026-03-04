@echo off
call conda activate voice-clone
cd /d "%~dp0"
python app.py
pause

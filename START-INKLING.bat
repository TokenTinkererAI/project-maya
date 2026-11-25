@echo off
rem Start Inkling-2-Mini on Windows (engine built with STRATA_IK2_WINDOWS=ON, run with IK2_UNSUPPORTED=1).
rem Canonical config: maya-inkling-2-mini.json (written by maya.py --setup; port 8095).
title Inkling-2-Mini (maya-ik2, Windows)
cd /d "%~dp0"
".venv\Scripts\python.exe" serve\server.py --engine maya --config maya-inkling-2-mini.json
if errorlevel 1 pause

@echo off
REM Detached wrapper for the periodic RAM trimmer. Stop: create scratchpad\ram_trim.stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Andrew\rwkv-state-quant\scratchpad\ram_trim.ps1"

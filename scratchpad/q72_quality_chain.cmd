@echo off
REM Dedicated: 72-b quality pass (EMA x2 + 2-seed soup x2), CPU-only, GPU stays free. Never edit while running.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\q72_quality.log
echo ===== q72 quality pass START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/q72_quality_evals.sh 14 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

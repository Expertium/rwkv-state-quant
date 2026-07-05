@echo off
REM Dedicated: 84-b int2-norm PTQ probe on q88N weights (no training). Never edit while running.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q80n2.log
echo ===== q80n2p PTQ probe START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q80n2_eval.sh 14 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

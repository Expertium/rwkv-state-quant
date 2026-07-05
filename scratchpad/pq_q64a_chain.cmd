@echo off
REM Dedicated: 64-b rotation-rung verdict chain (wait GPU eval -> convert -> CPU eval q64av). Never edit while running.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q64a.log
echo ===== q64a verdict chain START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q64a_eval.sh 14 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

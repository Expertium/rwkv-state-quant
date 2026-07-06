@echo off
REM Dedicated: 72-b joint-cb verdict chain (wait GPU eval -> convert -> CPU eval q72fv). Never edit while running.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q72f.log
echo ===== q72f verdict chain START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q72f_eval.sh 14 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

@echo off
REM Dedicated chain: wait for q224e2 training -> convert -> eval @224b. Never edit while running.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q224e2_chain.log
echo ===== q224e2 chain (wait-convert-eval) START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q224e2_eval.sh 14 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

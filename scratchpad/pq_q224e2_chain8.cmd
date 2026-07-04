@echo off
REM Dedicated: q224e2 eval RESTART, resume-aware, NPROC=8. Same log path as the killed chain.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q224e2_chain.log
echo ===== q224e2 chain RESTART (resume, NPROC=8) START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q224e2_eval8.sh 8 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

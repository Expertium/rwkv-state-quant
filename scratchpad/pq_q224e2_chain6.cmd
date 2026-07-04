@echo off
REM Dedicated: q224e2 eval restart #2, resume-aware, NPROC=6 (Andrew: CPU+GPU total = 14 threads;
REM running training is ~8). Same log path.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q224e2_chain.log
echo ===== q224e2 chain RESTART2 (resume, NPROC=6) START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_q224e2_eval8.sh 6 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

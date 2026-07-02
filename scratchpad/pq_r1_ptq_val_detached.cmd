@echo off
REM Detached wrapper for the F21 rank-1 PQ PTQ de-risk VAL eval.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_r1_ptq_val.log
echo ===== rank-1 PQ PTQ VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_r1_ptq_val.sh %1 %2 %3 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

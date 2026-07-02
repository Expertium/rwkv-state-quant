@echo off
REM Detached wrapper for the ep75 PQ+QAT VAL eval (CPU free — no wait needed).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep75_eval.log
echo ===== ep75 eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_ep75_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

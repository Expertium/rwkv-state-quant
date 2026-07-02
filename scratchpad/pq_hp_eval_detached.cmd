@echo off
REM Detached wrapper for the F24 HP-sweep PQ+QAT VAL deploy-eval (wd0, wd0_ema, cl01, cl01_ema).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_hp_eval.log
echo ===== HP PQ+QAT VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_hp_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

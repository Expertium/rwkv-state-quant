@echo off
REM Detached wrapper for the F22 PQ+QAT VAL deploy-eval.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_qat_val.log
echo ===== PQ+QAT VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_qat_val.sh %1 %2 %3 %4 %5 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

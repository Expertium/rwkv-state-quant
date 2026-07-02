@echo off
REM Detached wrapper for the F23 lower-LR PQ+QAT VAL deploy-eval (lr5e4 + lr2e4).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_qat_lr_eval.log
echo ===== lower-LR PQ+QAT VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_qat_lr_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

@echo off
cd /d C:\Users\Andrew\rwkv-state-quant
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_sweep_eval.sh > scratchpad\qat_sweep_eval.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> scratchpad\qat_sweep_eval.log

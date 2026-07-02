@echo off
cd /d C:\Users\Andrew\rwkv-state-quant
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_lr_dev_eval.sh %1 > scratchpad\qat_lr_dev.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> scratchpad\qat_lr_dev.log

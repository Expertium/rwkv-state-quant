@echo off
REM Detached wrapper for the 448 b VAL deploy eval. Args passed through to qat_448_val.sh.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_448_val.log
echo ===== 448b VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_448_val.sh %1 %2 %3 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

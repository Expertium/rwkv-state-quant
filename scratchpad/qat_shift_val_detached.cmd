@echo off
REM Detached wrapper for the shift-PTQ VAL eval. Args passed through to qat_shift_val.sh.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_shift_val.log
echo ===== shift-PTQ VAL eval START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_shift_val.sh %1 %2 %3 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

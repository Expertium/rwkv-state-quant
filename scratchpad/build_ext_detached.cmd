@echo off
REM Detached wrapper for the CUDA extension rebuild (survives Esc). Logs to a stable path.
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\build_ext.log
echo ===== build_ext START %DATE% %TIME% ===== > "%LOG%"
call C:\Users\Andrew\rwkv-state-quant\gpu_train\build_ext.cmd >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

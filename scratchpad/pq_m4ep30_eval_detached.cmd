@echo off
REM Detached wrapper for the F27 combo VAL eval. WAITS for the ep75 eval to finish first
REM (polls its log for DONE_EXIT, max ~5 h).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_m4ep30_eval.log
echo ===== F27 combo eval QUEUED (waiting for ep75 eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,600) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep75_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== F27 combo eval START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_m4ep30_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

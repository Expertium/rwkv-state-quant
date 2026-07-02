@echo off
REM Detached wrapper for the ep50 PQ+QAT VAL eval. WAITS for the wd0fix eval to finish first
REM (polls its log for DONE_EXIT, max ~5 h).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep50_eval.log
echo ===== ep50 eval QUEUED (waiting for wd0fix eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,600) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_wd0fix_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== ep50 eval START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_ep50_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

@echo off
REM Detached wrapper for the co-adaptation corpus dump + codebook retrain. WAITS for the ep75 dev-confirm
REM eval to finish first (polls its log for DONE_EXIT, max ~5 h).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\coad_dump.log
echo ===== co-adapt dump QUEUED (waiting for dev-confirm eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,600) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep75_dev_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== co-adapt dump START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/coad_dump.sh %1 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

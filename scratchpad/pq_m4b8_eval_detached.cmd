@echo off
REM Detached wrapper for the F26 m4b8 PQ+QAT VAL eval. WAITS for the F25 epoch eval to finish first
REM (polls its log for DONE_EXIT, max ~4 h) so the 14-thread evals run one at a time.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_m4b8_eval.log
echo ===== F26 m4b8 eval QUEUED (waiting for F25 eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,480) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== F26 m4b8 eval START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_m4b8_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

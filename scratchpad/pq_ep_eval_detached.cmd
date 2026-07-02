@echo off
REM Detached wrapper for the F25 epoch-sweep PQ+QAT VAL eval. WAITS for the F24 HP eval to finish first
REM (polls its log for DONE_EXIT, max ~3 h) so the two 14-thread evals never thrash the CPU together.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_ep_eval.log
echo ===== F25 epoch eval QUEUED (waiting for F24 eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,360) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_hp_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== F25 epoch eval START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_ep_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

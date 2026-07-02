@echo off
REM Detached wrapper for the wd0fix (TRUE WD=0) PQ+QAT VAL eval. WAITS for the F26 m4b8 eval to finish
REM first (polls its log for DONE_EXIT, max ~5 h) so the 14-thread evals run one at a time.
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_wd0fix_eval.log
echo ===== wd0fix eval QUEUED (waiting for F26 eval) %DATE% %TIME% ===== > "%LOG%"
for /L %%i in (1,1,600) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_m4b8_eval.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== wd0fix eval START %DATE% %TIME% ===== >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/pq_wd0fix_eval.sh %1 %2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

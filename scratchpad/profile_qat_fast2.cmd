@echo off
REM QAT speed A/B round 2: ALL FOUR flags (ROT_CACHE + FAST_EMB + EMA_FOREACH + NO_MEMFILL — the
REM last kills the deterministic-mode NaN-fill of every allocation, ~25%% of all launches). Runs in
REM the GPU-idle window after the q72c GPU eval. Compare profile_qat_out_fast2.txt vs _fast/baseline.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q72c.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_fast2.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_PROFILE_TAG=_fast2
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile SPEED-FLAG A/B ROUND 2 START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q56s.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

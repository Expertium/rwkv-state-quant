@echo off
REM QAT speed-flag A/B: rerun the step profiler with ROT_CACHE + FAST_EMB + EMA_FOREACH ON, in the
REM GPU-idle window after the q56b GPU eval ends (the q56b CPU deploy eval runs ~40 min on CPU only).
REM Compare scratchpad/profile_qat_out_fast.txt against profile_qat_out.txt (baseline, same batches).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q56b.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_fast.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_PROFILE_TAG=_fast
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile SPEED-FLAG A/B START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q56s.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

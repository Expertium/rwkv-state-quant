@echo off
REM QAT speed A/B round 3: all four flags + RWKV_QAT_COMPILE=1 (torch.compile dynamic=True on the
REM mixer forwards, triton-windows 3.7.1). Queued behind round 2 (same GPU-idle window after the
REM q72c GPU eval; nothing else is queued after q72c so overrun is harmless). First steps pay
REM compile latency -- the WARM phase absorbs it; segment numbers reflect steady state.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_fast2.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
REM Windows inductor needs MSVC cl.exe on PATH for host-side codegen (first attempt died with
REM "Compiler: cl is not found") -- load the VS 2022 Community x64 environment.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_fast3.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_QAT_COMPILE=1
set RWKV_PROFILE_TAG=_fast3
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile SPEED-FLAG A/B ROUND 3 (torch.compile) START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q56s.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

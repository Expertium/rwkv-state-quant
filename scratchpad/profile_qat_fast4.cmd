@echo off
REM QAT speed A/B round 4: all four flags + RWKV_QAT_COMPILE=student (round 3 showed compiling the
REM no_grad teacher REGRESSES it 166->343 ms/step while the student wins fwd+bwd). Queued behind the
REM q72d GPU eval (next GPU-idle window). Compare profile_qat_out_fast4.txt vs _fast3/_fast2.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q72d.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_fast4.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_QAT_COMPILE=student
set RWKV_PROFILE_TAG=_fast4
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile SPEED-FLAG A/B ROUND 4 (compile student-only) START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q56s.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

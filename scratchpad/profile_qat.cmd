@echo off
REM QAT step profiler: runs in the GPU-idle window AFTER the q52s GPU eval ends (the q52s CPU deploy
REM eval takes ~40 min, q56b training starts only after it -> ~35 min of free GPU). ~5 min run.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q52s.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q56s.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

@echo off
REM Serial queue: q192 GPU eval (m2b4 + int2-shift deploy env), starts after the q144L TRAINING ends.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,960) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_qat_pq_q144L.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q192.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_SHIFT_SCOPE=card:int2,note:int2
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m2b4.txt
set RWKV_QAT_FUSED=1
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== GPU eval q192 START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.get_result --config configs/gpu_eval_q192.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

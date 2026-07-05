@echo off
REM Serial queue: q64k GPU eval (learned cbs + FIXED loaded rotation), after the q64k TRAINING ends.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,960) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_qat_pq_q64k.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\gpu_eval_q64k.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\gpu_train\reference\qat_pq_q64k_wkvcb_6702.txt
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\gpu_train\reference\qat_pq_q64k_shiftcb_6702.txt
set RWKV_QAT_SHIFT_ROT=C:\Users\Andrew\rwkv-state-quant\gpu_train\reference\qat_pq_q64k_shiftrot_6702.txt
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== GPU eval q64k START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.get_result --config configs/gpu_eval_q64k.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

@echo off
REM task23: q72w = SEED-VARIANCE test of q72u (joint-uv WKV b10 + m2b12L shifts + 1-bit norms):
REM ONLY RWKV_AUGMENT_SEED 1234->4321. q72uv +0.001142/+0.000214 edges champion q72cv on every
REM axis (imm nbad 96 = best ever) but the margin is inside the family seed swing (~0.0004) ->
REM doctrine requires the seed pair before the champion title moves. SELF-QUEUED behind q72f.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2880) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q72f.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
  )
  timeout /t 30 /nobreak >nul
)
:run
REM Swap in the rebuilt RWKV_CUDA (joint-uv kernel). The old .pyd was file-locked by the q72d python
REM processes; by now they have all exited. Idempotent if the swap already happened.
copy /Y build\lib.win-amd64-cpython-312\rwkv\model\RWKV_CUDA.cp312-win_amd64.pyd rwkv\model\RWKV_CUDA.cp312-win_amd64.pyd
set CFG=qat_pq_q72w
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_juv_b10.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m2b12.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_NORM_BITS=1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_EMA_DECAY=0.99
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_FUSED=1
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== task23 72-b JOINT-UV SEED test (q72u recipe, seed 4321) 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

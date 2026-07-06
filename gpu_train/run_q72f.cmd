@echo off
REM task23: q72f = SHIFT ROTATION on the q72c champion recipe: m1b5L joint WKV + m2b12L big-catalog
REM shifts + 1-bit norms + LEARNED Cayley pre-rotation (RWKV_QAT_SHIFT_ROT=1). The rotation lever's only
REM remaining live target: m2b12 is m=2 chunks, rotation can move cross-chunk correlation the product
REM form cannot express (m=1 WKV catalogs provably absorb rotations - research log 2026-07-06 note).
REM Clean A/B vs q72cv +0.001295/+0.000212. SELF-QUEUED: waits for the q72u verdict chain.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2880) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q72u.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
REM Swap in the rebuilt RWKV_CUDA (joint-uv kernel). The old .pyd was file-locked by the q72d python
REM processes; by now they have all exited. Idempotent if the swap already happened.
copy /Y build\lib.win-amd64-cpython-312\rwkv\model\RWKV_CUDA.cp312-win_amd64.pyd rwkv\model\RWKV_CUDA.cp312-win_amd64.pyd
set CFG=qat_pq_q72f
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m1b5.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m2b12.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_SHIFT_ROT=1
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
set RWKV_AUGMENT_SEED=1234
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== task23 72-b SHIFT-ROT on q72c (m1b5 + m2b12 + Cayley) 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

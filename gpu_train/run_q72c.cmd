@echo off
REM task22: q72c = BIG-CATALOG shifts at IDENTICAL 72 b: exact q72j champion recipe (m1b5L joint WKV +
REM 1-bit norms, seed 1234, plain QAT) with the shift cb swapped m4b6 -> m2b12 (2 chunks x 16-dim,
REM 4096 entries = SAME 48 shift b/card, 64x the catalog; cb is global/amortized). The joint-cb lesson
REM (fewer chunks + bigger catalog beat the product form for WKV) applied to shifts. Clean A/B vs
REM q72jv +0.0018/+0.0016. SELF-QUEUED: waits for the CPU quality pass AND the m2b12 k-means fit.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\q72_quality.log" >nul 2>&1 && (
    findstr /C:"DONE_EXIT_0" "C:\Users\Andrew\rwkv-state-quant\scratchpad\fit_m2b12.log" >nul 2>&1 && goto :run
  )
  timeout /t 30 /nobreak >nul
)
:run
set CFG=qat_pq_q72c
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m1b5.txt
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
set RWKV_AUGMENT_SEED=1234
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== task22 72-b BIG-CATALOG shifts (m2b12, champion recipe) 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

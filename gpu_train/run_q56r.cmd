@echo off
REM task22: 64 b = m1b5 joint WKV (20, learnable) + m4b4 shifts (40, learnable) + LEARNED ROTATION + 1-bit norms (4).
REM GPU free at launch (serial queue). OMP 4 + fetch 4.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=qat_pq_q56r
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m1b5.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m4b4.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_ROT=1
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
echo ===== task22 56-b QAT (m1b5L + m4b4L + LEARNED ROTATION + 1-bit norms) 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

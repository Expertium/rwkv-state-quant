@echo off
REM PQ+QAT (F22): rank-1 low-rank WKV, factor directions codebook-encoded via the fused qat_lr_rank1 PQ
REM branch. arg1 = config basename, arg2 = codebook file (abs or repo-relative). RWKV_QAT_FUSED=1 is
REM MANDATORY -- PQ lives ONLY in the fused CUDA kernel; the Python-loop fallback would silently do int4
REM instead (train/deploy mismatch). Log -> scratchpad\qat_<basename>.log.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=%1
set RWKV_QAT_PQ=%2
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_FUSED=1
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== PQ+QAT cfg=%CFG% codebook=%RWKV_QAT_PQ% START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

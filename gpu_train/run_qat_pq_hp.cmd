@echo off
REM PQ+QAT with HP overrides (F24: base_drift reduction). %1=config basename, %2=weight_decay, %3=clip,
REM %4=ema_decay, %5=codebook file (optional, default champion-trained m2b8). Fused rank-1 PQ path as F22.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=%1
set RWKV_QAT_PQ=%5
if "%RWKV_QAT_PQ%"=="" set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m2b8.txt
set RWKV_WEIGHT_DECAY=%2
set RWKV_CLIP=%3
set RWKV_EMA_DECAY=%4
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_FUSED=1
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== PQ+QAT-HP cfg=%CFG% WD=%2 CLIP=%3 EMA=%4 START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

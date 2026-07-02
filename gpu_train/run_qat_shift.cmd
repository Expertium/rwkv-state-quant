@echo off
REM Rank-1 int4 WKV + intN token-shift QAT. arg1 = config basename, arg2 = shift level (int3/int2).
REM WKV rank-1 int4 via fused kernel; shifts fake-quant at %2 in the mixer forwards. Log -> scratchpad\qat_%1.log.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=%1
set SHLVL=%2
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_SHIFT_SCOPE=card:%SHLVL%,note:%SHLVL%
set RWKV_QAT_FUSED=1
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== rank-1 int4 WKV + %SHLVL% shift QAT cfg=%CFG% START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

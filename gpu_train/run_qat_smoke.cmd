@echo off
REM Smoke test the fused QAT kernel end-to-end. Arg1 = fused flag (1 or 0). Logs to scratchpad\qat_smoke_%1.log.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_smoke_%1.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int2
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_QAT_FUSED=%1
echo ===== QAT smoke fused=%1 START %DATE% %TIME% ===== > "%LOG%"
"C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe" -u -m rwkv.train_rwkv --config configs/qat_smoke.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

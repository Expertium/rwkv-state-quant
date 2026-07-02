@echo off
REM Generic QAT run: arg1 = config basename (in configs/, no .toml). Fused kernel, full-matrix int2 QAT.
REM Detached via WMI. Log -> scratchpad\qat_<basename>.log. Checkpoints -> reference\<SAVE_MODEL_PREFIX>_<step>.pth.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=%1
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int2
set RWKV_QAT_FUSED=1
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
echo ===== QAT run cfg=%CFG% START %DATE% %TIME% ===== > "%LOG%"
"C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

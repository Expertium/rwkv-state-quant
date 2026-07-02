@echo off
REM QAT fine-tune (rank-2 int2 fake-quant) — detached via WMI so it survives Esc/teardown. Trains the h2k16
REM WS-final into a compression-robust net; checkpoints land in gpu_train\reference\qat_r2i2_<step>.pth.
REM Monitor scratchpad\qat.log for DONE_EXIT. Uses the parent venv python + the parent LMDBs (referenced).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:2:int2,note:2:int2
set RWKV_CLIP=0.25
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
echo ===== QAT r2i2 decay START %DATE% %TIME% ===== > "%LOG%"
"C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe" -u -m rwkv.train_rwkv --config configs/qat_r2i2_decay.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

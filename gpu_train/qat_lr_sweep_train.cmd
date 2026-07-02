@echo off
REM Train the rank-1 int4 QAT recipe sweep (fused). Detached. One log, per-config markers.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_lr_sweep.log
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
echo ===== rank-1 int4 QAT sweep START %DATE% %TIME% ===== > "%LOG%"
for %%C in (qat_lr1i4_lr1e3 qat_lr1i4_lr5e4 qat_lr1i4_lr2e4 qat_lr1i4_lr1e4) do (
  echo ########## TRAIN %%C ########## >> "%LOG%"
  "%PY%" -u -m rwkv.train_rwkv --config configs/%%C.toml >> "%LOG%" 2>&1
)
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

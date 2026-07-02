@echo off
REM BASE-PRESERVING QAT tail (F14): champion h2k16d_904 + short low-LR int2 QAT tail (fused kernel).
REM Detached via WMI (survives Esc). Checkpoints -> reference\qat_tail_<step>.pth. Log scratchpad\qat_tail.log.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_tail.log
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
echo ===== QAT base-preserving tail START %DATE% %TIME% ===== > "%LOG%"
"C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe" -u -m rwkv.train_rwkv --config configs/qat_tail.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

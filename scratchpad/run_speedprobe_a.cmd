@echo off
REM task24 speed probe A: EXACT champion (run_q72u.cmd) env + the WARM-START joint-search kernel,
REM NO round-4 flags. Steps/min here vs the historical q72u run (49 steps/min) isolates the
REM warm-start gain in the real production loop. SAVE prefix = qat_speedprobe (never touches
REM champion artifacts). Killed externally after ~60 logged steps; kill the WRAPPER CMD FIRST,
REM then the python child (prevents a false DONE_EXIT echo).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CFG=qat_speedprobe
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_juv_b10.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m2b12.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_NORM_BITS=1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_EMA_DECAY=0.99
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\speedprobe_a.log
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
echo ===== task24 SPEED PROBE A (champion env + warm-start kernel, no round-4 flags) START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo PROBE_END_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

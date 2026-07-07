@echo off
REM task24 speed probe B: champion env + warm-start kernel + THE ROUND-4 SANCTIONED FLAG SET
REM (ROT_CACHE/FAST_EMB/EMA_FOREACH/NO_MEMFILL + COMPILE=student) = the full next-family stack.
REM Steps/min here = the production speed of the next champion-recipe training. Same probe
REM hygiene as probe A (qat_speedprobe prefix, external kill, wrapper-first).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CFG=qat_speedprobe
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_juv_b10.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m2b12.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_NORM_BITS=1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_EMA_DECAY=0.99
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_QAT_COMPILE=student
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\speedprobe_b.log
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
echo ===== task24 SPEED PROBE B (champion env + warm kernel + round-4 flags) START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo PROBE_END_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

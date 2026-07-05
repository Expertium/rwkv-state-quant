@echo off
REM task22: q56k = the 56-b config + the two NEW QAT levers: KD from the fp32 champion (lambda 0.5)
REM + dead-centroid resurrection. SELF-QUEUED: waits for the q56r verdict chain to finish first
REM (strict serial queue), then trains. OMP 4 + fetch 4.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,1200) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q56r.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set CFG=qat_pq_q56k
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m1b5.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m4b4.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_ROT=1
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_KD=0.5
set RWKV_QAT_CB_RESURRECT=1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_EMA_DECAY=0.99
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_%CFG%.log
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
echo ===== task22 56-b RETRY with KD 0.5 + RESURRECTION 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

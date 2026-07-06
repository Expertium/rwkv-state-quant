@echo off
REM task22: q52s = 52-b rung, BOTH 16-entry walls + full lever stack: m1b4 joint WKV (16 b) + m4b4
REM shifts (32 b) + rotation + ANNEAL (tau0 0.05, hard at 50%) + KD 0.2 + resurrection + 1-bit norms.
REM The stack cracked the 16-entry SHIFT wall at 56 b (q56sv WIN +0.0024/+0.0022); KD is the ahead
REM lever the m1b4 WKV wall needs (q68j failed ahead +0.0014 w/o levers). SELF-QUEUED: waits for the
REM q64b verdict chain (strict serial queue), then trains. OMP 4 + fetch 4.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q64b.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
set CFG=qat_pq_q52s
set RWKV_QAT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m1b4.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_shift_m4b4.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_ROT=1
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_SHIFT_ANNEAL=0.05
set RWKV_QAT_KD=0.2
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
echo ===== task22 52-b BOTH-WALLS (m1b4+m4b4, rot+anneal+KD0.2+resurrect) 2.0ep START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u -m rwkv.train_rwkv --config configs/%CFG%.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

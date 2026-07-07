@echo off
REM task24: JOINT-UV search cost profile. EXACT fast4 harness + flags (round-4 sanctioned set,
REM compile=student) with ONE variable changed: RWKV_PROFILE_PQ -> the joint 1024-entry catalog.
REM Compare the lr_forward/lr_backward kernel rows vs profile_qat_out_fast4.txt (product m1b5:
REM fwd 1.41 ms x18/step, bwd 2.57 ms x18/step) = the exact production cost of the in-kernel
REM 1024x32 joint search that smem tiling would attack. GPU idle at launch -> no wait loop.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat_juv2.log
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONUNBUFFERED=1
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_QAT_COMPILE=student
set RWKV_PROFILE_PQ=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_juv_b10.txt
set RWKV_PROFILE_TAG=_juv2
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
echo ===== QAT step profile JOINT-UV codebook (fast4 harness, PQ swapped m1b5 to juv_b10) START %DATE% %TIME% ===== > "%LOG%"
"%PY%" -u C:\Users\Andrew\rwkv-state-quant\scratchpad\profile_qat.py --config configs/qat_pq_q72u.toml >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

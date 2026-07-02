@echo off
REM Run a python script under the parent venv with gpu_train on PYTHONPATH.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set PYTHONPATH=C:\Users\Andrew\rwkv-state-quant\gpu_train
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_NO_JIT=1
"C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe" %*

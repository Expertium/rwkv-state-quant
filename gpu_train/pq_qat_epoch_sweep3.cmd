@echo off
REM F25c (Andrew): push the epoch trend — 1.0, 1.5, 2.0 ep sequentially (m2b8, LR1e3, WD0.01, EMA0.99).
REM WAITS for the m4ep50 training to free the GPU first (polls its log for DONE_EXIT, max ~4 h).
REM Trend so far: 0.5ep +0.0028/+0.0018 -> 0.75ep +0.0021/+0.0012 (WIN). Testing where it turns around.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,480) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_qat_pq_m4ep50.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
call run_qat_pq_hp.cmd qat_pq_ep100 0.01 0.25 0.99
call run_qat_pq_hp.cmd qat_pq_ep150 0.01 0.25 0.99
call run_qat_pq_hp.cmd qat_pq_ep200 0.01 0.25 0.99
echo EPOCH_SWEEP3_ALL_DONE

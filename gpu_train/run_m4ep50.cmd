@echo off
REM F27b: m4b8 x 0.5ep (margin run — clear-pass candidate at ~416 b if the 0.3ep combo lands close).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call run_qat_pq_hp.cmd qat_pq_m4ep50 0.01 0.25 0.99 C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m4b8.txt
echo M4EP50_DONE

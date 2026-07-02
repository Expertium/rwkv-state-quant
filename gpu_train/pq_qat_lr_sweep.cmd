@echo off
REM F23: lower-LR PQ+QAT sweep to reduce base_drift. Runs 5e-4 then 2e-4 sequentially (each ~40 min due to
REM the serial codebook search). Each writes its own scratchpad\qat_<cfg>.log via run_qat_pq.cmd.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set CB=C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m2b8.txt
call run_qat_pq.cmd qat_pq_m2b8_lr5e4 %CB%
call run_qat_pq.cmd qat_pq_m2b8_lr2e4 %CB%
echo SWEEP_ALL_DONE

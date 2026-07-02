@echo off
REM F25b: extend the epoch sweep upward — 0.5 then 0.75 ep (the F25 trend was monotone-improving through
REM 0.3 with no saturation: total imm +0.0050/0.05ep -> +0.0031/0.3ep). WD 0.01 (default, comparable to the
REM F25 series), clip 0.25, EMA 0.99 (free bonus ckpt; raw ckpt unaffected — EMA is saved separately).
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call run_qat_pq_hp.cmd qat_pq_ep50 0.01 0.25 0.99
call run_qat_pq_hp.cmd qat_pq_ep75 0.01 0.25 0.99
echo EPOCH_SWEEP2_ALL_DONE

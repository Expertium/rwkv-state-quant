@echo off
REM F25: PQ+QAT epoch sweep. 0.05 / 0.2 / 0.3 ep, all LR1e3, WD0.01, clip0.25, EMA OFF (pure epoch effect
REM vs F22's 0.1ep). Reuses run_qat_pq_hp.cmd (%wd %clip %ema). Each writes scratchpad\qat_<cfg>.log.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call run_qat_pq_hp.cmd qat_pq_ep05 0.01 0.25 0
call run_qat_pq_hp.cmd qat_pq_ep20 0.01 0.25 0
call run_qat_pq_hp.cmd qat_pq_ep30 0.01 0.25 0
echo EPOCH_SWEEP_ALL_DONE

@echo off
REM F24: PQ+QAT base_drift-reduction sweep. Run1 = WD=0 + EMA0.99; Run2 = CLIP=0.1 + EMA0.99. Both LR1e3,
REM 0.1ep from champion (clean ablation vs F22). EMA also saves a *_ema_* ckpt each -> raw+ema eval candidates.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
call run_qat_pq_hp.cmd qat_pq_wd0ema 0 0.25 0.99
call run_qat_pq_hp.cmd qat_pq_clip01ema 0.01 0.1 0.99
echo HP_SWEEP_ALL_DONE

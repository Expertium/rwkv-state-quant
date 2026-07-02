@echo off
REM F27: m4b8 x 0.3ep COMBO (the two independently-confirmed drift levers stacked). WAITS for the ep75
REM training to free the GPU (polls its log for DONE_EXIT, max ~5 h), then trains with the m4b8 codebook.
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
for /L %%i in (1,1,600) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\qat_qat_pq_ep75.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
call run_qat_pq_hp.cmd qat_pq_m4ep30 0.01 0.25 0.99 C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_cb_m4b8.txt
echo M4EP30_DONE

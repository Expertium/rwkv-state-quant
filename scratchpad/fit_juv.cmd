@echo off
REM Fit the JOINT-UV WKV codebook (task23): 1024 entries x 32-dim concat(u_unit,v_unit) per head,
REM 10-bit index = SAME 20 WKV b/card as m1b5, 32x the catalog + u/v correlation captured — the m2b12
REM "index bits != catalog size" principle applied to the WKV side. Fits from the champion card-state
REM corpus (scratchpad/corpus). SELF-QUEUED: waits for the q72d verdict chain (CPU free), then fits.
REM Glob expansion needs bash (cmd.exe passes *.txt literally).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\fit_juv.log
for /L %%i in (1,1,2400) do (
  findstr /C:"DONE_EXIT" "C:\Users\Andrew\rwkv-state-quant\scratchpad\pq_q72d.log" >nul 2>&1 && goto :run
  timeout /t 30 /nobreak >nul
)
:run
echo ===== joint-uv WKV codebook fit START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" -c "cd /c/Users/Andrew/rwkv-state-quant && OMP_NUM_THREADS=10 /c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe -u scratchpad/pq_train_juv.py scratchpad/pq_cb_juv_b10.txt scratchpad/corpus/*.txt --bits 10" >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

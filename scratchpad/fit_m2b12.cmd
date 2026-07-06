@echo off
REM Fit the m2b12 shift codebook (2 chunks x 16-dim, 4096 entries/chunk = SAME 48 b/card as m4b6,
REM 64x the catalog) from the champion shift corpus. CPU-only; OMP capped at 8 (quality evals run).
REM Glob expansion needs bash (cmd.exe passes *.txt literally).
cd /d C:\Users\Andrew\rwkv-state-quant
set LOG=C:\Users\Andrew\rwkv-state-quant\scratchpad\fit_m2b12.log
echo ===== m2b12 shift codebook fit START %DATE% %TIME% ===== > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" -c "cd /c/Users/Andrew/rwkv-state-quant && OMP_NUM_THREADS=8 /c/Users/Andrew/rwkv-anki-autoresearch/.venv/Scripts/python.exe -u scratchpad/pq_train_shift.py scratchpad/pq_cb_shift_m2b12.txt scratchpad/corpus_shift/*.txt --c 32 --m 2 --bits 12 --maxvec 80000" >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"

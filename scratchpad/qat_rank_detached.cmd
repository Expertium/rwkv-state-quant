@echo off
REM Detached qat_rank (qfp32+qi4r1 vs champion fp32). Args: NPROC weights tagprefix usersfile. Log = arg3.
cd /d C:\Users\Andrew\rwkv-state-quant
set TP=%3
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_rank.sh %1 %2 %3 %4 > scratchpad\qat_rank_%TP%.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> scratchpad\qat_rank_%TP%.log

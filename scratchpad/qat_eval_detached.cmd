@echo off
REM Detached QAT eval (survives Esc). Arg1 = NPROC. Logs to scratchpad\qat_eval.log, ends with DONE_EXIT.
cd /d C:\Users\Andrew\rwkv-state-quant
set ARGS=%*
if "%ARGS%"=="" set ARGS=14
"C:\Program Files\Git\bin\bash.exe" scratchpad/qat_eval.sh %ARGS% > scratchpad\qat_eval.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> scratchpad\qat_eval.log

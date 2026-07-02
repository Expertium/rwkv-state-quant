@echo off
REM Detached eval wrapper -- launched via WMI Win32_Process.Create so it is parented to WmiPrvSE
REM (a system service), NOT Claude's process tree, and therefore SURVIVES Esc / session teardown.
REM Writes to a STABLE log path (scratchpad\eval.log) -- not the session temp dir which rotates on Esc.
REM Arg 1 = NPROC (default 8). Ends with a DONE_EXIT_<code> marker so the poller knows it finished.
cd /d C:\Users\Andrew\rwkv-state-quant
set ARGS=%*
if "%ARGS%"=="" set ARGS=10 dev
"C:\Program Files\Git\bin\bash.exe" run_eval.sh %ARGS% > scratchpad\eval.log 2>&1
echo DONE_EXIT_%ERRORLEVEL%>> scratchpad\eval.log

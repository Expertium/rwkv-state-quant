# Launch a command FULLY DETACHED from Claude's process tree so it survives Esc / session teardown.
# A process created via WMI Win32_Process.Create is parented to WmiPrvSE (a system service), NOT Claude.
# The console window is HIDDEN (Win32_ProcessStartup ShowWindow=0) — 2026-07-06: a night of self-queued
# chain stages put ~10 black cmd windows on Andrew's desktop; all output goes to log files anyway.
# Usage:  powershell -NoProfile -File scratchpad/detach.ps1 -Script <abs path to .cmd> [-ArgList "8"]
param(
  [Parameter(Mandatory = $true)][string]$Script,
  [string]$ArgList = ""
)
$cmd = "cmd.exe /c `"$Script`" $ArgList"
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [UInt16]0 }
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd; ProcessStartupInformation = $startup
}
if ($res.ReturnValue -ne 0) { Write-Output "FAILED returnvalue=$($res.ReturnValue)"; exit 1 }
$pid_ = $res.ProcessId
Write-Output "detached_pid=$pid_"

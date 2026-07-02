# Resume the suspended rwkv-infer eval workers (counterpart to the NtSuspendProcess pause).
$sig = '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$nt = Add-Type -MemberDefinition $sig -Name NT -Namespace Win32 -PassThru
$n = 0
foreach ($p in (Get-Process rwkv-infer -ErrorAction SilentlyContinue)) { [void]$nt::NtResumeProcess($p.Handle); $n++ }
Write-Output "resumed $n rwkv-infer workers"

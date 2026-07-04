# Periodic working-set trimmer for OUR runs (task22 serial queue). Every 5 min: find python processes
# that are DESCENDANTS of our detached run cmds (cmd.exe whose command line contains rwkv-state-quant)
# and whose working set exceeds 3 GB, and EmptyWorkingSet them. This only evicts reclaimable file-backed
# LMDB pages (they fault back on demand) — no effect on numerics; prevents the 97%-RAM lag incident
# (2026-07-04: four fetchers held ~24 GB of mapped train_db pages). Sibling-repo processes are never
# touched (their roots don't contain the rwkv-state-quant path). Stop: create scratchpad\ram_trim.stop.
$sig = '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr hProcess);'
Add-Type -MemberDefinition $sig -Name PsApi -Namespace Win32
$stop = "C:\Users\Andrew\rwkv-state-quant\scratchpad\ram_trim.stop"
$log = "C:\Users\Andrew\rwkv-state-quant\scratchpad\ram_trim.log"
"ram_trim started $(Get-Date)" | Set-Content $log
while (-not (Test-Path $stop)) {
    try {
        $procs = Get-CimInstance Win32_Process
        $byId = @{}; foreach ($p in $procs) { $byId[$p.ProcessId] = $p }
        $roots = $procs | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*rwkv-state-quant*' } |
            Select-Object -ExpandProperty ProcessId
        $targets = New-Object System.Collections.Generic.HashSet[uint32]
        foreach ($r in $roots) { [void]$targets.Add([uint32]$r) }
        $grew = $true
        while ($grew) {
            $grew = $false
            foreach ($p in $procs) {
                if ($targets.Contains([uint32]$p.ParentProcessId) -and -not $targets.Contains([uint32]$p.ProcessId)) {
                    [void]$targets.Add([uint32]$p.ProcessId); $grew = $true
                }
            }
        }
        foreach ($pid_ in $targets) {
            $gp = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
            if ($gp -and $gp.ProcessName -eq 'python' -and $gp.WorkingSet64 -gt 3GB) {
                $gb = [math]::Round($gp.WorkingSet64/1GB, 1)
                [Win32.PsApi]::EmptyWorkingSet($gp.Handle) | Out-Null
                "$(Get-Date -Format HH:mm:ss) trimmed PID $pid_ ($gb GB)" | Add-Content $log
            }
        }
    } catch { "$(Get-Date -Format HH:mm:ss) error: $($_.Exception.Message)" | Add-Content $log }
    Start-Sleep -Seconds 300
}
"ram_trim stopped $(Get-Date)" | Add-Content $log

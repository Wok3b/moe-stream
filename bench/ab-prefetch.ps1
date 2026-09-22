<#
A/B the predictive prefetch under memory pressure: pin RAM to a target
"available" size, then run llama-moe-run with each config on the same prompt
(greedy, so every config decodes the same tokens). Appends to results\ab.csv.

Usage: .\ab-prefetch.ps1 -Model C:\models\x.gguf -AvailableGB 2 -Gen 64 -Configs "0 8","1 8","2 8","2 16"
       (each config = "<depth> <width>"; depth 0 = no prefetch)
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][double]$AvailableGB,
    [int]$Gen = 64,
    [int]$Threads = 10,
    [string[]]$Configs = @("0 8", "1 8", "2 8", "2 16", "3 16"),
    [string]$Prompt = "The history of computing is a story of ideas outrunning the machines built to hold them. Charles Babbage designed an Analytical Engine in the 1830s that could not be built with the tools of his century. Explain how each later transition changed who got to use computers:",
    [string]$Extra = "",
    [string]$Tag = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$exe  = Join-Path $root 'llama.cpp\build\bin\Release\llama-moe-run.exe'
$csv  = Join-Path $PSScriptRoot 'results\ab.csv'
$eater = Join-Path $PSScriptRoot 'ram-eater.py'
$name = [IO.Path]::GetFileNameWithoutExtension($Model)
if (-not (Test-Path $csv)) {
    'time,tag,model,avail_gb,depth,width,gen_tps,faults_per_token,prefetch_mb_per_token,hit_rate,predict_ms_per_token,issued_mb_per_token,skipped_mb_per_token,worker_ms_per_token,disk_read_MB,extra' | Out-File $csv -Encoding utf8
}

$availNow = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
$eatGB = [math]::Round($availNow - $AvailableGB, 2)
$eaterProc = $null
if ($eatGB -gt 0.1) {
    "pinning $eatGB GB to leave $AvailableGB GB available"
    $eaterOut = Join-Path $env:TEMP 'ram-eater.out'
    Remove-Item $eaterOut -ErrorAction SilentlyContinue
    $eaterProc = Start-Process python -ArgumentList $eater, $eatGB -RedirectStandardOutput $eaterOut -PassThru -NoNewWindow
    $null = $eaterProc.Handle
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Path $eaterOut) -or -not (Select-String -Path $eaterOut -Pattern '^LOCKED' -Quiet)) {
        if ($eaterProc.HasExited) { throw "ram-eater exited early" }
        if ((Get-Date) -gt $deadline) { Stop-Process $eaterProc -Force; throw "ram-eater timed out" }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Seconds 2
}
try {
    foreach ($cfg in $Configs) {
        $parts = $cfg.Split(' '); $depth = [int]$parts[0]; $width = [int]$parts[1]
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $json = Join-Path $PSScriptRoot "results\ab-$stamp-$name-d$depth-w$width.json"
        $log  = "$json.log"
        $args = @('-m', $Model, '-p', ('"' + $Prompt + '"'), '-n', $Gen, '-t', $Threads, '-lzm', 'on', '--json', $json,
                  '--prefetch-depth', $depth, '--prefetch-width', $width)
        if ($depth -gt 0) { $args += '--stats' }
        if ($Extra) { $args += $Extra.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) }
        $p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardError $log -PassThru -NoNewWindow
        $null = $p.Handle
        $disk = 0.0
        while (-not $p.HasExited) {
            try { $disk += (Get-Counter '\PhysicalDisk(_Total)\Disk Read Bytes/sec' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue } catch {}
        }
        $p.WaitForExit()
        if ($p.ExitCode -ne 0 -or -not (Test-Path $json)) { Write-Warning "config '$cfg' failed, see $log"; continue }
        $r = Get-Content $json -Raw | ConvertFrom-Json
        'depth {0} width {1,2}: {2,6:N2} tok/s  faults/tok {3,7:N0}  hit {4:N3}  predict {5,5:N1} ms  issued {6,5:N0} MB  skipped {7,5:N0} MB  worker {8,6:N1} ms/tok  disk {9,6:N0} MB' -f $depth, $width, $r.gen_tps, $r.faults_per_token, $r.hit_rate, $r.predict_ms_per_token, $r.issued_mb_per_token, $r.skipped_mb_per_token, $r.worker_ms_per_token, ($disk/1MB)
        (@($stamp, $Tag, $name, $AvailableGB, $depth, $width, ('{0:N3}' -f $r.gen_tps), ('{0:N0}' -f $r.faults_per_token), ('{0:N1}' -f $r.prefetch_mb_per_token), ('{0:N4}' -f $r.hit_rate), ('{0:N2}' -f $r.predict_ms_per_token), ('{0:N1}' -f $r.issued_mb_per_token), ('{0:N1}' -f $r.skipped_mb_per_token), ('{0:N2}' -f $r.worker_ms_per_token), ('{0:N0}' -f ($disk/1MB)), $Extra) -replace ',', '') -join ',' | Out-File $csv -Append -Encoding utf8
    }
} finally {
    if ($eaterProc -and -not $eaterProc.HasExited) { Stop-Process $eaterProc -Force }
}

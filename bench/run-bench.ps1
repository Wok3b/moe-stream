<#
Run llama-bench once and record, alongside its JSON results, what the machine
did during the run: bytes read from disk and hard page faults (system-wide, so
keep the machine idle) and available memory before/after.

Usage:
  .\run-bench.ps1 -Model C:\models\x.gguf [-Threads 8] [-Prompt 512] [-Gen 128]
                  [-Repeats 3] [-LoadMode auto|none|mmap|mlock|dio]
                  [-LazyMode auto|on|off] [-Tag label] [-ExtraArgs "..."]

Appends one row per test (pp / tg) to results\summary.csv.
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Threads = "8",
    [int]$Prompt = 512,
    [int]$Gen = 128,
    [int]$Repeats = 3,
    [string]$LoadMode = "auto",
    [string]$LazyMode = "auto",
    [string]$Tag = "",
    [string]$ExtraArgs = ""
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path $PSScriptRoot -Parent
$bench  = Join-Path $root 'llama.cpp\build\bin\Release\llama-bench.exe'
$outDir = Join-Path $PSScriptRoot 'results'
$csv    = Join-Path $outDir 'summary.csv'
New-Item -ItemType Directory -Force $outDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$name  = [IO.Path]::GetFileNameWithoutExtension($Model)
$label = "$stamp-$name-t$($Threads -replace ',','_')-$LoadMode-lazy$LazyMode$(if ($Tag) { "-$Tag" })"
$json  = Join-Path $outDir "$label.json"
$log   = Join-Path $outDir "$label.stderr.txt"

$args = @('-m', $Model, '-t', $Threads, '-p', $Prompt, '-n', $Gen, '-r', $Repeats,
          '-lm', $LoadMode, '-lzm', $LazyMode, '-o', 'json')
if ($ExtraArgs) { $args += $ExtraArgs.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) }

function Get-AvailMB { (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue }

$availBefore = Get-AvailMB
$t0 = Get-Date
$p = Start-Process -FilePath $bench -ArgumentList $args -RedirectStandardOutput $json -RedirectStandardError $log -PassThru -NoNewWindow
$null = $p.Handle   # cache the handle, otherwise ExitCode is null after the process exits

# integrate system-wide disk reads and hard faults while the benchmark runs
# the benchmark's working set includes the mapped model pages that are resident: it is the
# direct measure of "how much of the model fit"
$diskBytes = 0.0; $pagesIn = 0.0; $samples = 0; $wsSum = 0.0; $wsMax = 0.0
while (-not $p.HasExited) {
    try {
        $c = Get-Counter '\PhysicalDisk(_Total)\Disk Read Bytes/sec', '\Memory\Pages Input/sec', '\Process(llama-bench)\Working Set' -SampleInterval 1 -MaxSamples 1
        $diskBytes += $c.CounterSamples[0].CookedValue
        $pagesIn   += $c.CounterSamples[1].CookedValue
        $ws = $c.CounterSamples[2].CookedValue
        $wsSum += $ws; if ($ws -gt $wsMax) { $wsMax = $ws }
        $samples++
    } catch { Start-Sleep -Seconds 1 }
}
$wsAvg = if ($samples) { $wsSum / $samples } else { 0 }
$p.WaitForExit()
$runSec = ((Get-Date) - $t0).TotalSeconds
$availAfter = Get-AvailMB

if ($p.ExitCode -ne 0) {
    Write-Warning "llama-bench exited with $($p.ExitCode); see $log"
    Get-Content $log | Select-Object -Last 15
    exit $p.ExitCode
}

$results = Get-Content $json -Raw | ConvertFrom-Json
if (-not (Test-Path $csv)) {
    'time,tag,model,threads,load_mode,lazy_mode,test,avg_ts,stddev_ts,run_s,disk_read_MB,pages_in_MB,ws_avg_MB,ws_max_MB,avail_before_MB,avail_after_MB,json' | Out-File $csv -Encoding utf8
}
foreach ($r in $results) {
    $test = if ($r.n_gen -gt 0) { "tg$($r.n_gen)" } else { "pp$($r.n_prompt)" }
    $row = @(
        $stamp, $Tag, $name, $r.n_threads, $r.load_mode, $r.lazy_mode, $test,
        ('{0:N2}' -f $r.avg_ts), ('{0:N2}' -f $r.stddev_ts), ('{0:N0}' -f $runSec),
        ('{0:N0}' -f ($diskBytes / 1MB)), ('{0:N0}' -f ($pagesIn * 4096 / 1MB)),
        ('{0:N0}' -f ($wsAvg / 1MB)), ('{0:N0}' -f ($wsMax / 1MB)),
        ('{0:N0}' -f $availBefore), ('{0:N0}' -f $availAfter), (Split-Path $json -Leaf)
    ) -replace ',', ''
    ($row -join ',') | Out-File $csv -Append -Encoding utf8
    '{0,-6} t={1,-3} {2,-6} lazy={3,-5} {4,8:N2} tok/s  +/-{5,5:N2}' -f $test, $r.n_threads, $r.load_mode, $r.lazy_mode, $r.avg_ts, $r.stddev_ts
}
'run {0:N0}s  disk read {1:N0} MB  hard-fault pages-in {2:N0} MB  working set avg {5:N0} / max {6:N0} MB  avail {3:N0} -> {4:N0} MB' -f $runSec, ($diskBytes/1MB), ($pagesIn*4096/1MB), $availBefore, $availAfter, ($wsAvg/1MB), ($wsMax/1MB)

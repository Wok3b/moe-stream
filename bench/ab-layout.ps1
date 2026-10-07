<#
A/B the expert-bundled layout against the GGUF layout: same model, same cache
budget, same prompt, same prefetch settings, differing only in whether the
expert cache reads whole experts from the sidecar (one request) or three
ranges from the GGUF (three requests).

Runs are interleaved split/bundled/split/bundled so both layouts see the same
drive thermal state - this drive loses ~half its bandwidth once warm, which is
larger than the effect being measured. Appends to results\layout.csv.

Usage: .\ab-layout.ps1 -Model C:\models\x.gguf [-Experts C:\models\x.gguf.experts]
                       [-Slots 32] [-Rounds 2] [-AvailableGB 0] [-Gen 48] [-Threads 10]

-AvailableGB 0 leaves memory alone. Both layouts read with
FILE_FLAG_NO_BUFFERING, so neither depends on the page cache; pin RAM only to
reproduce the conditions of an earlier run.
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Experts = "",
    [int]$Slots = 32,
    [int]$Rounds = 2,
    [double]$AvailableGB = 0,
    [int]$Gen = 48,
    [int]$Threads = 10,
    [int]$Depth = 2,
    [switch]$Reverse,  # run bundled first, to cancel a drifting drive across paired runs
    [string]$Prompt = "The history of computing is a story of ideas outrunning the machines built to hold them. Charles Babbage designed an Analytical Engine in the 1830s that could not be built with the tools of his century. Explain how each later transition changed who got to use computers:",
    [string]$Tag = ""
)
$ErrorActionPreference = 'Stop'
# format numbers the same way everywhere: on a machine whose locale uses a
# decimal comma, -f would emit "6,21" and the comma stripping below would
# turn it into 621
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
$root = Split-Path $PSScriptRoot -Parent
$exe  = Join-Path $root 'llama.cpp\build\bin\Release\llama-moe-run.exe'
$csv  = Join-Path $PSScriptRoot 'results\layout.csv'
$eater = Join-Path $PSScriptRoot 'ram-eater.py'
$name = [IO.Path]::GetFileNameWithoutExtension($Model)
if (-not $Experts) { $Experts = "$Model.experts" }
if (-not (Test-Path $exe)) { throw "build llama-moe-run first: $exe" }
if (-not (Test-Path $Experts)) { throw "no sidecar at $Experts - run llama-moe-bundle -m $Model" }
if (-not (Test-Path $csv)) {
    'time,tag,model,avail_gb,slots,layout,gen_tps,faults_per_token,read_mb,reads_per_token,mb_per_request,io_mbps,wait_ms,hash,disk_read_MB' | Out-File $csv -Encoding utf8
}

$eaterProc = $null
if ($AvailableGB -gt 0) {
    $availNow = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    $eatGB = [math]::Round($availNow - $AvailableGB, 2)
    if ($eatGB -gt 0.1) {
        "pinning $eatGB GB to leave $AvailableGB GB available"
        $eaterOut = Join-Path $env:TEMP 'ram-eater.out'
        Remove-Item $eaterOut -ErrorAction SilentlyContinue
        $eaterProc = Start-Process python -ArgumentList ("`"$eater`""), $eatGB -RedirectStandardOutput $eaterOut -PassThru -NoNewWindow
        $null = $eaterProc.Handle
        $deadline = (Get-Date).AddSeconds(120)
        while (-not (Test-Path $eaterOut) -or -not (Select-String -Path $eaterOut -Pattern '^LOCKED' -Quiet)) {
            if ($eaterProc.HasExited) { throw "ram-eater exited early" }
            if ((Get-Date) -gt $deadline) { Stop-Process $eaterProc -Force; throw "ram-eater timed out" }
            Start-Sleep -Milliseconds 500
        }
        Start-Sleep -Seconds 2
    }
}

try {
    foreach ($round in 1..$Rounds) {
        $order = if ($Reverse) { @('bundled', 'split') } else { @('split', 'bundled') }
        foreach ($layout in $order) {
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $json = Join-Path $PSScriptRoot "results\layout-$stamp-$name-$layout-s$Slots.json"
            $log  = "$json.log"
            # every path is quoted: the repo can sit under a directory with spaces
            $a = @('-m', ("`"$Model`""), '-p', ("`"$Prompt`""), '-n', $Gen, '-t', $Threads, '-lzm', 'on',
                   '--json', ("`"$json`""), '--cache-slots', $Slots, '--io-threads', 2, '--prefetch-depth', $Depth)
            if ($layout -eq 'bundled') { $a += @('--experts-file', ("`"$Experts`"")) }
            $p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardError $log -PassThru -NoNewWindow
            $null = $p.Handle
            $disk = 0.0
            while (-not $p.HasExited) {
                try { $disk += (Get-Counter '\PhysicalDisk(_Total)\Disk Read Bytes/sec' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue } catch {}
            }
            $p.WaitForExit()
            if ($p.ExitCode -ne 0 -or -not (Test-Path $json)) { Write-Warning "$layout run failed, see $log"; continue }
            $r = Get-Content $json -Raw | ConvertFrom-Json
            $hash = (Select-String -Path $log -Pattern 'tokens hash ([0-9a-f]+)').Matches.Groups[1].Value
            $mbreq = if ($r.cache_reads_per_token -gt 0) { $r.cache_read_mb_per_token / $r.cache_reads_per_token } else { 0 }
            'round {0} {1,-7}: {2,6:N2} tok/s  {3,6:N1} MB/tok in {4,5:N1} reads ({5,4:N2} MB each)  I/O {6,5:N0} MB/s/thread  wait {7,6:N1} ms/tok  hash {8}' -f `
                $round, $layout, $r.gen_tps, $r.cache_read_mb_per_token, $r.cache_reads_per_token, $mbreq, $r.cache_io_mbps, $r.cache_wait_ms_per_token, $hash
            (@($stamp, $Tag, $name, $AvailableGB, $Slots, $layout, ('{0:N3}' -f $r.gen_tps),
               ('{0:N0}' -f $r.faults_per_token), ('{0:N1}' -f $r.cache_read_mb_per_token),
               ('{0:N2}' -f $r.cache_reads_per_token), ('{0:N3}' -f $mbreq), ('{0:N0}' -f $r.cache_io_mbps),
               ('{0:N1}' -f $r.cache_wait_ms_per_token), $hash, ('{0:N0}' -f ($disk/1MB))) -replace ',', '') -join ',' | Out-File $csv -Append -Encoding utf8
        }
    }
} finally {
    if ($eaterProc -and -not $eaterProc.HasExited) { Stop-Process $eaterProc -Force }
}

"-- results\layout.csv --"
Import-Csv $csv | Where-Object { $_.model -eq $name } | Select-Object -Last ($Rounds * 2) |
    Format-Table time, layout, slots, gen_tps, reads_per_token, mb_per_request, io_mbps, wait_ms, hash -AutoSize

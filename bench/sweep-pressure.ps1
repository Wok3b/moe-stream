<#
Memory-pressure sweep: pin RAM with ram-eater.py so that only a chosen amount
stays available, then run the benchmark. Produces the "tok/s vs. how much of
the model fits" curve for plain llama.cpp — the curve any streaming runtime
has to beat.

Usage:
  .\sweep-pressure.ps1 -Model C:\models\x.gguf -AvailableGB 6,4,3,2 [-Threads 8] [-Gen 128] [-Prompt 512] [-Repeats 3] [-LoadMode mmap] [-LazyMode auto]
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][double[]]$AvailableGB,
    [string]$Threads = "8",
    [int]$Prompt = 512,
    [int]$Gen = 128,
    [int]$Repeats = 3,
    [string]$LoadMode = "auto",
    [string]$LazyMode = "auto"
)
$ErrorActionPreference = 'Stop'
$eater = Join-Path $PSScriptRoot 'ram-eater.py'
$run   = Join-Path $PSScriptRoot 'run-bench.ps1'
$modelGB = (Get-Item $Model).Length / 1GB

foreach ($target in $AvailableGB) {
    $availNow = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    $eatGB = [math]::Round($availNow - $target, 2)
    if ($eatGB -le 0.1) {
        "target {0} GB available: already at {1:N1} GB, nothing to pin" -f $target, $availNow
        $eaterProc = $null
    } else {
        "target {0} GB available (model {1:N1} GB -> ~{2:P0} can be resident): pinning {3} GB" -f $target, $modelGB, [math]::Min(1.0, $target / $modelGB), $eatGB
        $eaterOut = Join-Path $env:TEMP 'ram-eater.out'
        Remove-Item $eaterOut -ErrorAction SilentlyContinue
        $eaterProc = Start-Process python -ArgumentList $eater, $eatGB -RedirectStandardOutput $eaterOut -PassThru -NoNewWindow
        $deadline = (Get-Date).AddSeconds(120)
        while (-not (Test-Path $eaterOut) -or -not (Select-String -Path $eaterOut -Pattern '^LOCKED' -Quiet)) {
            if ($eaterProc.HasExited) { Get-Content $eaterOut; throw "ram-eater exited early" }
            if ((Get-Date) -gt $deadline) { Stop-Process $eaterProc -Force; throw "ram-eater timed out" }
            Start-Sleep -Milliseconds 500
        }
        Start-Sleep -Seconds 2
    }
    try {
        & $run -Model $Model -Threads $Threads -Prompt $Prompt -Gen $Gen -Repeats $Repeats -LoadMode $LoadMode -LazyMode $LazyMode -Tag ("avail{0}g" -f $target)
    } finally {
        if ($eaterProc -and -not $eaterProc.HasExited) { Stop-Process $eaterProc -Force; Start-Sleep -Seconds 2 }
    }
    ""
}

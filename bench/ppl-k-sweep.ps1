<#
Perplexity vs. number of active experts (K), via --override-kv, no model changes.

Usage: .\ppl-k-sweep.ps1 -Model C:\models\x.gguf -Arch olmoe -Text C:\models\eval\wikitext-2-raw\wiki.test.raw
                         [-K 8,4] [-Ctx 512] [-Chunks 80] [-Threads 10] [-Tag wikitext]
Appends rows to results\ppl.csv and prints "K=<k> PPL <ppl> +/- <err>".
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Arch,
    [Parameter(Mandatory)][string]$Text,
    [int[]]$K = @(8, 4),
    [int]$Ctx = 512,
    [int]$Chunks = 80,
    [int]$Threads = 10,
    [string]$Tag = ""
)
$ErrorActionPreference = 'Continue' # a native exe writing to stderr under 2>&1 must not be fatal (PS 5.1)
$root = Split-Path $PSScriptRoot -Parent
$exe  = Join-Path $root 'llama.cpp\build\bin\Release\llama-perplexity.exe'
$csv  = Join-Path $PSScriptRoot 'results\ppl.csv'
$name = [IO.Path]::GetFileNameWithoutExtension($Model)
if (-not (Test-Path $csv)) {
    'time,tag,model,text,ctx,chunks,k,ppl,ppl_err,seconds' | Out-File $csv -Encoding utf8
}
foreach ($k in $K) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $log = Join-Path $PSScriptRoot "results\ppl-$stamp-$name-$Tag-k$k.log"
    $t0 = Get-Date
    & $exe -m $Model -f $Text -c $Ctx --chunks $Chunks -t $Threads --override-kv "$Arch.expert_used_count=int:$k" 2>&1 | Out-File $log -Encoding utf8
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    $m = (Select-String -Path $log -Pattern 'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)').Matches
    if ($m.Count -eq 0) { Write-Warning "K=${k}: no result, see $log"; continue }
    $ppl = $m[0].Groups[1].Value; $err = $m[0].Groups[2].Value
    "K=$k  PPL $ppl +/- $err  ($sec s)"
    "$stamp,$Tag,$name,$([IO.Path]::GetFileName($Text)),$Ctx,$Chunks,$k,$ppl,$err,$sec" | Out-File $csv -Append -Encoding utf8
}

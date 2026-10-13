# Grade the same journal entry with the plain mmap server and with the moe-stream server, and record
# wall time, prompt/decode speed, peak server working set and the lowest available RAM during the run.
#   .\bench\journal-ab.ps1 [-Rounds 1]
# Needs the journal-ai CLI and its test vault (C:\Users\Korisnik\journal-ai\samples\tests).
param([int]$Rounds = 1)

$journal = 'C:\Users\Korisnik\journal-ai\journal.cmd'
$tests   = 'C:\Users\Korisnik\journal-ai\samples\tests'
$log     = 'C:\Users\Korisnik\journal-ai\logs\llama-server.log'
$csv     = Join-Path $PSScriptRoot 'results\gemma4\journal-ab.csv'
if (-not (Test-Path $csv)) { 'time,config,wall_s,prompt_tokens,prompt_tps,gen_tokens,gen_tps,peak_ws_gb,min_avail_mb,grades' | Out-File $csv -Encoding utf8 }

foreach ($round in 1..$Rounds) {
    foreach ($config in 'mmap', 'moe-stream') {
        Copy-Item "$tests\C-mixed.md" "$tests\vault-C-mixed\Days\2026-10-14.md" -Force
        $env:JOURNAL_MOE_STREAM = if ($config -eq 'moe-stream') { '1' } else { '0' }
        $out = "$env:TEMP\journal-ab.out"
        $p = Start-Process -FilePath $journal -ArgumentList '--vault', "$tests\vault-C-mixed", '--text', "$tests\vault-C-mixed\Days\2026-10-14.md", '--no-chat', '--yes' -RedirectStandardOutput $out -PassThru -NoNewWindow
        $null = $p.Handle
        [long]$peak = 0; [double]$minAvail = [double]::MaxValue; $t0 = Get-Date
        while (-not $p.HasExited) {
            $s = Get-Process llama-server -ErrorAction SilentlyContinue
            if ($s) { $peak = [Math]::Max($peak, [long]$s.WorkingSet64) }
            $minAvail = [Math]::Min($minAvail, (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue)
            Start-Sleep -Milliseconds 500
        }
        $wall = ((Get-Date) - $t0).TotalSeconds
        $pe = Select-String -Path $log -Pattern 'prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens .*?([\d.]+) tokens per second' | Select-Object -Last 1
        $ge = Select-String -Path $log -Pattern '\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens .*?([\d.]+) tokens per second' | Where-Object { $_.Line -notmatch 'prompt' } | Select-Object -Last 1
        $grades = ((Get-Content $out) | Where-Object { $_ -match '^\s{2}(Productivity|Time management|Happiness|Discipline)\s+(\d+)' } | ForEach-Object { ($_ -split '\s+')[-1] }) -join '/'
        $row = [pscustomobject]@{
            config = $config; wall_s = [int]$wall
            prompt_tokens = $pe.Matches[0].Groups[1].Value; prompt_tps = $pe.Matches[0].Groups[2].Value
            gen_tokens = $ge.Matches[0].Groups[1].Value; gen_tps = $ge.Matches[0].Groups[2].Value
            peak_ws_gb = [Math]::Round($peak / 1GB, 2); min_avail_mb = [int]$minAvail; grades = $grades
        }
        $row | Format-List
        # invariant culture: with a decimal comma, 10.58 would split into two CSV columns (see Results 8)
        $inv = [Globalization.CultureInfo]::InvariantCulture
        ((Get-Date -Format s), $row.config, $row.wall_s, $row.prompt_tokens, $row.prompt_tps, $row.gen_tokens, $row.gen_tps,
         $row.peak_ws_gb.ToString($inv), $row.min_avail_mb, $row.grades) -join ',' | Out-File $csv -Append -Encoding utf8
    }
}
Remove-Item Env:JOURNAL_MOE_STREAM -ErrorAction SilentlyContinue

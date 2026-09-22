# Snapshot of the machine + build, saved next to results so numbers stay comparable.
$root = Split-Path $PSScriptRoot -Parent
$cs = Get-CimInstance Win32_ComputerSystem; $os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$mem = Get-CimInstance Win32_PhysicalMemory
$disk = Get-PhysicalDisk | Select-Object -First 1
$commit = & git -C (Join-Path $root 'llama.cpp') rev-parse --short HEAD 2>$null
[ordered]@{
    date        = (Get-Date).ToString('s')
    machine     = "$($cs.Manufacturer) $($cs.Model)"
    os          = "$($os.Caption) $($os.Version)"
    cpu         = $cpu.Name.Trim()
    cores       = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors
    ram_gb      = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    ram         = ($mem | ForEach-Object { "$([int]($_.Capacity/1GB))GB@$($_.Speed)" }) -join '+'
    disk        = "$($disk.FriendlyName) ($($disk.BusType))"
    llama_cpp   = $commit
} | ConvertTo-Json

<#
Empties the Windows standby list (the file cache) so the next model load is cold.
Needs admin: re-launches itself elevated (UAC prompt) when necessary.
Prints standby size before/after.
#>
param([switch]$Elevated)

$resultFile = Join-Path $env:TEMP 'purge-standby.txt'
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Remove-Item $resultFile -ErrorAction SilentlyContinue
    $p = Start-Process powershell -Verb RunAs -Wait -PassThru -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath,'-Elevated'
    if (Test-Path $resultFile) { Get-Content $resultFile } else { "purge failed (UAC declined?) exit=$($p.ExitCode)" }
    exit $p.ExitCode
}

$src = @"
using System;
using System.Runtime.InteropServices;
public static class Standby {
  [StructLayout(LayoutKind.Sequential, Pack=1)] struct TokPriv1Luid { public int Count; public long Luid; public int Attr; }
  [DllImport("advapi32.dll", SetLastError=true)] static extern bool OpenProcessToken(IntPtr h, int acc, ref IntPtr tok);
  [DllImport("advapi32.dll", SetLastError=true)] static extern bool LookupPrivilegeValue(string host, string name, ref long luid);
  [DllImport("advapi32.dll", SetLastError=true)] static extern bool AdjustTokenPrivileges(IntPtr tok, bool dis, ref TokPriv1Luid np, int len, IntPtr prev, IntPtr ret);
  [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
  [DllImport("ntdll.dll")] static extern int NtSetSystemInformation(int cls, ref int info, int len);
  static void Enable(string priv) {
    IntPtr tok = IntPtr.Zero; OpenProcessToken(GetCurrentProcess(), 0x28, ref tok);
    TokPriv1Luid tp; tp.Count = 1; tp.Luid = 0; tp.Attr = 2; LookupPrivilegeValue(null, priv, ref tp.Luid);
    AdjustTokenPrivileges(tok, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
  }
  public static int Purge() {
    Enable("SeProfileSingleProcessPrivilege"); Enable("SeIncreaseQuotaPrivilege");
    int cmd = 4;                                  // MemoryPurgeStandbyList
    return NtSetSystemInformation(80, ref cmd, 4); // SystemMemoryListInformation
  }
}
"@
Add-Type -TypeDefinition $src
function Get-StandbyGB {
    ((Get-Counter '\Memory\Standby Cache Normal Priority Bytes','\Memory\Standby Cache Reserve Bytes','\Memory\Standby Cache Core Bytes').CounterSamples | Measure-Object CookedValue -Sum).Sum / 1GB
}
$before = Get-StandbyGB
$rc = [Standby]::Purge()
Start-Sleep -Milliseconds 800
$after = Get-StandbyGB
$msg = "standby purge rc=0x{0:X}: {1:N2} GB -> {2:N2} GB" -f $rc, $before, $after
$msg | Out-File -FilePath $resultFile -Encoding utf8
if (-not $Elevated) { $msg }

# Copy our in-tree tools into the llama.cpp examples dir and register them.
# Re-run after editing tools/*; then: cmake --build llama.cpp\build --config Release --target llama-moe-trace
$root = Split-Path $PSScriptRoot -Parent
$examples = Join-Path $root 'llama.cpp\examples'
$cmake = Join-Path $examples 'CMakeLists.txt'
Get-ChildItem (Join-Path $root 'tools') -Directory | ForEach-Object {
    $dst = Join-Path $examples $_.Name
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $_.FullName '*') $dst -Force
    if (-not (Select-String -Path $cmake -Pattern "add_subdirectory\($($_.Name)\)" -Quiet)) {
        Add-Content $cmake "add_subdirectory($($_.Name))"
        "registered examples/$($_.Name)"
    } else {
        "updated examples/$($_.Name)"
    }
}

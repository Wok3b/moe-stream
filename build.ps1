# Build llama-moe-run, llama-moe-trace, llama-moe-bundle and llama-server with the expert cache.
# Usage: .\build.ps1 [extra cmake options]   e.g. .\build.ps1 -DGGML_AVX_VNNI=ON
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path llama.cpp\CMakeLists.txt)) {
    git submodule update --init llama.cpp
    if ($LASTEXITCODE) { exit 1 }
}
cmake -S llama.cpp -B llama.cpp\build -A x64 -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF @args
if ($LASTEXITCODE) { exit 1 }
cmake --build llama.cpp\build --config Release --parallel --target llama-moe-run llama-moe-trace llama-moe-bundle llama-server
if ($LASTEXITCODE) { exit 1 }
Write-Host "built into llama.cpp\build\bin\Release\"

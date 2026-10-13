#!/bin/sh
# Build llama-moe-run, llama-moe-trace, llama-moe-bundle and llama-server with the expert cache.
# Usage: ./build.sh [extra cmake options]   e.g. ./build.sh -DGGML_AVX_VNNI=ON
set -e
cd "$(dirname "$0")"
if [ ! -f llama.cpp/CMakeLists.txt ]; then
    git submodule update --init llama.cpp
fi
gen=
if [ ! -f llama.cpp/build/CMakeCache.txt ] && command -v ninja >/dev/null 2>&1; then
    gen="-G Ninja"
fi
cmake -S llama.cpp -B llama.cpp/build $gen -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF "$@"
cmake --build llama.cpp/build -j "$(nproc 2>/dev/null || echo 4)" --target llama-moe-run llama-moe-trace llama-moe-bundle llama-server
echo "built into llama.cpp/build/bin/"

#!/bin/sh
# Copy our in-tree tools into the llama.cpp examples dir and register them.
# Re-run after editing tools/*; then: cmake --build llama.cpp/build --target llama-moe-run
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
examples="$root/llama.cpp/examples"
for d in "$root"/tools/*/; do
    name=$(basename "$d")
    mkdir -p "$examples/$name"
    cp -f "$d"* "$examples/$name/"
    if ! grep -q "add_subdirectory($name)" "$examples/CMakeLists.txt"; then
        echo "add_subdirectory($name)" >> "$examples/CMakeLists.txt"
        echo "registered examples/$name"
    else
        echo "updated examples/$name"
    fi
done

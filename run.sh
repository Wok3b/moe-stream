#!/bin/sh
# Serve a MoE model with the expert cache: downloads it on first use, sizes the cache from free memory.
# Usage: ./run.sh qwen|gemma|path/to/model.gguf [--mem 8 | --no-limit] [--ctx 32k] [--no-think] [llama-server args...]
set -e
cd "$(dirname "$0")"
models=${MODELS:-$HOME/models}

usage() {
    cat <<EOF
usage: ./run.sh MODEL [--mem GB | --no-limit] [--ctx TOKENS] [--no-think] [llama-server args...]

  MODEL         qwen    Qwen3.6-35B-A3B UD-Q4_K_M, 22.1 GB download
                gemma   Gemma 4 26B-A4B UD-IQ3_S, 11.3 GB download
                or the path to any MoE .gguf
  --mem GB      stay inside this much memory, e.g. --mem 8 (or 8G, 7.5, 6144M);
                the expert cache sizes itself to fit
  --no-limit    use whatever RAM is free, the fastest (the default)
  --ctx TOKENS  context length, e.g. 32768, 32k, 128k or max (the model's own,
                256k for both models above); default 16k. Every 1k tokens costs
                qwen and gemma about 20 MiB, taken from the expert cache
  --no-think    answer straight away instead of reasoning first; both models
                think by default, which at ~10 tok/s can add minutes per reply

Models are kept in \$MODELS (default ~/models). Anything after the options goes
to llama-server, e.g. --port 9000 or -np 4.
EOF
    exit 1
}

[ $# -gt 0 ] || usage
file=
case $1 in
    qwen)
        repo=unsloth/Qwen3.6-35B-A3B-GGUF
        file=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
        size=22134528992
        sha=ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61
        fixed_mib=60 ;;   # linear-attention state, whatever the context
    gemma)
        repo=unsloth/gemma-4-26B-A4B-it-GGUF
        file=gemma-4-26B-A4B-it-UD-IQ3_S.gguf
        size=11289671136
        sha=878be93f9c238ea853b3fd1eb602637ce3cf1cddea56dc345d9a7bf2d6093e29
        fixed_mib=300 ;;  # the 25 sliding-window layers, 1024 tokens + one batch
    -h|--help) usage ;;
    *) model=$1 ;;
esac
shift

mem=
ctx=16384
think=
while [ $# -gt 0 ]; do
    case $1 in
        --mem) [ $# -ge 2 ] || usage; mem=$2; shift 2 ;;
        --mem=*) mem=${1#--mem=}; shift ;;
        --no-limit) mem=; shift ;;
        --ctx) [ $# -ge 2 ] || usage; ctx=$2; shift 2 ;;
        --ctx=*) ctx=${1#--ctx=}; shift ;;
        --no-think) think="-rea off"; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

# --mem: a bare number is GB; 8G, 8GB, 8g and 6144M all work
if [ -n "$mem" ]; then
    mem=$(echo "$mem" | tr 'a-z' 'A-Z' | sed 's/B$//; s/^\([0-9.]*\)$/\1G/')
    echo "$mem" | grep -Eq '^[0-9]+(\.[0-9]+)?[KMGT]$' || { echo "--mem: not a size: $mem" >&2; exit 1; }
fi
# --ctx: 32k = 32768, max = the model's own (llama.cpp's -c 0)
case $ctx in
    max) ctx=0 ;;
    *[kK]) echo "${ctx%?}" | grep -Eq '^[0-9]+$' && ctx=$(( ${ctx%?} * 1024 )) ;;
esac
echo "$ctx" | grep -Eq '^[0-9]+$' || { echo "--ctx: not a token count: $ctx" >&2; exit 1; }

if [ -n "$file" ]; then
    model=$models/$file
    if [ ! -f "$model" ]; then
        mkdir -p "$models"
        have=0
        [ -f "$model.part" ] && have=$(wc -c < "$model.part")
        need_kb=$(( (size - have) / 1024 + 1048576 ))
        free_kb=$(df -Pk "$models" | awk 'NR == 2 { print $4 }')
        gb=$(awk -v b=$size 'BEGIN { printf "%.1f", b / 1e9 }')
        if [ "$free_kb" -lt "$need_kb" ]; then
            echo "not enough disk in $models: $file needs $gb GB plus 1 GB spare, $((free_kb / 1048576)) GiB free" >&2
            exit 1
        fi
        printf 'Download %s (%s GB) to %s? [y/N] ' "$file" "$gb" "$models"
        read -r ans || ans=
        case $ans in y|Y|yes) ;; *) exit 1 ;; esac
        curl -fL -C - -o "$model.part" "https://huggingface.co/$repo/resolve/main/$file"
        echo "checking sha256..."
        if ! echo "$sha  $model.part" | sha256sum -c --quiet -; then
            echo "checksum mismatch: delete $model.part and run again" >&2
            exit 1
        fi
        mv "$model.part" "$model"
    fi
fi
[ -f "$model" ] || { echo "no such model: $model" >&2; exit 1; }

bin=llama.cpp/build/bin/llama-server
[ -x "$bin" ] || ./build.sh

export MOE_CACHE_SLOTS="${MOE_CACHE_SLOTS:-auto}" MOE_IO_THREADS="${MOE_IO_THREADS:-4}"
echo "model:   $model"
if [ -n "$mem" ]; then
    echo "memory:  $mem limit; the expert cache takes what is left"
else
    echo "memory:  no limit (free RAM less 15%); the expert cache takes what is left"
fi
if [ "$ctx" = 0 ]; then
    echo "context: the model's own maximum"
else
    echo "context: $ctx tokens"
fi
if [ -n "$file" ]; then
    # both reference models keep about 20 KiB of KV per token: only their full-attention layers grow with it
    n=$ctx; [ "$n" = 0 ] && n=262144
    echo "         about $(( n * 20 / 1024 + fixed_mib )) MiB of KV cache and state"
fi
if [ -n "$think" ]; then
    echo "thinking: off"
else
    echo "thinking: the model's default (--no-think to answer straight away)"
fi
echo "chat at http://127.0.0.1:8080 (or your --port), one request at a time (-np 4 for more); Ctrl+C stops it"
# -np 1: one sequence, as in every measurement; four (the server's auto) would reserve
# four sequences' recurrent / sliding-window state, taken from the expert cache
if [ -n "$mem" ]; then
    exec systemd-run --user --scope --quiet -p MemoryMax="$mem" -p MemorySwapMax=0 -- \
        "$bin" -m "$model" -c "$ctx" -np 1 -lzm on $think "$@"
fi
exec "$bin" -m "$model" -c "$ctx" -np 1 -lzm on $think "$@"

#!/bin/sh
# Download the Colibri Qwen3.6 int4 gs64 container with curl (resume + sha256 per LFS file).
set -e
repo=Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64
dir=$HOME/models/qwen36-colibri-i4-gs64
curl -s "https://huggingface.co/api/models/$repo/tree/main?recursive=true" |
  python3 -c "import json,sys; [print(x['path'], (x.get('lfs') or {}).get('oid','-')) for x in json.load(sys.stdin) if x['type']=='file']" > "$dir/.files"
while read -r path oid; do
  [ -f "$dir/$path" ] && continue
  curl -fsSL -C - --retry 5 -o "$dir/$path.part" "https://huggingface.co/$repo/resolve/main/$path"
  if [ "$oid" != - ]; then
    echo "$oid  $dir/$path.part" | sha256sum -c --quiet - || { echo "BAD $path"; exit 1; }
  fi
  mv "$dir/$path.part" "$dir/$path"
  echo "ok $path $(date +%T)"
done < "$dir/.files"
echo "DONE $(du -sb "$dir" | cut -f1) bytes"

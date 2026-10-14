"""Save each workload answer (greedy, 128 tokens, non-streamed) for a side-by-side quality check.
    capture.py BASE_URL MODEL WORKLOAD OUT.json"""
import json, sys, urllib.request
base, model, workload, out = sys.argv[1:]
res = []
for line in open(workload, encoding="utf-8"):
    body = dict(json.loads(line), model=model, max_tokens=128, temperature=0, stream=False)
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.load(r)["choices"][0]["message"]
    res.append({"content": msg.get("content"), "reasoning": msg.get("reasoning_content")})
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

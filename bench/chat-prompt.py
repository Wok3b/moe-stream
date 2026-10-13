"""Render a chat message with a model's own chat template, for tools that take a raw prompt.

    python bench/chat-prompt.py C:\\models\\model.gguf "question" --out prompt.txt
    python bench/chat-prompt.py C:\\models\\model.gguf @user.txt [@system.txt] --out prompt.txt

--out writes UTF-8 without a BOM; PowerShell 5.1 redirection would write UTF-16 or add a BOM.

Starts llama-server on a free port with no weights in use for long, asks /apply-template, prints the
prompt. Instruct models like Gemma 4 produce garbage on bare text, so the moe-run hash checks need this.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
PORT = 8091


def text_arg(a: str) -> str:
    return Path(a[1:]).read_text(encoding="utf-8") if a.startswith("@") else a


def main() -> None:
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i:i + 2]
    model, question = args[0], text_arg(args[1])
    system = text_arg(args[2]) if len(args) > 2 else None
    proc = subprocess.Popen([str(SERVER), "-m", model, "--device", "none", "-c", "512", "-np", "1",
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/apply-template", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        prompt = json.load(urllib.request.urlopen(req, timeout=60))["prompt"]
        if out:
            out.write_bytes(prompt.encode("utf-8"))  # no \r\n translation: it would change the tokens
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            print(prompt, end="")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()

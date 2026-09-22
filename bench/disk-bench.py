"""What can the SSD actually deliver, by access pattern?

    python disk-bench.py <big file> [--total-mb 1024]

Reads with FILE_FLAG_NO_BUFFERING (bypasses the Windows cache: measures the
device), for: sequential 8 MB, random reads of expert-sized (2 MB) and
page-fault-sized (64 KB, 4 KB) chunks, at queue depth 1 and 8. The gap between
"4 KB QD1" and "2 MB QD8" is the room an explicit expert reader has over
demand paging.
"""
import argparse
import ctypes
import ctypes.wintypes as w
import random
import threading
import time

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
INVALID_HANDLE = ctypes.c_void_p(-1).value
MEM_COMMIT_RESERVE = 0x3000
PAGE_READWRITE = 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = w.HANDLE
k32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
k32.ReadFile.restype = w.BOOL
k32.ReadFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k32.SetFilePointerEx.restype = w.BOOL
k32.SetFilePointerEx.argtypes = [w.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), w.DWORD]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]
k32.CloseHandle.argtypes = [w.HANDLE]
k32.GetFileSizeEx.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_longlong)]


def open_unbuffered(path):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None)
    if h == INVALID_HANDLE:
        raise OSError(ctypes.get_last_error(), "CreateFileW")
    return h


def file_size(h):
    n = ctypes.c_longlong()
    k32.GetFileSizeEx(h, ctypes.byref(n))
    return n.value


def worker(path, offsets, chunk, out, idx):
    h = open_unbuffered(path)
    buf = k32.VirtualAlloc(None, chunk, MEM_COMMIT_RESERVE, PAGE_READWRITE)  # 64 KB aligned: fine for unbuffered I/O
    got = w.DWORD()
    total = 0
    t0 = time.perf_counter()
    for off in offsets:
        k32.SetFilePointerEx(h, off, None, 0)
        if not k32.ReadFile(h, buf, chunk, ctypes.byref(got), None):
            raise OSError(ctypes.get_last_error(), "ReadFile")
        total += got.value
    out[idx] = (total, time.perf_counter() - t0)
    k32.CloseHandle(h)


def run(path, size, chunk, qd, total_bytes, sequential):
    n = total_bytes // chunk
    if sequential:
        offsets = [i * chunk for i in range(n)]
    else:
        # 4 KB aligned random offsets, as unbuffered I/O requires
        slots = (size - chunk) // 4096
        offsets = [random.randrange(slots) * 4096 for _ in range(n)]
    parts = [offsets[i::qd] for i in range(qd)]
    out = [None] * qd
    threads = [threading.Thread(target=worker, args=(path, parts[i], chunk, out, i)) for i in range(qd)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    read = sum(o[0] for o in out)
    return read / wall / 1e6, wall / n * 1e3 * qd  # MB/s, mean latency per request in ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--total-mb", type=int, default=1024)
    args = ap.parse_args()
    h = open_unbuffered(args.path)
    size = file_size(h)
    k32.CloseHandle(h)
    random.seed(1)
    total = args.total_mb * 1024 * 1024
    print(f"{args.path}: {size / 1e9:.2f} GB, reading {args.total_mb} MB per test, unbuffered\n")
    print(f"{'pattern':<28} {'MB/s':>8} {'ms/req':>8}")
    tests = [
        ("sequential 8 MB   QD1", 8 << 20, 1, True),
        ("random 2 MB       QD1", 2 << 20, 1, False),
        ("random 2 MB       QD4", 2 << 20, 4, False),
        ("random 2 MB       QD8", 2 << 20, 8, False),
        ("random 256 KB     QD1", 256 << 10, 1, False),
        ("random 256 KB     QD8", 256 << 10, 8, False),
        ("random 64 KB      QD1", 64 << 10, 1, False),
        ("random 64 KB      QD8", 64 << 10, 8, False),
        ("random 4 KB       QD1", 4 << 10, 1, False),
        ("random 4 KB       QD8", 4 << 10, 8, False),
    ]
    for name, chunk, qd, seq in tests:
        budget = total if chunk >= (256 << 10) else min(total, 128 << 20)  # small-chunk tests: fewer bytes, same signal
        mbs, ms = run(args.path, size, chunk, qd, budget, seq)
        print(f"{name:<28} {mbs:8.0f} {ms:8.3f}")


if __name__ == "__main__":
    main()

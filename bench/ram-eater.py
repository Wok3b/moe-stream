"""Pin N GB of RAM so a model under test sees a smaller machine.

Usage:  python ram-eater.py <GB>
Prints "LOCKED <GB>" once the memory is pinned, then sleeps until killed.
Pinned pages cannot be paged out or reclaimed, so `available` memory for the
model under test drops by N GB.
"""
import ctypes
import ctypes.wintypes as w
import sys
import time

GB = 1024 ** 3
CHUNK = 256 * 1024 * 1024
MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE = 0x1000, 0x2000, 0x04
QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x1
QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x8

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]
k32.VirtualLock.restype = w.BOOL
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessWorkingSetSizeEx.restype = w.BOOL
k32.SetProcessWorkingSetSizeEx.argtypes = [w.HANDLE, ctypes.c_size_t, ctypes.c_size_t, w.DWORD]
k32.GetCurrentProcess.restype = w.HANDLE


def main() -> None:
    total = int(float(sys.argv[1]) * GB)
    # VirtualLock can only pin up to the process's minimum working set, so raise it first.
    ok = k32.SetProcessWorkingSetSizeEx(
        k32.GetCurrentProcess(), total + 64 * 2**20, total + 512 * 2**20,
        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)
    if not ok:
        sys.exit(f"SetProcessWorkingSetSizeEx failed: {ctypes.get_last_error()}")

    locked = 0
    regions = []
    while locked < total:
        n = min(CHUNK, total - locked)
        p = k32.VirtualAlloc(None, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not p:
            sys.exit(f"VirtualAlloc failed at {locked / GB:.2f} GB: {ctypes.get_last_error()}")
        ctypes.memset(p, 0x5A, n)  # touch so the pages are real before locking
        if not k32.VirtualLock(p, n):
            sys.exit(f"VirtualLock failed at {locked / GB:.2f} GB: {ctypes.get_last_error()}")
        regions.append(p)
        locked += n
    print(f"LOCKED {locked / GB:.2f}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

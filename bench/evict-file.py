"""Drop one file's pages from the Linux page cache (no root needed): a cold start for that file
without `echo 3 > /proc/sys/vm/drop_caches`. Pages mapped by a running process stay."""
import os
import sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

# fs/blockio.py
from __future__ import annotations
import mmap
import struct
from typing import BinaryIO

class BoundsError(Exception): ...

class BlockReader:
    """
    Low-level reader for binary streams.

    When the stream is a real file (has a file descriptor), the entire file is
    memory-mapped so every read is a plain slice — no seek/read syscalls and no
    manual cache to manage.  The OS page cache handles eviction and read-ahead
    automatically, which is far more effective for random-access patterns than
    the old single-slot 1 MiB cache.

    For streams that have no file descriptor (e.g. BytesIO), the reader falls
    back to the original seek-and-read path with a 1 MiB block cache so that
    the rest of the codebase is unaffected.
    """

    def __init__(self, f: BinaryIO, size: int | None = None, cache_block: int = 1 << 20) -> None:
        self.size = size
        self._mm: mmap.mmap | None = None
        self._f: BinaryIO | None = None
        # Fallback cache state
        self._cache_off = -1
        self._cache = b""
        self._cache_block = cache_block

        try:
            fd = f.fileno()
            self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            if size is None:
                self.size = self._mm.size()
        except Exception:
            # BytesIO, SpooledTemporaryFile, or any stream without a real fd
            self._f = f

    def read(self, off: int, n: int) -> bytes:
        if off < 0 or n < 0:
            raise BoundsError("negative off/len")
        if self.size is not None and off + n > self.size:
            raise BoundsError("read beyond end")

        if self._mm is not None:
            data = self._mm[off:off + n]
            if len(data) != n:
                raise BoundsError("short read")
            return data

        # ── Fallback: cached seek-and-read path ──────────────────────────────
        assert self._f is not None
        if n <= self._cache_block:
            blk_off = (off // self._cache_block) * self._cache_block
            if blk_off != self._cache_off:
                to_read = self._cache_block
                if self.size is not None:
                    to_read = min(to_read, self.size - blk_off)
                self._f.seek(blk_off)
                self._cache = self._f.read(to_read)   # partial read at EOF is OK
                self._cache_off = blk_off
            rel = off - blk_off
            if rel + n <= len(self._cache):
                return self._cache[rel:rel + n]

        self._f.seek(off)
        data = self._f.read(n)
        if len(data) != n:
            raise BoundsError("short read")
        return data

    # ── endian helpers ────────────────────────────────────────────────────────
    # When mmap is active these call read() which returns a bytes slice;
    # struct.unpack_from works on any buffer so there is no extra copy.
    def u8(self,   off: int) -> int: return self.read(off, 1)[0]
    def u16le(self, off: int) -> int: return struct.unpack_from("<H", self.read(off, 2))[0]
    def u32le(self, off: int) -> int: return struct.unpack_from("<I", self.read(off, 4))[0]
    def u64le(self, off: int) -> int: return struct.unpack_from("<Q", self.read(off, 8))[0]
    def u16be(self, off: int) -> int: return struct.unpack_from(">H", self.read(off, 2))[0]
    def u32be(self, off: int) -> int: return struct.unpack_from(">I", self.read(off, 4))[0]
    def u64be(self, off: int) -> int: return struct.unpack_from(">Q", self.read(off, 8))[0]

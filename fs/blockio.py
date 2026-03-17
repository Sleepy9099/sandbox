# fs/blockio.py
from __future__ import annotations
import io
import mmap
import os
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
        self._fd: int | None = None   # kept for os.sendfile in copy_to()
        self._f: BinaryIO | None = None
        # Fallback cache state
        self._cache_off = -1
        self._cache = b""
        self._cache_block = cache_block

        try:
            fd = f.fileno()
            self._fd = fd
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

    def copy_to(self, out: BinaryIO, offset: int, size: int) -> int:
        """Copy ``size`` bytes starting at ``offset`` from the image to *out*.

        When both the source image and *out* are real files, ``os.sendfile``
        is used for a kernel zero-copy transfer — no data passes through Python
        memory at all.  This is the fast path for large-file extraction.

        Falls back to chunked ``read`` + ``write`` for any stream that does not
        have a file descriptor (e.g. BytesIO, sockets, pipes).

        Returns the number of bytes written.
        """
        if size <= 0:
            return 0
        if self.size is not None and offset + size > self.size:
            raise BoundsError("read beyond end")

        # ── Zero-copy path via os.sendfile ───────────────────────────────────
        if self._fd is not None and hasattr(os, "sendfile"):
            try:
                out_fd = out.fileno()
                sent = 0
                while sent < size:
                    n = os.sendfile(out_fd, self._fd, offset + sent, size - sent)
                    if n == 0:
                        break
                    sent += n
                return sent
            except (AttributeError, io.UnsupportedOperation, OSError):
                pass  # dest has no real fd — fall through to chunked copy

        # ── Chunked copy (mmap slice → write, or cached read → write) ────────
        chunk = 1 << 20  # 1 MiB
        sent = 0
        while sent < size:
            take = min(chunk, size - sent)
            out.write(self.read(offset + sent, take))
            sent += take
        return sent

    # ── endian helpers ────────────────────────────────────────────────────────
    def u8(self,   off: int) -> int: return self.read(off, 1)[0]
    def u16le(self, off: int) -> int: return struct.unpack_from("<H", self.read(off, 2))[0]
    def u32le(self, off: int) -> int: return struct.unpack_from("<I", self.read(off, 4))[0]
    def u64le(self, off: int) -> int: return struct.unpack_from("<Q", self.read(off, 8))[0]
    def u16be(self, off: int) -> int: return struct.unpack_from(">H", self.read(off, 2))[0]
    def u32be(self, off: int) -> int: return struct.unpack_from(">I", self.read(off, 4))[0]
    def u64be(self, off: int) -> int: return struct.unpack_from(">Q", self.read(off, 8))[0]

# fs/blockio.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO
import struct

class BoundsError(Exception): ...

@dataclass
class BlockReader:
    f: BinaryIO
    size: int | None = None   # optional total size
    cache_block: int = 1 << 20  # 1 MiB cache chunks

    def __post_init__(self) -> None:
        self._cache_off = -1
        self._cache = b""

    def _read_exact(self, off: int, n: int) -> bytes:
        if off < 0 or n < 0:
            raise BoundsError("negative off/len")
        if self.size is not None and off + n > self.size:
            raise BoundsError("read beyond end")
        self.f.seek(off)
        data = self.f.read(n)
        if len(data) != n:
            raise BoundsError("short read")
        return data

    def read(self, off: int, n: int) -> bytes:
        # tiny cache for locality
        if n <= self.cache_block:
            blk_off = (off // self.cache_block) * self.cache_block
            if blk_off != self._cache_off:
                self._cache = self._read_exact(blk_off, self.cache_block)
                self._cache_off = blk_off
            rel = off - blk_off
            if rel + n <= len(self._cache):
                return self._cache[rel:rel + n]
        return self._read_exact(off, n)

    # endian helpers
    def u8(self, off: int) -> int:  return self.read(off, 1)[0]
    def u16le(self, off: int) -> int: return struct.unpack_from("<H", self.read(off, 2))[0]
    def u32le(self, off: int) -> int: return struct.unpack_from("<I", self.read(off, 4))[0]
    def u64le(self, off: int) -> int: return struct.unpack_from("<Q", self.read(off, 8))[0]
    def u16be(self, off: int) -> int: return struct.unpack_from(">H", self.read(off, 2))[0]
    def u32be(self, off: int) -> int: return struct.unpack_from(">I", self.read(off, 4))[0]
    def u64be(self, off: int) -> int: return struct.unpack_from(">Q", self.read(off, 8))[0
]

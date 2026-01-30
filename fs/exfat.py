# fs/exfat.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Tuple
import struct

from .core import FSBase, Stat, DirEntry, NotFound, NotADirectory, IsADirectory, CorruptFs
from .blockio import BlockReader
from .pathutil import abspath

EXFAT_OEM = b"EXFAT   "

@dataclass(frozen=True)
class _Boot:
    bytes_per_sector: int
    sectors_per_cluster: int
    fat_offset_sectors: int
    fat_length_sectors: int
    cluster_heap_offset_sectors: int
    cluster_count: int
    root_dir_first_cluster: int

class ExFatFS(FSBase):
    def __init__(self, stream: BinaryIO, *, base_offset: int = 0, total_size: int | None = None) -> None:
        super().__init__()
        self.io = BlockReader(stream, size=total_size)
        self.base = base_offset
        self.boot = self._read_boot()
        self.bps = self.boot.bytes_per_sector
        self.spc = self.boot.sectors_per_cluster
        self.cluster_size = self.bps * self.spc

        self._dir_cache: Dict[int, Dict[str, Tuple[bool, int, int]]] = {}  # cluster -> name -> (is_dir, first_cluster, size)

    def _off(self, x: int) -> int:
        return self.base + x

    def _read_boot(self) -> _Boot:
        # exFAT VBR starts at sector 0 of the volume
        bs = self.io.read(self._off(0), 512)
        oem = bs[3:11]
        if oem != EXFAT_OEM:
            raise CorruptFs(f"exFAT OEM mismatch: {oem!r}")

        # Fields per Microsoft exFAT spec layout
        # 0x40: FAT offset (sectors), 0x44: FAT length (sectors)
        fat_off = struct.unpack_from("<I", bs, 0x50)[0]
        fat_len = struct.unpack_from("<I", bs, 0x54)[0]
        heap_off = struct.unpack_from("<I", bs, 0x58)[0]
        cluster_count = struct.unpack_from("<I", bs, 0x5C)[0]
        root_cluster = struct.unpack_from("<I", bs, 0x60)[0]
        bps_shift = bs[0x6C]
        spc_shift = bs[0x6D]
        bps = 1 << bps_shift
        spc = 1 << spc_shift

        return _Boot(
            bytes_per_sector=bps,
            sectors_per_cluster=spc,
            fat_offset_sectors=fat_off,
            fat_length_sectors=fat_len,
            cluster_heap_offset_sectors=heap_off,
            cluster_count=cluster_count,
            root_dir_first_cluster=root_cluster,
        )

    def _fat_off(self) -> int:
        return self._off(self.boot.fat_offset_sectors * self.bps)

    def _heap_off(self) -> int:
        return self._off(self.boot.cluster_heap_offset_sectors * self.bps)

    def _cluster_off(self, cluster: int) -> int:
        # cluster numbering starts at 2
        if cluster < 2:
            raise CorruptFs("bad cluster number")
        return self._heap_off() + (cluster - 2) * self.cluster_size

    def _fat_next(self, cluster: int) -> int:
        # FAT32-style 4-byte entries
        off = self._fat_off() + cluster * 4
        nxt = self.io.u32le(off)
        return nxt

    def _read_chain(self, first_cluster: int, max_clusters: int = 1_000_000) -> List[int]:
        out = []
        c = first_cluster
        for _ in range(max_clusters):
            if c < 2 or c >= self.boot.cluster_count + 2:
                break
            out.append(c)
            nxt = self._fat_next(c)
            if nxt >= 0xFFFFFFF8:  # end-of-chain
                break
            if nxt == 0x00000000 or nxt == 0xFFFFFFF7:
                break
            c = nxt
        return out

    def _read_clusters(self, chain: List[int], size: Optional[int] = None) -> bytes:
        buf = bytearray()
        remaining = size
        for c in chain:
            blk = self.io.read(self._cluster_off(c), self.cluster_size)
            if remaining is None:
                buf += blk
            else:
                take = min(len(blk), remaining)
                buf += blk[:take]
                remaining -= take
                if remaining <= 0:
                    break
        return bytes(buf)

    # ---- directory parsing ----
    def _parse_dir(self, first_cluster: int) -> Dict[str, Tuple[bool, int, int]]:
        if first_cluster in self._dir_cache:
            return self._dir_cache[first_cluster]

        chain = self._read_chain(first_cluster)
        raw = self._read_clusters(chain)

        # Directory is a sequence of 32-byte entries
        # We implement the "file directory entry set":
        # 0x85 File Directory Entry -> followed by:
        # 0xC0 Stream Extension Entry -> followed by N x 0xC1 File Name Entries
        out: Dict[str, Tuple[bool, int, int]] = {}
        i = 0
        while i + 32 <= len(raw):
            etype = raw[i]
            if etype == 0x00:
                break  # end marker
            if etype == 0x85:
                # File Directory Entry
                secondary_count = raw[i + 1]
                file_attrs = struct.unpack_from("<H", raw, i + 4)[0]
                is_dir = bool(file_attrs & 0x10)

                # Must have at least a stream extension next
                j = i + 32
                if j + 32 > len(raw) or raw[j] != 0xC0:
                    i += 32
                    continue

                # Stream Extension
                name_len = raw[j + 3]
                first_cluster_lo = struct.unpack_from("<I", raw, j + 20)[0]
                data_len = struct.unpack_from("<Q", raw, j + 24)[0]

                # Filename entries
                name_bytes = bytearray()
                k = j + 32
                # Each C1 contains 15 UTF-16LE chars = 30 bytes payload at offset 2
                while k + 32 <= len(raw) and raw[k] == 0xC1 and len(name_bytes) < name_len * 2:
                    name_bytes += raw[k + 2:k + 32]
                    k += 32

                name_utf16 = bytes(name_bytes[:name_len * 2])
                try:
                    name = name_utf16.decode("utf-16le", "strict")
                except Exception:
                    name = name_utf16.decode("utf-16le", "replace")

                if name not in (".", ".."):
                    out[name] = (is_dir, first_cluster_lo, int(data_len))

                # advance by the whole set
                i += 32 * (1 + secondary_count)
                continue

            # Skip other entry types (volume label, bitmap, upcase table, etc.)
            i += 32

        self._dir_cache[first_cluster] = out
        return out

    def _resolve(self, path: str) -> Tuple[bool, int, int]:
        p = abspath(self._cwd, path)
        if p == "/":
            return (True, self.boot.root_dir_first_cluster, 0)
        cur_is_dir, cur_cluster, _ = (True, self.boot.root_dir_first_cluster, 0)
        parts = [x for x in p.split("/") if x]
        for name in parts:
            if not cur_is_dir:
                raise NotADirectory(p)
            ents = self._parse_dir(cur_cluster)
            if name not in ents:
                raise NotFound(p)
            cur_is_dir, cur_cluster, size = ents[name]
        return (cur_is_dir, cur_cluster, size)

    # ---- public API ----
    def ls(self, path: str = ".") -> List[DirEntry]:
        is_dir, cluster, _ = self._resolve(path)
        if not is_dir:
            raise NotADirectory(path)
        ents = self._parse_dir(cluster)
        out = []
        for name, (isd, fc, sz) in sorted(ents.items()):
            out.append(DirEntry(name=name, is_dir=isd, inode=fc, size=sz))
        return out

    def stat(self, path: str) -> Stat:
        p = abspath(self._cwd, path)
        is_dir, cluster, size = self._resolve(p)
        return Stat(path=p, is_dir=is_dir, size=size, inode=cluster)

    def open(self, path: str):
        p = abspath(self._cwd, path)
        is_dir, first_cluster, size = self._resolve(p)
        if is_dir:
            raise IsADirectory(p)
        chain = self._read_chain(first_cluster)
        fs = self

        class _FH:
            def __init__(self) -> None:
                self._pos = 0
                self._closed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False  # do not suppress exceptions


            def read(self, n: int = -1) -> bytes:
                if self._closed:
                    return b""
                if n < 0:
                    n = size - self._pos
                if self._pos >= size:
                    return b""
                n = min(n, size - self._pos)

                # read whole file (simple + safe), then slice
                # Optimize later: partial cluster reads.
                raw = fs._read_clusters(chain, size=size)
                b = raw[self._pos:self._pos + n]
                self._pos += len(b)
                return b

            def seek(self, off: int, whence: int = 0) -> int:
                if whence == 0:
                    self._pos = max(0, off)
                elif whence == 1:
                    self._pos = max(0, self._pos + off)
                elif whence == 2:
                    self._pos = max(0, size + off)
                else:
                    raise ValueError("bad whence")
                return self._pos

            def tell(self) -> int:
                return self._pos

            def close(self) -> None:
                self._closed = True

        return _FH()

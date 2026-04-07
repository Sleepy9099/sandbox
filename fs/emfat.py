# fs/emfat.py
"""
FAT12 / FAT16 (emFAT) filesystem reader.

These are the "embedded FAT" formats used in small-capacity media
(floppy disks, small SD cards, USB sticks, ROM images, etc.).
FAT32 uses a different on-disk layout and is handled by fat32.py.

Key structural differences from FAT32:
  - Root directory is a fixed-size region (not cluster-based).
  - FAT entries are 12-bit (FAT12) or 16-bit (FAT16), not 28-bit.
  - Data area starts immediately after the root directory region.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Tuple
import struct

from .core import FSBase, Stat, DirEntry, NotFound, NotADirectory, IsADirectory, CorruptFs
from .blockio import BlockReader
from .pathutil import abspath

FAT_ATTR_VOLUMEID = 0x08
FAT_ATTR_DIR      = 0x10
FAT_ATTR_LFN      = 0x0F


@dataclass(frozen=True)
class _BPB:
    bytes_per_sector: int
    sectors_per_cluster: int
    reserved_sectors: int
    num_fats: int
    root_entry_count: int
    fat_size_sectors: int
    total_sectors: int
    fat_type: int             # 12 or 16
    fat_offset_bytes: int
    root_dir_offset_bytes: int
    data_offset_bytes: int


def _u16le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]

def _u32le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def _decode_83(name11: bytes) -> str:
    name = name11[:8].decode("ascii", "replace").rstrip(" ")
    ext  = name11[8:11].decode("ascii", "replace").rstrip(" ")
    return f"{name}.{ext}" if ext else name

def _lfn_checksum(short_name11: bytes) -> int:
    s = 0
    for c in short_name11:
        s = (((s & 1) << 7) | (s >> 1)) + c
        s &= 0xFF
    return s

def _parse_lfn_part(entry: bytes) -> str:
    raw = entry[1:11] + entry[14:26] + entry[28:32]
    chars = []
    for i in range(0, len(raw), 2):
        cp = struct.unpack_from("<H", raw, i)[0]
        if cp in (0x0000, 0xFFFF):
            break
        chars.append(cp)
    if not chars:
        return ""
    return struct.pack("<" + "H" * len(chars), *chars).decode("utf-16le", "replace")


class EmFatFS(FSBase):
    """Read-only FAT12/FAT16 (embedded FAT) filesystem."""

    def __init__(self, stream: BinaryIO, *, base_offset: int = 0, total_size: int | None = None) -> None:
        super().__init__()
        self.io = BlockReader(stream, size=total_size)
        self.base = base_offset
        self.bpb = self._read_bpb()
        self.bps = self.bpb.bytes_per_sector
        self.spc = self.bpb.sectors_per_cluster
        self.cluster_size = self.bps * self.spc
        self.fat_type = self.bpb.fat_type  # 12 or 16

        # Cache the entire FAT region for O(1) cluster lookups.
        fat_bytes = self.bpb.fat_size_sectors * self.bps
        self._fat: bytes = self.io.read(self._off(self.bpb.fat_offset_bytes), fat_bytes)

        # dir cache: "root" (str sentinel) or first_cluster (int) -> parsed entries
        self._dir_cache: Dict[object, Dict[str, Tuple[bool, int, int, int]]] = {}

    # ---- internal helpers ----

    def _off(self, x: int) -> int:
        return self.base + x

    def _read_bpb(self) -> _BPB:
        bs = self.io.read(self._off(0), 512)
        if bs[510:512] != b"\x55\xAA":
            raise CorruptFs("FAT boot signature 0x55AA missing")

        bps        = _u16le(bs, 11)
        spc        = bs[13]
        rsvd       = _u16le(bs, 14)
        nfats      = bs[16]
        root_cnt   = _u16le(bs, 17)
        tot16      = _u16le(bs, 19)
        fatsz16    = _u16le(bs, 22)
        tot32      = _u32le(bs, 32)

        if root_cnt == 0:
            raise CorruptFs("RootEntCnt is 0 — this is FAT32, not FAT12/16")
        if fatsz16 == 0:
            raise CorruptFs("FATSz16 is 0 — not a valid FAT12/16 image")
        if bps not in (512, 1024, 2048, 4096):
            raise CorruptFs(f"Unexpected bytes/sector: {bps}")
        if spc == 0 or (spc & (spc - 1)) != 0:
            raise CorruptFs(f"Unexpected sectors/cluster: {spc}")

        total_sectors = tot16 if tot16 != 0 else tot32
        fat_off       = rsvd * bps
        root_dir_off  = (rsvd + nfats * fatsz16) * bps
        root_dir_bytes = root_cnt * 32
        data_off      = root_dir_off + root_dir_bytes

        root_dir_sectors = (root_dir_bytes + bps - 1) // bps
        first_data_sector = rsvd + nfats * fatsz16 + root_dir_sectors
        count_of_clusters = (total_sectors - first_data_sector) // spc

        if count_of_clusters < 4085:
            fat_type = 12
        elif count_of_clusters < 65525:
            fat_type = 16
        else:
            raise CorruptFs("Cluster count too large for FAT12/16 — use Fat32FS")

        return _BPB(
            bytes_per_sector=bps,
            sectors_per_cluster=spc,
            reserved_sectors=rsvd,
            num_fats=nfats,
            root_entry_count=root_cnt,
            fat_size_sectors=fatsz16,
            total_sectors=total_sectors,
            fat_type=fat_type,
            fat_offset_bytes=fat_off,
            root_dir_offset_bytes=root_dir_off,
            data_offset_bytes=data_off,
        )

    def _cluster_off(self, cluster: int) -> int:
        if cluster < 2:
            raise CorruptFs(f"bad cluster number: {cluster}")
        return self._off(self.bpb.data_offset_bytes + (cluster - 2) * self.cluster_size)

    def _fat_next(self, cluster: int) -> int:
        if self.fat_type == 12:
            # Two 12-bit entries share 3 bytes; odd clusters use the upper 12 bits.
            byte_off = cluster + (cluster // 2)
            if byte_off + 1 >= len(self._fat):
                return 0x0FF8  # treat as EOC
            raw = _u16le(self._fat, byte_off)
            return (raw >> 4) if (cluster & 1) else (raw & 0x0FFF)
        else:
            byte_off = cluster * 2
            if byte_off + 1 >= len(self._fat):
                return 0xFFF8
            return _u16le(self._fat, byte_off)

    def _is_eoc(self, val: int) -> bool:
        return val >= (0x0FF8 if self.fat_type == 12 else 0xFFF8)

    def _bad_cluster(self, val: int) -> bool:
        return val == (0x0FF7 if self.fat_type == 12 else 0xFFF7)

    def _read_chain(self, first_cluster: int) -> List[int]:
        out: List[int] = []
        c = first_cluster
        for _ in range(2_000_000):
            if c < 2:
                break
            out.append(c)
            nxt = self._fat_next(c)
            if self._is_eoc(nxt) or self._bad_cluster(nxt) or nxt == 0:
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

    def _fat_datetime_to_epoch(self, d: int, t: int) -> int:
        year   = 1980 + ((d >> 9) & 0x7F)
        month  = (d >> 5) & 0x0F
        day    = d & 0x1F
        hour   = (t >> 11) & 0x1F
        minute = (t >> 5) & 0x3F
        sec    = (t & 0x1F) * 2
        if not (1980 <= year <= 2107 and 1 <= month <= 12 and 1 <= day <= 31):
            return 0
        if hour > 23 or minute > 59 or sec > 59:
            return 0
        import datetime
        return int(datetime.datetime(year, month, day, hour, minute, sec,
                                     tzinfo=datetime.timezone.utc).timestamp())

    def _parse_raw_dir(self, raw: bytes) -> Dict[str, Tuple[bool, int, int, int]]:
        """Parse a sequence of 32-byte directory entries from a raw bytes buffer."""
        out: Dict[str, Tuple[bool, int, int, int]] = {}
        i = 0
        lfn_parts: List[str] = []
        lfn_expected_chk: Optional[int] = None

        while i + 32 <= len(raw):
            ent = raw[i:i + 32]
            first = ent[0]
            if first == 0x00:
                break                    # no more entries
            if first == 0xE5:           # deleted
                lfn_parts.clear()
                lfn_expected_chk = None
                i += 32
                continue

            attr = ent[11]
            if attr == FAT_ATTR_LFN:
                seq = ent[0]
                chk = ent[13]
                if seq & 0x40:          # first LFN entry (highest sequence number)
                    lfn_parts = []
                    lfn_expected_chk = chk
                if lfn_expected_chk == chk:
                    lfn_parts.append(_parse_lfn_part(ent))
                i += 32
                continue

            short11 = ent[0:11]
            chk = _lfn_checksum(short11)
            if lfn_parts and lfn_expected_chk == chk:
                name = "".join(reversed(lfn_parts))
            else:
                name = _decode_83(short11)
            lfn_parts.clear()
            lfn_expected_chk = None

            if attr & FAT_ATTR_VOLUMEID or name in (".", "..") or not name.strip():
                i += 32
                continue

            is_dir = bool(attr & FAT_ATTR_DIR)
            # FAT12/16 cluster field: high word at offset 20 is typically 0 but keep it.
            cl_hi = _u16le(ent, 20)
            cl_lo = _u16le(ent, 26)
            first_cluster = (cl_hi << 16) | cl_lo
            size  = _u32le(ent, 28)
            mtime = self._fat_datetime_to_epoch(_u16le(ent, 24), _u16le(ent, 22))

            out[name] = (is_dir, first_cluster, size, mtime)
            i += 32

        return out

    def _root_entries(self) -> Dict[str, Tuple[bool, int, int, int]]:
        if "root" not in self._dir_cache:
            size = self.bpb.root_entry_count * 32
            raw  = self.io.read(self._off(self.bpb.root_dir_offset_bytes), size)
            self._dir_cache["root"] = self._parse_raw_dir(raw)
        return self._dir_cache["root"]

    def _dir_entries(self, first_cluster: int) -> Dict[str, Tuple[bool, int, int, int]]:
        if first_cluster not in self._dir_cache:
            raw = self._read_clusters(self._read_chain(first_cluster))
            self._dir_cache[first_cluster] = self._parse_raw_dir(raw)
        return self._dir_cache[first_cluster]

    def _resolve(self, path: str) -> Tuple[bool, int, int, int]:
        """Return (is_dir, first_cluster, size, mtime).
        first_cluster == 0 is the sentinel for the fixed root directory.
        """
        p = abspath(self._cwd, path)
        if p == "/":
            return (True, 0, 0, 0)

        cur_is_dir, cur_cluster, cur_size, cur_mtime = (True, 0, 0, 0)
        for name in (x for x in p.split("/") if x):
            if not cur_is_dir:
                raise NotADirectory(p)
            ents = self._root_entries() if cur_cluster == 0 else self._dir_entries(cur_cluster)
            if name not in ents:
                raise NotFound(p)
            cur_is_dir, cur_cluster, cur_size, cur_mtime = ents[name]
        return (cur_is_dir, cur_cluster, cur_size, cur_mtime)

    # ---- public FSBase API ----

    def ls(self, path: str = ".") -> List[DirEntry]:
        is_dir, cluster, size, _ = self._resolve(path)
        if not is_dir:
            raise NotADirectory(path)
        ents = self._root_entries() if cluster == 0 else self._dir_entries(cluster)
        return [DirEntry(name=n, is_dir=v[0], inode=v[1], size=v[2])
                for n, v in sorted(ents.items())]

    def stat(self, path: str) -> Stat:
        p = abspath(self._cwd, path)
        is_dir, cluster, size, mtime = self._resolve(p)
        return Stat(path=p, is_dir=is_dir, size=size, mtime=mtime,
                    inode=cluster if cluster else None)

    def open(self, path: str):
        p = abspath(self._cwd, path)
        is_dir, first_cluster, size, _ = self._resolve(p)
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
                return False

            def read(self, n: int = -1) -> bytes:
                if self._closed or self._pos >= size:
                    return b""
                if n < 0:
                    n = size - self._pos
                n = min(n, size - self._pos)
                if n == 0:
                    return b""
                out = bytearray()
                remaining = n
                cur = self._pos
                cs = fs.cluster_size
                while remaining > 0 and cur < size:
                    ci    = cur // cs
                    inner = cur % cs
                    if ci >= len(chain):
                        break
                    take = min(cs - inner, remaining)
                    out += fs.io.read(fs._cluster_off(chain[ci]) + inner, take)
                    remaining -= take
                    cur += take
                self._pos += len(out)
                return bytes(out)

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

# fs/fat32.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Tuple
import struct

from .core import FSBase, Stat, DirEntry, NotFound, NotADirectory, IsADirectory, CorruptFs
from .blockio import BlockReader
from .pathutil import abspath

FAT_ATTR_READONLY = 0x01
FAT_ATTR_HIDDEN   = 0x02
FAT_ATTR_SYSTEM   = 0x04
FAT_ATTR_VOLUMEID = 0x08
FAT_ATTR_DIR      = 0x10
FAT_ATTR_ARCHIVE  = 0x20
FAT_ATTR_LFN      = 0x0F

@dataclass(frozen=True)
class _BPB:
    bytes_per_sector: int
    sectors_per_cluster: int
    reserved_sectors: int
    num_fats: int
    fat_size_sectors: int
    root_cluster: int
    total_sectors: int
    fat_offset_bytes: int
    data_offset_bytes: int

def _u16le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]

def _u32le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def _decode_83(name11: bytes) -> str:
    name = name11[:8].decode("ascii", "replace").rstrip(" ")
    ext  = name11[8:11].decode("ascii", "replace").rstrip(" ")
    return f"{name}.{ext}" if ext else name

def _is_eoc(val: int) -> bool:
    # FAT32 EOC markers are 0x0FFFFFF8..0x0FFFFFFF (top 4 bits ignored)
    return (val & 0x0FFFFFFF) >= 0x0FFFFFF8

def _fat32_mask(val: int) -> int:
    return val & 0x0FFFFFFF

def _lfn_checksum(short_name11: bytes) -> int:
    s = 0
    for c in short_name11:
        s = (((s & 1) << 7) | (s >> 1)) + c
        s &= 0xFF
    return s

def _parse_lfn_part(entry: bytes) -> str:
    # LFN entry stores 13 UTF-16LE chars across fields:
    # name1 (5 chars @ 1), name2 (6 chars @ 14), name3 (2 chars @ 28)
    def take_utf16(off: int, nbytes: int) -> bytes:
        return entry[off:off+nbytes]
    raw = take_utf16(1, 10) + take_utf16(14, 12) + take_utf16(28, 4)
    # strip 0x0000 and 0xFFFF padding
    chars = []
    for i in range(0, len(raw), 2):
        cp = struct.unpack_from("<H", raw, i)[0]
        if cp in (0x0000, 0xFFFF):
            break
        chars.append(cp)
    return bytes(struct.pack("<" + "H"*len(chars), *chars)).decode("utf-16le", "replace") if chars else ""

class Fat32FS(FSBase):
    def __init__(self, stream: BinaryIO, *, base_offset: int = 0, total_size: int | None = None) -> None:
        super().__init__()
        self.io = BlockReader(stream, size=total_size)
        self.base = base_offset
        self.bpb = self._read_bpb()

        self.bps = self.bpb.bytes_per_sector
        self.spc = self.bpb.sectors_per_cluster
        self.cluster_size = self.bps * self.spc

        # cache: dir_cluster -> name -> (is_dir, first_cluster, size, mtime_epoch)
        self._dir_cache: Dict[int, Dict[str, Tuple[bool, int, int, int]]] = {}

    def _off(self, x: int) -> int:
        return self.base + x

    def _read_bpb(self) -> _BPB:
        bs = self.io.read(self._off(0), 512)
        if bs[510:512] != b"\x55\xAA":
            raise CorruptFs("FAT boot signature 0x55AA missing")

        bps = _u16le(bs, 11)
        spc = bs[13]
        rsvd = _u16le(bs, 14)
        nfats = bs[16]
        tot16 = _u16le(bs, 19)
        tot32 = _u32le(bs, 32)
        fatsz16 = _u16le(bs, 22)
        fatsz32 = _u32le(bs, 36)
        rootclus = _u32le(bs, 44)

        total_sectors = tot32 if tot16 == 0 else tot16
        fat_size_sectors = fatsz32 if fatsz16 == 0 else fatsz16

        # quick FAT32 validation: RootEntCnt must be 0 (FAT12/16 use it)
        root_ent_cnt = _u16le(bs, 17)
        if root_ent_cnt != 0 or fat_size_sectors == 0:
            raise CorruptFs("Not FAT32 BPB (looks like FAT12/16 or invalid)")

        fat_off = rsvd * bps
        data_off = (rsvd + nfats * fat_size_sectors) * bps

        if bps not in (512, 1024, 2048, 4096):
            raise CorruptFs(f"Unexpected bytes/sector: {bps}")
        if spc == 0 or (spc & (spc - 1)) != 0:
            raise CorruptFs(f"Unexpected sectors/cluster: {spc}")

        return _BPB(
            bytes_per_sector=bps,
            sectors_per_cluster=spc,
            reserved_sectors=rsvd,
            num_fats=nfats,
            fat_size_sectors=fat_size_sectors,
            root_cluster=rootclus,
            total_sectors=total_sectors,
            fat_offset_bytes=fat_off,
            data_offset_bytes=data_off,
        )

    def _cluster_off(self, cluster: int) -> int:
        if cluster < 2:
            raise CorruptFs("bad cluster number")
        return self._off(self.bpb.data_offset_bytes + (cluster - 2) * self.cluster_size)

    def _fat_next(self, cluster: int) -> int:
        # FAT entry is 4 bytes (FAT32), masked to 28 bits
        off = self._off(self.bpb.fat_offset_bytes + cluster * 4)
        val = self.io.u32le(off)
        return _fat32_mask(val)

    def _read_chain(self, first_cluster: int, max_clusters: int = 2_000_000) -> List[int]:
        out = []
        c = first_cluster
        for _ in range(max_clusters):
            if c < 2:
                break
            out.append(c)
            nxt = self._fat_next(c)
            if _is_eoc(nxt) or nxt == 0x0000000 or nxt == 0x0FFFFFF7:
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
        # FAT date: bits: Y(1980+)7, M4, D5 ; time: H5, M6, S/2 5
        year = 1980 + ((d >> 9) & 0x7F)
        month = (d >> 5) & 0x0F
        day = d & 0x1F
        hour = (t >> 11) & 0x1F
        minute = (t >> 5) & 0x3F
        sec = (t & 0x1F) * 2
        # avoid importing datetime for speed; keep simple and safe:
        # if invalid, return 0
        if not (1980 <= year <= 2107 and 1 <= month <= 12 and 1 <= day <= 31):
            return 0
        if hour > 23 or minute > 59 or sec > 59:
            return 0
        import datetime
        return int(datetime.datetime(year, month, day, hour, minute, sec, tzinfo=datetime.timezone.utc).timestamp())

    def _parse_dir(self, first_cluster: int) -> Dict[str, Tuple[bool, int, int, int]]:
        if first_cluster in self._dir_cache:
            return self._dir_cache[first_cluster]

        chain = self._read_chain(first_cluster)
        raw = self._read_clusters(chain)

        out: Dict[str, Tuple[bool, int, int, int]] = {}
        i = 0

        lfn_parts: List[str] = []
        lfn_expected_chk: Optional[int] = None

        while i + 32 <= len(raw):
            ent = raw[i:i+32]
            first = ent[0]
            if first == 0x00:
                break
            if first == 0xE5:
                # deleted
                lfn_parts.clear()
                lfn_expected_chk = None
                i += 32
                continue

            attr = ent[11]
            if attr == FAT_ATTR_LFN:
                # Long File Name entry
                seq = ent[0]
                chk = ent[13]
                if (seq & 0x40) != 0:
                    lfn_parts = []
                    lfn_expected_chk = chk
                if lfn_expected_chk == chk:
                    lfn_parts.append(_parse_lfn_part(ent))
                i += 32
                continue

            # Standard entry
            short11 = ent[0:11]
            chk = _lfn_checksum(short11)
            name = None
            if lfn_parts and lfn_expected_chk == chk:
                # LFN parts were encountered in reverse order; rebuild
                name = "".join(reversed(lfn_parts))
            else:
                name = _decode_83(short11)

            lfn_parts.clear()
            lfn_expected_chk = None

            if attr & FAT_ATTR_VOLUMEID:
                i += 32
                continue

            is_dir = bool(attr & FAT_ATTR_DIR)
            cl_hi = _u16le(ent, 20)
            cl_lo = _u16le(ent, 26)
            first_cluster_child = (cl_hi << 16) | cl_lo
            size = _u32le(ent, 28)

            wrt_time = _u16le(ent, 22)
            wrt_date = _u16le(ent, 24)
            mtime = self._fat_datetime_to_epoch(wrt_date, wrt_time)

            if name not in (".", "..") and name != "":
                out[name] = (is_dir, first_cluster_child, size, mtime)

            i += 32

        self._dir_cache[first_cluster] = out
        return out

    def _resolve(self, path: str) -> Tuple[bool, int, int, int]:
        p = abspath(self._cwd, path)
        if p == "/":
            return (True, self.bpb.root_cluster, 0, 0)

        cur_is_dir, cur_cluster, cur_size, cur_mtime = (True, self.bpb.root_cluster, 0, 0)
        parts = [x for x in p.split("/") if x]
        for name in parts:
            if not cur_is_dir:
                raise NotADirectory(p)
            ents = self._parse_dir(cur_cluster)
            if name not in ents:
                raise NotFound(p)
            cur_is_dir, cur_cluster, cur_size, cur_mtime = ents[name]
        return (cur_is_dir, cur_cluster, cur_size, cur_mtime)

    # ---- public API ----
    def ls(self, path: str = ".") -> List[DirEntry]:
        is_dir, cluster, _, _ = self._resolve(path)
        if not is_dir:
            raise NotADirectory(path)
        ents = self._parse_dir(cluster)
        return [DirEntry(name=n, is_dir=v[0], inode=v[1], size=v[2]) for n, v in sorted(ents.items())]

    def stat(self, path: str) -> Stat:
        p = abspath(self._cwd, path)
        is_dir, cluster, size, mtime = self._resolve(p)
        return Stat(path=p, is_dir=is_dir, size=size, mtime=mtime, inode=cluster)

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
                return False  # do not suppress exceptions

            def read(self, n: int = -1) -> bytes:
                if self._closed:
                    return b""
                if self._pos >= size:
                    return b""
                if n < 0:
                    n = size - self._pos
                n = min(n, size - self._pos)
                if n == 0:
                    return b""
                # Stream directly from the cluster chain — never load the whole
                # file into memory.  Each io.read is bounded to one cluster so
                # the BlockReader cache remains effective.
                out = bytearray()
                remaining = n
                cur = self._pos
                cs = fs.cluster_size
                while remaining > 0 and cur < size:
                    ci = cur // cs
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

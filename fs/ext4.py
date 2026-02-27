# fs/ext4.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, List, Dict, Tuple, Optional
import struct

from .core import FSBase, Stat, DirEntry, NotFound, NotADirectory, IsADirectory, CorruptFs
from .blockio import BlockReader
from .pathutil import abspath, normpath

EXT4_SUPERBLOCK_OFF = 1024
EXT4_SUPER_MAGIC = 0xEF53

# Inode modes
S_IFDIR = 0x4000
S_IFREG = 0x8000

# i_flags EXTENTS
EXT4_EXTENTS_FL = 0x00080000

@dataclass(frozen=True)
class _Super:
    block_size: int
    inodes_per_group: int
    inode_size: int
    blocks_per_group: int
    first_data_block: int
    groups_count: int

@dataclass(frozen=True)
class _GroupDesc:
    inode_table_block: int

@dataclass(frozen=True)
class _Inode:
    inode: int
    mode: int
    size: int
    uid: int
    gid: int
    atime: int
    ctime: int
    mtime: int
    flags: int
    i_block: bytes  # 60 bytes

class Ext4FS(FSBase):
    def __init__(self, stream: BinaryIO, *, base_offset: int = 0, total_size: int | None = None) -> None:
        super().__init__()
        self.io = BlockReader(stream, size=total_size)
        self.base = base_offset
        self.sb = self._read_super()
        self.gdt = self._read_group_desc()
        # Cache for resolved directories: (inode -> {name: inode})
        self._dir_cache: Dict[int, Dict[str, int]] = {}

    def _off(self, x: int) -> int:
        return self.base + x

    def _read_super(self) -> _Super:
        off = self._off(EXT4_SUPERBLOCK_OFF)
        magic = self.io.u16le(off + 0x38)
        if magic != EXT4_SUPER_MAGIC:
            raise CorruptFs(f"ext4 magic mismatch: 0x{magic:04x}")

        log_block_size = self.io.u32le(off + 0x18)
        block_size = 1024 << log_block_size

        inodes_per_group = self.io.u32le(off + 0x28)
        inode_size = self.io.u16le(off + 0x58)
        blocks_per_group = self.io.u32le(off + 0x20)
        first_data_block = self.io.u32le(off + 0x14)

        blocks_count_lo = self.io.u32le(off + 0x04)
        # groups_count = ceil(blocks / blocks_per_group)
        groups_count = (blocks_count_lo + blocks_per_group - 1) // blocks_per_group

        return _Super(
            block_size=block_size,
            inodes_per_group=inodes_per_group,
            inode_size=inode_size,
            blocks_per_group=blocks_per_group,
            first_data_block=first_data_block,
            groups_count=groups_count,
        )

    def _read_group_desc(self) -> List[_GroupDesc]:
        # GDT location depends on block size. If block_size == 1024, superblock at block 1, GDT at block 2.
        # Otherwise superblock is inside block 0 and GDT starts at block 1.
        bs = self.sb.block_size
        if bs == 1024:
            gdt_block = 2
        else:
            gdt_block = 1
        gdt_off = self._off(gdt_block * bs)

        desc_size = 32  # classic
        out: List[_GroupDesc] = []
        for i in range(self.sb.groups_count):
            d = self.io.read(gdt_off + i * desc_size, desc_size)
            inode_table_lo = struct.unpack_from("<I", d, 0x08)[0]
            out.append(_GroupDesc(inode_table_block=inode_table_lo))
        return out

    def _inode_loc(self, ino: int) -> int:
        if ino <= 0:
            raise CorruptFs("invalid inode number")
        ino0 = ino - 1
        grp = ino0 // self.sb.inodes_per_group
        idx = ino0 % self.sb.inodes_per_group
        if grp >= len(self.gdt):
            raise CorruptFs("inode group out of range")
        table_block = self.gdt[grp].inode_table_block
        return self._off(table_block * self.sb.block_size + idx * self.sb.inode_size)

    def _read_inode(self, ino: int) -> _Inode:
        off = self._inode_loc(ino)
        data = self.io.read(off, self.sb.inode_size)
        mode = struct.unpack_from("<H", data, 0x00)[0]
        uid = struct.unpack_from("<H", data, 0x02)[0]
        size_lo = struct.unpack_from("<I", data, 0x04)[0]
        atime = struct.unpack_from("<I", data, 0x08)[0]
        ctime = struct.unpack_from("<I", data, 0x0C)[0]
        mtime = struct.unpack_from("<I", data, 0x10)[0]
        gid = struct.unpack_from("<H", data, 0x18)[0]
        flags = struct.unpack_from("<I", data, 0x20)[0]
        i_block = data[0x28:0x28 + 60]
        # For simplicity: ignore high size bits (huge files). Add later if needed.
        return _Inode(
            inode=ino, mode=mode, size=size_lo, uid=uid, gid=gid,
            atime=atime, ctime=ctime, mtime=mtime, flags=flags,
            i_block=i_block,
        )

    def _inode_is_dir(self, ino: _Inode) -> bool:
        return (ino.mode & 0xF000) == S_IFDIR

    def _inode_is_reg(self, ino: _Inode) -> bool:
        return (ino.mode & 0xF000) == S_IFREG

    # -------- extents --------
    def _iter_extents(self, inode: _Inode) -> List[Tuple[int, int, int]]:
        """
        Return list of (logical_block, phys_block, length_blocks)
        Supports extent header in i_block (depth 0 only) and 1-level index.
        """
        b = inode.i_block
        eh_magic, eh_entries, eh_max, eh_depth = struct.unpack_from("<HHHH", b, 0x00)
        if eh_magic != 0xF30A:
            raise CorruptFs("inode not using extents (or unsupported)")
        if eh_depth == 0:
            extents = []
            off = 0x0C
            for _ in range(eh_entries):
                ee_block, ee_len, ee_start_hi, ee_start_lo = struct.unpack_from("<IHHI", b, off)
                ee_start = (ee_start_hi << 32) | ee_start_lo
                extents.append((ee_block, ee_start, ee_len & 0x7FFF))
                off += 12
            return extents
        elif eh_depth == 1:
            # index points to extent leaf blocks
            indexes = []
            off = 0x0C
            for _ in range(eh_entries):
                ei_block, ei_leaf_lo, ei_leaf_hi, _ = struct.unpack_from("<IIHH", b, off)
                leaf = (ei_leaf_hi << 32) | ei_leaf_lo
                indexes.append((ei_block, leaf))
                off += 12
            # read each leaf block and gather extents
            out = []
            for _, leaf_block in indexes:
                leaf_off = self._off(leaf_block * self.sb.block_size)
                leaf = self.io.read(leaf_off, self.sb.block_size)
                lm, le, _, ld = struct.unpack_from("<HHHH", leaf, 0)
                if lm != 0xF30A or ld != 0:
                    raise CorruptFs("bad extent leaf")
                o2 = 0x0C
                for _ in range(le):
                    ee_block, ee_len, ee_start_hi, ee_start_lo = struct.unpack_from("<IHHI", leaf, o2)
                    ee_start = (ee_start_hi << 32) | ee_start_lo
                    out.append((ee_block, ee_start, ee_len & 0x7FFF))
                    o2 += 12
            return out
        else:
            raise CorruptFs(f"extent depth {eh_depth} unsupported")

    def _read_u32_ptr_block(self, block_no: int) -> list[int]:
        bs = self.sb.block_size
        data = self.io.read(self._off(block_no * bs), bs)
        return list(struct.unpack_from("<" + "I" * (bs // 4), data, 0))

    def _legacy_logical_to_phys(self, inode: _Inode, lb: int) -> int | None:
        """
        Map logical block -> physical block for non-extents inodes.
        Returns None for holes (0 pointers).
        """
        bs = self.sb.block_size
        ptrs_per = bs // 4

        # i_block holds 15 u32 pointers in legacy format
        p = list(struct.unpack_from("<15I", inode.i_block, 0))
        direct = p[0:12]
        ind1, ind2, ind3 = p[12], p[13], p[14]

        # direct
        if lb < 12:
            return direct[lb] or None
        lb -= 12

        # single indirect
        if lb < ptrs_per:
            if ind1 == 0:
                return None
            a = self._read_u32_ptr_block(ind1)
            return a[lb] or None
        lb -= ptrs_per

        # double indirect
        span2 = ptrs_per * ptrs_per
        if lb < span2:
            if ind2 == 0:
                return None
            l1 = self._read_u32_ptr_block(ind2)
            i1 = lb // ptrs_per
            i2 = lb % ptrs_per
            blk1 = l1[i1]
            if blk1 == 0:
                return None
            l2 = self._read_u32_ptr_block(blk1)
            return l2[i2] or None
        lb -= span2

        # triple indirect (rare, but implement for completeness)
        span3 = ptrs_per * ptrs_per * ptrs_per
        if lb < span3:
            if ind3 == 0:
                return None
            l1 = self._read_u32_ptr_block(ind3)
            i1 = lb // span2
            rem = lb % span2
            i2 = rem // ptrs_per
            i3 = rem % ptrs_per
            blk1 = l1[i1]
            if blk1 == 0:
                return None
            l2 = self._read_u32_ptr_block(blk1)
            blk2 = l2[i2]
            if blk2 == 0:
                return None
            l3 = self._read_u32_ptr_block(blk2)
            return l3[i3] or None

        return None

    # -------- directories + path resolve --------
    def _read_dir(self, dir_ino: int) -> Dict[str, int]:
        if dir_ino in self._dir_cache:
            return self._dir_cache[dir_ino]

        ino = self._read_inode(dir_ino)
        if not self._inode_is_dir(ino):
            raise NotADirectory(str(dir_ino))

        data = self._read_file_bytes_as_dirblob(ino)
        m: Dict[str, int] = {}
        pos = 0
        while pos + 8 <= len(data):
            inode, rec_len, name_len, file_type = struct.unpack_from("<IHBB", data, pos)
            if rec_len < 8 or pos + rec_len > len(data):
                break
            name = data[pos + 8:pos + 8 + name_len].decode("utf-8", "replace")
            if inode != 0 and name not in (".", ".."):
                m[name] = inode
            pos += rec_len
        self._dir_cache[dir_ino] = m
        return m

    def _read_inode_bytes(self, inode: _Inode, offset: int, n: int) -> bytes:
        """
        Read bytes from inode's data stream.
        Valid for regular files AND directories (directories are stored as data blocks too).
        """
        if not (self._inode_is_reg(inode) or self._inode_is_dir(inode)):
            raise CorruptFs("unsupported inode type for byte reads")

        if offset < 0:
            offset = 0
        if offset >= inode.size:
            return b""
        n = min(n, inode.size - offset)

        bs = self.sb.block_size
        start_block = offset // bs
        end_block = (offset + n + bs - 1) // bs

        use_extents = bool(inode.flags & EXT4_EXTENTS_FL)

        if use_extents:
            extents = self._iter_extents(inode)

            def map_lb(lb: int) -> int | None:
                for l0, p0, ln in extents:
                    if l0 <= lb < l0 + ln:
                        return p0 + (lb - l0)
                return None
        else:
            def map_lb(lb: int) -> int | None:
                return self._legacy_logical_to_phys(inode, lb)

        out = bytearray()
        remaining = n
        cur_off = offset
        for lb in range(start_block, end_block):
            phys = map_lb(lb)

            inner = cur_off % bs
            take = min(bs - inner, remaining)

            if phys is None:
                # hole/unmapped => return zeros
                out += b"\x00" * take
            else:
                blk_off = self._off(phys * bs)
                blk = self.io.read(blk_off, bs)
                out += blk[inner:inner + take]

            remaining -= take
            cur_off += take
            if remaining <= 0:
                break

        return bytes(out)

    def _read_file_bytes_as_dirblob(self, inode: _Inode) -> bytes:
        return self._read_inode_bytes(inode, 0, inode.size)

    def _resolve(self, path: str) -> int:
        p = abspath(self._cwd, path)
        if p == "/":
            return 2  # ext4 root inode is 2
        cur = 2
        parts = [x for x in p.split("/") if x]
        for name in parts:
            ents = self._read_dir(cur)
            if name not in ents:
                raise NotFound(p)
            cur = ents[name]
        return cur

    # -------- public API --------
    def ls(self, path: str = ".") -> List[DirEntry]:
        ino_num = self._resolve(path)
        ino = self._read_inode(ino_num)
        if not self._inode_is_dir(ino):
            raise NotADirectory(path)
        ents = self._read_dir(ino_num)
        out = []
        for name, child in sorted(ents.items()):
            cino = self._read_inode(child)
            out.append(DirEntry(
                name=name,
                is_dir=self._inode_is_dir(cino),
                inode=child,
                size=cino.size,
            ))
        return out

    def stat(self, path: str) -> Stat:
        p = abspath(self._cwd, path)
        ino_num = self._resolve(p)
        ino = self._read_inode(ino_num)
        is_dir = self._inode_is_dir(ino)
        return Stat(
            path=p,
            is_dir=is_dir,
            size=ino.size,
            mtime=ino.mtime,
            atime=ino.atime,
            ctime=ino.ctime,
            uid=ino.uid,
            gid=ino.gid,
            inode=ino_num,
            mode=ino.mode & 0x0FFF,
        )

    def open(self, path: str):
        p = abspath(self._cwd, path)
        ino_num = self._resolve(p)
        ino = self._read_inode(ino_num)
        if self._inode_is_dir(ino):
            raise IsADirectory(p)
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
                if self._closed:
                    return b""
                if n < 0:
                    n = ino.size - self._pos
                b = fs._read_inode_bytes(ino, self._pos, n)
                self._pos += len(b)
                return b

            def seek(self, off: int, whence: int = 0) -> int:
                if whence == 0:
                    self._pos = max(0, off)
                elif whence == 1:
                    self._pos = max(0, self._pos + off)
                elif whence == 2:
                    self._pos = max(0, ino.size + off)
                else:
                    raise ValueError("bad whence")
                return self._pos

            def tell(self) -> int:
                return self._pos

            def close(self) -> None:
                self._closed = True


        return _FH()

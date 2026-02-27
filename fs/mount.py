# fs/mount.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Optional, Literal, Union

from .core import FsError, CorruptFs
from .ext4 import Ext4FS
from .exfat import ExFatFS
from .fat32 import Fat32FS

FsType = Literal["ext4", "exfat", "fat32", "auto"]

@dataclass(frozen=True)
class MountInfo:
    fs_type: str
    base_offset: int
    # optionally expose FS-specific parameters later (block size, cluster size, etc.)

def mount(
    stream: BinaryIO,
    *,
    offset: int = 0,
    size: int | None = None,
    fs_type: FsType = "auto",
) -> tuple[Union[Ext4FS, ExFatFS], MountInfo]:
    """
    Mount a filesystem located at `offset` bytes within `stream`.

    Args:
        stream: Seekable binary stream (file handle, BytesIO, etc.)
        offset: Byte offset where the filesystem volume starts.
        size: Optional maximum readable size for the volume (bounds checks).
        fs_type: "ext4", "exfat", or "auto" to probe.

    Returns:
        (fs_instance, MountInfo)

    Raises:
        CorruptFs / FsError if probing or mount fails.
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if size is not None and size <= 0:
        raise ValueError("size must be > 0 if provided")

    if fs_type == "ext4":
        fs = Ext4FS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="ext4", base_offset=offset)

    if fs_type == "exfat":
        fs = ExFatFS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="exfat", base_offset=offset)
    
    if fs_type == "fat32":
        fs = Fat32FS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="fat32", base_offset=offset)

    # auto-probe: ext4 → exfat → fat32
    errors: list[Exception] = []
    for t, cls in (("ext4", Ext4FS), ("exfat", ExFatFS), ("fat32", Fat32FS)):
        try:
            fs = cls(stream, base_offset=offset, total_size=size)
            return fs, MountInfo(fs_type=t, base_offset=offset)
        except Exception as e:
            errors.append(e)

    raise CorruptFs(f"auto-mount failed at offset 0x{offset:x} (ext4/exfat/fat32 probes failed)") from errors[-1]

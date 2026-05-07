# fs/mount.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Optional, Literal, Union, Sequence

from .core import FsError, CorruptFs
from .ext4 import Ext4FS
from .exfat import ExFatFS
from .fat32 import Fat32FS
from .ewf import EwfStream, check_signature as is_ewf

FsType = Literal["ext4", "exfat", "fat32", "auto"]

@dataclass(frozen=True)
class MountInfo:
    fs_type: str
    base_offset: int
    container: Optional[str] = None  # e.g. "ewf" when wrapped in an E01/Ex01 stream
    # optionally expose FS-specific parameters later (block size, cluster size, etc.)

def mount(
    stream: Union[BinaryIO, Sequence[BinaryIO]],
    *,
    offset: int = 0,
    size: int | None = None,
    fs_type: FsType = "auto",
) -> tuple[Union[Ext4FS, ExFatFS], MountInfo]:
    """
    Mount a filesystem located at `offset` bytes within `stream`.

    Args:
        stream: Seekable binary stream (file handle, BytesIO, etc.), or a
            sequence of streams for a multi-segment E01/Ex01 evidence file
            (one stream per segment, in ascending order).
        offset: Byte offset where the filesystem volume starts (relative to
            the raw image, i.e. inside the EWF container if one is present).
        size: Optional maximum readable size for the volume (bounds checks).
        fs_type: "ext4", "exfat", "fat32", or "auto" to probe.

    Returns:
        (fs_instance, MountInfo)

    Raises:
        CorruptFs / FsError if probing or mount fails.
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if size is not None and size <= 0:
        raise ValueError("size must be > 0 if provided")

    # Normalize input and transparently wrap E01/Ex01 evidence containers.
    if isinstance(stream, (list, tuple)):
        segments = list(stream)
        if not segments:
            raise ValueError("stream sequence must contain at least one segment")
    else:
        segments = [stream]

    container: Optional[str] = None
    if is_ewf(segments[0]):
        stream = EwfStream(segments)
        container = "ewf"
    else:
        if len(segments) > 1:
            raise ValueError(
                "multiple streams are only valid for E01/Ex01 evidence files"
            )
        stream = segments[0]

    if fs_type == "ext4":
        fs = Ext4FS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="ext4", base_offset=offset, container=container)

    if fs_type == "exfat":
        fs = ExFatFS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="exfat", base_offset=offset, container=container)

    if fs_type == "fat32":
        fs = Fat32FS(stream, base_offset=offset, total_size=size)
        return fs, MountInfo(fs_type="fat32", base_offset=offset, container=container)

    # auto-probe: ext4 → exfat → fat32
    errors: list[Exception] = []
    for t, cls in (("ext4", Ext4FS), ("exfat", ExFatFS), ("fat32", Fat32FS)):
        try:
            fs = cls(stream, base_offset=offset, total_size=size)
            return fs, MountInfo(fs_type=t, base_offset=offset, container=container)
        except Exception as e:
            errors.append(e)

    raise CorruptFs(f"auto-mount failed at offset 0x{offset:x} (ext4/exfat/fat32 probes failed)") from errors[-1]

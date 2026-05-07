# fs/ewf.py
from __future__ import annotations
import io
from typing import BinaryIO, Iterable, List


def check_signature(stream: BinaryIO) -> bool:
    """Return True if `stream` looks like an E01 or Ex01 segment."""
    try:
        import pyewf
    except ImportError:
        return False
    pos = stream.tell()
    try:
        stream.seek(0)
        return bool(pyewf.check_file_signature_file_object(stream))
    finally:
        stream.seek(pos)


class EwfStream(io.RawIOBase):
    """
    Read-only, seekable stream presenting the raw image inside one or more
    E01 / Ex01 evidence segments.

    The caller passes already-open binary streams (one per segment, in
    ascending order). Nothing is written to disk — pyewf reads the segments
    via file-object IO.

    Note: a multi-segment image must be presented in full; pyewf cannot
    discover sibling segments without filesystem access.
    """

    def __init__(self, segments: Iterable[BinaryIO]) -> None:
        try:
            import pyewf
        except ImportError as e:
            raise RuntimeError(
                "E01/Ex01 support requires pyewf "
                "(`pip install libewf-python`)"
            ) from e

        segs: List[BinaryIO] = list(segments)
        if not segs:
            raise ValueError("EwfStream requires at least one segment")

        handle = pyewf.handle()
        handle.open_file_objects(segs)
        self._handle = handle
        self._segments = segs
        self._size = handle.get_media_size()

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False

    def read(self, n: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O on closed stream")
        if n is None or n < 0:
            n = self._size - self._handle.get_offset()
        if n <= 0:
            return b""
        return self._handle.read_buffer(n)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, off: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O on closed stream")
        return self._handle.seek_offset(off, whence)

    def tell(self) -> int:
        return self._handle.get_offset()

    @property
    def size(self) -> int:
        return self._size

    def close(self) -> None:
        if not self.closed:
            try:
                self._handle.close()
            finally:
                super().close()

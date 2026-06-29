# tools/_io.py
"""Input normalization shared by the offline extraction modules.

Every module in this package accepts the same range of inputs so the caller
can feed either a file produced by this project's ``fs`` package
(``fs.open(path)``) or a plain Python file object, and so on:

* a file-like object exposing ``read`` (and ideally ``seek``/``tell``) --
  this covers ``fs`` handles, ``open(path, "rb")`` objects, ``io.BytesIO``,
  ``EwfStream``, etc.
* a filesystem path (``str`` or ``os.PathLike``)
* a ``bytes``/``bytearray`` blob

The helpers here turn any of those into a seekable binary stream and take
care of closing only the streams we opened ourselves.
"""
from __future__ import annotations

import io
import os
from typing import BinaryIO, Union

# Anything we are willing to treat as a source of bytes.
Source = Union[BinaryIO, str, "os.PathLike[str]", bytes, bytearray, memoryview]


def _is_fileobj(src: object) -> bool:
    return hasattr(src, "read") and callable(getattr(src, "read"))


def _is_seekable(stream: object) -> bool:
    seekable = getattr(stream, "seekable", None)
    if callable(seekable):
        try:
            return bool(seekable())
        except Exception:
            return False
    # Fall back to probing for the methods we need.
    return hasattr(stream, "seek") and hasattr(stream, "tell")


class OpenStream:
    """Context manager yielding a seekable binary stream from any ``Source``.

    ``owned`` streams (paths we opened, or in-memory buffers we created from a
    non-seekable source) are closed on exit; caller-supplied file objects are
    left open so the caller keeps ownership of their ``fs`` handle.

    Usage::

        with OpenStream(src) as stream:
            stream.seek(0)
            ...
    """

    def __init__(self, src: Source, *, need_seek: bool = True) -> None:
        self._src = src
        self._need_seek = need_seek
        self._owned = False
        self.stream: BinaryIO

    def __enter__(self) -> BinaryIO:
        src = self._src
        if isinstance(src, (bytes, bytearray, memoryview)):
            self.stream = io.BytesIO(bytes(src))
            self._owned = True
        elif isinstance(src, (str, os.PathLike)):
            self.stream = open(os.fspath(src), "rb")
            self._owned = True
        elif _is_fileobj(src):
            stream = src  # type: ignore[assignment]
            if self._need_seek and not _is_seekable(stream):
                # Materialize a non-seekable stream (e.g. a pipe) so modules
                # that need random access keep working.
                self.stream = io.BytesIO(stream.read())
                self._owned = True
            else:
                self.stream = stream  # type: ignore[assignment]
        else:
            raise TypeError(
                f"unsupported source type {type(src).__name__!r}; expected a "
                "file-like object, path, or bytes"
            )
        return self.stream

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._owned:
            try:
                self.stream.close()
            except Exception:
                pass
        return False  # never suppress exceptions


def read_all(src: Source) -> bytes:
    """Return the full contents of ``src`` as ``bytes``."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    with OpenStream(src, need_seek=False) as stream:
        return stream.read()

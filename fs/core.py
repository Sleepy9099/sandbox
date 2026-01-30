# fs/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Optional, Iterable, List, Protocol
import time
class FsError(Exception): ...
class NotFound(FsError): ...
class NotADirectory(FsError): ...
class IsADirectory(FsError): ...
class PermissionDenied(FsError): ...
class CorruptFs(FsError): ...

@dataclass(frozen=True)
class Stat:
    path: str
    is_dir: bool
    size: int
    mtime: int = 0
    atime: int = 0
    ctime: int = 0
    mode: int = 0o444
    uid: int = 0
    gid: int = 0
    inode: Optional[int] = None

@dataclass(frozen=True)
class DirEntry:
    name: str
    is_dir: bool
    inode: Optional[int] = None
    size: Optional[int] = None

class FileHandle(Protocol):
    def read(self, n: int = -1) -> bytes: ...
    def seek(self, off: int, whence: int = 0) -> int: ...
    def tell(self) -> int: ...
    def close(self) -> None: ...

class FSBase:
    """
    Read-only, path-based filesystem interface.
    Implementations should treat paths as POSIX-style: /a/b/c
    """
    def __init__(self) -> None:
        self._cwd = "/"

    def pwd(self) -> str:
        return self._cwd

    # Implement these
    def ls(self, path: str = ".") -> List[DirEntry]:
        raise NotImplementedError

    def cd(self, path: str) -> None:
        st = self.stat(path)
        if not st.is_dir:
            raise NotADirectory(path)
        self._cwd = self._abspath(path)

    def stat(self, path: str) -> Stat:
        raise NotImplementedError

    def open(self, path: str) -> FileHandle:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        try:
            self.stat(path)
            return True
        except NotFound:
            return False

    # Helpers
    def _abspath(self, path: str) -> str:
        from .pathutil import abspath
        return abspath(self._cwd, path)

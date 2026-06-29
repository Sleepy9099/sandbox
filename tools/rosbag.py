# tools/rosbag.py
"""Extract embedded (compressed) images from ROS 1 ``.bag`` files, offline.

This is a native, dependency-free reader for the ROS 1 bag *2.0* format. It
walks the record/chunk structure straight off a file-like object -- an ``fs``
handle from this project, a plain ``open(path, "rb")`` object, ``BytesIO`` or
raw ``bytes`` -- so nothing is written to disk and no ROS install is required.

For every ``sensor_msgs/CompressedImage`` message it yields the raw payload,
which for the common case *is already a JPEG (or PNG) file* -- exactly the
"embedded compressed jpeg" stored in the bag. Raw ``sensor_msgs/Image``
messages are skipped unless numpy + OpenCV (cv2) are importable, in which case
they are transcoded to JPEG on the fly (this mirrors the usual
``cv_bridge`` + ``cv2`` workflow named in the request).

``bz2`` chunk compression is handled natively; ``lz4`` is handled when the
``lz4`` package is importable.

Example::

    from tools.rosbag import extract_jpegs
    with fs.open("/logs/drive.bag") as fh:
        for name, jpeg in extract_jpegs(fh):
            ...                      # name like 'cam_front_0003.jpg'
"""
from __future__ import annotations

import bz2
import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ._io import OpenStream, Source

__all__ = [
    "Connection", "ImageMessage", "Ros1BagReader",
    "read_bag", "iter_images", "extract_jpegs",
]

# Record opcodes (the 'op' header field).
_OP_MSG_DATA = 0x02
_OP_BAG_HEADER = 0x03
_OP_INDEX_DATA = 0x04
_OP_CHUNK = 0x05
_OP_CHUNK_INFO = 0x06
_OP_CONNECTION = 0x07

_COMPRESSED_TYPE = "sensor_msgs/CompressedImage"
_IMAGE_TYPE = "sensor_msgs/Image"


@dataclass(frozen=True)
class Connection:
    conn_id: int
    topic: str
    msgtype: str
    md5sum: str = ""


@dataclass
class ImageMessage:
    """One decoded image-bearing message."""

    topic: str
    msgtype: str
    conn_id: int
    seq: int
    stamp_ns: int                 # ROS timestamp in nanoseconds
    frame_id: str
    format: str                   # CompressedImage 'format' or Image 'encoding'
    codec: str                    # 'jpeg' | 'png' | 'raw' | 'unknown'
    data: bytes                   # image bytes (JPEG/PNG payload)

    def suggested_ext(self) -> str:
        return {"jpeg": "jpg", "png": "png"}.get(self.codec, "bin")


# --------------------------------------------------------------------------- #
# Low-level binary helpers
# --------------------------------------------------------------------------- #
def _parse_fields(buf: bytes) -> Dict[str, bytes]:
    """Parse a ROS bag record header / connection header field block."""
    fields: Dict[str, bytes] = {}
    i, n = 0, len(buf)
    while i + 4 <= n:
        (flen,) = struct.unpack_from("<I", buf, i)
        i += 4
        field = buf[i:i + flen]
        i += flen
        key, sep, val = field.partition(b"=")
        if sep:
            fields[key.decode("latin-1")] = val
    return fields


class _Cursor:
    """Little-endian sequential reader over a message buffer."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.buf, self.pos)
        self.pos += 4
        return v

    def time_ns(self) -> int:
        sec, nsec = struct.unpack_from("<II", self.buf, self.pos)
        self.pos += 8
        return sec * 1_000_000_000 + nsec

    def string(self) -> str:
        n = self.u32()
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", "replace")

    def blob(self) -> bytes:
        n = self.u32()
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


def _detect_codec(fmt: str, data: bytes) -> str:
    f = fmt.lower()
    if data[:3] == b"\xff\xd8\xff" or "jpeg" in f or "jpg" in f:
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n" or "png" in f:
        return "png"
    return "unknown"


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
class Ros1BagReader:
    """Streaming reader for ROS 1 bag (v2.0) files."""

    MAGIC = b"#ROSBAG V2.0\n"

    def __init__(self, stream):
        self._s = stream
        magic = stream.read(len(self.MAGIC))
        if magic != self.MAGIC:
            raise ValueError(
                f"not a ROS1 bag (bad magic {magic!r}); "
                "ROS2 bags (sqlite3/mcap directories) are not supported"
            )
        self.connections: Dict[int, Connection] = {}

    # -- record framing ----------------------------------------------------
    def _read_record(self) -> Optional[Tuple[Dict[str, bytes], bytes]]:
        s = self._s
        raw = s.read(4)
        if len(raw) < 4:
            return None
        (hlen,) = struct.unpack("<I", raw)
        header = s.read(hlen)
        (dlen,) = struct.unpack("<I", s.read(4))
        data = s.read(dlen)
        return _parse_fields(header), data

    @staticmethod
    def _decompress(compression: str, data: bytes, size: int) -> bytes:
        if compression == "none" or not compression:
            return data
        if compression == "bz2":
            return bz2.decompress(data)
        if compression == "lz4":
            try:
                import lz4.frame  # optional
            except ImportError as e:
                raise RuntimeError(
                    "lz4-compressed bag chunk requires the 'lz4' package "
                    "(`pip install lz4`)"
                ) from e
            return lz4.frame.decompress(data)
        raise RuntimeError(f"unsupported chunk compression {compression!r}")

    def _register_connection(self, header: Dict[str, bytes], data: bytes) -> None:
        conn_id = struct.unpack("<I", header["conn"])[0]
        topic = header.get("topic", b"").decode("utf-8", "replace")
        info = _parse_fields(data)
        msgtype = info.get("type", b"").decode("utf-8", "replace")
        md5 = info.get("md5sum", b"").decode("latin-1")
        topic = info.get("topic", topic.encode()).decode("utf-8", "replace") \
            if "topic" in info else topic
        self.connections[conn_id] = Connection(conn_id, topic, msgtype, md5)

    # -- message decoding --------------------------------------------------
    def _decode_message(
        self, conn: Connection, data: bytes, *, transcode_raw: bool
    ) -> Optional[ImageMessage]:
        if conn.msgtype == _COMPRESSED_TYPE:
            return self._decode_compressed(conn, data)
        if conn.msgtype == _IMAGE_TYPE and transcode_raw:
            return self._decode_raw(conn, data)
        return None

    @staticmethod
    def _read_header(c: _Cursor) -> Tuple[int, int, str]:
        seq = c.u32()
        stamp = c.time_ns()
        frame_id = c.string()
        return seq, stamp, frame_id

    def _decode_compressed(self, conn, data) -> ImageMessage:
        c = _Cursor(data)
        seq, stamp, frame_id = self._read_header(c)
        fmt = c.string()
        payload = c.blob()
        return ImageMessage(
            topic=conn.topic, msgtype=conn.msgtype, conn_id=conn.conn_id,
            seq=seq, stamp_ns=stamp, frame_id=frame_id, format=fmt,
            codec=_detect_codec(fmt, payload), data=payload,
        )

    def _decode_raw(self, conn, data) -> Optional[ImageMessage]:
        try:
            import numpy as np
            import cv2
        except ImportError:
            return None
        c = _Cursor(data)
        seq, stamp, frame_id = self._read_header(c)
        height = c.u32()
        width = c.u32()
        encoding = c.string()
        _is_bigendian = c.u8()
        _step = c.u32()
        buf = c.blob()
        arr = self._reshape(np, buf, height, width, encoding)
        if arr is None:
            return None
        if encoding == "rgb8":
            arr = arr[:, :, ::-1]                       # cv2 wants BGR
        elif encoding == "rgba8":
            arr = arr[:, :, [2, 1, 0, 3]]
        ok, enc = cv2.imencode(".jpg", arr)
        if not ok:
            return None
        return ImageMessage(
            topic=conn.topic, msgtype=conn.msgtype, conn_id=conn.conn_id,
            seq=seq, stamp_ns=stamp, frame_id=frame_id, format=encoding,
            codec="jpeg", data=enc.tobytes(),
        )

    @staticmethod
    def _reshape(np, buf, height, width, encoding):
        spec = {
            "mono8": ("uint8", 1), "8UC1": ("uint8", 1),
            "mono16": ("uint16", 1), "16UC1": ("uint16", 1),
            "bgr8": ("uint8", 3), "rgb8": ("uint8", 3), "8UC3": ("uint8", 3),
            "bgra8": ("uint8", 4), "rgba8": ("uint8", 4),
        }.get(encoding)
        if spec is None:
            return None
        dtype, ch = spec
        arr = np.frombuffer(buf, dtype=dtype)
        need = height * width * ch
        if arr.size < need:
            return None
        arr = arr[:need].reshape((height, width, ch) if ch > 1 else (height, width))
        return arr

    # -- public iteration --------------------------------------------------
    def iter_images(
        self,
        topics: Optional[Sequence[str]] = None,
        *,
        transcode_raw: bool = True,
    ) -> Iterator[ImageMessage]:
        """Yield image messages in bag order.

        ``topics`` optionally restricts output to specific topics. Raw
        ``Image`` messages are transcoded to JPEG when ``transcode_raw`` is set
        and numpy + cv2 are available; otherwise only ``CompressedImage``
        messages are produced.
        """
        wanted = set(topics) if topics else None
        while True:
            rec = self._read_record()
            if rec is None:
                break
            header, data = rec
            op = header.get("op", b"\x00")[0] if header.get("op") else 0
            if op == _OP_CONNECTION:
                self._register_connection(header, data)
            elif op == _OP_MSG_DATA:
                yield from self._emit(header, data, wanted, transcode_raw)
            elif op == _OP_CHUNK:
                comp = header.get("compression", b"none").decode("latin-1")
                size = struct.unpack("<I", header["size"])[0] if "size" in header else 0
                yield from self._iter_chunk(
                    self._decompress(comp, data, size), wanted, transcode_raw)
            # INDEX_DATA / CHUNK_INFO / BAG_HEADER carry no image payloads.

    def _iter_chunk(self, buf, wanted, transcode_raw):
        i, n = 0, len(buf)
        while i + 4 <= n:
            (hlen,) = struct.unpack_from("<I", buf, i)
            i += 4
            header = _parse_fields(buf[i:i + hlen])
            i += hlen
            (dlen,) = struct.unpack_from("<I", buf, i)
            i += 4
            data = buf[i:i + dlen]
            i += dlen
            op = header.get("op", b"\x00")[0] if header.get("op") else 0
            if op == _OP_CONNECTION:
                self._register_connection(header, data)
            elif op == _OP_MSG_DATA:
                yield from self._emit(header, data, wanted, transcode_raw)

    def _emit(self, header, data, wanted, transcode_raw):
        conn_id = struct.unpack("<I", header["conn"])[0]
        conn = self.connections.get(conn_id)
        if conn is None:
            return
        if wanted is not None and conn.topic not in wanted:
            return
        msg = self._decode_message(conn, data, transcode_raw=transcode_raw)
        if msg is not None:
            yield msg


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #
def read_bag(src: Source) -> Ros1BagReader:
    """Open ``src`` as a ROS1 bag. Caller keeps ownership of file objects."""
    stream = OpenStream(src).__enter__()  # kept open for streaming reads
    return Ros1BagReader(stream)


def iter_images(
    src: Source,
    topics: Optional[Sequence[str]] = None,
    *,
    transcode_raw: bool = True,
) -> Iterator[ImageMessage]:
    """Yield :class:`ImageMessage` objects from a ROS1 bag source."""
    with OpenStream(src) as stream:
        reader = Ros1BagReader(stream)
        yield from reader.iter_images(topics, transcode_raw=transcode_raw)


def extract_jpegs(
    src: Source,
    out_dir: Optional[str] = None,
    topics: Optional[Sequence[str]] = None,
) -> List[Tuple[str, bytes]]:
    """Extract all embedded JPEG/PNG images from a ROS1 bag.

    Returns a list of ``(filename, image_bytes)``. When ``out_dir`` is given
    the images are also written there. Filenames are derived from the topic
    and a per-topic running index, e.g. ``cam_front_0003.jpg``.
    """
    import os

    results: List[Tuple[str, bytes]] = []
    counters: Dict[str, int] = {}
    for msg in iter_images(src, topics):
        slug = msg.topic.strip("/").replace("/", "_") or "image"
        idx = counters.get(slug, 0)
        counters[slug] = idx + 1
        name = f"{slug}_{idx:04d}.{msg.suggested_ext()}"
        results.append((name, msg.data))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(msg.data)
    return results

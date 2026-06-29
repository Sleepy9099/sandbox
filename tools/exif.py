# tools/exif.py
"""Offline metadata / EXIF extraction for common media containers.

Pure Python, no third-party dependencies and no bundled assets -- everything
is parsed natively from the byte stream, so it runs fully offline.

Supported containers:

* **JPEG/JFIF**     -- Exif (APP1/TIFF) tags, GPS, image dimensions, XMP blob.
* **TIFF**          -- the same IFD machinery used for JPEG Exif.
* **PNG**           -- ``IHDR`` geometry, ``tEXt``/``zTXt``/``iTXt`` text,
                       ``tIME`` timestamp and an embedded ``eXIf`` block.
* **MP4 / MOV /     -- ISO base-media boxes: ``ftyp`` brand, ``mvhd`` creation
  ISO-BMFF**           time/duration, track geometry and ``udta`` ISO-6709 GPS.

Entry point::

    from tools.exif import read_metadata

    with fs.open("/DCIM/IMG_0001.JPG") as fh:
        meta = read_metadata(fh)
    print(meta.gps)            # {'latitude': 37.77, 'longitude': -122.41, ...}
    print(meta.tags["Model"])  # 'iPhone 12'

The function accepts ``fs`` handles, plain file objects, paths or ``bytes``.
"""
from __future__ import annotations

import datetime as _dt
import io
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._io import OpenStream, Source

__all__ = ["Metadata", "read_metadata", "detect_format"]

# Seconds between the 1904-01-01 (QuickTime/MP4) and 1970-01-01 (Unix) epochs.
_MP4_EPOCH_DELTA = 2082844800


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class Metadata:
    """Structured result of :func:`read_metadata`."""

    format: str = "unknown"          # 'jpeg' | 'tiff' | 'png' | 'mp4' | ...
    mime: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    tags: Dict[str, Any] = field(default_factory=dict)   # EXIF / IFD tags
    gps: Dict[str, Any] = field(default_factory=dict)    # decoded lat/lon/alt
    datetime: Optional[str] = None   # best-effort capture timestamp (ISO/EXIF)
    extra: Dict[str, Any] = field(default_factory=dict)  # container specifics

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "tags": self.tags,
            "gps": self.gps,
            "datetime": self.datetime,
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #
def detect_format(head: bytes) -> str:
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    # ISO base media (mp4/mov/m4v/3gp/heic): bytes 4..8 == 'ftyp'
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


_MIME = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff",
    "mp4": "video/mp4",
}


# --------------------------------------------------------------------------- #
# TIFF / EXIF IFD parsing (shared by JPEG APP1, standalone TIFF, PNG eXIf)
# --------------------------------------------------------------------------- #
# TIFF field type -> (struct code, byte size)
_TIFF_TYPES = {
    1: ("B", 1),   # BYTE
    2: ("c", 1),   # ASCII
    3: ("H", 2),   # SHORT
    4: ("I", 4),   # LONG
    5: ("II", 8),  # RATIONAL (num/den)
    6: ("b", 1),   # SBYTE
    7: ("c", 1),   # UNDEFINED
    8: ("h", 2),   # SSHORT
    9: ("i", 4),   # SLONG
    10: ("ii", 8),  # SRATIONAL
    11: ("f", 4),  # FLOAT
    12: ("d", 8),  # DOUBLE
}

_TAG_NAMES = {
    0x0100: "ImageWidth", 0x0101: "ImageLength", 0x0102: "BitsPerSample",
    0x0103: "Compression", 0x010E: "ImageDescription", 0x010F: "Make",
    0x0110: "Model", 0x0112: "Orientation", 0x011A: "XResolution",
    0x011B: "YResolution", 0x0128: "ResolutionUnit", 0x0131: "Software",
    0x0132: "DateTime", 0x013B: "Artist", 0x8298: "Copyright",
    0x8769: "ExifOffset", 0x8825: "GPSInfo",
    0x829A: "ExposureTime", 0x829D: "FNumber", 0x8822: "ExposureProgram",
    0x8827: "ISOSpeedRatings", 0x9000: "ExifVersion",
    0x9003: "DateTimeOriginal", 0x9004: "DateTimeDigitized",
    0x9201: "ShutterSpeedValue", 0x9202: "ApertureValue",
    0x9204: "ExposureBiasValue", 0x9207: "MeteringMode", 0x9209: "Flash",
    0x920A: "FocalLength", 0xA002: "PixelXDimension", 0xA003: "PixelYDimension",
    0xA405: "FocalLengthIn35mmFilm", 0xA433: "LensMake", 0xA434: "LensModel",
}

_GPS_TAG_NAMES = {
    0x0000: "GPSVersionID", 0x0001: "GPSLatitudeRef", 0x0002: "GPSLatitude",
    0x0003: "GPSLongitudeRef", 0x0004: "GPSLongitude", 0x0005: "GPSAltitudeRef",
    0x0006: "GPSAltitude", 0x0007: "GPSTimeStamp", 0x0008: "GPSSatellites",
    0x0009: "GPSStatus", 0x0010: "GPSImgDirectionRef",
    0x0011: "GPSImgDirection", 0x001D: "GPSDateStamp",
}


def _ascii(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _parse_tiff(buf: bytes) -> Dict[str, Any]:
    """Parse a TIFF/EXIF byte block into a flat ``{name: value}`` dict."""
    if len(buf) < 8:
        return {}
    bo = buf[:2]
    if bo == b"II":
        e = "<"
    elif bo == b"MM":
        e = ">"
    else:
        return {}
    (ifd0_off,) = struct.unpack_from(e + "I", buf, 4)
    out: Dict[str, Any] = {}
    _read_ifd(buf, ifd0_off, e, _TAG_NAMES, out, set())
    return out


def _read_ifd(
    buf: bytes,
    off: int,
    e: str,
    names: Dict[int, str],
    out: Dict[str, Any],
    seen: set,
) -> None:
    if off <= 0 or off + 2 > len(buf) or off in seen:
        return
    seen.add(off)
    (count,) = struct.unpack_from(e + "H", buf, off)
    pos = off + 2
    for _ in range(count):
        if pos + 12 > len(buf):
            break
        tag, ftype, n = struct.unpack_from(e + "HHI", buf, pos)
        value = _read_value(buf, pos + 8, e, ftype, n)
        pos += 12

        name = names.get(tag, f"0x{tag:04X}")
        if tag == 0x8769:  # Exif sub-IFD
            if isinstance(value, int):
                _read_ifd(buf, value, e, _TAG_NAMES, out, seen)
            continue
        if tag == 0x8825:  # GPS sub-IFD
            if isinstance(value, int):
                gps: Dict[str, Any] = {}
                _read_ifd(buf, value, e, _GPS_TAG_NAMES, gps, seen)
                out["GPSInfo"] = gps
            continue
        out[name] = value


def _read_value(buf: bytes, entry_off: int, e: str, ftype: int, n: int):
    spec = _TIFF_TYPES.get(ftype)
    if spec is None:
        return None
    code, size = spec
    total = size * n
    if total <= 4:
        data_off = entry_off
    else:
        (data_off,) = struct.unpack_from(e + "I", buf, entry_off)
    if data_off + total > len(buf):
        return None
    raw = buf[data_off:data_off + total]

    if ftype in (2, 7):  # ASCII / UNDEFINED
        if ftype == 2:
            return _ascii(raw)
        return raw

    vals: List[Any] = []
    if ftype in (5, 10):  # RATIONAL / SRATIONAL
        rc = "II" if ftype == 5 else "ii"
        for i in range(n):
            num, den = struct.unpack_from(e + rc, raw, i * 8)
            vals.append((num / den) if den else 0.0)
    else:
        for i in range(n):
            (v,) = struct.unpack_from(e + code, raw, i * size)
            vals.append(v)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _decode_gps(gps: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw GPS IFD tags into decimal-degree latitude/longitude/alt."""
    out: Dict[str, Any] = {}

    def _dms(value, ref, neg_ref) -> Optional[float]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        deg, minute, sec = value[0], value[1], value[2]
        dd = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
        if isinstance(ref, str) and ref.upper().startswith(neg_ref):
            dd = -dd
        return dd

    lat = _dms(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"), "S")
    lon = _dms(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"), "W")
    if lat is not None:
        out["latitude"] = round(lat, 8)
    if lon is not None:
        out["longitude"] = round(lon, 8)

    alt = gps.get("GPSAltitude")
    if isinstance(alt, (int, float)):
        ref = gps.get("GPSAltitudeRef")
        ref_val = ref[0] if isinstance(ref, (list, tuple)) else ref
        out["altitude"] = -float(alt) if ref_val == 1 else float(alt)

    if gps.get("GPSDateStamp"):
        out["datestamp"] = gps["GPSDateStamp"]
    ts = gps.get("GPSTimeStamp")
    if isinstance(ts, (list, tuple)) and len(ts) >= 3:
        out["timestamp"] = "%02d:%02d:%06.3f" % (ts[0], ts[1], ts[2])
    return out


def _finalize_exif(meta: Metadata, tags: Dict[str, Any]) -> None:
    gps = tags.pop("GPSInfo", None)
    meta.tags.update(tags)
    if isinstance(gps, dict):
        meta.tags["GPSInfo"] = gps
        decoded = _decode_gps(gps)
        if decoded:
            meta.gps.update(decoded)
    meta.datetime = (
        tags.get("DateTimeOriginal")
        or tags.get("DateTime")
        or tags.get("DateTimeDigitized")
        or meta.datetime
    )


# --------------------------------------------------------------------------- #
# JPEG
# --------------------------------------------------------------------------- #
def _parse_jpeg(stream) -> Metadata:
    meta = Metadata(format="jpeg", mime=_MIME["jpeg"])
    stream.seek(0)
    data = stream.read()
    i = 2  # skip SOI (FFD8)
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan -> compressed data follows
            break
        seg_len = struct.unpack_from(">H", data, i + 2)[0]
        seg_start = i + 4
        seg = data[seg_start:seg_start + seg_len - 2]

        if marker == 0xE1:  # APP1: Exif or XMP
            if seg.startswith(b"Exif\x00\x00"):
                _finalize_exif(meta, _parse_tiff(seg[6:]))
            elif seg.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                meta.extra["xmp"] = seg[29:].decode("utf-8", "replace")
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            # SOFn frame header carries the real pixel dimensions.
            if len(seg) >= 5:
                meta.height = struct.unpack_from(">H", seg, 1)[0]
                meta.width = struct.unpack_from(">H", seg, 3)[0]

        i = seg_start + seg_len - 2
    # EXIF PixelXDimension is more authoritative than SOF when present.
    meta.width = meta.tags.get("PixelXDimension", meta.width)
    meta.height = meta.tags.get("PixelYDimension", meta.height)
    return meta


# --------------------------------------------------------------------------- #
# TIFF (standalone)
# --------------------------------------------------------------------------- #
def _parse_tiff_file(stream) -> Metadata:
    meta = Metadata(format="tiff", mime=_MIME["tiff"])
    stream.seek(0)
    _finalize_exif(meta, _parse_tiff(stream.read()))
    meta.width = meta.tags.get("ImageWidth", meta.width)
    meta.height = meta.tags.get("ImageLength", meta.height)
    return meta


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def _parse_png(stream) -> Metadata:
    meta = Metadata(format="png", mime=_MIME["png"])
    stream.seek(8)  # skip signature
    text: Dict[str, str] = {}
    while True:
        hdr = stream.read(8)
        if len(hdr) < 8:
            break
        length, ctype = struct.unpack(">I4s", hdr)
        body = stream.read(length)
        stream.read(4)  # CRC
        if ctype == b"IHDR":
            meta.width, meta.height = struct.unpack(">II", body[:8])
            meta.extra["bit_depth"] = body[8]
            meta.extra["color_type"] = body[9]
        elif ctype == b"tEXt":
            key, _, val = body.partition(b"\x00")
            text[key.decode("latin-1")] = val.decode("latin-1", "replace")
        elif ctype == b"zTXt":
            key, _, rest = body.partition(b"\x00")
            if rest:
                try:
                    text[key.decode("latin-1")] = zlib.decompress(
                        rest[1:]
                    ).decode("latin-1", "replace")
                except Exception:
                    pass
        elif ctype == b"iTXt":
            text.update(_parse_itxt(body))
        elif ctype == b"tIME" and len(body) >= 7:
            y, mo, d, h, mi, s = struct.unpack(">HBBBBB", body[:7])
            meta.datetime = "%04d:%02d:%02d %02d:%02d:%02d" % (y, mo, d, h, mi, s)
        elif ctype == b"eXIf":
            _finalize_exif(meta, _parse_tiff(body))
        elif ctype == b"IEND":
            break
    if text:
        meta.extra["text"] = text
    return meta


def _parse_itxt(body: bytes) -> Dict[str, str]:
    try:
        key, _, rest = body.partition(b"\x00")
        comp_flag = rest[0]
        # rest[1] = compression method, then langTag\0 transKeyword\0 text
        parts = rest[2:].split(b"\x00", 2)
        if len(parts) < 3:
            return {}
        payload = parts[2]
        if comp_flag == 1:
            payload = zlib.decompress(payload)
        return {key.decode("latin-1"): payload.decode("utf-8", "replace")}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# MP4 / MOV / ISO base media
# --------------------------------------------------------------------------- #
# Boxes whose payload is itself a sequence of child boxes.
_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta"}


def _parse_mp4(stream) -> Metadata:
    meta = Metadata(format="mp4", mime=_MIME["mp4"])
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    brands: List[str] = []
    _walk_boxes(stream, 0, size, meta, brands, depth=0)
    if brands:
        meta.extra["brands"] = brands
        meta.mime = "video/quicktime" if brands[0] == "qt  " else meta.mime
    return meta


def _walk_boxes(stream, start, end, meta, brands, depth):
    if depth > 12:
        return
    pos = start
    while pos + 8 <= end:
        stream.seek(pos)
        hdr = stream.read(8)
        if len(hdr) < 8:
            break
        box_size, btype = struct.unpack(">I4s", hdr)
        header_len = 8
        if box_size == 1:  # 64-bit extended size
            ext = stream.read(8)
            if len(ext) < 8:
                break
            box_size = struct.unpack(">Q", ext)[0]
            header_len = 16
        elif box_size == 0:  # extends to end of file
            box_size = end - pos
        if box_size < header_len or pos + box_size > end:
            break
        body_off = pos + header_len
        body_len = box_size - header_len

        if btype == b"ftyp":
            stream.seek(body_off)
            ft = stream.read(min(body_len, 256))
            if len(ft) >= 4:
                brands.append(ft[:4].decode("latin-1", "replace"))
                for j in range(8, len(ft) - 3, 4):
                    brands.append(ft[j:j + 4].decode("latin-1", "replace"))
        elif btype == b"mvhd":
            _read_mvhd(stream, body_off, body_len, meta)
        elif btype == b"\xa9xyz" or btype == b"xyz ":
            _read_iso6709(stream, body_off, body_len, meta)
        elif btype == b"meta":
            # 'meta' has a 4-byte version/flags prefix before its children.
            _walk_boxes(stream, body_off + 4, body_off + body_len, meta, brands,
                        depth + 1)
        elif btype in _CONTAINER_BOXES:
            _walk_boxes(stream, body_off, body_off + body_len, meta, brands,
                        depth + 1)
        pos += box_size


def _read_mvhd(stream, off, length, meta) -> None:
    stream.seek(off)
    data = stream.read(min(length, 120))
    if not data:
        return
    version = data[0]
    try:
        if version == 1:
            ctime, mtime, timescale, duration = struct.unpack_from(">QQIQ", data, 4)
        else:
            ctime, mtime, timescale, duration = struct.unpack_from(">IIII", data, 4)
    except struct.error:
        return
    if timescale:
        meta.extra["duration_seconds"] = round(duration / timescale, 3)
    iso = _mp4_time_to_iso(ctime)
    if iso:
        meta.extra["creation_time"] = iso
        meta.datetime = meta.datetime or iso
    mod = _mp4_time_to_iso(mtime)
    if mod:
        meta.extra["modification_time"] = mod


def _mp4_time_to_iso(t: int) -> Optional[str]:
    if not t:
        return None
    unix = t - _MP4_EPOCH_DELTA
    try:
        return _dt.datetime.fromtimestamp(unix, _dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _read_iso6709(stream, off, length, meta) -> None:
    import re

    stream.seek(off)
    raw = stream.read(min(length, 128))
    # QuickTime stores: 2-byte size, 2-byte language code, then the string.
    text = raw[4:].decode("latin-1", "replace") if len(raw) > 4 else ""
    nums = re.findall(r"[+-]\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        meta.gps["latitude"] = float(nums[0])
        meta.gps["longitude"] = float(nums[1])
        if len(nums) >= 3:
            meta.gps["altitude"] = float(nums[2])


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_PARSERS = {
    "jpeg": _parse_jpeg,
    "tiff": _parse_tiff_file,
    "png": _parse_png,
    "mp4": _parse_mp4,
}


def read_metadata(src: Source) -> Metadata:
    """Extract metadata from ``src`` (fs handle, file object, path or bytes)."""
    with OpenStream(src) as stream:
        stream.seek(0)
        head = stream.read(16)
        fmt = detect_format(head)
        parser = _PARSERS.get(fmt)
        if parser is None:
            return Metadata(format="unknown")
        stream.seek(0)
        return parser(stream)

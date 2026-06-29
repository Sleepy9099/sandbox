# tools/selftest.py
"""Self-contained smoke tests for the offline utility modules.

Builds tiny media / ROS-bag fixtures in memory (no image or ROS libraries
needed) and exercises every module against the bundled assets. Run with::

    python -m tools.selftest
"""
from __future__ import annotations

import bz2
import datetime
import io
import struct
import zlib

from . import exif, geocode, geoid, rosbag

JPEG_MAGIC = b"\xff\xd8\xff"


# --------------------------------------------------------------------------- #
# Fixture builders (native; mirror the on-disk byte layouts)
# --------------------------------------------------------------------------- #
def _build_tiff() -> bytes:
    """A little-endian TIFF with IFD0 + Exif + GPS sub-IFDs."""
    def enc(tag, tcode, count, val4):
        return struct.pack("<HHI4s", tag, tcode, count, val4)

    ifd0_off = 8
    ifd0_size = 2 + 12 * 5 + 4
    exif_off = ifd0_off + ifd0_size
    exif_size = 2 + 12 * 3 + 4
    gps_off = exif_off + exif_size
    gps_size = 2 + 12 * 7 + 4
    pool_base = gps_off + gps_size
    pool = bytearray()

    def ascii_entry(tag, s):
        raw = s.encode() + b"\x00"
        if len(raw) <= 4:
            return enc(tag, 2, len(raw), raw + b"\x00" * (4 - len(raw)))
        off = pool_base + len(pool)
        pool.extend(raw + (b"\x00" if len(raw) % 2 else b""))
        return struct.pack("<HHII", tag, 2, len(raw), off)

    def rational_entry(tag, pairs):
        off = pool_base + len(pool)
        for num, den in pairs:
            pool.extend(struct.pack("<II", num, den))
        return struct.pack("<HHII", tag, 5, len(pairs), off)

    ifd0 = struct.pack("<H", 5)
    ifd0 += ascii_entry(0x010F, "TestCam")
    ifd0 += ascii_entry(0x0110, "Model-X")
    ifd0 += ascii_entry(0x0132, "2026:06:29 12:00:00")
    ifd0 += struct.pack("<HHII", 0x8769, 4, 1, exif_off)
    ifd0 += struct.pack("<HHII", 0x8825, 4, 1, gps_off)
    ifd0 += struct.pack("<I", 0)

    exif = struct.pack("<H", 3)
    exif += ascii_entry(0x9003, "2026:06:29 11:59:30")
    exif += struct.pack("<HHII", 0xA002, 4, 1, 640)
    exif += struct.pack("<HHII", 0xA003, 4, 1, 480)
    exif += struct.pack("<I", 0)

    gps = struct.pack("<H", 7)
    gps += ascii_entry(0x0001, "N")
    gps += rational_entry(0x0002, [(37, 1), (46, 1), (2964, 100)])
    gps += ascii_entry(0x0003, "W")
    gps += rational_entry(0x0004, [(122, 1), (25, 1), (990, 100)])
    gps += struct.pack("<HHI4s", 0x0005, 1, 1, b"\x00\x00\x00\x00")
    gps += rational_entry(0x0006, [(1000, 10)])
    gps += ascii_entry(0x001D, "2026:06:29")
    gps += struct.pack("<I", 0)

    header = b"II" + struct.pack("<HI", 42, ifd0_off)
    return header + ifd0 + exif + gps + bytes(pool)


def _build_jpeg(tiff: bytes) -> bytes:
    app1 = b"Exif\x00\x00" + tiff
    seg = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    sof_body = (bytes([8]) + struct.pack(">HH", 480, 640) + bytes([3])
                + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    sof = b"\xff\xc0" + struct.pack(">H", len(sof_body) + 2) + sof_body
    return b"\xff\xd8" + seg + sof + b"\xff\xda\x00\x02" + b"\xff\xd9"


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + ctype + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))


def _build_png(tiff: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)
    return (sig + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"tEXt", b"Author\x00Claude")
            + _png_chunk(b"eXIf", tiff) + _png_chunk(b"IEND", b""))


def _box(btype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + btype + payload


def _build_mp4() -> bytes:
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2mp41")
    secs = int(datetime.datetime(2026, 6, 29, 12, 0, 0,
                                 tzinfo=datetime.timezone.utc).timestamp()) + 2082844800
    mvhd_body = b"\x00\x00\x00\x00" + struct.pack(">IIII", secs, secs, 600, 7200)
    mvhd_body += b"\x00" * 80
    xyz = "+37.7749-122.4194/".encode("latin-1")
    xyz_box = _box(b"\xa9xyz", struct.pack(">HH", len(xyz), 0x15C7) + xyz)
    moov = _box(b"moov", _box(b"mvhd", mvhd_body) + _box(b"udta", xyz_box))
    return ftyp + moov


def _build_bag(compression: str = "none") -> bytes:
    topic = "/cam/front/compressed"
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 20 + b"\xff\xd9"

    def field(name, val):
        if isinstance(val, str):
            val = val.encode()
        f = name.encode() + b"=" + val
        return struct.pack("<I", len(f)) + f

    def record(fields, data):
        hdr = b"".join(fields)
        return struct.pack("<I", len(hdr)) + hdr + struct.pack("<I", len(data)) + data

    def rstr(s):
        b = s.encode()
        return struct.pack("<I", len(b)) + b

    bag_header = record(
        [field("op", b"\x03"), field("index_pos", struct.pack("<Q", 0)),
         field("conn_count", struct.pack("<I", 1)),
         field("chunk_count", struct.pack("<I", 1))], b"\x00" * 64)
    conn_data = (field("topic", topic)
                 + field("type", "sensor_msgs/CompressedImage")
                 + field("md5sum", "8f7a12909da2c9d3")
                 + field("message_definition", "# def"))
    conn = record([field("op", b"\x07"), field("conn", struct.pack("<I", 0)),
                   field("topic", topic)], conn_data)
    msgs = b""
    for i in range(3):
        body = (struct.pack("<III", i, 1700000000 + i, 0) + rstr("cam_link")
                + rstr("jpeg") + struct.pack("<I", len(jpeg)) + jpeg)
        msgs += record([field("op", b"\x02"), field("conn", struct.pack("<I", 0)),
                        field("time", struct.pack("<II", 1700000000 + i, 0))], body)
    inner = conn + msgs
    cdata = bz2.compress(inner) if compression == "bz2" else inner
    chunk = record([field("op", b"\x05"), field("compression", compression),
                    field("size", struct.pack("<I", len(inner)))], cdata)
    return b"#ROSBAG V2.0\n" + bag_header + chunk


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_exif():
    tiff = _build_tiff()
    for fmt, data in [("jpeg", _build_jpeg(tiff)), ("png", _build_png(tiff)),
                      ("tiff", tiff), ("mp4", _build_mp4())]:
        # Feed a file object, like an fs handle.
        meta = exif.read_metadata(io.BytesIO(data))
        assert meta.format == fmt, (fmt, meta.format)
        assert abs(meta.gps["latitude"] - 37.7749) < 1e-3, meta.gps
        assert abs(meta.gps["longitude"] - (-122.4194)) < 1e-3, meta.gps
    print("exif      : jpeg/png/tiff/mp4 parsed, GPS decoded  OK")


def test_geocode():
    rg = geocode.ReverseGeocoder()
    cases = {
        (37.7749, -122.4194): ("San Francisco", "California", "US"),
        (40.7128, -74.0060): ("New York City", "New York", "US"),
        (35.6895, 139.6917): ("Tokyo", "Tokyo", "JP"),
    }
    for (lat, lon), (name, a1, cc) in cases.items():
        r = rg.query(lat, lon)
        assert (r.name, r.admin1, r.cc) == (name, a1, cc), r
        assert r.distance_km < 10.0, r
    print("geocode   : nearest city + state correct (numpy=%s)  OK" % (rg._np is not None))


def test_geoid():
    g = geoid.Geoid()
    assert abs(g.undulation(0.0, 0.0) - 17.16) < 0.5, g.undulation(0, 0)
    # round-trip
    for lat, lon in [(37.7749, -122.4194), (51.5, 0.0), (-33.87, 151.2)]:
        msl = g.hae_to_msl(123.4, lat, lon)
        assert abs(g.msl_to_hae(msl, lat, lon) - 123.4) < 1e-6
    print("geoid     : EGM96 N(0,0)=%.3f m, round-trip exact  OK" % g.undulation(0, 0))


def test_rosbag():
    for comp in ("none", "bz2"):
        out = rosbag.extract_jpegs(_build_bag(comp))
        assert len(out) == 3, (comp, len(out))
        assert all(d[:3] == JPEG_MAGIC for _, d in out), comp
    print("rosbag    : CompressedImage JPEGs extracted (none/bz2)  OK")


def main():
    test_exif()
    test_geocode()
    test_geoid()
    test_rosbag()
    print("\nAll selftests passed.")


if __name__ == "__main__":
    main()

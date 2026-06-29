# tools/geoid.py
"""Convert ellipsoidal heights (HAE) to mean-sea-level (MSL / orthometric).

The vertical separation between the WGS84 ellipsoid and the geoid (the
"geoid undulation" ``N``) is read from the bundled EGM96 5-arc-minute grid
(``_assets/egm96-5.pgm.bz2``, the GeographicLib ``egm96-5`` model). The model
is interpolated natively -- no third-party libraries and no network access:

    MSL = HAE - N            (orthometric height = ellipsoidal - undulation)
    HAE = MSL + N

The ``.pgm`` is GeographicLib's geoid format: a big-endian 16-bit raster whose
real undulation in metres is ``offset + scale * sample``. We reproduce
GeographicLib's bilinear sampling (longitude wraps, latitude runs +90..-90).

Example::

    from tools.geoid import hae_to_msl
    msl = hae_to_msl(100.0, 37.7749, -122.4194)   # metres above MSL
"""
from __future__ import annotations

import array
import bz2
import os
import sys
from typing import Optional, Union

__all__ = ["Geoid", "hae_to_msl", "msl_to_hae", "undulation"]

_ASSET = os.path.join(os.path.dirname(__file__), "_assets", "egm96-5.pgm.bz2")


class Geoid:
    """An EGM96 geoid model loaded from a GeographicLib ``.pgm`` raster.

    ``source`` may be a path to a ``.pgm`` or ``.pgm.bz2`` file; it defaults to
    the bundled ``egm96-5`` grid.
    """

    def __init__(self, source: Optional[Union[str, os.PathLike]] = None):
        self._load(source or _ASSET)

    # -- loading -----------------------------------------------------------
    def _load(self, source) -> None:
        path = os.fspath(source)
        raw = (bz2.open(path, "rb") if path.endswith(".bz2")
               else open(path, "rb"))
        with raw as fh:
            data = fh.read()
        self.width, self.height, self.offset, self.scale, body = \
            self._parse_pgm(data)
        samples = array.array("H")
        samples.frombytes(body[: self.width * self.height * 2])
        if sys.byteorder == "little":   # PGM is big-endian
            samples.byteswap()
        if len(samples) != self.width * self.height:
            raise ValueError("geoid grid truncated or malformed")
        self.samples = samples
        # GeographicLib resolution factors.
        self._rlon = self.width / 360.0
        self._rlat = (self.height - 1) / 180.0

    @staticmethod
    def _parse_pgm(data: bytes):
        if data[:2] != b"P5":
            raise ValueError("not a binary PGM (P5) geoid file")
        offset = -108.0
        scale = 0.003
        # Tokenize the header, honouring '#' comment lines (which is where
        # GeographicLib stores Offset/Scale).
        i = 2
        tokens = []
        n = len(data)
        while len(tokens) < 3:
            # skip whitespace
            while i < n and data[i] in b" \t\r\n":
                i += 1
            if i < n and data[i:i + 1] == b"#":
                eol = data.find(b"\n", i)
                line = data[i:eol if eol != -1 else n]
                parts = line.split()
                if len(parts) >= 3 and parts[1] == b"Offset":
                    offset = float(parts[2])
                elif len(parts) >= 3 and parts[1] == b"Scale":
                    scale = float(parts[2])
                i = eol + 1 if eol != -1 else n
                continue
            start = i
            while i < n and data[i] not in b" \t\r\n":
                i += 1
            tokens.append(data[start:i])
        width, height, _maxval = (int(t) for t in tokens)
        # Exactly one whitespace byte separates the header from the raster.
        body_start = i + 1
        return width, height, offset, scale, data[body_start:]

    # -- sampling ----------------------------------------------------------
    def _raw(self, ix: int, iy: int) -> int:
        return self.samples[iy * self.width + ix]

    def undulation(self, lat: float, lon: float) -> float:
        """Geoid undulation ``N`` (metres) at ``(lat, lon)`` via bilinear interp."""
        if lat > 90.0 or lat < -90.0:
            raise ValueError("latitude out of range [-90, 90]")
        lon %= 360.0
        if lon < 0:
            lon += 360.0

        fx = lon * self._rlon
        fy = (90.0 - lat) * self._rlat
        ix = int(fx)
        iy = int(fy)
        if iy >= self.height - 1:
            iy = self.height - 2
        if iy < 0:
            iy = 0
        fx -= ix
        fy -= iy
        ix %= self.width
        ix1 = (ix + 1) % self.width

        v00 = self._raw(ix, iy)
        v01 = self._raw(ix1, iy)
        v10 = self._raw(ix, iy + 1)
        v11 = self._raw(ix1, iy + 1)

        a = v00 + fx * (v01 - v00)
        b = v10 + fx * (v11 - v10)
        h = a + fy * (b - a)
        return self.offset + self.scale * h

    # -- conversions -------------------------------------------------------
    def hae_to_msl(self, hae: float, lat: float, lon: float) -> float:
        """Ellipsoidal height -> orthometric (MSL) height, metres."""
        return hae - self.undulation(lat, lon)

    def msl_to_hae(self, msl: float, lat: float, lon: float) -> float:
        """Orthometric (MSL) height -> ellipsoidal height, metres."""
        return msl + self.undulation(lat, lon)


# --------------------------------------------------------------------------- #
# Module-level convenience (lazily-loaded shared model)
# --------------------------------------------------------------------------- #
_DEFAULT: Optional[Geoid] = None


def _default() -> Geoid:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Geoid()
    return _DEFAULT


def undulation(lat: float, lon: float) -> float:
    """Geoid undulation ``N`` (metres) at ``(lat, lon)`` from the EGM96 grid."""
    return _default().undulation(lat, lon)


def hae_to_msl(hae: float, lat: float, lon: float) -> float:
    """Convert ellipsoidal height to MSL height (metres) at ``(lat, lon)``."""
    return _default().hae_to_msl(hae, lat, lon)


def msl_to_hae(msl: float, lat: float, lon: float) -> float:
    """Convert MSL height to ellipsoidal height (metres) at ``(lat, lon)``."""
    return _default().msl_to_hae(msl, lat, lon)

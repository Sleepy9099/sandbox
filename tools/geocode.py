# tools/geocode.py
"""Offline reverse geocoding: city + state (and country) from lat/lon.

The module ships a GeoNames *cities1000* dataset
(``_assets/rg_cities1000.csv.gz``, ~144k populated places) and resolves the
nearest place to a coordinate entirely offline.

Two interchangeable nearest-neighbour back ends are selected automatically:

* **numpy** (if importable) -- coordinates are projected onto the unit sphere
  and the nearest place is the one with the largest dot product. Vectorised
  and exact.
* **pure Python KD-tree** -- a balanced 3-D tree over the same unit-sphere
  points, built once and cached. No third-party dependency at all.

Both back ends return identical results; the great-circle (haversine) distance
to the matched place is included so callers can gauge confidence.

Example::

    from tools.geocode import search
    r = search(37.7749, -122.4194)
    print(r.name, r.admin1, r.cc)   # San Francisco California US
"""
from __future__ import annotations

import csv
import gzip
import io
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

__all__ = ["GeoResult", "ReverseGeocoder", "search", "search_many"]

_ASSET = os.path.join(os.path.dirname(__file__), "_assets",
                      "rg_cities1000.csv.gz")

Coord = Tuple[float, float]


@dataclass(frozen=True)
class GeoResult:
    """A matched populated place. ``admin1`` is the state/region name."""

    name: str            # city / place name
    admin1: str          # first-level admin division (state / region)
    admin2: str          # second-level admin division (county / district)
    cc: str              # ISO-3166 alpha-2 country code
    lat: float
    lon: float
    distance_km: float   # great-circle distance from the query point

    def as_dict(self) -> dict:
        return {
            "name": self.name, "admin1": self.admin1, "admin2": self.admin2,
            "cc": self.cc, "lat": self.lat, "lon": self.lon,
            "distance_km": self.distance_km,
        }


def _to_unit_vec(lat: float, lon: float) -> Tuple[float, float, float]:
    rlat = math.radians(lat)
    rlon = math.radians(lon)
    cl = math.cos(rlat)
    return (cl * math.cos(rlon), cl * math.sin(rlon), math.sin(rlat))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# Pure-Python KD-tree (3-D, unit sphere)
# --------------------------------------------------------------------------- #
class _KDTree:
    __slots__ = ("xs", "ys", "zs", "root")

    def __init__(self, xs, ys, zs):
        self.xs, self.ys, self.zs = xs, ys, zs
        axes = (xs, ys, zs)
        # Build over an index array; recursion returns nested tuples:
        #   (point_index, axis, left_node, right_node)
        import sys
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

        def build(idx: List[int], depth: int):
            if not idx:
                return None
            axis = depth % 3
            idx.sort(key=axes[axis].__getitem__)
            mid = len(idx) // 2
            return (idx[mid], axis,
                    build(idx[:mid], depth + 1),
                    build(idx[mid + 1:], depth + 1))

        self.root = build(list(range(len(xs))), 0)

    def nearest(self, q: Tuple[float, float, float]) -> int:
        qx, qy, qz = q
        xs, ys, zs = self.xs, self.ys, self.zs
        best = [-1, float("inf")]  # [index, squared distance]

        def visit(node):
            if node is None:
                return
            i, axis, left, right = node
            dx, dy, dz = xs[i] - qx, ys[i] - qy, zs[i] - qz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best[1]:
                best[1] = d2
                best[0] = i
            if axis == 0:
                diff = qx - xs[i]
            elif axis == 1:
                diff = qy - ys[i]
            else:
                diff = qz - zs[i]
            near, far = (left, right) if diff < 0 else (right, left)
            visit(near)
            if diff * diff < best[1]:
                visit(far)

        visit(self.root)
        return best[0]


# --------------------------------------------------------------------------- #
# Reverse geocoder
# --------------------------------------------------------------------------- #
class ReverseGeocoder:
    """Loads the cities dataset once and answers nearest-place queries."""

    def __init__(self, source: Optional[Union[str, os.PathLike]] = None):
        self._lat: List[float] = []
        self._lon: List[float] = []
        self._rows: List[Tuple[str, str, str, str]] = []  # name, a1, a2, cc
        self._load(source or _ASSET)

        self._np = None
        self._kdtree: Optional[_KDTree] = None
        try:
            import numpy as np  # optional acceleration
            self._np = np
            vecs = [_to_unit_vec(la, lo) for la, lo in zip(self._lat, self._lon)]
            self._mat = np.asarray(vecs, dtype="float64")  # (N, 3)
        except Exception:
            self._np = None  # fall back to the KD-tree on first query

    # -- loading -----------------------------------------------------------
    def _load(self, source) -> None:
        path = os.fspath(source)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            # Expected columns: lat,lon,name,admin1,admin2,cc
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    la = float(row[0]); lo = float(row[1])
                except ValueError:
                    continue
                self._lat.append(la)
                self._lon.append(lo)
                self._rows.append((row[2], row[3], row[4], row[5]))
        if not self._rows:
            raise ValueError(f"no city records loaded from {path!r}")

    def _ensure_kdtree(self) -> _KDTree:
        if self._kdtree is None:
            xs, ys, zs = [], [], []
            for la, lo in zip(self._lat, self._lon):
                x, y, z = _to_unit_vec(la, lo)
                xs.append(x); ys.append(y); zs.append(z)
            self._kdtree = _KDTree(xs, ys, zs)
        return self._kdtree

    # -- queries -----------------------------------------------------------
    def _result(self, idx: int, lat: float, lon: float) -> GeoResult:
        name, a1, a2, cc = self._rows[idx]
        clat, clon = self._lat[idx], self._lon[idx]
        return GeoResult(
            name=name, admin1=a1, admin2=a2, cc=cc, lat=clat, lon=clon,
            distance_km=round(_haversine_km(lat, lon, clat, clon), 4),
        )

    def query(self, lat: float, lon: float) -> GeoResult:
        """Return the nearest populated place to ``(lat, lon)``."""
        if self._np is not None:
            qx, qy, qz = _to_unit_vec(lat, lon)
            # Nearest point on the unit sphere == largest dot product.
            dots = self._mat[:, 0] * qx + self._mat[:, 1] * qy + self._mat[:, 2] * qz
            idx = int(self._np.argmax(dots))
        else:
            idx = self._ensure_kdtree().nearest(_to_unit_vec(lat, lon))
        return self._result(idx, lat, lon)

    def query_many(self, coords: Sequence[Coord]) -> List[GeoResult]:
        """Vectorised batch query; ``coords`` is a sequence of (lat, lon)."""
        coords = list(coords)
        if self._np is not None and coords:
            np = self._np
            q = np.asarray([_to_unit_vec(la, lo) for la, lo in coords])
            idxs = np.argmax(q @ self._mat.T, axis=1)
            return [self._result(int(i), coords[k][0], coords[k][1])
                    for k, i in enumerate(idxs)]
        return [self.query(la, lo) for la, lo in coords]


# --------------------------------------------------------------------------- #
# Module-level convenience (lazily-built shared instance)
# --------------------------------------------------------------------------- #
_DEFAULT: Optional[ReverseGeocoder] = None


def _default() -> ReverseGeocoder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ReverseGeocoder()
    return _DEFAULT


def search(lat: float, lon: float) -> GeoResult:
    """Reverse geocode a single coordinate using the shared dataset."""
    return _default().query(lat, lon)


def search_many(coords: Sequence[Coord]) -> List[GeoResult]:
    """Reverse geocode many ``(lat, lon)`` pairs at once."""
    return _default().query_many(coords)

# tools/__init__.py
"""Offline media / geospatial utility modules.

Each submodule is independent and works fully offline -- either natively in
code or by carrying its data asset under ``tools/_assets/``. They all accept
the same inputs as this project's ``fs`` package: an ``fs.open(...)`` handle, a
plain Python file object, a path, or raw ``bytes``.

Submodules
----------
* :mod:`tools.exif`    -- EXIF / metadata extraction (JPEG, PNG, MP4/MOV, TIFF).
* :mod:`tools.geocode` -- reverse geocoding (city + state) from lat/lon.
* :mod:`tools.rosbag`  -- extract embedded JPEGs from ROS 1 ``.bag`` files.
* :mod:`tools.geoid`   -- HAE -> MSL height conversion via the EGM96 geoid.

The heavy data assets and optional third-party accelerators (numpy, cv2,
pyproj, lz4) are imported lazily, so importing this package stays cheap.
"""
from __future__ import annotations

__all__ = ["exif", "geocode", "rosbag", "geoid"]

__version__ = "0.1.0"

# `tools` — offline media & geospatial utilities

Four independent, fully-offline modules. Each works either **natively in code**
or by **carrying its data asset** under `tools/_assets/`, so none of them needs
network access at runtime.

Every module that reads a file accepts the same inputs as the project's `fs`
package, interchangeably:

- an `fs` handle: `fs.open("/DCIM/IMG.JPG")`
- a plain Python file object: `open("img.jpg", "rb")`
- a path (`str` / `os.PathLike`)
- raw `bytes` / `bytearray`

(Internally normalised by `tools/_io.py`; caller-supplied file objects are
never closed by the module, so you keep ownership of your `fs` handle.)

| Module | What it does | Offline strategy | Optional accelerators |
|--------|--------------|------------------|-----------------------|
| `tools.exif`    | EXIF / metadata + GPS from JPEG, PNG, TIFF, DNG, MP4/MOV/M4A | pure-Python parsers, no assets | — |
| `tools.geocode` | nearest city + state from lat/lon | bundled GeoNames `cities1000` CSV | `numpy` (vectorised query) |
| `tools.rosbag`  | extract embedded compressed JPEGs from ROS 1 `.bag` | native ROS1 v2.0 + `bz2` reader | `lz4` (lz4 chunks), `numpy`+`cv2` (transcode raw `Image`) |
| `tools.geoid`   | HAE → MSL height via EGM96 geoid | bundled `egm96-5` grid + bilinear interp | — |

## Examples

```python
from tools.exif import read_metadata
from tools.geocode import search
from tools.geoid import hae_to_msl
from tools.rosbag import extract_jpegs

# 1. EXIF / metadata (works on jpeg, png, mp4/mov, tiff)
with fs.open("/DCIM/IMG_0001.JPG") as fh:
    meta = read_metadata(fh)
lat, lon, hae = meta.gps["latitude"], meta.gps["longitude"], meta.gps["altitude"]

# 2. Reverse geocode -> city + state
place = search(lat, lon)            # GeoResult(name='San Francisco', admin1='California', cc='US', ...)

# 3. Ellipsoidal height -> mean sea level
msl = hae_to_msl(hae, lat, lon)     # metres above MSL

# 4. Pull every embedded JPEG out of a ROS 1 bag
with open("drive.bag", "rb") as fh:
    for name, jpeg in extract_jpegs(fh):
        open(name, "wb").write(jpeg)
```

### `tools.exif` formats

- **JPEG** — Exif (APP1/TIFF) tags, GPS, dimensions, XMP blob.
- **TIFF / DNG** — shared IFD parser; DNG (TIFF-based raw) is reported as
  `format="dng"` and exposes `DNGVersion` plus camera Make/Model and GPS.
- **PNG** — `IHDR` geometry, `tEXt`/`zTXt`/`iTXt` text, `tIME`, embedded `eXIf`.
- **MP4 / MOV / M4A** — ISO-BMFF boxes: `ftyp` brand → format/mime
  (`m4a` ⇒ `audio/mp4`), `mvhd` creation time + duration, `udta` ISO-6709 GPS,
  and iTunes `ilst` tags (title, artist, album, track, genre, bpm, …).

## Bundled assets (`tools/_assets/`)

- `rg_cities1000.csv.gz` — GeoNames *cities1000* (≈144k places; CC-BY 4.0).
- `egm96-5.pgm.bz2` — GeographicLib EGM96 5-arc-minute geoid grid (public domain).

To swap in a different dataset/grid, pass a path to `ReverseGeocoder(source=...)`
or `Geoid(source=...)`.

## Notes

- `tools.rosbag` handles the **ROS 1** bag format (a single `.bag` file, which
  maps cleanly onto a file object). ROS 2 bags are sqlite/mcap *directories*
  and are out of scope for the file-object input model.
- `sensor_msgs/CompressedImage` payloads are already complete JPEG/PNG files and
  are emitted verbatim. Raw `sensor_msgs/Image` messages are transcoded to JPEG
  only when `numpy` + `cv2` are importable.
- The EGM96 conversion reproduces GeographicLib's bilinear sampling; spot checks
  agree with published undulations (e.g. `N(0,0) ≈ 17.16 m`).

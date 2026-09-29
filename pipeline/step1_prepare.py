"""
STEP 1 - Prepare the images.

What it does:
  1. Reads every raw drone photo (7152 x 5368 px, ~13 MB each).
  2. Shrinks it so the long side is MAX_SIDE pixels (full size would not fit
     in 8 GB of RAM once we start matching and computing depth).
  3. Reads the EXIF metadata: GPS position + camera focal length.
  4. Converts the focal length from millimetres to pixels, because every
     later step works in pixels.
  5. Writes the small images to data/images/ and a table of metadata to
     data/metadata.csv.

Run:  .venv/bin/python pipeline/step1_prepare.py
"""
import csv
import math
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "images"
META_CSV = ROOT / "data" / "metadata.csv"

MAX_SIDE = 1600  # long side of the working images, in pixels


def dms_to_degrees(dms, ref):
    """GPS is stored as (degrees, minutes, seconds). Convert to one decimal number."""
    deg = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif(img):
    exif = img.getexif()
    cam = {TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}  # camera settings block
    gps = exif.get_ifd(0x8825)                                         # GPS block
    # GPS tag ids: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon, 6=Altitude
    lat = dms_to_degrees(gps[2], gps[1])
    lon = dms_to_degrees(gps[4], gps[3])
    alt = float(gps[6])
    return cam, lat, lon, alt


def focal_in_pixels(cam, width_px):
    """
    focal_px = focal_mm * pixels_per_mm.
    FocalPlaneXResolution tells us how many pixels fit in one unit on the sensor;
    FocalPlaneResolutionUnit says which unit (2 = inch, 3 = cm, 4 = mm).
    """
    unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}[int(cam["FocalPlaneResolutionUnit"])]
    px_per_mm = float(cam["FocalPlaneXResolution"]) / unit_mm
    return float(cam["FocalLength"]) * px_per_mm


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_files = sorted(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(raw_files)} raw images")

    for i, path in enumerate(raw_files, 1):
        with Image.open(path) as img:
            cam, lat, lon, alt = read_exif(img)
            full_w, full_h = img.size
            f_full = focal_in_pixels(cam, full_w)

            scale = MAX_SIDE / max(full_w, full_h)
            small = img.resize((round(full_w * scale), round(full_h * scale)), Image.LANCZOS)
            small.save(OUT_DIR / path.name, quality=95)

        # File names look like EX-01-10917_0285_0030.jpg -> flight "0285"
        flight = path.stem.split("_")[1]
        rows.append({
            "name": path.name,
            "flight": flight,
            "lat": lat, "lon": lon, "alt": alt,
            "width": small.width, "height": small.height,
            "focal_px": f_full * scale,   # focal length scales with the image
        })
        print(f"[{i}/{len(raw_files)}] {path.name}  flight={flight}  alt={alt:.1f} m")

    with open(META_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fov = 2 * math.degrees(math.atan(rows[0]["width"] / 2 / rows[0]["focal_px"]))
    print(f"\nDone. {len(rows)} images -> {OUT_DIR}")
    print(f"Working size {rows[0]['width']}x{rows[0]['height']}, "
          f"focal {rows[0]['focal_px']:.0f} px, horizontal field of view {fov:.0f} deg")


if __name__ == "__main__":
    main()

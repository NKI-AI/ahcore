# ahcore utilities

## `convert_wsi_to_tiles.py`:
Utility to pretile WSIs into small tiles for use with dinov2.
Combine with `detect_problems.py` to detect images which were not correctly parsed.

Process for v1 is:
- run `convert_wsi_to_tiles.py` on all *diagnostic* slides
- run detect_problems.py on the output and remove cases with `parsing` error
-

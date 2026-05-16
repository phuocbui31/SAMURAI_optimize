"""Compatibility entrypoint for the demo prediction renderer.

The renderer now lives in scripts/render_output.py because it supports both
legacy bbox TXT and mask JSONL outputs. Keep this file so older commands that
call render_bounding_box.py still work.
"""

from render_output import main


if __name__ == "__main__":
    main()

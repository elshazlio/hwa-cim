"""Console entry: `hwa-dashboard` -> Streamlit lab UI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    pkg_dir = Path(__file__).resolve().parent
    home = pkg_dir / "Home.py"
    # Ensure .streamlit/config.toml is picked up and relative paths resolve.
    try:
        from hwa_gui.paths import project_root

        os.chdir(project_root())
    except Exception:
        os.chdir(pkg_dir.parent.parent)
    cmd = [sys.executable, "-m", "streamlit", "run", str(home)]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

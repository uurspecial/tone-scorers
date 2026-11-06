"""Download private GitHub Release assets using an authenticated GitHub CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from artifacts import ROOT, install_archive, read_manifest, verify_adapter


def main():
    manifest = read_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub owner/tone-scorers")
    parser.add_argument("--model", action="append", choices=[m["id"] for m in manifest["models"]])
    parser.add_argument("--directory", type=Path, default=ROOT / "adapters")
    parser.add_argument("--gh", default=os.environ.get("GH_BIN", "gh"))
    args = parser.parse_args()
    for item in manifest["models"]:
        if args.model and item["id"] not in args.model:
            continue
        destination = args.directory / item["id"]
        if destination.exists():
            verify_adapter(destination, item)
            print(f"Already verified: {item['id']}")
            continue
        with tempfile.TemporaryDirectory(prefix="tone-scorers-download-") as temporary:
            subprocess.run([
                args.gh, "release", "download", manifest["release_tag"], "--repo", args.repo,
                "--pattern", item["archive"], "--dir", temporary,
            ], check=True)
            install_archive(Path(temporary) / item["archive"], args.directory, item)
            print(f"Installed and verified: {item['id']}")


if __name__ == "__main__":
    main()

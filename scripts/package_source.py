"""Export committed source, not local recordings or virtual environments.

Uses only Python's standard library and Git. The ZIP carries SOURCE_REVISION.txt;
its adjacent .sha256 file can be checked with `sha256sum -c <file>.sha256`.
This is a source release, not an offline bundle of third-party dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise RuntimeError(f"Git command failed: {detail.decode('utf-8', errors='replace')}") from exc


def package_source(root: Path, output: Path) -> dict[str, str]:
    """Package a clean committed tree; refuse overwriting either deliverable."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Source archive path must end in .zip")
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("Archive/checksum exists; choose a new output filename")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root:
        raise ValueError("--repo must be the repository root, not a nested directory")
    if _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Tracked files have uncommitted changes; commit before making a release")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    archive = io.BytesIO(_git(root, "archive", "--format=zip", "--prefix=gesture/", revision))
    with zipfile.ZipFile(archive, "a") as bundle:
        name = "gesture/SOURCE_REVISION.txt"
        if name in bundle.namelist():
            raise ValueError("SOURCE_REVISION.txt is generated; do not track it in Git")
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        bundle.writestr(info, revision + "\n")
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    created_output = created_checksum = False
    try:
        with output.open("xb") as stream:
            created_output = True
            stream.write(payload)
        with checksum.open("x", encoding="utf-8") as stream:
            created_checksum = True
            stream.write(f"{digest}  {output.name}\n")
    except BaseException:
        # Remove only artifacts created by this call, never an existing user's file.
        if created_checksum:
            checksum.unlink(missing_ok=True)
        if created_output:
            output.unlink(missing_ok=True)
        raise
    return {"revision": revision, "archive": str(output), "sha256": digest,
            "checksum_file": str(checksum)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/gesture-source.zip"))
    args = parser.parse_args()
    result = package_source(args.repo, args.output)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

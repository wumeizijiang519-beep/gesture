"""Release artifact tests do not need a camera, a model, or a simulator."""
from pathlib import Path
import hashlib
import importlib.util
import subprocess
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_source", Path(__file__).resolve().parents[1] / "scripts/package_source.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
package_source = MODULE.package_source


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    (root / "README.md").write_text("Source release fixture\n", encoding="utf-8")
    git(root, "add", "README.md")
    git(root, "-c", "user.name=Archive Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Initial fixture")
    return root


def test_committed_content_revision_checksum_and_no_untracked_files(repo, tmp_path):
    (repo / "private_recording.txt").write_text("not tracked", encoding="utf-8")
    output = tmp_path / "delivery/source.zip"
    result = package_source(repo, output)
    with zipfile.ZipFile(output) as bundle:
        assert bundle.read("gesture/README.md") == b"Source release fixture\n"
        assert bundle.read("gesture/SOURCE_REVISION.txt").decode().strip() == result["revision"]
        assert not any("private_recording" in name or "/.git/" in name for name in bundle.namelist())
        assert bundle.testzip() is None
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest == result["sha256"]
    assert Path(result["checksum_file"]).read_text().strip() == f"{digest}  source.zip"


def test_deterministic_zip_for_same_revision(repo, tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    assert package_source(repo, a)["sha256"] == package_source(repo, b)["sha256"]
    assert a.read_bytes() == b.read_bytes()


def test_refuses_existing_zip(repo, tmp_path):
    output = tmp_path / "source.zip"
    output.write_bytes(b"keep existing data")
    with pytest.raises(FileExistsError):
        package_source(repo, output)
    assert output.read_bytes() == b"keep existing data"


def test_refuses_existing_checksum(repo, tmp_path):
    output = tmp_path / "source.zip"
    checksum = output.with_suffix(".zip.sha256")
    checksum.write_text("keep existing checksum", encoding="utf-8")
    with pytest.raises(FileExistsError):
        package_source(repo, output)
    assert not output.exists()
    assert checksum.read_text() == "keep existing checksum"


def test_refuses_uncommitted_tracked_changes(repo, tmp_path):
    (repo / "README.md").write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted"):
        package_source(repo, tmp_path / "source.zip")


def test_refuses_non_zip_output(repo, tmp_path):
    with pytest.raises(ValueError, match=".zip"):
        package_source(repo, tmp_path / "source.tar")


def test_refuses_nested_root(repo, tmp_path):
    subdir = repo / "subdir"
    subdir.mkdir()
    with pytest.raises(ValueError, match="repository root"):
        package_source(subdir, tmp_path / "source.zip")


def test_requires_git_repository(tmp_path):
    source = tmp_path / "not_a_repo"
    source.mkdir()
    with pytest.raises(RuntimeError, match="Git command failed"):
        package_source(source, tmp_path / "source.zip")

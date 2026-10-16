"""Preflight on the GPU host (rev. 3): host-local lock directories and a runnable detector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qcal.config import load_config
from qcal_lab import preflight, status
from tests.conftest import REPO_ROOT
from tests.lab_support import lab_config

pytestmark = pytest.mark.rule("C3")

MOUNTS = """\
/dev/sda1 / ext4 rw 0 0
server:/export /mnt/shared nfs4 rw 0 0
/dev/sdb1 /mnt/shared/local ext4 rw 0 0
host:/x /mnt/with\\040space fuse.sshfs rw 0 0
host:/y /mnt/tab\\011and\\134slash nfs4 rw 0 0
"""
NETWORK = frozenset({"nfs4", "fuse.sshfs"})


def test_the_mount_table_is_parsed_with_escaped_spaces() -> None:
    assert preflight.mount_points(MOUNTS)[-2:] == [
        ("/mnt/with space", "fuse.sshfs"),
        ("/mnt/tab\tand\\slash", "nfs4"),
    ]
    assert preflight.mount_points("short line\n") == []


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/mnt/shared/runs/inflight", "nfs4"),
        ("/mnt/shared/local/cache", "ext4"),  # the deeper mount point wins
        ("/mnt/sharedness/x", "ext4"),  # a prefix of a name is not inside it
        ("/mnt/with space/a", "fuse.sshfs"),
    ],
)
def test_the_filesystem_is_the_deepest_mount_containing_the_path(path: str, kind: str) -> None:
    assert preflight.filesystem_type(Path(path), preflight.mount_points(MOUNTS)) == kind


def test_directories_on_network_filesystems_are_reported() -> None:
    paths = [("a", Path("/mnt/shared/x")), ("b", Path("/mnt/shared/local/y"))]
    assert preflight.host_local_problems(paths, NETWORK, MOUNTS) == [
        "a (/mnt/shared/x) is on nfs4: locks need a host-local filesystem"
    ]


def _importer(*, mmcv: bool = True, cuda: bool = True) -> Any:
    def load(name: str) -> Any:
        if name == preflight.MMCV_EXT and not mmcv:
            raise ImportError("libcudart.so.12: cannot open shared object file")
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))
        return SimpleNamespace()

    return load


@pytest.mark.parametrize(
    ("device", "importer", "problems"),
    [
        ("cuda:0", _importer(), []),
        ("cpu", _importer(cuda=False), []),  # no CUDA needed on the CPU
        ("cuda:0", _importer(cuda=False), ["device cuda:0 but CUDA is not available"]),
        (
            "cuda:1",
            _importer(mmcv=False),
            [
                (
                    "mmcv._ext does not import (ImportError: libcudart.so.12: cannot open shared "
                    "object file)"
                )
            ],
        ),
    ],
)
def test_a_detector_runtime_is_checked(device: str, importer: Any, problems: list[str]) -> None:
    assert preflight.runtime_problems(device, importer) == problems


def test_the_settings_come_from_the_tooling_file() -> None:
    network, mounts = preflight.settings()
    assert {"nfs", "nfs4", "cifs", "fuse.sshfs"} <= network
    assert mounts == Path("/proc/mounts")


def _items(report: status.StatusReport) -> dict[str, dict[str, Any]]:
    return {i["name"]: i for i in report.to_dict()["items"]}


def test_status_reports_the_preflight(tmp_path: Path) -> None:
    mounts = tmp_path / "mounts"
    mounts.write_text(f"/dev/x / ext4 rw 0 0\nsrv:/e {REPO_ROOT / 'runs'} nfs4 rw 0 0\n")
    report = status.StatusReport()
    preflight.add_preflight(
        report,
        load_config(REPO_ROOT, environ={}),
        lab_config(REPO_ROOT),
        tooling={"preflight": {"network_filesystems": ["nfs4"], "mounts_file": str(mounts)}},
        importer=_importer(cuda=False),
    )
    items = _items(report)
    assert items["lock directories host-local"]["ready"] is False
    assert "paths.inflight_dir" in items["lock directories host-local"]["detail"]
    assert items["detector atss_r50 runtime"]["detail"] == "device cuda:0 but CUDA is not available"
    assert "detector fixture runtime" not in items  # only MMDetection detectors


def test_status_without_a_mount_table_says_it_did_not_check(tmp_path: Path) -> None:
    report = status.StatusReport()
    preflight.add_preflight(
        report,
        load_config(REPO_ROOT, environ={}),
        lab_config(REPO_ROOT),
        tooling={"preflight": {"network_filesystems": [], "mounts_file": str(tmp_path / "no")}},
        importer=_importer(),
    )
    item = _items(report)["lock directories host-local"]
    assert (item["ready"], item["detail"].startswith("not checked")) == (True, True)


def test_build_status_includes_the_preflight() -> None:
    report = status.build_status(load_config(REPO_ROOT, environ={}), lab_config(REPO_ROOT))
    assert {"lock directories host-local", "detector atss_r50 runtime"} <= set(_items(report))

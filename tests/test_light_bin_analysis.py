from pathlib import Path

import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_light_bins import add_light_bins, collect_folder_light_stats, summarize_light_rows


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_collect_folder_light_stats_reads_identity_folders(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write_rgb(root / "id_a" / "dark.jpg", 20)
    _write_rgb(root / "id_a" / "mid.jpg", 128)
    _write_rgb(root / "id_b" / "bright.jpg", 235)

    rows = collect_folder_light_stats(str(root), split="query")

    assert [row["identity"] for row in rows] == ["id_a", "id_a", "id_b"]
    assert rows[0]["brightness"] < rows[1]["brightness"] < rows[2]["brightness"]
    assert all(row["split"] == "query" for row in rows)


def test_add_light_bins_assigns_dark_mid_bright_and_hard_light(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write_rgb(root / "id_a" / "dark.jpg", 10)
    _write_rgb(root / "id_b" / "mid.jpg", 128)
    _write_rgb(root / "id_c" / "bright.jpg", 245)

    rows = collect_folder_light_stats(str(root), split="query")
    thresholds = add_light_bins(rows, low_quantile=0.34, high_quantile=0.66)

    by_name = {Path(row["path"]).stem: row for row in rows}
    assert thresholds["brightness_low"] < thresholds["brightness_high"]
    assert by_name["dark"]["light_bin"] == "dark"
    assert by_name["mid"]["light_bin"] == "mid"
    assert by_name["bright"]["light_bin"] == "bright"
    assert by_name["dark"]["hard_light"] is True
    assert by_name["bright"]["hard_light"] is True


def test_summarize_light_rows_reports_group_counts(tmp_path: Path) -> None:
    root = tmp_path / "query"
    _write_rgb(root / "id_a" / "dark.jpg", 10)
    _write_rgb(root / "id_b" / "mid.jpg", 128)
    _write_rgb(root / "id_c" / "bright.jpg", 245)

    rows = collect_folder_light_stats(str(root), split="query")
    add_light_bins(rows, low_quantile=0.34, high_quantile=0.66)
    summary = summarize_light_rows(rows)

    assert summary["all"]["count"] == 3
    assert summary["dark"]["count"] == 1
    assert summary["mid"]["count"] == 1
    assert summary["bright"]["count"] == 1
    assert summary["hard_light"]["count"] == 2

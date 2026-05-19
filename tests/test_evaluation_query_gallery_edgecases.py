from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("torchvision")

from app.core.evaluation import compute_cmc_map, evaluate_openset


def test_compute_cmc_map_skips_queries_that_only_match_junk_same_image() -> None:
    distmat = np.array([[0.0, 1.0]], dtype=np.float32)
    query_ids = [1]
    gallery_ids = [1, 2]
    query_paths = ["query/0001.jpg"]
    gallery_paths = ["gallery/0001.jpg", "gallery/0002.jpg"]

    cmc, m_ap = compute_cmc_map(
        distmat=distmat,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_paths=query_paths,
        gallery_paths=gallery_paths,
        max_rank=2,
        exclude_same_camera=False,
    )

    assert np.all(np.isfinite(cmc))
    assert float(m_ap) == 0.0
    assert np.array_equal(cmc, np.zeros(2, dtype=float))


def test_evaluate_openset_returns_finite_metrics_when_same_image_match_is_removed() -> None:
    distmat = np.array([[0.0, 1.0]], dtype=np.float32)
    query_ids = [1]
    gallery_ids = [1, 2]
    query_paths = ["query/0001.jpg"]
    gallery_paths = ["gallery/0001.jpg", "gallery/0002.jpg"]

    results = evaluate_openset(
        distmat=distmat,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_paths=query_paths,
        gallery_paths=gallery_paths,
        seen_ids=[1],
        unseen_ids=[],
        max_rank=2,
    )

    for key, value in results.items():
        assert np.isfinite(value), key
    assert results["mAP"] == 0.0
    assert results["rank1"] == 0.0
    assert results["mAP_seen"] == 0.0

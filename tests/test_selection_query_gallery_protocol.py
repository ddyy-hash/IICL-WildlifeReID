from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_cross_species_paper_ablation import DEFAULT_SELECTION_PROTOCOL
from tools.selection_protocols import (
    CROSS_SPECIES_SELECTION_PROTOCOL,
    resolve_selection_query_gallery_eval_spec,
)


def test_selection_query_gallery_protocol_is_supported() -> None:
    assert DEFAULT_SELECTION_PROTOCOL == CROSS_SPECIES_SELECTION_PROTOCOL


def test_resolve_selection_query_gallery_eval_spec_reads_explicit_paths() -> None:
    eval_cfg = {
        "protocol": DEFAULT_SELECTION_PROTOCOL,
        "selection_query_dir": "data/processed/foo_openset/selection_query",
        "selection_gallery_dir": "data/processed/foo_openset/selection_gallery",
        "selection_info": "data/processed/foo_openset/selection_info.json",
    }

    spec = resolve_selection_query_gallery_eval_spec(eval_cfg)

    assert spec == {
        "query_dir": "data/processed/foo_openset/selection_query",
        "gallery_dir": "data/processed/foo_openset/selection_gallery",
        "info_path": "data/processed/foo_openset/selection_info.json",
        "exclude_same_camera": False,
        "force_standard_eval": True,
    }

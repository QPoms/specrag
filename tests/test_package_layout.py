from pathlib import Path

from scripts.check_structure import (
    _contains_forbidden_fragment,
    main as check_structure,
)
from specrag.config import load_config


def test_repository_structure_contract() -> None:
    assert check_structure() == 0


def test_legacy_import_check_uses_package_boundaries() -> None:
    assert _contains_forbidden_fragment("from specrag.eval import run", "specrag.eval")
    assert not _contains_forbidden_fragment(
        "from specrag.evaluation import run",
        "specrag.eval",
    )


def test_retrieval_configuration_is_domain_neutral() -> None:
    config = load_config()

    assert config.retrieval.index_dir == ".local/indexes/specifications"
    assert config.retrieval.specification_root == "data/3gpp"
    assert config.corpus_sync.supported_releases == (17, 18, 19)
    assert config.corpus_sync.automatic_sync_enabled is False
    assert not hasattr(config.retrieval, "threat_model_path")
    assert not hasattr(config, "capability_retrieval")


def test_legacy_screening_package_is_absent() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "specagent").exists()
    assert not (root / "src" / "specrag" / "screening").exists()

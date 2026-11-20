import pytest

from pipeline.config import ConfigError, load_config, validate_slug
from tests.conftest import ROOT


@pytest.mark.parametrize("slug", ["appartement", "loft", "studio-lyon-3", "t2-paris11"])
def test_slugs_valides(slug):
    assert validate_slug(slug) == slug


@pytest.mark.parametrize("slug", ["", "Studio", "studio_lyon", "-studio", "studio-", "a--b", "é", "a b",
                                  "demo", "images", "a" * 61, "../x"])
def test_slugs_invalides(slug):
    with pytest.raises(ConfigError):
        validate_slug(slug)


def test_config_du_repo_charge():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg["train"]["cap_max"] == 1_000_000
    assert cfg["publish"]["prefix"] == "tours/"
    assert cfg["export"]["variants"]["mobile"]["max_splats"] == 500_000


def test_cle_inconnue(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("frames:\n  fpss: 3\n")
    with pytest.raises(ConfigError, match="frames.fpss"):
        load_config(p)


def test_surcharge_partielle(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  max_steps: 7000\n")
    cfg = load_config(p)
    assert cfg["train"]["max_steps"] == 7000
    assert cfg["train"]["cap_max"] == 1_000_000


def test_opacite_doit_etre_lineaire(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("export:\n  min_opacity: -2.94\n")
    with pytest.raises(ConfigError):
        load_config(p)

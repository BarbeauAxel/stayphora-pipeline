import json

from pipeline.provenance import code_info, fingerprint


def _pkg(tmp_path):
    d = tmp_path / "pipeline"
    (d / "steps" / "__pycache__").mkdir(parents=True)
    (d / "a.py").write_text("x = 1\n")
    (d / "steps" / "b.py").write_text("y = 2\n")
    (d / "steps" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"junk")
    return d


def test_empreinte_ignore_cache_et_origin(tmp_path):
    d = _pkg(tmp_path)
    fp = fingerprint(d)
    assert fp["files"] == 2
    (d / "ORIGIN.json").write_text("{}")
    (d / "steps" / "__pycache__" / "c.pyc").write_bytes(b"autre")
    assert fingerprint(d) == fp
    (d / "a.py").write_text("x = 3\n")
    assert fingerprint(d)["sha256"] != fp["sha256"]


def test_code_info_compare_a_l_envoi(tmp_path):
    d = _pkg(tmp_path)
    sha = fingerprint(d)["sha256"]
    (d / "ORIGIN.json").write_text(json.dumps({"source": "envoi-local", "git_commit": None,
                                               "git_dirty": None, "sha256_sent": sha}))
    info = code_info(d)
    assert info["matches_sent"] is True and info["source"] == "envoi-local"
    (d / "a.py").write_text("modifié\n")
    assert code_info(d)["matches_sent"] is False

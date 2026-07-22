from guardrail_audit.utils.config import load_config


def test_load_and_override(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("a:\n  b: 1\nx: hello\n")
    cfg = load_config(cfg_file, ["a.b=5", "x=world"])
    assert cfg.a.b == 5
    assert cfg.x == "world"


def test_override_coerces_types(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("n: 1\nf: 0.0\n")
    cfg = load_config(cfg_file, ["n=10", "f=0.35"])
    assert cfg.n == 10 and isinstance(cfg.n, int)
    assert cfg.f == 0.35 and isinstance(cfg.f, float)

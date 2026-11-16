"""The settings: where they are read from, how bad values are reported, and that .env.example
lists every setting at its default. Also the method switch of search.find_similar_products."""

from pathlib import Path

import pytest

from similarity import config
from similarity.search import METHOD_WEIGHTS, WEIGHTS, find_similar_products, load_features, rank, search
from similarity.preprocessing import PROCESSED_PATH

EXAMPLE = config.ROOT / ".env.example"


def test_env_file_lines(tmp_path):
    file = tmp_path / ".env"
    file.write_text("# a comment\n\nSIMILARITY_METHOD=minilm\nIMAGE_DIR = \"~/My Photos\"\nEMPTY=\n", encoding="utf-8")
    assert config.read_env_file(file) == {"SIMILARITY_METHOD": "minilm", "IMAGE_DIR": "~/My Photos", "EMPTY": ""}
    assert config.read_env_file(tmp_path / "missing.env") == {}
    assert config.read_env_file("") == {}
    file.write_text("NO_EQUALS_SIGN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        config.read_env_file(file)


def test_environment_wins_over_the_file_which_wins_over_the_default(monkeypatch):
    monkeypatch.setattr(config, "DEFAULTS", dict(config.DEFAULTS))   # undone after the test
    monkeypatch.setattr(config, "_FILE", {"A_SETTING": "2"})
    assert config.setting("A_SETTING", 1, int) == 2                  # from the file
    monkeypatch.setenv("A_SETTING", "3")
    assert config.setting("A_SETTING", 1, int) == 3                  # the environment wins
    monkeypatch.setenv("A_SETTING", "")
    assert config.setting("A_SETTING", 1, int) == 1                  # empty means the default


def test_bad_values_name_the_setting(monkeypatch):
    monkeypatch.setattr(config, "DEFAULTS", dict(config.DEFAULTS))
    monkeypatch.setenv("SOME_COUNT", "ten")
    with pytest.raises(ValueError, match="SOME_COUNT='ten' is not a valid int"):
        config.setting("SOME_COUNT", 10, int)
    monkeypatch.setenv("SOME_COUNT", "-1")
    with pytest.raises(ValueError, match="at least 0"):
        config.at_least("SOME_COUNT", 10, 0)
    monkeypatch.setenv("SOME_MODE", "fast")
    with pytest.raises(ValueError, match="choose one of exact, ann"):
        config.choice("SOME_MODE", "exact", config.SEARCH_MODES)


def test_the_example_file_lists_every_setting_at_its_default():
    example = config.read_env_file(EXAMPLE)
    assert set(example) == set(config.DEFAULTS)
    for name, default in config.DEFAULTS.items():
        value = config.path(example[name]) if isinstance(default, Path) else type(default)(example[name])
        assert value == default, name


def test_the_methods_match_the_settings():
    assert set(METHOD_WEIGHTS) == set(config.SIMILARITY_METHODS)
    assert METHOD_WEIGHTS["tfidf"] == WEIGHTS


def test_the_defaults_are_the_api_method():
    assert (config.SIMILARITY_METHOD, config.SEARCH_MODE) == ("tfidf", "exact")


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="run `python -m similarity.preprocessing` first")
def test_the_switch_picks_the_method():
    features = load_features()
    row = features.row_of["26d41bdc1495de290bc8e6062d927729"]
    assert search(features, row)[0].tolist() == rank(features, row, WEIGHTS)[0].tolist()
    with pytest.raises(ValueError, match="choose one of tfidf, minilm"):
        find_similar_products(features.ids[row], 5, method="word2vec")
    with pytest.raises(ValueError, match="choose one of exact, ann"):
        find_similar_products(features.ids[row], 5, mode="fast")

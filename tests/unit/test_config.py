from __future__ import annotations

from pathlib import Path

import pytest

from qcal.config import (
    CONFIG_PATH_ENV,
    REPO_CONFIG_NAME,
    ROOT_ENV,
    ConfigError,
    deep_merge,
    find_root,
    load_config,
    load_defaults,
    parse_env_overrides,
)


def test_packaged_defaults_have_every_documented_section() -> None:
    defaults = load_defaults()
    expected = {
        "paths",
        "logging",
        "git",
        "policy",
        "signing",
        "hooks",
        "registry",
        "executor",
        "experiments",
        "tables",
        "claims",
        "data",
        "licenses",
        "agent_layer",
        "init",
    }
    assert expected <= set(defaults)


def test_load_defaults_returns_independent_copies() -> None:
    first = load_defaults()
    first["paths"]["experiments"] = "changed"
    assert load_defaults()["paths"]["experiments"] != "changed"


def test_deep_merge_merges_tables_and_replaces_lists_and_scalars() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged = deep_merge(base, {"a": {"y": [3]}, "b": {"nested": True}})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": {"nested": True}}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        ("QCAL__LOGGING__LEVEL", "DEBUG", {"logging": {"level": "DEBUG"}}),
        ("QCAL__LOGGING__LEVEL", '"INFO"', {"logging": {"level": "INFO"}}),
        ("QCAL__REGISTRY__MAX_RUNS_PER_BATCH", "5", {"registry": {"max_runs_per_batch": 5}}),
        (
            "QCAL__GIT__PROTECTED_BRANCHES",
            '["a", "b"]',
            {"git": {"protected_branches": ["a", "b"]}},
        ),
        ("QCAL__HOOKS__X__Y", "true", {"hooks": {"x": {"y": True}}}),
    ],
)
def test_env_overrides_parse_toml_values_with_string_fallback(
    name: str, raw: str, expected: dict
) -> None:
    assert parse_env_overrides({name: raw, "UNRELATED": "1"}) == expected


@pytest.mark.parametrize("name", ["QCAL__", "QCAL__A____B"])
def test_malformed_override_names_are_rejected(name: str) -> None:
    with pytest.raises(ConfigError, match="malformed"):
        parse_env_overrides({name: "1"})


def test_override_conflicting_with_scalar_is_rejected() -> None:
    with pytest.raises(ConfigError, match="conflicts"):
        parse_env_overrides({"QCAL__A": "1", "QCAL__A__B": "2"})


def test_find_root_prefers_environment(tmp_path: Path) -> None:
    assert find_root(environ={ROOT_ENV: str(tmp_path)}) == tmp_path.resolve()


@pytest.mark.parametrize("marker", [REPO_CONFIG_NAME, ".git"])
def test_find_root_walks_up_to_a_marker(tmp_path: Path, marker: str) -> None:
    (tmp_path / marker).mkdir() if marker == ".git" else (tmp_path / marker).write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_root(nested, environ={}) == tmp_path.resolve()


def test_find_root_falls_back_to_start(tmp_path: Path) -> None:
    lonely = tmp_path / "x"
    lonely.mkdir()
    assert find_root(lonely, environ={}) in {lonely.resolve(), *lonely.resolve().parents}


def test_repo_file_overrides_defaults_and_env_overrides_repo(repo: Path) -> None:
    (repo / REPO_CONFIG_NAME).write_text('[logging]\nlevel = "WARNING"\nformat = "json"\n')
    config = load_config(repo, environ={"QCAL__LOGGING__LEVEL": "ERROR"})
    assert config.str_value("logging.level") == "ERROR"
    assert config.str_value("logging.format") == "json"
    assert config.sources == ("defaults", str(repo / REPO_CONFIG_NAME), "environment")


def test_config_path_env_selects_another_file(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.toml"
    other.write_text('[paths]\nexperiments = "alt.yaml"\n')
    config = load_config(repo, environ={CONFIG_PATH_ENV: str(other)})
    assert config.path("experiments") == repo.resolve() / "alt.yaml"


def test_repo_text_layer_replaces_the_file(repo: Path) -> None:
    (repo / REPO_CONFIG_NAME).write_text('[signing]\nmode = "bootstrap"\n')
    config = load_config(repo, repo_text='[signing]\nmode = "enforce"\n', environ={})
    assert config.str_value("signing.mode") == "enforce"
    assert "repo_text" in config.sources


def test_use_repo_file_false_ignores_the_repository_layer(repo: Path) -> None:
    (repo / REPO_CONFIG_NAME).write_text('[signing]\nmode = "enforce"\n')
    assert (
        load_config(repo, environ={}, use_repo_file=False).str_value("signing.mode") == "bootstrap"
    )


def test_invalid_toml_names_the_file(repo: Path) -> None:
    (repo / REPO_CONFIG_NAME).write_text("not = [valid\n")
    with pytest.raises(ConfigError, match=REPO_CONFIG_NAME):
        load_config(repo, environ={})


@pytest.mark.parametrize(
    ("accessor", "key", "body"),
    [
        ("str_value", "x.v", "[x]\nv = 1\n"),
        ("int_value", "x.v", "[x]\nv = true\n"),
        ("int_value", "x.v", '[x]\nv = "1"\n'),
        ("float_value", "x.v", "[x]\nv = false\n"),
        ("bool_value", "x.v", "[x]\nv = 1\n"),
        ("str_list", "x.v", "[x]\nv = [1]\n"),
        ("table_list", "x.v", "[x]\nv = [1]\n"),
        ("section", "x.v", "[x]\nv = 1\n"),
    ],
)
def test_typed_accessors_name_the_key_on_type_errors(
    make_config, accessor: str, key: str, body: str
) -> None:
    config = make_config(body)
    with pytest.raises(ConfigError, match=key):
        getattr(config, accessor)(key)


def test_typed_accessors_return_values(make_config) -> None:
    config = make_config('[x]\ns = "a"\ni = 2\nf = 1.5\nb = true\nl = ["a"]\nt = [{k = 1}]\n')
    assert config.str_value("x.s") == "a"
    assert config.int_value("x.i") == 2
    assert config.float_value("x.i") == 2.0
    assert config.float_value("x.f") == 1.5
    assert config.bool_value("x.b") is True
    assert config.str_list("x.l") == ["a"]
    assert config.table_list("x.t") == [{"k": 1}]
    assert config.section("x")["s"] == "a"


def test_missing_key_raises_unless_a_default_is_given(config) -> None:
    with pytest.raises(ConfigError, match=r"no\.such"):
        config.get("no.such")
    assert config.get("no.such", 7) == 7
    assert config.get("paths.experiments.deeper", None) is None

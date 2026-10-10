"""Claims checker: tagged values must match the run index; strict files may not hold
untagged decimals outside configured layout, citation and version contexts."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from qcal.config import Config
from qcal.integrity.claims import Finding, check_claims, format_findings
from qcal.registry.index import write_index
from qcal.registry.records import RunRecord
from qcal.registry.store import RegistryStore
from tests.conftest import make_record, write

BASE_TOML = '[registry]\nwrite_parquet = "never"\n'
SECTION = "paper/sections/results.tex"
TABLE = "paper/tables/h1.tex"
CLAIMS = "CLAIMS.md"
ABSTRACT = "paper/sections/abstract.tex"
README = "README.md"

RECORDS: tuple[RunRecord, ...] = (
    make_record("R1", metrics={"AP": 41.0, "LaECE0": 12.3456}),
    make_record("R2", metrics={"AP": 43.0, "LaECE0": 10.0}),
    make_record("R3", metrics={"AP": 45.5, "LaECE0": 8.0, "OCE": 0.75}),
    make_record("R4", metrics={"AP": 39.0, "LaECE0": -0.25}),
    make_record("R0", metrics={"AP": 30.0}),
    make_record("R0b", supersedes="R0", metrics={"AP": 31.0}),
)

Scan = Callable[..., list[Finding]]


def build_index(config: Config, records: Iterable[RunRecord]) -> None:
    store = RegistryStore(config.path("registry_dir"))
    for record in records:
        store.write(record)
    write_index(config, store)


def summary(findings: Iterable[Finding]) -> list[tuple[int, str, str]]:
    return [(f.line, f.kind, f.message) for f in findings]


@pytest.fixture
def configure(make_config: Callable[[str], Config]) -> Callable[[str], Config]:
    def _configure(extra: str = "") -> Config:
        return make_config(BASE_TOML + textwrap.dedent(extra))

    return _configure


@pytest.fixture
def indexed(configure: Callable[[str], Config]) -> Config:
    config = configure("")
    build_index(config, RECORDS)
    return config


@pytest.fixture
def scan(indexed: Config, configure: Callable[[str], Config]) -> Scan:
    """Write one file and run the checker, optionally with extra qcal.toml settings."""

    def _scan(relative: str, text: str, extra: str = "") -> list[Finding]:
        write(indexed.root, relative, text)
        return check_claims(configure(extra) if extra else indexed)

    return _scan


# --- tagged values ----------------------------------------------------------------------


def test_matching_tagged_tex_value_has_no_finding(scan: Scan) -> None:
    assert scan(SECTION, r"AP reaches \qcalval{run:R1:AP}{41.0} on COCO") == []


@pytest.mark.parametrize("shown", ["12", "12.3", "12.35", "12.346", "12.3456", "12.34560"])
def test_tagged_value_is_compared_at_its_displayed_precision(scan: Scan, shown: str) -> None:
    assert scan(SECTION, rf"\qcalval{{run:R1:LaECE0}}{{{shown}}}") == []


@pytest.mark.parametrize(
    ("shown", "registry"),
    [("12.34", "12.35"), ("12.4", "12.3"), ("13", "12"), ("12.3457", "12.3456")],
)
def test_value_differing_at_displayed_precision_is_a_mismatch(
    scan: Scan, shown: str, registry: str
) -> None:
    findings = scan(SECTION, rf"\qcalval{{run:R1:LaECE0}}{{{shown}}}")

    assert summary(findings) == [
        (1, "mismatch", f"run:R1:LaECE0 shows {shown} but the registry gives {registry}")
    ]


@pytest.mark.parametrize("value", ["$41.0$", r"41.0\,\%", " 41.0 "])
def test_number_is_extracted_from_value_markup(scan: Scan, value: str) -> None:
    assert scan(SECTION, rf"\qcalval{{run:R1:AP}}{{{value}}}") == []


@pytest.mark.parametrize(("value", "ok"), [("-0.25", True), ("$-0.25$", True), ("0.25", False)])
def test_sign_of_negative_metric_is_compared(scan: Scan, value: str, ok: bool) -> None:
    findings = scan(SECTION, rf"\qcalval{{run:R4:LaECE0}}{{{value}}}")

    assert (findings == []) is ok


def test_mismatch_reports_the_line_it_occurs_on(scan: Scan) -> None:
    text = "\\section{Results}\nIntro text.\nAP is \\qcalval{run:R1:AP}{40.0} here.\n"

    findings = scan(SECTION, text)

    assert summary(findings) == [
        (3, "mismatch", "run:R1:AP shows 40.0 but the registry gives 41.0")
    ]


def test_every_reference_on_a_line_is_verified(scan: Scan) -> None:
    findings = scan(SECTION, r"\qcalval{run:R1:AP}{40.0} and \qcalval{run:R2:AP}{44.0}")

    assert [f.message for f in findings] == [
        "run:R1:AP shows 40.0 but the registry gives 41.0",
        "run:R2:AP shows 44.0 but the registry gives 43.0",
    ]


@pytest.mark.parametrize(
    ("ref", "value", "message"),
    [
        pytest.param("run:R9:AP", "41.0", "run:R9:AP names unknown run 'R9'", id="unknown-run"),
        pytest.param("run:R1:Nope", "1.0", "run 'R1' has no metric 'Nope'", id="no-column"),
        pytest.param("run:R1:OCE", "0.75", "run 'R1' has no metric 'OCE'", id="empty-cell"),
        pytest.param("run:R1", "41.0", "malformed reference 'run:R1'", id="run-too-short"),
        pytest.param("run:R1:AP:x", "41.0", "malformed reference 'run:R1:AP:x'", id="run-too-long"),
        pytest.param("agg:mean:AP", "41.0", "malformed reference 'agg:mean:AP'", id="agg-no-ids"),
        pytest.param("tab:R1:AP", "41.0", "malformed reference 'tab:R1:AP'", id="bad-kind"),
        pytest.param(
            "run:R1:AP", "--", "value '--' for run:R1:AP must be exactly one number", id="nan"
        ),
        pytest.param("agg:mode:AP:R1+R2", "1.0", "unknown aggregate 'mode'", id="unknown-agg"),
        pytest.param("agg:std:AP:R1", "1.0", "std needs at least two values", id="std-one"),
        pytest.param("agg:mean:AP:", "1.0", "cannot aggregate mean over no values", id="agg-empty"),
    ],
)
def test_unverifiable_reference_is_a_mismatch(
    scan: Scan, ref: str, value: str, message: str
) -> None:
    findings = scan(SECTION, rf"\qcalval{{{ref}}}{{{value}}}")

    assert [f.kind for f in findings] == ["mismatch"]
    assert message in findings[0].message


def test_malformed_reference_message_names_the_expected_forms(scan: Scan) -> None:
    findings = scan(SECTION, r"\qcalval{run:R1}{41.0}")

    assert "expected run:<id>:<metric> or agg:<fn>:<metric>:<ids>" in findings[0].message


@pytest.mark.parametrize(
    ("fn", "shown"),
    [
        ("mean", "43.17"),
        ("median", "43.0"),
        ("min", "41.0"),
        ("max", "45.5"),
        ("sum", "129.5"),
        ("count", "3"),
        ("std", "2.255"),
    ],
)
def test_aggregate_reference_matches_registry(scan: Scan, fn: str, shown: str) -> None:
    assert scan(SECTION, rf"\qcalval{{agg:{fn}:AP:R1+R2+R3}}{{{shown}}}") == []


def test_aggregate_mismatch_reports_the_aggregated_value(scan: Scan) -> None:
    findings = scan(SECTION, r"\qcalval{agg:mean:AP:R1+R2}{41.0}")

    assert [f.message for f in findings] == [
        "agg:mean:AP:R1+R2 shows 41.0 but the registry gives 42.0"
    ]


def test_empty_run_ids_between_separators_are_ignored(scan: Scan) -> None:
    assert scan(SECTION, r"\qcalval{agg:mean:AP:R1++R2+}{42.0}") == []


def test_reference_to_superseded_run_is_a_mismatch(scan: Scan) -> None:
    findings = scan(SECTION, r"\qcalval{run:R0:AP}{30.0}")

    assert [f.message for f in findings] == ["run:R0:AP uses superseded run 'R0'"]


def test_aggregate_over_a_superseded_run_is_a_mismatch(scan: Scan) -> None:
    findings = scan(SECTION, r"\qcalval{agg:mean:AP:R0+R1}{35.5}")

    assert [f.message for f in findings] == ["agg:mean:AP:R0+R1 uses superseded run 'R0'"]


def test_reference_to_the_superseding_run_passes(scan: Scan) -> None:
    assert scan(SECTION, r"\qcalval{run:R0b:AP}{31.0}") == []


def test_superseded_run_is_accepted_when_not_configured_as_error(scan: Scan) -> None:
    findings = scan(
        SECTION, r"\qcalval{run:R0:AP}{30.0}", extra="[claims]\nsuperseded_is_error = false\n"
    )

    assert findings == []


@pytest.mark.parametrize(("tolerance", "kinds"), [("0.0", ["mismatch"]), ("0.15", [])])
def test_abs_tolerance_bounds_the_accepted_difference(
    scan: Scan, tolerance: str, kinds: list[str]
) -> None:
    findings = scan(
        SECTION, r"\qcalval{run:R1:AP}{41.1}", extra=f"[claims]\nabs_tolerance = {tolerance}\n"
    )

    assert [f.kind for f in findings] == kinds


def test_macro_name_comes_from_tables_config(scan: Scan) -> None:
    findings = scan(
        SECTION, r"\resultval{run:R1:AP}{40.0}", extra='[tables]\nmacro = "resultval"\n'
    )

    assert [f.message for f in findings] == ["run:R1:AP shows 40.0 but the registry gives 41.0"]


def test_metric_columns_follow_the_configured_prefix(configure: Callable[[str], Config]) -> None:
    config = configure('[registry.column_prefixes]\nmetrics = "m_"\n')
    build_index(config, RECORDS)
    write(config.root, SECTION, r"\qcalval{run:R1:AP}{41.0} \qcalval{run:R2:AP}{40.0}")

    findings = check_claims(config)

    assert [f.message for f in findings] == ["run:R2:AP shows 40.0 but the registry gives 43.0"]


def test_without_an_index_every_reference_is_unknown(configure: Callable[[str], Config]) -> None:
    config = configure("")
    write(config.root, SECTION, r"\qcalval{run:R1:AP}{41.0}")

    findings = check_claims(config)

    assert [f.message for f in findings] == ["run:R1:AP names unknown run 'R1'"]


# --- Markdown ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "AP is 41.0 (run:R1:AP) on COCO.",
        "AP is 41.0(run:R1:AP)",
        "Mean AP is 42.0 (agg:mean:AP:R1+R2)",
        "LaECE0 is -0.25 (run:R4:LaECE0)",
    ],
)
def test_matching_markdown_tag_has_no_finding(scan: Scan, line: str) -> None:
    assert scan(CLAIMS, line) == []


@pytest.mark.parametrize("relative", [CLAIMS, README])
def test_markdown_tag_mismatch_is_reported(scan: Scan, relative: str) -> None:
    findings = scan(relative, "AP is 41.5 (run:R1:AP)")

    assert summary(findings) == [
        (1, "mismatch", "run:R1:AP shows 41.5 but the registry gives 41.0")
    ]


def test_markdown_reference_pattern_is_configurable(scan: Scan) -> None:
    extra = (
        "[claims]\nmarkdown_ref_pattern = '(?P<value>\\d+\\.\\d+)\\s*\\[(?P<ref>run:[^\\]]+)\\]'\n"
    )

    findings = scan(README, "AP is 40.0 [run:R1:AP]", extra=extra)

    assert [f.message for f in findings] == ["run:R1:AP shows 40.0 but the registry gives 41.0"]


# --- untagged decimals --------------------------------------------------------------------


def test_untagged_decimal_in_strict_tex_is_reported(scan: Scan) -> None:
    findings = scan(SECTION, "AP improves to 41.0 on COCO")

    assert summary(findings) == [(1, "untagged", "number 41.0 has no run reference")]


@pytest.mark.parametrize("relative", [SECTION, "paper/sections/appendix/extra.tex", TABLE, CLAIMS])
def test_strict_globs_flag_untagged_decimals(scan: Scan, relative: str) -> None:
    findings = scan(relative, "AP improves to 41.0 on COCO")

    assert [f.kind for f in findings] == ["untagged"]


@pytest.mark.parametrize("relative", ["paper/main.tex", "paper/appendix/extra.tex", README])
def test_untagged_decimal_outside_strict_globs_is_allowed(scan: Scan, relative: str) -> None:
    assert scan(relative, "AP improves to 41.0 on COCO") == []


@pytest.mark.parametrize(
    ("relative", "line"),
    [("paper/main.tex", r"\qcalval{run:R1:AP}{40.0}"), (README, "40.0 (run:R1:AP)")],
)
def test_tags_in_non_strict_files_are_still_verified(scan: Scan, relative: str, line: str) -> None:
    assert [f.kind for f in scan(relative, line)] == ["mismatch"]


def test_files_outside_the_claims_globs_are_not_read(scan: Scan) -> None:
    assert scan("docs/notes.md", "AP is 40.0 (run:R1:AP) and 3.5 untagged") == []


def test_tagged_value_is_not_also_reported_as_untagged(scan: Scan) -> None:
    findings = scan(SECTION, r"AP \qcalval{run:R1:AP}{40.0} and 1.5 more")

    assert [f.kind for f in findings] == ["mismatch", "untagged"]
    assert findings[1].message == "number 1.5 has no run reference"


@pytest.mark.parametrize(
    "line",
    [
        "We use 3 detectors, 5000 images and seed 0",
        "Table 2 and Section 4 use COCO 2017 with 3 seeds",
        "speedup of 1.5x over FP32",
        "checkpoint yolox-0.5 from the zoo",
        "see eq:2.5 and https://example.org/1.5",
    ],
)
def test_integers_and_embedded_decimals_are_not_results(scan: Scan, line: str) -> None:
    assert scan(SECTION, line) == []


@pytest.mark.parametrize(
    ("relative", "line", "shown"),
    [
        (SECTION, "AP improves by 7 points", "7 points"),
        (SECTION, "AP improves by 1 point", "1 point"),
        (SECTION, r"a 3\% gain over FP32", r"3\%"),
        (SECTION, "INT8 is 2 pp worse", "2 pp"),
        (SECTION, "INT8 is 2pp worse", "2pp"),
        (SECTION, "a drop of 5 percentage points", "5 percentage points"),
        (CLAIMS, "AP rises 3% over FP32", "3%"),
        (CLAIMS, "LaECE0 changes by -2% overall", "-2%"),
    ],
)
def test_whole_number_written_as_a_result_is_untagged(
    scan: Scan, relative: str, line: str, shown: str
) -> None:
    findings = scan(relative, line)

    assert summary(findings) == [(1, "untagged", f"number {shown} has no run reference")]


@pytest.mark.parametrize("line", ["AP improves by 2.5 points", r"a 2.5\% gain"])
def test_decimal_result_with_a_unit_is_reported_once(scan: Scan, line: str) -> None:
    findings = scan(SECTION, line)

    assert [f.message for f in findings] == ["number 2.5 has no run reference"]


@pytest.mark.parametrize(
    ("relative", "line"),
    [
        (SECTION, r"AP improves by \qcalval{run:R1:AP}{41} points"),
        (CLAIMS, "AP improves by 41 (run:R1:AP) points"),
    ],
)
def test_tagged_whole_number_result_is_not_untagged(scan: Scan, relative: str, line: str) -> None:
    assert scan(relative, line) == []


def test_overlapping_number_patterns_report_a_number_once(scan: Scan) -> None:
    extra = "[claims]\ninteger_result_pattern = '-?\\d+(?:\\.\\d+)?\\s*points'\n"

    findings = scan(SECTION, "AP improves by 2.5 points and 3 points", extra=extra)

    assert [f.message for f in findings] == [
        "number 2.5 has no run reference",
        "number 3 points has no run reference",
    ]


def test_tex_unescaped_percent_starts_a_comment_not_a_result(scan: Scan) -> None:
    assert scan(SECTION, "half the runs % 50% of runs were excluded") == []


def test_integer_result_pattern_is_configurable(scan: Scan) -> None:
    extra = "[claims]\ninteger_result_pattern = '\\b\\d+ boxes\\b'\n"

    findings = scan(SECTION, "7 points and 12 boxes", extra=extra)

    assert [f.message for f in findings] == ["number 12 boxes has no run reference"]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(r"\vspace{0.5 em}", id="vspace"),
        pytest.param(r"\hspace*{1.5 cm}", id="hspace-star"),
        pytest.param(r"\scalebox{0.85}{x}", id="scalebox"),
        pytest.param(r"\resizebox{0.9\linewidth}{!}{x}", id="resizebox"),
        pytest.param(r"\includegraphics[width=0.48\linewidth]{fig.pdf}", id="includegraphics"),
        pytest.param(r"\begin{minipage}{0.48\textwidth}", id="relative-width"),
        pytest.param(r"\setlength{\tabcolsep}{4.5 pt}", id="setlength-unit"),
        pytest.param(r"\cite[p.~12]{smith2020}", id="cite-integer-page"),
        pytest.param(r"\label{tab:0.5} \ref{fig:1.5}", id="label-ref"),
        pytest.param(r"see arXiv: 2609.16085 for details", id="arxiv"),
        pytest.param(r"built with TensorRT 10.3.0 and v2.1.4", id="version"),
        pytest.param(r"text % tuned from 0.75 earlier", id="comment"),
    ],
)
def test_tex_layout_citation_and_version_contexts_are_ignored(scan: Scan, line: str) -> None:
    assert scan(SECTION, line) == []


def test_escaped_percent_does_not_start_a_comment(scan: Scan) -> None:
    findings = scan(SECTION, r"41.0\% of boxes")

    assert [f.message for f in findings] == ["number 41.0 has no run reference"]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("set `lr = 0.001` in the config", id="code-span"),
        pytest.param("see arXiv: 2609.16085", id="arxiv"),
        pytest.param("pinned to v1.2.3", id="version"),
        pytest.param("[table](tables.md?scale=0.5)", id="link-target"),
    ],
)
def test_markdown_code_link_and_version_contexts_are_ignored(scan: Scan, line: str) -> None:
    assert scan(CLAIMS, line) == []


# --- escape hatches (ignore marker, \qcalfixed) -------------------------------------------


@pytest.mark.parametrize(
    ("relative", "line"),
    [
        (CLAIMS, "threshold 0.5 by convention <!-- qcal:ignore -->"),
        (ABSTRACT, "threshold 0.5 by convention % qcal:ignore"),
    ],
)
def test_ignore_marker_skips_untagged_numbers_in_escape_hatch_files(
    scan: Scan, relative: str, line: str
) -> None:
    assert scan(relative, line) == []


@pytest.mark.parametrize("relative", [SECTION, TABLE])
def test_ignore_marker_is_not_honoured_outside_escape_hatch_files(
    scan: Scan, relative: str
) -> None:
    findings = scan(relative, "threshold 0.5 by convention % qcal:ignore")

    assert [f.message for f in findings] == ["number 0.5 has no run reference"]


@pytest.mark.parametrize(
    ("relative", "line"),
    [
        (CLAIMS, "40.0 (run:R1:AP) <!-- qcal:ignore -->"),
        (ABSTRACT, r"\qcalval{run:R1:AP}{40.0} % qcal:ignore"),
        (SECTION, r"\qcalval{run:R1:AP}{40.0} % qcal:ignore"),
    ],
)
def test_ignore_marker_never_skips_tag_verification(scan: Scan, relative: str, line: str) -> None:
    findings = scan(relative, line)

    assert [f.message for f in findings] == ["run:R1:AP shows 40.0 but the registry gives 41.0"]


@pytest.mark.parametrize(
    "relative",
    [ABSTRACT, "paper/sections/conclusion.tex", "paper/sections/limitations.tex", CLAIMS],
)
def test_qcalfixed_is_honoured_in_escape_hatch_files(scan: Scan, relative: str) -> None:
    assert scan(relative, r"nominal \qcalfixed{0.95} coverage") == []


@pytest.mark.parametrize("relative", [SECTION, TABLE])
def test_qcalfixed_is_not_honoured_outside_escape_hatch_files(scan: Scan, relative: str) -> None:
    findings = scan(relative, r"nominal \qcalfixed{0.95} coverage")

    assert [f.message for f in findings] == ["number 0.95 has no run reference"]


def test_escape_hatch_categories_are_configurable(scan: Scan) -> None:
    extra = """
        [policy.categories]
        results = ["paper/sections/results.tex"]
        [claims]
        escape_hatch_categories = ["results"]
    """

    assert scan(SECTION, r"nominal \qcalfixed{0.95} and 0.5 % qcal:ignore", extra=extra) == []


def test_no_escape_hatch_categories_disables_every_escape_hatch(scan: Scan) -> None:
    findings = scan(
        CLAIMS,
        "0.5 qcal:ignore\n\\qcalfixed{0.95}\n",
        extra="[claims]\nescape_hatch_categories = []\n",
    )

    assert summary(findings) == [
        (1, "untagged", "number 0.5 has no run reference"),
        (2, "untagged", "number 0.95 has no run reference"),
    ]


def test_escape_patterns_are_configurable(scan: Scan) -> None:
    extra = "[claims]\nescape_patterns = ['\\\\constant\\{[^}]*\\}']\n"

    findings = scan(ABSTRACT, r"\constant{0.95} but \qcalfixed{0.9}", extra=extra)

    assert [f.message for f in findings] == ["number 0.9 has no run reference"]


def test_ignore_marker_only_affects_its_own_line(scan: Scan) -> None:
    findings = scan(CLAIMS, "kept 0.5 qcal:ignore\nflagged 0.5\n")

    assert summary(findings) == [(2, "untagged", "number 0.5 has no run reference")]


@pytest.mark.parametrize(
    ("marker", "line", "kinds"),
    [
        ("nocheck", "0.5 nocheck", []),
        ("nocheck", "0.5 qcal:ignore", ["untagged"]),
        ("", "0.5 qcal:ignore", ["untagged"]),
    ],
)
def test_ignore_marker_is_configurable(
    scan: Scan, marker: str, line: str, kinds: list[str]
) -> None:
    findings = scan(CLAIMS, line, extra=f'[claims]\nignore_line_marker = "{marker}"\n')

    assert [f.kind for f in findings] == kinds


def test_claims_globs_are_configurable(scan: Scan, indexed: Config) -> None:
    write(indexed.root, "manuscript/body.tex", "AP improves to 41.0")
    extra = """
        [claims]
        tex_globs = ["manuscript/**/*.tex"]
        strict_tex_globs = ["manuscript/**/*.tex"]
        markdown_globs = []
    """

    findings = scan(SECTION, "AP improves to 41.0", extra=extra)

    assert [f.path for f in findings] == [indexed.root / "manuscript/body.tex"]


def test_findings_are_ordered_by_file_then_line(scan: Scan, indexed: Config) -> None:
    write(indexed.root, TABLE, "x 1.5\ny 2.5\n")
    write(indexed.root, "paper/sections/a.tex", "z 3.5\n")

    findings = scan(CLAIMS, "w 4.5\n")

    assert [(f.path.relative_to(indexed.root).as_posix(), f.line) for f in findings] == [
        ("paper/sections/a.tex", 1),
        (TABLE, 1),
        (TABLE, 2),
        (CLAIMS, 1),
    ]


def test_packaged_claims_template_passes_the_strict_check(indexed: Config) -> None:
    from qcal.initdocs import init_documents

    init_documents(indexed)

    assert check_claims(indexed) == []


# --- rendering ----------------------------------------------------------------------------


def test_finding_renders_path_relative_to_root(tmp_path: Path) -> None:
    finding = Finding(tmp_path / "CLAIMS.md", 3, "untagged", "number 1.5 has no run reference")

    assert finding.render(tmp_path) == "CLAIMS.md:3: [untagged] number 1.5 has no run reference"


def test_finding_outside_root_renders_its_full_path(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "x.tex"
    finding = Finding(outside, 1, "mismatch", "m")

    assert finding.render(tmp_path / "root") == f"{outside}:1: [mismatch] m"


def test_format_findings_renders_one_finding_per_line(tmp_path: Path) -> None:
    findings = [
        Finding(tmp_path / "a.tex", 1, "untagged", "one"),
        Finding(tmp_path / "b.md", 2, "mismatch", "two"),
    ]

    assert format_findings(findings, tmp_path) == "a.tex:1: [untagged] one\nb.md:2: [mismatch] two"


def test_format_findings_of_nothing_is_empty(tmp_path: Path) -> None:
    assert format_findings([], tmp_path) == ""


# --- regressions in the default patterns -------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "text"),
    [
        (CLAIMS, "LaECE0 drops to 10.3."),
        (CLAIMS, "LaECE0 drops to 10.3. FP32 is unchanged."),
        (SECTION, "LaECE0 drops to 10.3."),
    ],
)
def test_untagged_decimal_ending_a_sentence_is_reported(
    scan: Scan, relative: str, text: str
) -> None:
    findings = scan(relative, text)

    assert summary(findings) == [(1, "untagged", "number 10.3 has no run reference")]


@pytest.mark.parametrize("line", ["AP drops by 2.5 in the INT8 setting", "a gain of 2.1 pt in AP"])
def test_decimal_followed_by_a_unit_like_word_is_reported(scan: Scan, line: str) -> None:
    assert [f.kind for f in scan(SECTION, line)] == ["untagged"]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(r"\vspace{-0.5em} gap", id="unit-attached"),
        pytest.param(r"\addtolength{\textwidth}{1.5 cm}", id="addtolength"),
        pytest.param(r"\setlength {\tabcolsep} {4.5 pt}", id="setlength-spaced"),
        pytest.param(r"\definecolor{shade}{rgb}{0.9,0.9,0.95}", id="definecolor-rgb"),
        pytest.param(r"\definecolor {ink} {cmyk} {0.1, 0.2, 0.3, 0.4}", id="definecolor-spaced"),
    ],
)
def test_multi_argument_layout_commands_are_ignored(scan: Scan, line: str) -> None:
    assert scan(SECTION, line) == []


def test_prose_after_a_layout_command_is_still_checked(scan: Scan) -> None:
    findings = scan(SECTION, r"\definecolor{shade}{rgb}{0.9,0.9,0.95} AP is 41.0")

    assert [f.message for f in findings] == ["number 41.0 has no run reference"]

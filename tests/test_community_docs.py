"""The repository must not claim work it has not done.

Two of these documents were recorded as delivered in ``docs/ROADMAP.md`` while
neither file existed, and the README described a CI pipeline that is not in the
repository. Both are the same class of defect — prose drifting away from the
tree — and prose is not something the rest of the suite looks at.

The CI check below disarms itself: the moment a workflow lands, it stops
constraining what the documentation may say.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A stub is the failure mode here, so each document has to carry some substance.
# CHANGELOG.md is the exception: it is deliberately empty until the first public
# release, so only its presence and its format note are checked.
_REQUIRED_DOCS = {
    "README.md": 1000,
    "LICENSE": 500,
    "CONTRIBUTING.md": 800,
    "SECURITY.md": 800,
    "CODE_OF_CONDUCT.md": 800,
    "CHANGELOG.md": 0,
    "CITATION.cff": 500,
    "AUTHORS": 500,
    "NOTICE": 800,
}

def _published_prose() -> list[str]:
    """Every tracked document, because the claim leaked into `docs/` once already.

    An earlier version of this guard named README.md and pyproject.toml
    explicitly, and three files under `docs/` kept the claim for exactly that
    reason. Asking git rather than keeping a list means a new document is
    covered the day it is added, and the internal working notes stay out on
    their own — they are gitignored, so they never appear here.
    """
    result = subprocess.run(
        ["git", "ls-files", "*.md", "pyproject.toml"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:  # not a checkout, or no git on PATH
        return []
    return sorted(result.stdout.split())


_PUBLISHED_PROSE = _published_prose()


def _tracked_paths() -> set[str]:
    """Everything git knows about, as forward-slash paths from the root.

    Presence on disk is not the question a reader of the published repository
    has. A file that exists only in the author's working tree is, to everyone
    else, a broken link.
    """
    result = subprocess.run(
        ["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return set()
    tracked = set(result.stdout.split())
    # A directory counts as present when it contains something tracked, so that
    # a link to `specialists/conversion/` resolves.
    return tracked | {parent for path in tracked for parent in _ancestors(path)}


def _ancestors(path: str) -> list[str]:
    parts = path.split("/")[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


_TRACKED = _tracked_paths()


@pytest.mark.parametrize(("name", "min_bytes"), sorted(_REQUIRED_DOCS.items()))
def test_community_documents_are_present_and_substantial(name: str, min_bytes: int):
    path = _REPO_ROOT / name

    assert path.is_file(), f"{name} is missing"
    assert path.stat().st_size >= min_bytes, f"{name} looks like a stub"


def test_the_security_policy_names_a_reporting_channel():
    """A security policy with no contact is worse than none: it looks handled."""
    text = (_REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "@" in text or "http" in text


def test_the_prose_discovery_actually_found_the_published_documents():
    """A guard that silently covers nothing is worse than no guard.

    `_published_prose` returns an empty list when git is unavailable, which
    would parametrize the test below into nothing at all. Fail loudly instead.
    """
    assert _REPO_ROOT.joinpath(".git").exists(), "not a checkout — cannot verify prose"
    for expected in ("README.md", "pyproject.toml", "docs/tck.md"):
        assert expected in _PUBLISHED_PROSE


# --- what the documentation says CI does, against what the workflow does -----
#
# This replaces a guard that forbade any document from mentioning CI while
# .github/workflows was empty. It disarmed itself the day the workflow landed,
# and the risk then inverted: the danger is no longer prose describing a
# pipeline that does not exist, but prose describing one that does not do this.
# README.md makes five specific, checkable promises about ci.yml. Each test
# below defends one of them and quotes it.

_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _ci() -> dict:
    """The CI workflow, parsed."""
    return yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))


def _ci_commands() -> str:
    """Every shell command the workflow runs, across all jobs, as one string."""
    return " ".join(
        step["run"]
        for job in _ci()["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    )


def test_the_workflow_runs_on_pushes_and_pull_requests():
    """README: the checks "run on every push and pull request"."""
    workflow = _ci()
    # YAML 1.1 reads a bare `on:` key as the boolean True, so asking for the
    # string alone would silently find nothing and pass.
    triggers = workflow.get("on", workflow.get(True))

    assert triggers is not None, "ci.yml declares no triggers at all"
    assert {"push", "pull_request"} <= set(triggers), (
        f"ci.yml triggers on {sorted(triggers)}, but README.md tells the reader "
        "the checks run on every push and pull request."
    )


def test_the_workflow_covers_linux_and_windows():
    """README: the two commands run "on Linux and Windows"."""
    systems = _ci()["jobs"]["test"]["strategy"]["matrix"]["os"]

    assert any("ubuntu" in s for s in systems), f"no Linux runner in {systems}"
    assert any("windows" in s for s in systems), (
        f"the test matrix is {systems}, with no Windows runner. README.md "
        "promises both, and the one platform-specific failure this repository "
        "has had was Windows-only."
    )


def test_the_workflow_runs_the_two_documented_commands():
    """README's Development section shows pytest and ruff as *the same* two."""
    commands = _ci_commands()

    for tool in ("pytest", "ruff"):
        assert tool in commands, (
            f"no job runs {tool}, but README.md presents it as one of the two "
            "commands CI runs, so a reader expects it to be checked for them."
        )


def test_the_wheel_job_imports_every_supported_path():
    """README: a job "imports every supported path"; ci.yml says it mirrors the table.

    Without this the two drift silently and in the quiet direction: a path is
    added to the Supported table, nobody adds it here, and it can then break in
    a published wheel with CI staying green. That had already happened to three
    of the twelve when this test was written.
    """
    supported = (
        (_REPO_ROOT / "docs" / "api-stability.md")
        .read_text(encoding="utf-8")
        .split("## Supported", 1)[1]
        .split("## Internal", 1)[0]
    )
    documented = re.findall(r"^\|\s*`(erebots[\w.]*)`\s*\|", supported, re.MULTILINE)
    assert documented, "no import paths parsed out of the Supported table"

    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    exercised = set(
        re.findall(r"^\s*(?:from|import)\s+(erebots[\w.]*)", workflow, re.MULTILINE)
    )

    missing = [path for path in documented if path not in exercised]
    assert not missing, (
        f"the wheel job never imports {missing}, which docs/api-stability.md "
        "lists as supported. Add them to the import block in ci.yml, or drop "
        "the row -- an unexercised promise is the kind that breaks."
    )


def test_the_workflow_needs_no_secret_and_no_service():
    """README: "Nothing there needs a secret or a service, so it runs unchanged on a fork".

    That sentence is an invitation to contributors, and it is only true while it
    stays true. A fork cannot read the upstream repository's secrets, so a
    workflow that grows one fails for exactly the people this promises.
    """
    assert "secrets." not in _CI_WORKFLOW.read_text(encoding="utf-8"), (
        "ci.yml reads a secret, but README.md tells contributors the checks run "
        "unchanged on a fork -- where secrets are not available."
    )

    with_services = [name for name, job in _ci()["jobs"].items() if "services" in job]
    assert not with_services, (
        f"jobs {with_services} declare service containers, but README.md says "
        "nothing in CI needs a service."
    )


# A backticked token is only treated as a path claim when it contains a "/" and
# ends in a source extension. A bare `__init__.py` or `client.py` is a kind of
# file, not a location, and checking those would produce nothing but noise.
_PATH_CLAIM = re.compile(
    r"`([\w.][\w./-]*/[\w./-]*\.(?:py|md|yaml|yml|toml|json|j2|html|css|txt|sh|lock|cff))`"
)

# Paths that are deliberately not in the tree.
_NOT_IN_THE_TREE = {
    # The agent docs/adding-an-agent.md walks you through writing. It is the
    # reader's file, not ours.
    "specialists/weather/agent.py",
    # Lives in the separately cloned a2a-tck checkout, not here.
    "./run_tck.py",
}

@pytest.mark.parametrize("name", _PUBLISHED_PROSE)
def test_documents_only_point_at_files_that_exist(name: str):
    """Prose that names a path is making a checkable claim about the tree.

    This is how `docs/adding-an-agent.md` came to instruct readers to edit a
    dict in `bootstrap.py` that lives in `specialists/registry.py`, and how
    `demo.py` kept pointing at an endpoint that had been renamed. A path in
    backticks either resolves from the repository root or from the document's
    own directory, or it is wrong.
    """
    document = _REPO_ROOT / name
    dangling = sorted(
        {
            claim
            for claim in _PATH_CLAIM.findall(document.read_text(encoding="utf-8"))
            if claim not in _NOT_IN_THE_TREE
            and not (_REPO_ROOT / claim).exists()
            and not (document.parent / claim).exists()
        }
    )

    assert not dangling, f"{name} points at paths that do not exist: {dangling}"


# [text](target) where the target is a relative path rather than a URL or an
# in-page anchor.
_RELATIVE_LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)\s#]+)")


@pytest.mark.parametrize("name", _PUBLISHED_PROSE)
def test_relative_links_resolve(name: str):
    """A link to a file that is not in the repository is a broken promise.

    The backtick guard above does not see these: it reads code spans, and a
    link is markup. `docs/architecture.md` was written and linked from the
    README while `.gitignore` quietly excluded it, so the repository shipped a
    link to a page that had never been added — which is exactly the failure
    both guards exist to prevent, arriving through the door the first one does
    not watch.
    """
    document = _REPO_ROOT / name
    here = Path(name).parent
    broken = sorted(
        {
            target
            for target in _RELATIVE_LINK.findall(document.read_text(encoding="utf-8"))
            if _resolve_from(here, target) not in _TRACKED
        }
    )

    assert not broken, f"{name} links to targets that are not in the repository: {broken}"


def _resolve_from(directory: Path, target: str) -> str:
    """Normalize a link target to a repository-root path with forward slashes.

    ``posixpath.normpath`` resolves the ``../`` that documents under `docs/`
    use to reach the tree, and drops the trailing slash on a directory link.
    """
    return posixpath.normpath(posixpath.join(directory.as_posix(), target))


# Rows of the file table in specialists/conversion/README.md:
#   | [`agent.py`](agent.py) | 47 | The BaseSpecialist subclass. ... |
_FILE_TABLE_ROW = re.compile(r"^\|\s*\[`([\w./-]+)`\]\([^)]*\)\s*\|\s*(\d+)\s*\|")


def test_the_example_walkthrough_counts_the_lines_it_claims():
    """A line count in prose rots silently, and this repository has proof.

    `docs/ROADMAP.md` records the orchestrator at "545 LOC" and the A2A client
    at "174 LOC" where the files are now 913 and 427. Those went unnoticed
    because nothing read them. The conversion walkthrough leads with a table of
    per-file counts, so the table is checked instead.
    """
    readme = _REPO_ROOT / "specialists" / "conversion" / "README.md"
    text = readme.read_text(encoding="utf-8")

    wrong: list[str] = []
    counted = 0
    for line in text.splitlines():
        row = _FILE_TABLE_ROW.match(line)
        if row is None:
            continue
        name, claimed = row.group(1), int(row.group(2))
        actual = len((readme.parent / name).read_text(encoding="utf-8").splitlines())
        counted += actual
        if actual != claimed:
            wrong.append(f"{name}: says {claimed}, is {actual}")

    assert counted, "the file table was not found — did the README change shape?"
    assert not wrong, f"specialists/conversion/README.md miscounts: {wrong}"
    assert f"{counted} lines" in text, (
        f"the table now totals {counted} lines, which the prose no longer states"
    )


# "255 lines" — a size claim about the example, wherever it is repeated.
_LINE_COUNT_CLAIM = re.compile(r"\b(\d+) lines\b")


def test_the_example_size_is_quoted_consistently_everywhere():
    """The figure is repeated in five documents; only one of them owns it.

    Quoting a measurement in several places is how `docs/ROADMAP.md` ended up
    three times off on the orchestrator's size. Any document that talks about
    the conversion example and states a line count has to state the same one
    the files add up to.
    """
    readme = _REPO_ROOT / "specialists" / "conversion" / "README.md"
    total = sum(
        len((readme.parent / name).read_text(encoding="utf-8").splitlines())
        for name, _ in (
            (row.group(1), row.group(2))
            for row in (
                _FILE_TABLE_ROW.match(line)
                for line in readme.read_text(encoding="utf-8").splitlines()
            )
            if row is not None
        )
    )

    wrong: list[str] = []
    for name in _PUBLISHED_PROSE:
        if not name.endswith(".md"):
            continue
        text = (_REPO_ROOT / name).read_text(encoding="utf-8")
        if "conversion" not in text:
            continue
        for quoted in _LINE_COUNT_CLAIM.findall(text):
            if int(quoted) != total:
                wrong.append(f"{name}: says {quoted} lines, the example is {total}")

    assert not wrong, f"the example's size is quoted inconsistently: {wrong}"


def test_every_setting_is_documented():
    """A setting nobody wrote down is one nobody can deploy against.

    Fourteen of these reached this point undocumented, including the two that
    decide how far a turn fans out. `env.example` carried them, but a template
    is not a reference. Binding the document to the dataclass is what stops the
    next one from being added silently.
    """
    import dataclasses

    from erebots.config import Settings

    documented = (_REPO_ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    undocumented = sorted(
        field.name
        for field in dataclasses.fields(Settings)
        if f"`{field.name}`" not in documented
    )

    assert not undocumented, (
        f"docs/configuration.md does not mention: {undocumented}. "
        "Add a row for each, or the setting exists only for whoever wrote it."
    )


def test_the_documentation_index_lists_every_page():
    """A page nobody links to is a page nobody reads.

    `docs/` grew to nine pages while the README linked three, which is how two
    conformance documents came to be reachable only by browsing the directory.
    The index is also the URL handed out as the developer documentation, so a
    page missing from it is effectively unpublished.
    """
    index = _REPO_ROOT / "docs" / "README.md"
    listed = index.read_text(encoding="utf-8")

    pages = sorted(
        path.name
        for path in (_REPO_ROOT / "docs").glob("*.md")
        if path.name != "README.md" and f"docs/{path.name}" in _TRACKED
    )
    unlisted = [name for name in pages if f"({name})" not in listed]

    assert pages, "no pages found under docs/ — did the layout change?"
    assert not unlisted, f"docs/README.md does not link: {unlisted}"


def test_the_run_record_template_survives():
    """The procedure is the repository's contribution; keep somewhere to write it down.

    Results are reported in the paper rather than here, so the record in
    `tck.md` is a blank form. A blank form is easy to delete as clutter, and
    then the next person measuring has nowhere to state the commit and the tool
    versions a number is only meaningful against.
    """
    record = (_REPO_ROOT / "docs" / "tck.md").read_text(encoding="utf-8")

    assert "## Run record" in record, "docs/tck.md lost its run record"
    for field in ("Erebots4.0 commit", "a2a-tck commit", "Endpoints tested"):
        assert field in record, f"the run record no longer asks for {field!r}"


def test_any_conformance_count_quoted_here_is_backed_by_the_record():
    """Arms itself if numbers come back, and passes while there are none.

    The two pages once disagreed -- 94 against 97 -- and both were right, one
    being a single endpoint and the other the union, with neither saying so.
    The numbers have since moved to the paper, so there is nothing to compare
    today. This does not therefore go away: the moment somebody quotes a count
    in the summary again, it has to appear in the run record it points at.

    The record's own section is searched rather than the whole page, and each
    count has to stand as a number of its own. Searching the page for the digits
    as a substring is how the first version of this check passed against a
    document that contained no record at all -- "94" turns up inside a version
    string or a percentage sooner or later.
    """
    summary = (_REPO_ROOT / "docs" / "a2a-conformance.md").read_text(encoding="utf-8")
    page = (_REPO_ROOT / "docs" / "tck.md").read_text(encoding="utf-8")

    row = re.search(r"^\|\s*Requirements passed\s*\|(.+)\|\s*$", summary, re.MULTILINE)
    if row is None:
        return  # no counts quoted; nothing to keep in step

    start = page.index("## Run record")
    end = page.find("\n## ", start + 1)
    record = page[start : end if end != -1 else len(page)]

    quoted = re.findall(r"\d+", row.group(1))
    missing = [n for n in quoted if not re.search(rf"(?<!\d){n}(?!\d)", record)]
    assert not missing, (
        f"docs/a2a-conformance.md quotes {missing}, which docs/tck.md's run record "
        "does not. Fill in the record, or say which endpoint each count is for."
    )


# --- citation metadata: the three copies of the author list agree ------------


def _authors_from_citation_cff() -> list[tuple[str, str]]:
    """(full name, ORCID) for each author in CITATION.cff, in file order."""
    data = yaml.safe_load((_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    return [
        (f"{a['given-names']} {a['family-names']}", a["orcid"]) for a in data["authors"]
    ]


def _numbered_authors_from_authors_file() -> list[tuple[int, str, str]]:
    """(position, full name, ORCID) for each line of AUTHORS' numbered list."""
    listing = (_REPO_ROOT / "AUTHORS").read_text(encoding="utf-8")
    return [
        (int(position), name.strip(), orcid)
        for position, name, orcid in re.findall(
            r"^\s*(\d+)\.\s+(.+?)\s{2,}(https://orcid\.org/\S+)\s*$",
            listing,
            re.MULTILINE,
        )
    ]


def _authors_from_authors_file() -> list[tuple[str, str]]:
    """The same list as the CFF's, without the printed position."""
    return [(name, orcid) for _, name, orcid in _numbered_authors_from_authors_file()]


def test_every_author_carries_an_affiliation():
    """An author added later must not arrive without one.

    The seven here share an institution, so the field is easy to forget the day
    an eighth name is added -- and a citation record with six affiliations and
    one blank reads as an oversight about that one person rather than as a
    missing field.
    """
    data = yaml.safe_load((_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    missing = [
        f"{a['given-names']} {a['family-names']}"
        for a in data["authors"]
        if not (a.get("affiliation") or "").strip()
    ]
    assert not missing, f"no affiliation for {missing} in CITATION.cff"


def test_the_authors_file_is_numbered_in_order():
    """AUTHORS prints a position against each name, so the digits have to be right.

    The list is read by people, and a human reader takes the printed number at
    face value rather than counting lines. Nothing else here would notice a
    renumbering, because every other check works from the order of the lines.
    """
    numbered = _numbered_authors_from_authors_file()
    printed = [position for position, _, _ in numbered]
    assert printed == list(range(1, len(numbered) + 1)), (
        f"AUTHORS is numbered {printed}, which is not 1..{len(numbered)} in order. "
        "The printed positions are what a reader trusts; renumber the whole list."
    )


def _orcid_check_digit(orcid: str) -> str:
    """The ORCID checksum, ISO 7064 mod 11-2 over the first fifteen digits."""
    digits = orcid.rsplit("/", 1)[-1].replace("-", "")
    total = 0
    for digit in digits[:15]:
        total = (total + int(digit)) * 2
    remainder = (12 - total % 11) % 11
    return "X" if remainder == 10 else str(remainder)


def test_the_author_list_is_the_same_in_every_file():
    """CITATION.cff, AUTHORS and pyproject.toml carry one list, in one order.

    Three files repeat the same names because three consumers read different
    ones: GitHub reads the CFF, a person reads AUTHORS, and a package index
    reads the metadata. Order is part of the claim in an author list, so
    this compares sequences rather than sets.
    """
    cff = _authors_from_citation_cff()
    assert cff, "CITATION.cff lists no authors"

    assert _authors_from_authors_file() == cff, (
        "AUTHORS and CITATION.cff disagree on the author list or its order. "
        "Both files are the same claim written twice; change them together."
    )

    metadata = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = [author["name"] for author in metadata["project"]["authors"]]
    assert names == [name for name, _ in cff], (
        "pyproject.toml's `authors` disagrees with CITATION.cff. The packaging "
        "metadata has no ORCID field, so it carries the names and the order only."
    )


@pytest.mark.parametrize("name,orcid", _authors_from_citation_cff())
def test_each_orcid_is_well_formed(name: str, orcid: str):
    """An ORCID carries its own check digit, so a typo is detectable here.

    Worth checking because the cost is asymmetric: a wrong identifier in a
    citation file propagates into everything that cites this, and into the
    wrong person's publication record, where nobody is looking for it.
    """
    assert re.fullmatch(r"https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid), (
        f"{name}'s ORCID is not a bare https://orcid.org/ identifier: {orcid}"
    )
    assert orcid[-1] == _orcid_check_digit(orcid), (
        f"{name}'s ORCID {orcid} fails its own checksum, so a digit is wrong. "
        "Check it against the person's ORCID record rather than adjusting it."
    )

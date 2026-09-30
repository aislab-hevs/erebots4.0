# Contributing

Thanks for looking. The most useful contributions here are small and specific: a
failing case, a sharper error message, a fix with a test. Anything that changes a
path listed in [docs/api-stability.md](docs/api-stability.md) is a breaking
change and needs to be discussed first.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv run pytest -q          # the whole suite, offline, under half a minute
uvx ruff check .          # lint
```

The suite needs no network, no model and no database. If it wants any of those,
something is wrong with the change, not with your machine.

Running the framework itself needs an OpenAI-compatible LLM; see the README
quickstart, or `demo.py` for the smallest deployment.

## What runs on a pull request

`.github/workflows/ci.yml`, in three jobs so a failure names itself:

- **tests** — `uv sync --frozen` then `uv run pytest -q`, on Linux *and* Windows.
  Both, because the one platform-specific failure this project has had was a
  Windows-only import error that a Linux-only matrix would have missed.
- **lint** — `uvx ruff@0.16.9 check .`, the same check as locally, with the
  version pinned in the workflow rather than in `uv.lock` (see below).
- **wheel builds and imports** — building is not the check; importing is. The
  wheel is installed into an empty environment and every path in
  [docs/api-stability.md](docs/api-stability.md) is imported, because a
  published wheel was once uninstallable while `uv build` was perfectly happy.

Run the same commands locally before opening a pull request. Nothing in CI needs
a secret, a service or a model, so it runs on a fork unchanged.

## Two rules that are not obvious

**The framework never imports the example.** `erebots/` must not import
`specialists`, `gradio`, or any other application-level package.
`tests/test_import_boundaries.py` enforces this with an AST sweep, and it is the
invariant the whole framework claim rests on. If your change needs the framework
to know something about an application, the answer is a parameter or a setting,
not an import.

**Every change carries a test.** Including a documentation claim about the code,
where feasible — see `tests/test_community_docs.py` for the pattern.

## Public API

`docs/api-stability.md` lists the import paths the framework supports. If you
change one, update that document, `tests/test_public_api.py` and `CHANGELOG.md`
in the same commit.

After a packaging change, confirm the wheel is still right:

```bash
uv build --wheel
# the wheel must contain erebots/py.typed and the .j2 prompt templates,
# and must not contain specialists/, app.py or demo.py
```

And after changing a dependency bound, install it somewhere else entirely:

```bash
cd $(mktemp -d) && uv venv && uv pip install /path/to/erebots-*.whl
python -c "import erebots.server, erebots.agents"
```

The test suite runs against `uv.lock`, so it cannot see this. A fresh install
resolves whatever is newest within the bounds in `pyproject.toml`, and that is
what someone installing the package gets. `mcp 2.0` renamed a function the
framework imports; with no upper bound, the published wheel did not import at
all while every test was green.

## Commits

`type: description`, lower case, imperative. `feat:`, `fix:`, `refactor:`,
`test:`, `docs:`, `chore:`.

Explain *why* in the body when the change is not obvious from the diff. The
existing history is the reference — several commits are worth reading before you
write one.

## Pull requests

Target the development branch, not `main`. Keep one concern per pull request; if
you found three things, three pull requests are easier to review and to revert.

Say what you verified and what you did not. "Tested on Linux, not on Windows" is
more useful than silence — the project has had exactly that bug.

## Reporting a vulnerability

Do not open an issue. See [SECURITY.md](SECURITY.md).

"""Pytest session setup.

Lives at the repository root, not under ``tests/``, so it is imported before any
collected module anywhere in ``testpaths`` — including the co-located skill tests
under ``specialists/`` — and therefore before anything can import
``erebots.config`` and build its settings singleton.

Its one job: keep a developer's ``.env`` out of the test run. ``Settings`` is
built at import time from ``os.environ``, which ``_load_env_file`` seeds from
``.env``; without this, the suite silently inherits whatever the developer has
configured locally. CI runs on a clean checkout with no ``.env``, so the declared
defaults are the intended baseline, and a test that needs a different value says
so explicitly (see the ``_NoAuth`` stubs in ``tests/test_server_app.py``).

To run the suite against a real ``.env`` — to reproduce a configuration-specific
problem — clear the flag: ``EREBOTS_IGNORE_DOTENV= uv run pytest``.
"""

import os

os.environ.setdefault("EREBOTS_IGNORE_DOTENV", "1")

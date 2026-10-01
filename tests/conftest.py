"""
Shared pytest fixtures.

Importing app.py loads SBERT + all 10 trained models (~15-20s) -- that only
needs to happen once per test run, not once per test, so the Flask app/client
are session-scoped fixtures.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def flask_app():
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture(scope="session")
def app_module():
    import app as app_module
    return app_module


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()

"""Pytest fixtures for web backend testing (P08).
"""
from __future__ import annotations

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from web.backend.app import create_app


@pytest.fixture
def app():
    app_instance = create_app({"TESTING": True})
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()

"""Suite-wide guard: DQN tests must not be skipped silently.

torch is a core requirement (requirements.txt). Without it every DQN test
skips, so a green run would say nothing about the DQN agent. The run therefore
stops at start-up unless the skip is requested explicitly with
WALLZ_SKIP_DQN=1, and the skip is then reported in the summary.
"""
import importlib.util
import os

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DQN_SKIP_REQUESTED = os.environ.get("WALLZ_SKIP_DQN") == "1"


def pytest_configure(config):
    if not TORCH_AVAILABLE and not DQN_SKIP_REQUESTED:
        raise pytest.UsageError(
            "torch is not installed, so all DQN tests would be skipped. "
            "Install requirements.txt (e.g. `.venv/bin/pip install -r requirements-dev.txt`) "
            "or set WALLZ_SKIP_DQN=1 to run without DQN coverage."
        )


def pytest_terminal_summary(terminalreporter):
    if not TORCH_AVAILABLE:
        terminalreporter.write_sep(
            "!", "WALLZ_SKIP_DQN=1: torch missing, DQN tests were NOT run", red=True
        )

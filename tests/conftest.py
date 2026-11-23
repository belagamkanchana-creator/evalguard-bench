import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evalguard.core.agent import SimulatedAgent, load_scenarios  # noqa: E402
from evalguard.core.judge import LLMBackend  # noqa: E402


@pytest.fixture(scope="session")
def scenarios():
    return load_scenarios(ROOT / "data" / "scenarios.yaml")


@pytest.fixture
def agent():
    return SimulatedAgent()


class ScriptedBackend(LLMBackend):
    """Test backend that returns canned replies (or a reply computed from the prompt)."""

    name = "scripted"

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply(system, user) if callable(self.reply) else self.reply


@pytest.fixture
def scripted():
    return ScriptedBackend

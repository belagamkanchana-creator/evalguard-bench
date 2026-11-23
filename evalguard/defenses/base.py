"""Base class for defenses.

A defense can act at three points of the judging pipeline:

1. ``screen``                 drop an input before the judge sees it (input filtering)
2. ``transform_system_prompt``  change the judge's instructions (prompt hardening, reminders)
3. ``render_untrusted``       change how external content is shown to the judge (boundaries)
"""

from __future__ import annotations

from collections.abc import Sequence

from evalguard.core.models import AgentTrajectory, JudgeMode, ToolReturn


class Defense:
    name: str = "base"

    def fit(self, clean_texts: Sequence[str]) -> None:
        """Calibrate on known-clean external content. Most defenses need no calibration."""

    def screen(self, trajectory: AgentTrajectory) -> str | None:
        """Return a reason string to drop this input before judging, or None to let it through."""
        return None

    def transform_system_prompt(self, system_prompt: str, mode: JudgeMode) -> str:
        return system_prompt

    def render_untrusted(self, tool_returns: Sequence[ToolReturn]) -> str | None:
        """Return the judge-facing rendering of external content, or None to use the default."""
        return None

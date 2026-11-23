"""Adversarial attack suite."""

from evalguard.attacks.base import Attack, AttackContext
from evalguard.attacks.ipi_attacks import (
    AuthoritySpoofAttack,
    DelimiterHijackAttack,
    IndirectPromptInjector,
    PayloadSplitAttack,
)
from evalguard.attacks.justification_manip import JustificationManipulator
from evalguard.attacks.universal_suffix import UniversalSuffixAttacker

__all__ = [
    "Attack",
    "AttackContext",
    "AuthoritySpoofAttack",
    "DelimiterHijackAttack",
    "IndirectPromptInjector",
    "JustificationManipulator",
    "PayloadSplitAttack",
    "UniversalSuffixAttacker",
]

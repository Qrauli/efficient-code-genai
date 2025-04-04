"""
Efficient Code Generation using AI
==================================

A package for rule-based code generation using AI.
"""

from .config import Config
from .agents.rule_orchestrator import RuleOrchestrator

__all__ = ["RuleOrchestrator", "Config"]
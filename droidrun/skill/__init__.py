"""
Enhanced Skill abstraction module for DroidRun.

This module provides functionality to convert trajectories into reusable,
parameterized skills with Claude-like agent patterns:

- Pre/post-conditions for execution validation
- Skill dependencies and composition
- Context-aware execution with retry mechanisms
- Enhanced error handling and rollback
- Execution telemetry and monitoring
- Pattern recognition and intelligent extraction
"""

from droidrun.skill.skill import (
    Skill,
    SkillParameter,
    ParameterType,
    SkillComplexity,
    Condition,
    SkillDependency,
    ExecutionContext,
)
from droidrun.skill.skill_library import SkillLibrary
from droidrun.skill.skill_extractor import SkillExtractor
from droidrun.skill.skill_executor import SkillExecutor, ExecutionResult
from droidrun.skill.code_generator import SkillCodeGenerator

__all__ = [
    # Core classes
    "Skill",
    "SkillParameter",
    "SkillLibrary",
    "SkillExtractor",
    "SkillExecutor",
    "SkillCodeGenerator",
    
    # Enums and types
    "ParameterType",
    "SkillComplexity",
    
    # Advanced features
    "Condition",
    "SkillDependency",
    "ExecutionContext",
    "ExecutionResult",
]

__version__ = "2.0.0"

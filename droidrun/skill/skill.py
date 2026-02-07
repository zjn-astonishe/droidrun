"""
Skill class for representing parameterized, reusable action sequences.

Enhanced with Claude-like agent skill patterns:
- Pre/post-conditions for execution validation
- Skill dependencies and composition
- Context awareness and state management
- Enhanced error handling and recovery
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("droidrun")


class ParameterType(str, Enum):
    """Type of skill parameter."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    COORDINATE = "coordinate"
    LIST = "list"
    DICT = "dict"


class SkillComplexity(str, Enum):
    """Complexity level of a skill."""
    SIMPLE = "simple"      # Single atomic action
    MODERATE = "moderate"  # Multiple related actions
    COMPLEX = "complex"    # Multi-step workflow


@dataclass
class SkillParameter:
    """A parameter that can be customized when executing a skill."""

    name: str
    description: str
    param_type: ParameterType
    default_value: Any = None
    required: bool = True
    validator: Optional[Callable[[Any], bool]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)  # min, max, pattern, enum, etc.

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Type validation
        if self.param_type == ParameterType.STRING and not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        elif self.param_type == ParameterType.INTEGER and not isinstance(value, int):
            return False, f"Expected integer, got {type(value).__name__}"
        elif self.param_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False, f"Expected float, got {type(value).__name__}"
        elif self.param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False, f"Expected boolean, got {type(value).__name__}"
        elif self.param_type == ParameterType.COORDINATE:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, "Expected coordinate [x, y]"
        
        # Constraint validation
        if "min" in self.constraints and value < self.constraints["min"]:
            return False, f"Value {value} below minimum {self.constraints['min']}"
        if "max" in self.constraints and value > self.constraints["max"]:
            return False, f"Value {value} above maximum {self.constraints['max']}"
        if "enum" in self.constraints and value not in self.constraints["enum"]:
            return False, f"Value must be one of {self.constraints['enum']}"
        if "pattern" in self.constraints and isinstance(value, str):
            import re
            if not re.match(self.constraints["pattern"], value):
                return False, f"Value does not match pattern {self.constraints['pattern']}"
        
        # Custom validator
        if self.validator and not self.validator(value):
            return False, f"Custom validation failed for {self.name}"
        
        return True, None


@dataclass
class Condition:
    """Represents a pre or post condition for skill execution."""
    
    name: str
    description: str
    check: Callable[[Dict[str, Any]], bool]  # Function that checks the condition
    required: bool = True  # If False, warning only
    error_message: Optional[str] = None


@dataclass
class SkillDependency:
    """Represents a dependency on another skill."""
    
    skill_name: str
    version_constraint: Optional[str] = None  # e.g., ">=1.0,<2.0"
    optional: bool = False
    fallback_skill: Optional[str] = None  # Alternative skill if primary unavailable


@dataclass
class ExecutionContext:
    """Context information available during skill execution."""
    
    device_state: Dict[str, Any] = field(default_factory=dict)
    app_state: Dict[str, Any] = field(default_factory=dict)
    previous_actions: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context with fallback order."""
        return (
            self.metadata.get(key) or
            self.device_state.get(key) or
            self.app_state.get(key) or
            self.environment.get(key) or
            default
        )


@dataclass
class Skill:
    """
    Enhanced skill representing a reusable sequence of actions.
    
    Features:
    - Pre/post-conditions for validation
    - Dependencies on other skills
    - Context-aware execution
    - Composition support
    - Enhanced error recovery
    """

    name: str
    description: str
    actions: List[Dict[str, Any]]
    parameters: List[SkillParameter] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    # Enhanced features
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)
    dependencies: List[SkillDependency] = field(default_factory=list)
    complexity: SkillComplexity = SkillComplexity.MODERATE
    estimated_duration: Optional[float] = None  # seconds
    reliability_score: float = 1.0  # 0.0 to 1.0
    tags: Set[str] = field(default_factory=set)
    
    # Retry and recovery
    max_retries: int = 0
    retry_delay: float = 1.0  # seconds
    rollback_actions: Optional[List[Dict[str, Any]]] = None

    def validate_parameters(self, param_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameter values against parameter definitions.
        
        Returns:
            Dictionary with 'valid', 'errors', and 'warnings' keys
        """
        errors = []
        warnings = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in param_values:
                if param.default_value is not None:
                    param_values[param.name] = param.default_value
                else:
                    errors.append(f"Required parameter '{param.name}' not provided")
        
        # Validate each provided parameter
        for param_name, param_value in param_values.items():
            param = next((p for p in self.parameters if p.name == param_name), None)
            if param is None:
                warnings.append(f"Unknown parameter '{param_name}'")
                continue
            
            is_valid, error_msg = param.validate(param_value)
            if not is_valid:
                errors.append(f"Parameter '{param_name}': {error_msg}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def check_preconditions(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Check if preconditions are met.
        
        Returns:
            Dictionary with 'met', 'failed', and 'warnings' keys
        """
        failed = []
        warnings = []
        
        for condition in self.preconditions:
            try:
                if not condition.check(context.__dict__):
                    msg = condition.error_message or f"Precondition '{condition.name}' not met"
                    if condition.required:
                        failed.append(msg)
                    else:
                        warnings.append(msg)
            except Exception as e:
                error_msg = f"Error checking precondition '{condition.name}': {str(e)}"
                if condition.required:
                    failed.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        return {
            "met": len(failed) == 0,
            "failed": failed,
            "warnings": warnings,
        }

    def check_postconditions(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Check if postconditions are met after execution.
        
        Returns:
            Dictionary with 'met', 'failed', and 'warnings' keys
        """
        failed = []
        warnings = []
        
        for condition in self.postconditions:
            try:
                if not condition.check(context.__dict__):
                    msg = condition.error_message or f"Postcondition '{condition.name}' not met"
                    if condition.required:
                        failed.append(msg)
                    else:
                        warnings.append(msg)
            except Exception as e:
                error_msg = f"Error checking postcondition '{condition.name}': {str(e)}"
                if condition.required:
                    failed.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        return {
            "met": len(failed) == 0,
            "failed": failed,
            "warnings": warnings,
        }

    def apply_parameters(self, param_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply parameter values to the skill's actions.
        
        Args:
            param_values: Dictionary mapping parameter names to their values
            
        Returns:
            List of actions with parameters applied
        """
        # Validate parameters first
        validation = self.validate_parameters(param_values)
        if not validation["valid"]:
            raise ValueError(f"Parameter validation failed: {validation['errors']}")

        # Deep copy actions to avoid modifying the original
        parameterized_actions = copy.deepcopy(self.actions)

        # Replace parameter placeholders in actions
        for action in parameterized_actions:
            self._replace_placeholders(action, param_values)

        return parameterized_actions

    def _replace_placeholders(
        self, obj: Any, param_values: Dict[str, Any]
    ) -> None:
        """
        Recursively replace parameter placeholders in an object.
        
        Placeholders are in the format: {{param_name}}
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # Replace placeholders
                    for param_name, param_value in param_values.items():
                        placeholder = f"{{{{{param_name}}}}}"
                        if placeholder in value:
                            if value == placeholder:
                                # Entire value is the parameter, use actual type
                                obj[key] = param_value
                            else:
                                # Partial replacement, convert to string
                                obj[key] = value.replace(placeholder, str(param_value))
                elif isinstance(value, (dict, list)):
                    self._replace_placeholders(value, param_values)
        elif isinstance(obj, list):
            for item in obj:
                self._replace_placeholders(item, param_values)

    def compose_with(self, other: "Skill", merge_parameters: bool = True) -> "Skill":
        """
        Compose this skill with another skill to create a new composite skill.
        
        Args:
            other: Another skill to compose with
            merge_parameters: Whether to merge parameters from both skills
            
        Returns:
            New composite skill
        """
        # Merge actions
        combined_actions = self.actions + other.actions
        
        # Merge parameters if requested
        combined_parameters = list(self.parameters)
        if merge_parameters:
            for param in other.parameters:
                if not any(p.name == param.name for p in combined_parameters):
                    combined_parameters.append(param)
        
        # Merge conditions
        combined_preconditions = list(self.preconditions) + list(other.preconditions)
        combined_postconditions = list(self.postconditions) + list(other.postconditions)
        
        # Merge dependencies
        combined_dependencies = list(self.dependencies) + list(other.dependencies)
        
        # Create new skill
        return Skill(
            name=f"{self.name}_then_{other.name}",
            description=f"{self.description} followed by {other.description}",
            actions=combined_actions,
            parameters=combined_parameters,
            preconditions=combined_preconditions,
            postconditions=combined_postconditions,
            dependencies=combined_dependencies,
            complexity=SkillComplexity.COMPLEX,
            version="1.0",
            metadata={
                "composed_from": [self.name, other.name],
                "composition_timestamp": self.metadata.get("created_at"),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary format for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "complexity": self.complexity.value if isinstance(self.complexity, SkillComplexity) else self.complexity,
            "estimated_duration": self.estimated_duration,
            "reliability_score": self.reliability_score,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "tags": list(self.tags),
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "param_type": p.param_type.value if isinstance(p.param_type, ParameterType) else p.param_type,
                    "default_value": p.default_value,
                    "required": p.required,
                    "constraints": p.constraints,
                }
                for p in self.parameters
            ],
            "actions": self.actions,
            "rollback_actions": self.rollback_actions,
            "dependencies": [
                {
                    "skill_name": d.skill_name,
                    "version_constraint": d.version_constraint,
                    "optional": d.optional,
                    "fallback_skill": d.fallback_skill,
                }
                for d in self.dependencies
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create a skill from dictionary format."""
        parameters = [
            SkillParameter(
                name=p["name"],
                description=p["description"],
                param_type=ParameterType(p["param_type"]) if isinstance(p["param_type"], str) else p["param_type"],
                default_value=p.get("default_value"),
                required=p.get("required", True),
                constraints=p.get("constraints", {}),
            )
            for p in data.get("parameters", [])
        ]
        
        dependencies = [
            SkillDependency(
                skill_name=d["skill_name"],
                version_constraint=d.get("version_constraint"),
                optional=d.get("optional", False),
                fallback_skill=d.get("fallback_skill"),
            )
            for d in data.get("dependencies", [])
        ]

        complexity_value = data.get("complexity", "moderate")
        complexity = SkillComplexity(complexity_value) if isinstance(complexity_value, str) else complexity_value

        return cls(
            name=data["name"],
            description=data["description"],
            actions=data["actions"],
            parameters=parameters,
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
            dependencies=dependencies,
            complexity=complexity,
            estimated_duration=data.get("estimated_duration"),
            reliability_score=data.get("reliability_score", 1.0),
            max_retries=data.get("max_retries", 0),
            retry_delay=data.get("retry_delay", 1.0),
            rollback_actions=data.get("rollback_actions"),
            tags=set(data.get("tags", [])),
        )

    def save(self, filepath: str) -> None:
        """Save skill to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved skill '{self.name}' to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Skill":
        """Load skill from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"📖 Loaded skill '{data['name']}' from {filepath}")
        return cls.from_dict(data)

    def get_summary(self) -> str:
        """Get a human-readable summary of the skill."""
        summary = [
            f"Skill: {self.name}",
            f"Description: {self.description}",
            f"Version: {self.version}",
            f"Complexity: {self.complexity.value if isinstance(self.complexity, SkillComplexity) else self.complexity}",
            f"Reliability: {self.reliability_score:.2f}",
            f"Total actions: {len(self.actions)}",
        ]
        
        if self.estimated_duration:
            summary.append(f"Estimated duration: {self.estimated_duration}s")

        if self.parameters:
            summary.append(f"\nParameters ({len(self.parameters)}):")
            for param in self.parameters:
                required = "required" if param.required else "optional"
                default = (
                    f", default={param.default_value}"
                    if param.default_value is not None
                    else ""
                )
                summary.append(
                    f"  - {param.name} ({param.param_type.value if isinstance(param.param_type, ParameterType) else param.param_type}, {required}{default}): {param.description}"
                )
        
        if self.dependencies:
            summary.append(f"\nDependencies:")
            for dep in self.dependencies:
                opt = " (optional)" if dep.optional else ""
                summary.append(f"  - {dep.skill_name}{opt}")
        
        if self.preconditions:
            summary.append(f"\nPreconditions: {len(self.preconditions)}")
        
        if self.postconditions:
            summary.append(f"Postconditions: {len(self.postconditions)}")

        # Action type breakdown
        action_types: Dict[str, int] = {}
        for action in self.actions:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        if action_types:
            summary.append("\nAction breakdown:")
            for action_type, count in action_types.items():
                summary.append(f"  - {action_type}: {count}")

        return "\n".join(summary)

    def __str__(self) -> str:
        return self.get_summary()

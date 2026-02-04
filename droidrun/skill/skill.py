"""
Skill class for representing parameterized, reusable action sequences.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("droidrun")


@dataclass
class SkillParameter:
    """A parameter that can be customized when executing a skill."""

    name: str
    description: str
    param_type: str  # 'text', 'coordinate', 'number', 'boolean'
    default_value: Any = None
    required: bool = True


@dataclass
class Skill:
    """
    A skill represents a reusable sequence of actions extracted from a trajectory.
    
    Skills can be parameterized to work across different contexts (e.g., different
    text inputs, coordinates, or app states).
    """

    name: str
    description: str
    actions: List[Dict[str, Any]]
    parameters: List[SkillParameter] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary format for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "param_type": p.param_type,
                    "default_value": p.default_value,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "actions": self.actions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create a skill from dictionary format."""
        parameters = [
            SkillParameter(
                name=p["name"],
                description=p["description"],
                param_type=p["param_type"],
                default_value=p.get("default_value"),
                required=p.get("required", True),
            )
            for p in data.get("parameters", [])
        ]

        return cls(
            name=data["name"],
            description=data["description"],
            actions=data["actions"],
            parameters=parameters,
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
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

    def apply_parameters(self, param_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply parameter values to the skill's actions.
        
        Args:
            param_values: Dictionary mapping parameter names to their values
            
        Returns:
            List of actions with parameters applied
        """
        # Validate required parameters
        for param in self.parameters:
            if param.required and param.name not in param_values:
                if param.default_value is not None:
                    param_values[param.name] = param.default_value
                else:
                    raise ValueError(
                        f"Required parameter '{param.name}' not provided"
                    )

        # Deep copy actions to avoid modifying the original
        import copy

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
                            obj[key] = value.replace(placeholder, str(param_value))
                elif isinstance(value, (dict, list)):
                    self._replace_placeholders(value, param_values)
        elif isinstance(obj, list):
            for item in obj:
                self._replace_placeholders(item, param_values)

    def get_summary(self) -> str:
        """Get a human-readable summary of the skill."""
        summary = [
            f"Skill: {self.name}",
            f"Description: {self.description}",
            f"Version: {self.version}",
            f"Total actions: {len(self.actions)}",
        ]

        if self.parameters:
            summary.append(f"Parameters ({len(self.parameters)}):")
            for param in self.parameters:
                required = "required" if param.required else "optional"
                default = (
                    f", default={param.default_value}"
                    if param.default_value is not None
                    else ""
                )
                summary.append(
                    f"  - {param.name} ({param.param_type}, {required}{default}): {param.description}"
                )

        # Action type breakdown
        action_types: Dict[str, int] = {}
        for action in self.actions:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        if action_types:
            summary.append("Action breakdown:")
            for action_type, count in action_types.items():
                summary.append(f"  - {action_type}: {count}")

        return "\n".join(summary)

    def __str__(self) -> str:
        return self.get_summary()

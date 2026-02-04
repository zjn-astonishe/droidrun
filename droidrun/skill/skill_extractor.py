"""
Skill extractor for converting trajectories into reusable skills.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from droidrun.skill.skill import Skill, SkillParameter

logger = logging.getLogger("droidrun")


class SkillExtractor:
    """
    Extracts skills from trajectory data (macro.json files).
    
    Supports:
    - Converting raw action sequences into skills
    - Identifying parameterizable elements
    - Suggesting skill names and descriptions
    """

    def __init__(self):
        self.action_type_keywords = {
            "tap": ["click", "press", "select"],
            "input_text": ["type", "enter", "input"],
            "swipe": ["scroll", "swipe", "drag"],
            "wait": ["wait", "pause"],
            "open_app": ["launch", "open", "start"],
        }

    def extract_from_trajectory(
        self,
        trajectory_path: str,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
        auto_parameterize: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """
        Extract a skill from a trajectory directory.
        
        Args:
            trajectory_path: Path to the trajectory directory (containing macro.json)
            skill_name: Name for the skill (auto-generated if not provided)
            description: Description of the skill (auto-generated if not provided)
            auto_parameterize: Whether to automatically identify parameters
            tags: Tags to associate with the skill
            
        Returns:
            The extracted skill
        """
        trajectory_dir = Path(trajectory_path)
        macro_file = trajectory_dir / "macro.json"

        if not macro_file.exists():
            raise FileNotFoundError(f"macro.json not found in {trajectory_path}")

        # Load trajectory data
        with open(macro_file, "r", encoding="utf-8") as f:
            trajectory_data = json.load(f)

        actions = trajectory_data.get("actions", [])
        if not actions:
            raise ValueError("No actions found in trajectory")

        # Auto-generate name and description if not provided
        if skill_name is None:
            skill_name = self._generate_skill_name(actions, trajectory_dir.name)

        if description is None:
            description = self._generate_description(actions)

        # Identify parameters
        parameters = []
        if auto_parameterize:
            parameters = self._identify_parameters(actions)

        # Create metadata
        metadata = {
            "source_trajectory": str(trajectory_path),
            "total_actions": len(actions),
            "tags": tags or [],
        }

        # Add action statistics
        action_stats = self._compute_action_stats(actions)
        metadata["action_stats"] = action_stats

        skill = Skill(
            name=skill_name,
            description=description,
            actions=actions,
            parameters=parameters,
            metadata=metadata,
        )

        logger.info(
            f"✨ Extracted skill '{skill_name}' with {len(actions)} actions "
            f"and {len(parameters)} parameters"
        )

        return skill

    def extract_from_action_sequence(
        self,
        actions: List[Dict[str, Any]],
        skill_name: str,
        description: str,
        auto_parameterize: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """
        Extract a skill from a raw action sequence.
        
        Args:
            actions: List of action dictionaries
            skill_name: Name for the skill
            description: Description of the skill
            auto_parameterize: Whether to automatically identify parameters
            tags: Tags to associate with the skill
            
        Returns:
            The extracted skill
        """
        if not actions:
            raise ValueError("Action sequence cannot be empty")

        # Identify parameters
        parameters = []
        if auto_parameterize:
            parameters = self._identify_parameters(actions)

        # Create metadata
        metadata = {
            "total_actions": len(actions),
            "tags": tags or [],
            "action_stats": self._compute_action_stats(actions),
        }

        return Skill(
            name=skill_name,
            description=description,
            actions=actions,
            parameters=parameters,
            metadata=metadata,
        )

    def _generate_skill_name(self, actions: List[Dict[str, Any]], fallback: str) -> str:
        """Generate a skill name based on actions."""
        # Use the most common action type
        action_types = [action.get("action_type", "") for action in actions]
        if action_types:
            most_common = max(set(action_types), key=action_types.count)
            # Clean up the name
            name = most_common.replace("_", " ").title().replace(" ", "")
            return f"{name}Skill"

        # Fallback to trajectory directory name
        return f"Skill_{fallback}"

    def _generate_description(self, actions: List[Dict[str, Any]]) -> str:
        """Generate a description based on actions."""
        action_types = [action.get("action_type", "") for action in actions]
        unique_types = list(dict.fromkeys(action_types))  # Preserve order

        if len(unique_types) == 1:
            return f"Performs {len(actions)} {unique_types[0]} action(s)"
        elif len(unique_types) <= 3:
            type_str = ", ".join(unique_types)
            return f"Performs actions: {type_str}"
        else:
            return f"Performs {len(actions)} actions including {', '.join(unique_types[:3])}, and more"

    def _identify_parameters(
        self, actions: List[Dict[str, Any]]
    ) -> List[SkillParameter]:
        """
        Automatically identify potential parameters in actions.
        
        Looks for:
        - Text input values
        - Coordinates that might vary
        - Repeated values that could be parameterized
        """
        parameters = []
        seen_text_inputs: Set[str] = set()

        for i, action in enumerate(actions):
            action_type = action.get("action_type", "")

            # Identify text input parameters
            if action_type == "input_text":
                text = action.get("text", "")
                if text and text not in seen_text_inputs:
                    seen_text_inputs.add(text)
                    param = SkillParameter(
                        name=f"text_input_{len(parameters) + 1}",
                        description=f"Text to input (original: '{text}')",
                        param_type="text",
                        default_value=text,
                        required=True,
                    )
                    parameters.append(param)

                    # Replace with placeholder in action
                    action["text"] = f"{{{{text_input_{len(parameters)}}}}}"

            # Identify coordinate parameters for tap actions
            elif action_type == "tap":
                x = action.get("x")
                y = action.get("y")
                if x is not None and y is not None:
                    # Only parameterize if this looks like a specific UI element
                    # (you might want to add more sophisticated logic here)
                    param_x = SkillParameter(
                        name=f"tap_x_{len(parameters) + 1}",
                        description=f"X coordinate for tap action {i + 1}",
                        param_type="number",
                        default_value=x,
                        required=False,
                    )
                    param_y = SkillParameter(
                        name=f"tap_y_{len(parameters) + 2}",
                        description=f"Y coordinate for tap action {i + 1}",
                        param_type="number",
                        default_value=y,
                        required=False,
                    )

                    # Note: Commenting out automatic coordinate parameterization
                    # as it might be too aggressive. Uncomment if needed.
                    # parameters.extend([param_x, param_y])
                    # action["x"] = f"{{{{tap_x_{len(parameters) - 1}}}}}"
                    # action["y"] = f"{{{{tap_y_{len(parameters)}}}}}"

        return parameters

    def _compute_action_stats(self, actions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute statistics about action types."""
        stats: Dict[str, int] = {}
        for action in actions:
            action_type = action.get("action_type", "unknown")
            stats[action_type] = stats.get(action_type, 0) + 1
        return stats

    def suggest_parameterization(
        self, skill: Skill
    ) -> List[Dict[str, Any]]:
        """
        Suggest additional parameterization opportunities for a skill.
        
        Args:
            skill: The skill to analyze
            
        Returns:
            List of suggestions with details about potential parameters
        """
        suggestions = []

        # Look for repeated values that could be parameterized
        value_counts: Dict[str, List[int]] = {}

        for i, action in enumerate(skill.actions):
            for key, value in action.items():
                if isinstance(value, (str, int, float)) and key not in [
                    "action_type",
                    "timestamp",
                ]:
                    value_str = str(value)
                    if value_str not in value_counts:
                        value_counts[value_str] = []
                    value_counts[value_str].append(i)

        # Suggest parameters for values that appear multiple times
        for value, occurrences in value_counts.items():
            if len(occurrences) > 1:
                suggestions.append(
                    {
                        "value": value,
                        "occurrences": len(occurrences),
                        "action_indices": occurrences,
                        "suggestion": f"Consider parameterizing '{value}' which appears {len(occurrences)} times",
                    }
                )

        return suggestions

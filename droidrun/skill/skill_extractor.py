"""
Enhanced skill extractor for converting trajectories into reusable skills.

Improvements:
- Automatic complexity detection
- Enhanced parameter identification
- Pre/post-condition generation
- Reliability scoring based on trajectory analysis
- Intelligent skill naming and description
- Optional LLM-powered description generation
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from droidrun.skill.skill import (
    Skill,
    SkillParameter,
    ParameterType,
    SkillComplexity,
    Condition,
)

logger = logging.getLogger("droidrun")


class SkillExtractor:
    """
    Enhanced extractor for converting trajectories into skills.
    
    Features:
    - Automatic complexity assessment
    - Smart parameter identification
    - Condition generation
    - Metadata enrichment
    - Pattern recognition
    """

    def __init__(
        self,
    ):
        """
        Initialize the skill extractor.
        """
        self.action_type_keywords = {
            "tap": ["click", "press", "select", "touch"],
            "input_text": ["type", "enter", "input", "write"],
            "swipe": ["scroll", "swipe", "drag", "slide"],
            "wait": ["wait", "pause", "delay", "sleep"],
            "open_app": ["launch", "open", "start", "run"],
        }

    def extract_from_trajectory(
        self,
        trajectory_path: str,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
        auto_parameterize: bool = True,
        tags: Optional[List[str]] = None,
        analyze_complexity: bool = True,
    ) -> Skill:
        """
        Extract an enhanced skill from a trajectory directory.
        
        Args:
            trajectory_path: Path to the trajectory directory (containing macro.json)
            skill_name: Name for the skill (auto-generated if not provided)
            description: Description of the skill (auto-generated if not provided)
            auto_parameterize: Whether to automatically identify parameters
            tags: Tags to associate with the skill
            analyze_complexity: Whether to analyze and set complexity level
            
        Returns:
            The extracted skill with enhanced metadata
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

        # Analyze complexity
        complexity = SkillComplexity.MODERATE
        if analyze_complexity:
            complexity = self._assess_complexity(actions)

        # Estimate duration from trajectory timestamps
        estimated_duration = self._estimate_duration(actions)

        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(actions)

        # Create metadata
        metadata = {
            "source_trajectory": str(trajectory_path),
            "total_actions": len(actions),
            "created_at": trajectory_data.get("timestamp"),
            "tags": tags or [],
        }

        # Add action statistics
        action_stats = self._compute_action_stats(actions)
        metadata["action_stats"] = action_stats

        # Add pattern analysis
        metadata["patterns"] = self._analyze_patterns(actions)

        skill = Skill(
            name=skill_name,
            description=description,
            actions=actions,
            parameters=parameters,
            metadata=metadata,
            complexity=complexity,
            estimated_duration=estimated_duration,
            reliability_score=reliability_score,
            tags=set(tags or []),
        )

        logger.info(
            f"✨ Extracted skill '{skill_name}' with {len(actions)} actions, "
            f"{len(parameters)} parameters, complexity: {complexity.value}"
        )

        return skill

    def extract_from_action_sequence(
        self,
        actions: List[Dict[str, Any]],
        skill_name: str,
        description: str,
        auto_parameterize: bool = True,
        tags: Optional[List[str]] = None,
        analyze_complexity: bool = True,
    ) -> Skill:
        """
        Extract an enhanced skill from a raw action sequence.
        
        Args:
            actions: List of action dictionaries
            skill_name: Name for the skill
            description: Description of the skill
            auto_parameterize: Whether to automatically identify parameters
            tags: Tags to associate with the skill
            analyze_complexity: Whether to analyze complexity
            
        Returns:
            The extracted skill
        """
        if not actions:
            raise ValueError("Action sequence cannot be empty")

        # Identify parameters
        parameters = []
        if auto_parameterize:
            parameters = self._identify_parameters(actions)

        # Analyze complexity
        complexity = SkillComplexity.MODERATE
        if analyze_complexity:
            complexity = self._assess_complexity(actions)

        # Estimate duration
        estimated_duration = self._estimate_duration(actions)

        # Calculate reliability
        reliability_score = self._calculate_reliability_score(actions)

        # Create metadata
        metadata = {
            "total_actions": len(actions),
            "tags": tags or [],
            "action_stats": self._compute_action_stats(actions),
            "patterns": self._analyze_patterns(actions),
        }

        return Skill(
            name=skill_name,
            description=description,
            actions=actions,
            parameters=parameters,
            metadata=metadata,
            complexity=complexity,
            estimated_duration=estimated_duration,
            reliability_score=reliability_score,
            tags=set(tags or []),
        )

    def _generate_skill_name(self, actions: List[Dict[str, Any]], fallback: str) -> str:
        """Generate a descriptive skill name based on actions."""
        action_types = [action.get("action_type", "") for action in actions]
        
        # Analyze action patterns
        if action_types:
            # Check for specific patterns
            if "open_app" in action_types:
                return "open_app_workflow"
            elif action_types.count("input_text") > 0 and "tap" in action_types:
                return "form_input_workflow"
            elif "swipe" in action_types:
                return "navigation_workflow"
            else:
                # Use most common action
                most_common = Counter(action_types).most_common(1)[0][0]
                name = most_common.replace("_", " ").title().replace(" ", "")
                return f"{name}Workflow"

        # Fallback to trajectory directory name
        clean_fallback = fallback.replace("-", "_").replace(" ", "_")
        return f"skill_{clean_fallback}"

    def _assess_complexity(self, actions: List[Dict[str, Any]]) -> SkillComplexity:
        """
        Assess the complexity level of a skill based on its actions.
        
        Factors:
        - Number of actions
        - Variety of action types
        - Presence of conditional logic (inferred)
        - Coordination requirements
        """
        action_count = len(actions)
        action_types = [action.get("action_type", "") for action in actions]
        unique_types = set(action_types)

        # Simple: 1-3 actions, single type
        if action_count <= 3 and len(unique_types) == 1:
            return SkillComplexity.SIMPLE

        # Complex: Many actions or high variety
        if action_count > 10 or len(unique_types) > 4:
            return SkillComplexity.COMPLEX

        # Moderate: Everything else
        return SkillComplexity.MODERATE

    def _estimate_duration(self, actions: List[Dict[str, Any]]) -> Optional[float]:
        """
        Estimate skill duration based on action timestamps.
        
        Returns estimated duration in seconds, or None if not available.
        """
        try:
            timestamps = [
                action.get("timestamp")
                for action in actions
                if action.get("timestamp") is not None
            ]
            
            if len(timestamps) >= 2:
                # Calculate duration from first to last timestamp
                duration = timestamps[-1] - timestamps[0]
                return max(duration, 0.1)  # Minimum 0.1 seconds
        except Exception as e:
            logger.debug(f"Could not estimate duration: {e}")
        
        # Fallback: estimate based on action count and type
        base_duration = 0.5  # seconds per action
        wait_actions = sum(1 for a in actions if a.get("action_type") == "wait")
        other_actions = len(actions) - wait_actions
        
        estimated = (other_actions * base_duration) + (wait_actions * 2.0)
        return estimated

    def _calculate_reliability_score(self, actions: List[Dict[str, Any]]) -> float:
        """
        Calculate a reliability score based on action characteristics.
        
        Factors:
        - Action diversity (more diverse = potentially less reliable)
        - Presence of wait actions (good for stability)
        - Action count (very long sequences may be less reliable)
        
        Returns score from 0.0 to 1.0
        """
        score = 1.0
        action_count = len(actions)
        action_types = [action.get("action_type", "") for action in actions]
        
        # Penalize very long sequences (reduced reliability)
        if action_count > 20:
            score *= 0.9
        if action_count > 50:
            score *= 0.8
        
        # Reward presence of wait actions (stability)
        wait_count = action_types.count("wait")
        if wait_count > 0:
            score *= 1.05
        
        # Penalize very high action diversity
        unique_types = len(set(action_types))
        if unique_types > 5:
            score *= 0.95
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    def _identify_parameters(
        self, actions: List[Dict[str, Any]]
    ) -> List[SkillParameter]:
        """
        Automatically identify potential parameters in actions with enhanced logic.
        借鉴 code_generator 的思路，改进参数识别和动作标准化。
        
        Identifies:
        - Text input values
        - App launch parameters  
        - Repeated values
        - Configurable delays
        """
        parameters = []
        seen_text_inputs: Set[str] = set()
        seen_app_names: Set[str] = set()

        for i, action in enumerate(actions):
            action_type = action.get("action_type", "")
            
            # 处理文本输入动作
            if action_type in ["type", "input_text"]:
                text = action.get("text", "")
                
                # 参数化文本输入
                if text and text not in seen_text_inputs:
                    seen_text_inputs.add(text)
                    
                    param_name = self._generate_param_name("input_text", len(parameters))
                    param = SkillParameter(
                        name=param_name,
                        description=f"Text to input (default: '{text}')",
                        param_type=ParameterType.STRING,
                        default_value=text,
                        required=True,
                    )
                    parameters.append(param)
                    action["text"] = f"{{{{{param_name}}}}}"
            
            # 处理应用启动动作
            elif action_type in ["launch", "open_app"]:
                app = action.get("app", "") or action.get("package", "")
                if app and app not in seen_app_names:
                    seen_app_names.add(app)
                    
                    # 标准化为 "app" 字段
                    if "app" not in action:
                        action["app"] = app
                    
                    # 可选：参数化 app 名称（默认禁用）
                    # param_name = f"app_name_{len(parameters) + 1}"
                    # param = SkillParameter(
                    #     name=param_name,
                    #     description=f"Application to launch (default: '{app}')",
                    #     param_type=ParameterType.STRING,
                    #     default_value=app,
                    #     required=False,
                    # )
                    # parameters.append(param)
                    # action["app"] = f"{{{{{param_name}}}}}"

            # 可选：参数化 wait 时间（默认禁用）
            elif action_type == "wait":
                duration = action.get("duration")
                if duration is not None:
                    # 可以在这里参数化 wait 时间
                    pass

        return parameters

    def _generate_param_name(self, base: str, index: int) -> str:
        """Generate a descriptive parameter name."""
        if index == 0:
            return base
        return f"{base}_{index + 1}"

    def _compute_action_stats(self, actions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute detailed statistics about action types."""
        stats: Dict[str, int] = {}
        for action in actions:
            action_type = action.get("action_type", "unknown")
            stats[action_type] = stats.get(action_type, 0) + 1
        return stats

    def _analyze_patterns(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze action patterns to detect common workflows.
        
        Returns:
            Dictionary with pattern analysis results
        """
        action_types = [action.get("action_type", "") for action in actions]
        
        patterns = {
            "sequential_taps": self._count_sequential_pattern(action_types, "tap"),
            "input_then_submit": self._has_input_submit_pattern(action_types),
            "scroll_and_tap": self._has_scroll_tap_pattern(action_types),
            "repeated_actions": self._find_repeated_sequences(action_types),
        }
        
        return patterns

    def _count_sequential_pattern(self, action_types: List[str], pattern: str) -> int:
        """Count maximum consecutive occurrences of a pattern."""
        max_count = 0
        current_count = 0
        
        for action_type in action_types:
            if action_type == pattern:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count

    def _has_input_submit_pattern(self, action_types: List[str]) -> bool:
        """Check if there's an input followed by tap pattern (form submission)."""
        for i in range(len(action_types) - 1):
            if action_types[i] == "input_text" and action_types[i + 1] == "tap":
                return True
        return False

    def _has_scroll_tap_pattern(self, action_types: List[str]) -> bool:
        """Check if there's a scroll followed by tap pattern (navigation)."""
        for i in range(len(action_types) - 1):
            if action_types[i] in ["swipe", "scroll"] and action_types[i + 1] == "tap":
                return True
        return False

    def _find_repeated_sequences(self, action_types: List[str]) -> List[Dict[str, Any]]:
        """Find repeated action sequences."""
        repeated = []
        
        # Check for sequences of length 2-4
        for seq_len in range(2, 5):
            for i in range(len(action_types) - seq_len * 2 + 1):
                sequence = action_types[i : i + seq_len]
                # Check if this sequence repeats immediately after
                next_sequence = action_types[i + seq_len : i + seq_len * 2]
                if sequence == next_sequence:
                    repeated.append({
                        "sequence": sequence,
                        "length": seq_len,
                        "position": i,
                    })
        
        return repeated

    def suggest_parameterization(self, skill: Skill) -> List[Dict[str, Any]]:
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
                    # Skip if already parameterized
                    if isinstance(value, str) and "{{" in value:
                        continue
                    
                    value_str = str(value)
                    if value_str not in value_counts:
                        value_counts[value_str] = []
                    value_counts[value_str].append(i)

        # Suggest parameters for values that appear multiple times
        for value, occurrences in value_counts.items():
            if len(occurrences) > 1:
                suggestions.append({
                    "value": value,
                    "occurrences": len(occurrences),
                    "action_indices": occurrences,
                    "suggestion": f"Consider parameterizing '{value}' which appears {len(occurrences)} times",
                    "parameter_type": self._infer_parameter_type(value),
                })

        # Suggest parameterizing coordinates if they vary slightly
        coord_suggestions = self._suggest_coordinate_parameterization(skill.actions)
        suggestions.extend(coord_suggestions)

        return suggestions

    def _infer_parameter_type(self, value: Any) -> str:
        """Infer the parameter type from a value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Check if it looks like a coordinate pair
            if value.startswith("(") and "," in value:
                return "coordinate"
            return "string"
        else:
            return "unknown"

    def _suggest_coordinate_parameterization(
        self, actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest parameterization for coordinates that vary."""
        suggestions = []
        coord_groups: Dict[str, List[tuple]] = {}

        # Group coordinates by action type
        for i, action in enumerate(actions):
            action_type = action.get("action_type")
            if action_type in ["tap", "swipe"]:
                x = action.get("x")
                y = action.get("y")
                if x is not None and y is not None:
                    key = f"{action_type}_coords"
                    if key not in coord_groups:
                        coord_groups[key] = []
                    coord_groups[key].append((x, y, i))

        # Check for variation in coordinates
        for action_type, coords in coord_groups.items():
            if len(coords) > 1:
                x_values = [c[0] for c in coords]
                y_values = [c[1] for c in coords]
                
                # If coordinates vary significantly, suggest parameterization
                x_range = max(x_values) - min(x_values)
                y_range = max(y_values) - min(y_values)
                
                if x_range > 50 or y_range > 50:
                    suggestions.append({
                        "value": f"coordinates for {action_type}",
                        "occurrences": len(coords),
                        "action_indices": [c[2] for c in coords],
                        "suggestion": f"Coordinates for {action_type} vary significantly (x range: {x_range}, y range: {y_range}), consider parameterizing",
                        "parameter_type": "coordinate",
                    })

        return suggestions

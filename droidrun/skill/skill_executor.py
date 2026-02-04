"""
Skill executor for running skills with parameter substitution.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from droidrun.skill.skill import Skill

logger = logging.getLogger("droidrun")


class SkillExecutor:
    """
    Executes skills with parameter substitution and validation.
    
    Supports:
    - Parameter validation and substitution
    - Action execution with callbacks
    - Execution status tracking
    """

    def __init__(self, action_executor: Optional[Any] = None):
        """
        Initialize the skill executor.
        
        Args:
            action_executor: Optional executor that handles individual actions
                            (e.g., MacroPlayer from droidrun.macro.replay)
        """
        self.action_executor = action_executor

    def execute(
        self,
        skill: Skill,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a skill with the given parameters.
        
        Args:
            skill: The skill to execute
            parameters: Dictionary of parameter values to use
            dry_run: If True, only validate and prepare actions without executing
            
        Returns:
            Execution result with status and details
        """
        parameters = parameters or {}

        # Validate parameters
        validation_result = skill.validate_parameters(parameters)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": "Parameter validation failed",
                "details": validation_result,
            }

        # Substitute parameters in actions
        try:
            resolved_actions = self._substitute_parameters(
                skill.actions, parameters, skill.parameters
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Parameter substitution failed: {str(e)}",
            }

        logger.info(
            f"🚀 Executing skill '{skill.name}' with {len(resolved_actions)} actions"
        )

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "actions": resolved_actions,
                "message": f"Dry run: {len(resolved_actions)} actions ready to execute",
            }

        # Execute actions
        if self.action_executor is None:
            return {
                "success": True,
                "actions": resolved_actions,
                "message": "No action executor configured. Actions prepared but not executed.",
            }

        execution_results = []
        for i, action in enumerate(resolved_actions):
            try:
                result = self._execute_action(action, i)
                execution_results.append(result)

                if not result.get("success", True):
                    logger.warning(
                        f"Action {i + 1}/{len(resolved_actions)} failed: {result.get('error')}"
                    )
            except Exception as e:
                error_msg = f"Error executing action {i + 1}: {str(e)}"
                logger.error(error_msg)
                execution_results.append({"success": False, "error": error_msg})

        # Compute overall result
        total_actions = len(resolved_actions)
        successful_actions = sum(
            1 for r in execution_results if r.get("success", True)
        )

        return {
            "success": successful_actions == total_actions,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "failed_actions": total_actions - successful_actions,
            "execution_results": execution_results,
        }

    def _substitute_parameters(
        self,
        actions: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        skill_parameters: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Substitute parameters in action list.
        
        Args:
            actions: List of action dictionaries
            parameters: Dictionary of parameter values
            skill_parameters: List of skill parameter definitions
            
        Returns:
            List of actions with substituted values
        """
        # Get default values from skill parameters
        defaults = {}
        for param in skill_parameters:
            if hasattr(param, "name") and hasattr(param, "default_value"):
                defaults[param.name] = param.default_value

        # Merge defaults with provided parameters
        all_params = {**defaults, **parameters}

        # Deep copy actions to avoid modifying original
        import copy

        resolved_actions = copy.deepcopy(actions)

        # Substitute parameters in each action
        for action in resolved_actions:
            for key, value in action.items():
                if isinstance(value, str):
                    # Replace {{param_name}} placeholders
                    action[key] = self._substitute_string(value, all_params)

        return resolved_actions

    def _substitute_string(self, template: str, parameters: Dict[str, Any]) -> Any:
        """
        Substitute parameters in a string template.
        
        Supports:
        - Simple substitution: {{param_name}}
        - Type conversion based on parameter type
        
        Args:
            template: String with placeholders
            parameters: Dictionary of parameter values
            
        Returns:
            Substituted value (may be converted to int/float if applicable)
        """
        # Pattern to match {{param_name}}
        pattern = r"\{\{(\w+)\}\}"

        def replace_func(match):
            param_name = match.group(1)
            if param_name in parameters:
                return str(parameters[param_name])
            return match.group(0)  # Keep original if parameter not found

        result = re.sub(pattern, replace_func, template)

        # If the entire string was a single parameter, try to convert to original type
        if template.startswith("{{") and template.endswith("}}"):
            param_name = template[2:-2]
            if param_name in parameters:
                return parameters[param_name]

        # Try to convert to number if applicable
        try:
            if "." in result:
                return float(result)
            return int(result)
        except ValueError:
            return result

    def _execute_action(self, action: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action dictionary
            index: Action index
            
        Returns:
            Execution result
        """
        if self.action_executor is None:
            return {"success": True, "message": "No executor configured"}

        try:
            # If action_executor has a method to execute single actions
            if hasattr(self.action_executor, "execute_action"):
                return self.action_executor.execute_action(action)
            else:
                # Generic execution
                logger.info(f"Executing action {index + 1}: {action.get('action_type')}")
                return {"success": True, "action": action}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_execution_plan(
        self, skill: Skill, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get an execution plan without actually executing the skill.
        
        Args:
            skill: The skill to plan
            parameters: Dictionary of parameter values
            
        Returns:
            Execution plan with resolved actions
        """
        return self.execute(skill, parameters, dry_run=True)

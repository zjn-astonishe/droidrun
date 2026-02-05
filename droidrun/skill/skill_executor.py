"""
Enhanced skill executor for running skills with parameter substitution.

Improvements:
- Context-aware execution with pre/post-condition checking
- Retry and rollback mechanisms
- Dependency resolution and chaining
- Enhanced error handling and recovery
- Execution telemetry and monitoring
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from droidrun.skill.skill import (
    Skill,
    ExecutionContext,
    SkillDependency,
    ParameterType,
)

logger = logging.getLogger("droidrun")


class ExecutionResult:
    """Structured result from skill execution."""
    
    def __init__(
        self,
        success: bool,
        skill_name: str,
        total_actions: int = 0,
        successful_actions: int = 0,
        failed_actions: int = 0,
        execution_time: float = 0.0,
        error: Optional[str] = None,
        warnings: List[str] = None,
        context: Optional[ExecutionContext] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        execution_results: Optional[List[Dict[str, Any]]] = None,
    ):
        self.success = success
        self.skill_name = skill_name
        self.total_actions = total_actions
        self.successful_actions = successful_actions
        self.failed_actions = failed_actions
        self.execution_time = execution_time
        self.error = error
        self.warnings = warnings or []
        self.context = context
        self.actions = actions
        self.execution_results = execution_results or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "execution_time": self.execution_time,
            "error": self.error,
            "warnings": self.warnings,
            "actions": self.actions,
            "execution_results": self.execution_results,
        }


class SkillExecutor:
    """
    Enhanced executor for skills with context awareness and error handling.
    
    Features:
    - Pre/post-condition validation
    - Automatic retry with exponential backoff
    - Rollback on failure
    - Dependency resolution
    - Execution context tracking
    - Performance telemetry
    """

    def __init__(
        self,
        action_executor: Optional[Any] = None,
        skill_library: Optional[Any] = None,
        enable_telemetry: bool = True,
    ):
        """
        Initialize the skill executor.
        
        Args:
            action_executor: Optional executor that handles individual actions
            skill_library: Optional skill library for dependency resolution
            enable_telemetry: Whether to collect execution telemetry
        """
        self.action_executor = action_executor
        self.skill_library = skill_library
        self.enable_telemetry = enable_telemetry
        self.execution_history: List[ExecutionResult] = []

    def execute(
        self,
        skill: Skill,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> ExecutionResult:
        """
        Execute a skill with enhanced error handling and context awareness.
        
        Args:
            skill: The skill to execute
            parameters: Dictionary of parameter values to use
            context: Execution context (created if not provided)
            dry_run: If True, only validate and prepare actions without executing
            force: If True, skip precondition checks
            
        Returns:
            ExecutionResult with detailed status and telemetry
        """
        start_time = time.time()
        parameters = parameters or {}
        context = context or ExecutionContext()
        
        # Step 1: Validate parameters
        validation_result = skill.validate_parameters(parameters)
        if not validation_result["valid"]:
            return ExecutionResult(
                success=False,
                skill_name=skill.name,
                error="Parameter validation failed",
                warnings=validation_result.get("errors", []),
                execution_time=time.time() - start_time,
            )
        
        if validation_result.get("warnings"):
            logger.warning(f"Parameter warnings: {validation_result['warnings']}")
        
        # Step 2: Resolve dependencies
        if skill.dependencies:
            dep_result = self._resolve_dependencies(skill, context)
            if not dep_result["success"]:
                return ExecutionResult(
                    success=False,
                    skill_name=skill.name,
                    error=f"Dependency resolution failed: {dep_result['error']}",
                    execution_time=time.time() - start_time,
                )
        
        # Step 3: Check preconditions
        if not force and skill.preconditions:
            precond_result = skill.check_preconditions(context)
            if not precond_result["met"]:
                return ExecutionResult(
                    success=False,
                    skill_name=skill.name,
                    error="Preconditions not met",
                    warnings=precond_result["failed"],
                    execution_time=time.time() - start_time,
                )
            if precond_result.get("warnings"):
                logger.warning(f"Precondition warnings: {precond_result['warnings']}")
        
        # Step 4: Substitute parameters in actions
        try:
            resolved_actions = skill.apply_parameters(parameters)
        except Exception as e:
            return ExecutionResult(
                success=False,
                skill_name=skill.name,
                error=f"Parameter substitution failed: {str(e)}",
                execution_time=time.time() - start_time,
            )
        
        logger.info(
            f"🚀 Executing skill '{skill.name}' with {len(resolved_actions)} actions"
        )
        
        if dry_run:
            return ExecutionResult(
                success=True,
                skill_name=skill.name,
                total_actions=len(resolved_actions),
                actions=resolved_actions,
                execution_time=time.time() - start_time,
                context=context,
            )
        
        # Step 5: Execute with retry logic
        execution_result = self._execute_with_retry(
            skill, resolved_actions, context
        )
        
        # Step 6: Check postconditions
        if execution_result.success and skill.postconditions:
            postcond_result = skill.check_postconditions(context)
            if not postcond_result["met"]:
                logger.error(f"Postconditions failed: {postcond_result['failed']}")
                execution_result.success = False
                execution_result.error = "Postconditions not met after execution"
                execution_result.warnings.extend(postcond_result["failed"])
                
                # Attempt rollback if postconditions fail
                if skill.rollback_actions:
                    logger.info("Attempting rollback...")
                    self._execute_rollback(skill.rollback_actions, context)
        
        execution_result.execution_time = time.time() - start_time
        
        # Step 7: Record telemetry
        if self.enable_telemetry:
            self.execution_history.append(execution_result)
        
        return execution_result

    def _resolve_dependencies(
        self, skill: Skill, context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Resolve skill dependencies.
        
        Args:
            skill: The skill with dependencies
            context: Execution context
            
        Returns:
            Dictionary with success status and error message if failed
        """
        if not self.skill_library:
            logger.warning("No skill library available for dependency resolution")
            return {"success": True}  # Proceed without dependency checking
        
        for dependency in skill.dependencies:
            dep_skill = self.skill_library.get_skill(dependency.skill_name)
            
            if dep_skill is None:
                if dependency.optional:
                    logger.warning(
                        f"Optional dependency '{dependency.skill_name}' not found"
                    )
                    # Try fallback if available
                    if dependency.fallback_skill:
                        dep_skill = self.skill_library.get_skill(
                            dependency.fallback_skill
                        )
                        if dep_skill:
                            logger.info(
                                f"Using fallback skill '{dependency.fallback_skill}'"
                            )
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Required dependency '{dependency.skill_name}' not found",
                    }
            
            # Check version constraint if specified
            if dependency.version_constraint:
                if not self._check_version_constraint(
                    dep_skill.version, dependency.version_constraint
                ):
                    return {
                        "success": False,
                        "error": f"Dependency '{dependency.skill_name}' version {dep_skill.version} does not meet constraint {dependency.version_constraint}",
                    }
        
        return {"success": True}

    def _check_version_constraint(
        self, version: str, constraint: str
    ) -> bool:
        """
        Check if a version meets a constraint.
        
        Simple implementation supporting: >=, <=, >, <, ==
        For complex constraints like ">=1.0,<2.0", split and check all
        """
        try:
            constraints = constraint.split(",")
            for c in constraints:
                c = c.strip()
                if c.startswith(">="):
                    if not (version >= c[2:]):
                        return False
                elif c.startswith("<="):
                    if not (version <= c[2:]):
                        return False
                elif c.startswith(">"):
                    if not (version > c[1:]):
                        return False
                elif c.startswith("<"):
                    if not (version < c[1:]):
                        return False
                elif c.startswith("=="):
                    if not (version == c[2:]):
                        return False
            return True
        except Exception as e:
            logger.warning(f"Error checking version constraint: {e}")
            return False

    def _execute_with_retry(
        self,
        skill: Skill,
        actions: List[Dict[str, Any]],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute actions with retry logic.
        
        Args:
            skill: The skill being executed
            actions: List of resolved actions
            context: Execution context
            
        Returns:
            ExecutionResult
        """
        max_attempts = skill.max_retries + 1
        attempt = 0
        last_error = None
        
        while attempt < max_attempts:
            if attempt > 0:
                delay = skill.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(
                    f"Retry attempt {attempt}/{skill.max_retries}, "
                    f"waiting {delay}s..."
                )
                time.sleep(delay)
            
            try:
                result = self._execute_actions(actions, context)
                
                if result.success:
                    if attempt > 0:
                        logger.info(
                            f"✅ Skill succeeded on attempt {attempt + 1}"
                        )
                    return result
                
                last_error = result.error
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Execution error on attempt {attempt + 1}: {e}")
            
            attempt += 1
        
        # All attempts failed
        logger.error(
            f"❌ Skill '{skill.name}' failed after {max_attempts} attempts"
        )
        
        # Attempt rollback if configured
        if skill.rollback_actions:
            logger.info("Attempting rollback...")
            self._execute_rollback(skill.rollback_actions, context)
        
        return ExecutionResult(
            success=False,
            skill_name=skill.name,
            total_actions=len(actions),
            error=f"Failed after {max_attempts} attempts: {last_error}",
        )

    def _execute_actions(
        self,
        actions: List[Dict[str, Any]],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a list of actions.
        
        Args:
            actions: List of action dictionaries
            context: Execution context
            
        Returns:
            ExecutionResult
        """
        if self.action_executor is None:
            return ExecutionResult(
                success=True,
                skill_name="unknown",
                total_actions=len(actions),
                successful_actions=len(actions),
                actions=actions,
                warnings=["No action executor configured. Actions prepared but not executed."],
            )
        
        execution_results = []
        successful_count = 0
        
        for i, action in enumerate(actions):
            try:
                result = self._execute_action(action, i, context)
                execution_results.append(result)
                
                if result.get("success", True):
                    successful_count += 1
                    # Track action in context
                    context.previous_actions.append(action)
                else:
                    logger.warning(
                        f"Action {i + 1}/{len(actions)} failed: {result.get('error')}"
                    )
                    # Stop on first failure (can be configurable)
                    break
                    
            except Exception as e:
                error_msg = f"Error executing action {i + 1}: {str(e)}"
                logger.error(error_msg)
                execution_results.append({"success": False, "error": error_msg})
                break
        
        total_actions = len(actions)
        success = successful_count == total_actions
        
        return ExecutionResult(
            success=success,
            skill_name="unknown",
            total_actions=total_actions,
            successful_actions=successful_count,
            failed_actions=total_actions - successful_count,
            execution_results=execution_results,
            context=context,
        )

    def _execute_action(
        self,
        action: Dict[str, Any],
        index: int,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute a single action with context tracking.
        
        Args:
            action: Action dictionary
            index: Action index
            context: Execution context
            
        Returns:
            Execution result dictionary
        """
        if self.action_executor is None:
            return {"success": True, "message": "No executor configured"}
        
        try:
            # Update context with current action
            context.metadata["current_action_index"] = index
            context.metadata["current_action"] = action
            
            # If action_executor has a method to execute single actions
            if hasattr(self.action_executor, "execute_action"):
                result = self.action_executor.execute_action(action)
            else:
                # Generic execution
                logger.info(
                    f"Executing action {index + 1}: {action.get('action_type')}"
                )
                result = {"success": True, "action": action}
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_rollback(
        self,
        rollback_actions: List[Dict[str, Any]],
        context: ExecutionContext,
    ) -> None:
        """
        Execute rollback actions to undo changes.
        
        Args:
            rollback_actions: List of actions to execute for rollback
            context: Execution context
        """
        logger.info(f"Executing {len(rollback_actions)} rollback actions...")
        
        try:
            for i, action in enumerate(rollback_actions):
                self._execute_action(action, i, context)
            logger.info("✅ Rollback completed")
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")

    def execute_chain(
        self,
        skills: List[Skill],
        parameters_list: Optional[List[Dict[str, Any]]] = None,
        context: Optional[ExecutionContext] = None,
        stop_on_failure: bool = True,
    ) -> List[ExecutionResult]:
        """
        Execute a chain of skills sequentially with shared context.
        
        Args:
            skills: List of skills to execute
            parameters_list: List of parameter dictionaries for each skill
            context: Shared execution context
            stop_on_failure: Whether to stop execution on first failure
            
        Returns:
            List of ExecutionResults
        """
        context = context or ExecutionContext()
        parameters_list = parameters_list or [{}] * len(skills)
        
        if len(parameters_list) != len(skills):
            raise ValueError(
                f"Parameters list length ({len(parameters_list)}) must match "
                f"skills list length ({len(skills)})"
            )
        
        results = []
        
        for i, (skill, params) in enumerate(zip(skills, parameters_list)):
            logger.info(
                f"Executing skill {i + 1}/{len(skills)}: {skill.name}"
            )
            
            result = self.execute(skill, params, context)
            results.append(result)
            
            if not result.success and stop_on_failure:
                logger.error(
                    f"Chain execution stopped at skill {i + 1} due to failure"
                )
                break
            
            # Update context with skill result
            context.metadata[f"skill_{i}_result"] = result.to_dict()
        
        return results

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
        result = self.execute(skill, parameters, dry_run=True)
        return result.to_dict()

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get execution telemetry and statistics.
        
        Returns:
            Dictionary with telemetry data
        """
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        failed = total - successful
        
        total_time = sum(r.execution_time for r in self.execution_history)
        avg_time = total_time / total if total > 0 else 0
        
        skill_stats = {}
        for result in self.execution_history:
            name = result.skill_name
            if name not in skill_stats:
                skill_stats[name] = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_time": 0,
                }
            
            skill_stats[name]["executions"] += 1
            if result.success:
                skill_stats[name]["successes"] += 1
            else:
                skill_stats[name]["failures"] += 1
            skill_stats[name]["avg_time"] = (
                skill_stats[name].get("avg_time", 0) * (skill_stats[name]["executions"] - 1)
                + result.execution_time
            ) / skill_stats[name]["executions"]
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "skill_statistics": skill_stats,
        }

    def clear_telemetry(self) -> None:
        """Clear execution telemetry history."""
        self.execution_history.clear()
        logger.info("Execution telemetry cleared")

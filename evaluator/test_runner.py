"""Test runner for Android World tasks using DroidRun."""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

from droidrun import DroidAgent, ResultEvent
from droidrun.config_manager import ConfigLoader, DroidrunConfig
from droidrun.cli.main import _setup_portal

from .task_loader import AndroidWorldTaskLoader
from .evaluator import AndroidWorldEvaluator
from .result_reporter import AndroidWorldResultReporter

class AndroidWorldTestRunner:
    """Runs Android World tests using DroidRun agent."""
    
    def __init__(self, 
                 config: Optional[DroidrunConfig] = None,
                 config_path: Optional[str] = None,
                 device_id: str = "emulator-5554",
                 timeout_per_task: int = 3600,
                 verbose: bool = True):
        """
        Initialize the test runner.
        
        Args:
            config: Pre-loaded DroidrunConfig instance (if provided, config_path is ignored)
            config_path: Path to droidrun config file
            device_id: Android device ID/serial
            timeout_per_task: Default timeout per task in seconds
            verbose: Whether to print verbose output
        """
        self.device_id = device_id
        self.timeout_per_task = timeout_per_task
        self.verbose = verbose
        
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = ConfigLoader.load(config_path)
        
        # Override device serial if provided
        if device_id:
            self.config.device.serial = device_id
        
        # Initialize other components
        self.task_loader = AndroidWorldTaskLoader()
        self.evaluator = AndroidWorldEvaluator(device_id=device_id)
        self.result_reporter = AndroidWorldResultReporter()
        
        if self.verbose:
            print(f"Android World Test Runner initialized for device {device_id}")
            print(f"Config: max_steps={self.config.agent.max_steps}, "
                  f"vision={self.config.agent.executor.vision}")
    
    async def run_single_task(self, 
                       task_name: str, 
                       family: Optional[str] = None,
                       timeout: int = 3600) -> Dict[str, Any]:
        """
        Run a single Android World task.
        
        Args:
            task_name: Name of the task to run
            family: Task family (optional)
            timeout: Timeout in seconds
            
        Returns:
            Task execution and evaluation result
        """
        print(f"\n🎯 Running task: {task_name}")
        
        start_time = time.time()
        
        try:
            # Load the task
            task_goal, task_instance, metadata = self.task_loader.get_task(task_name, family)
            print(f"📋 Task goal: {task_goal}")
            print(f"📊 Task metadata: {metadata}")
            
            # Initialize the task in AndroidWorld environment BEFORE agent execution
            print("🔧 Initializing AndroidWorld task...")
            if self.evaluator.env and task_instance:
                try:
                    task_instance.initialize_task(self.evaluator.env)
                    print("✅ AndroidWorld task initialized successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to initialize AndroidWorld task: {e}")
            
            # Setup Portal after Android World environment initialization
            print("🔧 Verifying Portal setup...")
            try:
                await _setup_portal(
                    path=None, 
                    device=self.device_id, 
                    debug=self.verbose
                )
                print("✅ Portal setup verified")
            except Exception as e:
                print(f"⚠️ Warning: Portal setup check failed: {e}")
            
            # Create new DroidAgent instance for this task
            print("🤖 Creating DroidAgent...")
            agent = DroidAgent(
                goal=task_goal,
                config=self.config,
                timeout=timeout,
            )
            
            # Run the agent on the task
            print("🤖 Starting agent execution...")
            agent_start_time = time.time()
            actions_taken = []
            
            try:
                # Get the event handler
                handler = agent.run()
                
                # Stream events to collect actions
                async for event in handler.stream_events():
                    # Collect action information if available
                    if hasattr(event, 'action'):
                        actions_taken.append(str(event.action))
                
                # Wait for final result
                result_event: ResultEvent = await handler
                agent_execution_time = time.time() - agent_start_time
                
                # Create agent_result dict compatible with evaluator
                agent_result = {
                    'finished': result_event.success,
                    'actions': actions_taken,
                    'execution_time': agent_execution_time,
                    'result_message': result_event.message if hasattr(result_event, 'message') else "Task completed",
                    'success': result_event.success
                }
                
                print(f"✅ Agent execution completed in {agent_execution_time:.2f}s")
                print(f"📝 Agent result: {agent_result.get('result_message', 'Task completed')}")
                
            except Exception as e:
                agent_execution_time = time.time() - agent_start_time
                print(f"❌ Agent execution failed: {e}")
                agent_result = {
                    'finished': False,
                    'actions': actions_taken,
                    'execution_time': agent_execution_time,
                    'error': str(e),
                    'result_message': f"Agent execution failed: {e}",
                    'success': False
                }
            
            # Evaluate the task (evaluator will NOT re-initialize the task)
            print("📏 Evaluating task completion...")
            evaluation_result = self.evaluator.evaluate_task(task_instance, agent_result, metadata)
            
            # Combine results
            total_time = time.time() - start_time
            
            result = {
                'task_name': task_name,
                'family': metadata.get('family', 'unknown'),
                'task_goal': task_goal,
                'metadata': metadata,
                'agent_result': agent_result,
                'evaluation': evaluation_result,
                'total_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'success': evaluation_result['success']
            }
            
            # Print result summary
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{status} - {task_name}")
            print(f"   Score: {evaluation_result['evaluation_score']:.2f}")
            print(f"   Actions: {evaluation_result['num_actions']}")
            print(f"   Time: {total_time:.2f}s")
            
            if evaluation_result.get('error'):
                print(f"   Error: {evaluation_result['error']}")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            
            return {
                'task_name': task_name,
                'family': family or 'unknown',
                'task_goal': f"Failed to load task {task_name}",
                'metadata': {'task_name': task_name, 'family': family},
                'agent_result': {'finished': False, 'actions': [], 'execution_time': 0, 'error': error_msg},
                'evaluation': {'success': False, 'evaluation_score': 0.0, 'error': error_msg},
                'total_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': error_msg
            }
    
    async def run_task_list(self, 
                     task_names: List[str], 
                     family: Optional[str] = None,
                     timeout_per_task: int = 300) -> Dict[str, Any]:
        """
        Run a list of tasks.
        
        Args:
            task_names: List of task names to run
            family: Task family (optional)
            timeout_per_task: Timeout per task in seconds
            
        Returns:
            Aggregated results for all tasks
        """
        print(f"\n🚀 Running {len(task_names)} tasks from family: {family or 'all'}")
        
        results = []
        start_time = time.time()
        
        for i, task_name in enumerate(task_names, 1):
            print(f"\n--- Task {i}/{len(task_names)} ---")
            
            try:
                result = await self.run_single_task(task_name, family, timeout_per_task)
                results.append(result)
                
            except KeyboardInterrupt:
                print("\n⚠️ Test run interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error running task {task_name}: {e}")
                # Add error result
                results.append({
                    'task_name': task_name,
                    'family': family or 'unknown',
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        total_time = time.time() - start_time
        
        # Generate summary report
        summary = self._generate_summary(results, total_time)
        
        return {
            'summary': summary,
            'results': results,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_benchmark_suite(self, 
                           family: str = "android_world",
                           task_names: Optional[List[str]] = None,
                           n_combinations: int = 1,
                           timeout_per_task: int = 3600) -> Dict[str, Any]:
        """
        Run the full Android World benchmark suite.
        
        Args:
            family: Task family to run
            task_names: Specific tasks to run (if None, run all in family)
            n_combinations: Number of parameter combinations per task
            timeout_per_task: Timeout per task in seconds
            
        Returns:
            Complete benchmark results
        """
        print(f"\n🏆 Running Android World Benchmark Suite")
        print(f"   Family: {family}")
        print(f"   Tasks: {len(task_names) if task_names else 'all'}")
        print(f"   Combinations per task: {n_combinations}")
        
        start_time = time.time()
        
        try:
            # Create task suite
            task_suite = self.task_loader.create_task_suite(
                family=family,
                task_names=task_names,
                n_combinations=n_combinations
            )
            
            print(f"📋 Created test suite with {len(task_suite)} task instances")
            
            results = []
            
            for i, (task_goal, task_instance, metadata) in enumerate(task_suite, 1):
                task_name = metadata['task_name']
                combination_id = metadata.get('combination_id', 0)
                
                print(f"\n--- Task {i}/{len(task_suite)}: {task_name} (combination {combination_id}) ---")
                
                try:
                    # Initialize the task in AndroidWorld environment BEFORE agent execution
                    print("🔧 Initializing AndroidWorld task...")
                    if self.evaluator.env and task_instance:
                        try:
                            task_instance.initialize_task(self.evaluator.env)
                            print("✅ AndroidWorld task initialized successfully")
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to initialize AndroidWorld task: {e}")
                    
                    # Setup Portal
                    try:
                        await _setup_portal(
                            path=None, 
                            device=self.device_id, 
                            debug=self.verbose
                        )
                    except Exception as e:
                        print(f"⚠️ Portal setup: {e}")
                    
                    # Create new DroidAgent for this task
                    agent = DroidAgent(
                        goal=task_goal,
                        config=self.config,
                        timeout=timeout_per_task,
                    )
                    
                    # Run agent
                    print(f"🎯 Goal: {task_goal}")
                    agent_start_time = time.time()
                    actions_taken = []
                    
                    try:
                        # Get the event handler
                        handler = agent.run()
                        
                        # Stream events to collect actions
                        async for event in handler.stream_events():
                            if hasattr(event, 'action'):
                                actions_taken.append(str(event.action))
                        
                        # Wait for final result
                        result_event: ResultEvent = await handler
                        agent_execution_time = time.time() - agent_start_time
                        
                        agent_result = {
                            'finished': result_event.success,
                            'actions': actions_taken,
                            'execution_time': agent_execution_time,
                            'result_message': result_event.message if hasattr(result_event, 'message') else str(result_event),
                            'success': result_event.success
                        }
                            
                    except Exception as e:
                        agent_execution_time = time.time() - agent_start_time
                        agent_result = {
                            'finished': False,
                            'actions': actions_taken,
                            'execution_time': agent_execution_time,
                            'error': str(e),
                            'success': False
                        }
                    
                    # Evaluate
                    evaluation_result = self.evaluator.evaluate_task(task_instance, agent_result, metadata)
                    
                    # Calculate total time for this task
                    task_total_time = time.time() - agent_start_time
                    
                    result = {
                        'task_name': task_name,
                        'combination_id': combination_id,
                        'family': metadata.get('family'),
                        'task_goal': task_goal,
                        'metadata': metadata,
                        'agent_result': agent_result,
                        'evaluation': evaluation_result,
                        'total_time': task_total_time,
                        'timestamp': datetime.now().isoformat(),
                        'success': evaluation_result['success']
                    }
                    
                    results.append(result)
                    
                    # Print progress
                    status = "✅" if result['success'] else "❌"
                    print(f"{status} {task_name} - Score: {evaluation_result['evaluation_score']:.2f}")
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Benchmark interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Error in task {task_name}: {e}")
                    results.append({
                        'task_name': task_name,
                        'combination_id': combination_id,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            total_time = time.time() - start_time
            
            # Generate comprehensive report
            benchmark_report = self.result_reporter.generate_benchmark_report(results, total_time)
            
            return benchmark_report
            
        except Exception as e:
            print(f"❌ Benchmark suite failed: {e}")
            return {
                'error': str(e),
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_summary(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Generate summary statistics for a set of results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get('success', False))
        failed_tasks = total_tasks - successful_tasks
        
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': success_rate,
            'total_time': total_time,
            'average_time_per_task': total_time / total_tasks if total_tasks > 0 else 0.0
        }
    
    def list_available_tasks(self, family: Optional[str] = None) -> List[str]:
        """List all available tasks."""
        return self.task_loader.get_all_task_names(family)
    
    def list_available_families(self) -> List[str]:
        """List all available task families."""
        return self.task_loader.get_available_families()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.evaluator:
            self.evaluator.cleanup()

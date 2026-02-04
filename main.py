import asyncio
import argparse
import sys
from droidrun.config_manager import ConfigLoader
from evaluator import AndroidWorldTestRunner


async def list_tasks(runner: AndroidWorldTestRunner):
    """List available tasks."""
    print("\n" + "="*60)
    print("📋 Available Task Families:")
    families = runner.list_available_families()
    for family in families:
        print(f"  - {family}")
    
    print("\n📋 Available Tasks:")
    all_tasks = runner.list_available_tasks()
    for i, task in enumerate(all_tasks, 1):
        print(f"  {i:3d}. {task}")
    print(f"\nTotal: {len(all_tasks)} tasks available")


async def run_single_task(runner: AndroidWorldTestRunner, task_name: str, family: str, timeout: int):
    """Run a single task."""
    print(f"\n🎯 Running Task: {task_name}")
    print("="*60)
    
    try:
        result = await runner.run_single_task(
            task_name=task_name,
            family=family,
            timeout=timeout
        )
        
        print(f"\n✅ Task Result:")
        print(f"  Task: {result['task_name']}")
        print(f"  Success: {result['success']}")
        
        evaluation = result.get('evaluation', {})
        print(f"  Score: {evaluation.get('evaluation_score', 0.0):.2f}")
        
        if 'num_actions' in evaluation:
            print(f"  Actions: {evaluation['num_actions']}")
        
        print(f"  Time: {result.get('total_time', 0.0):.2f}s")
        
        if result.get('error'):
            print(f"  Error: {result['error']}")
        
        if evaluation.get('error'):
            print(f"  Evaluation Error: {evaluation['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"\n❌ Task failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_multiple_tasks(runner: AndroidWorldTestRunner, task_names: list, family: str, timeout: int):
    """Run multiple tasks."""
    print(f"\n🚀 Running {len(task_names)} Tasks")
    print("="*60)
    
    try:
        results = await runner.run_task_list(
            task_names=task_names,
            family=family,
            timeout_per_task=timeout
        )
        
        print(f"\n📊 Results Summary:")
        summary = results.get('summary', {})
        print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
        print(f"  Successful: {summary.get('successful_tasks', 0)}")
        print(f"  Failed: {summary.get('failed_tasks', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Total Time: {summary.get('total_time', 0):.1f}s")
        
        return summary.get('success_rate', 0) > 0
        
    except Exception as e:
        print(f"\n❌ Failed to run tasks: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_benchmark(runner: AndroidWorldTestRunner, task_names: list, family: str, 
                       combinations: int, timeout: int):
    """Run benchmark suite."""
    print(f"\n🏆 Running Benchmark Suite")
    print("="*60)
    print(f"  Family: {family}")
    print(f"  Tasks: {len(task_names) if task_names else 'all'}")
    print(f"  Combinations: {combinations}")
    print(f"  Timeout: {timeout}s per task")
    
    try:
        benchmark_results = await runner.run_benchmark_suite(
            family=family,
            task_names=task_names,
            n_combinations=combinations,
            timeout_per_task=timeout
        )
        
        print(f"\n📊 Benchmark Results:")
        summary = benchmark_results.get('summary', {})
        print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
        print(f"  Successful: {summary.get('successful_tasks', 0)}")
        print(f"  Failed: {summary.get('failed_tasks', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Total Time: {summary.get('total_time', 0):.1f}s")
        
        return summary.get('success_rate', 0) > 0
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Android World benchmark using DroidRun."""
    parser = argparse.ArgumentParser(
        description='Run Android World benchmark using DroidRun',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Task selection options
    parser.add_argument('--list-tasks', action='store_true',
                       help='List available tasks and exit')
    parser.add_argument('--task', type=str,
                       help='Run a specific task')
    parser.add_argument('--tasks', nargs='+',
                       help='Run multiple specific tasks')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run full benchmark suite')
    
    # Configuration options
    parser.add_argument('--family', type=str, default='android_world',
                       help='Task family to use (default: android_world)')
    parser.add_argument('--combinations', type=int, default=1,
                       help='Parameter combinations per task (default: 1)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per task in seconds (default: 3600)')
    parser.add_argument('--device', type=str, default='emulator-5554',
                       help='Device serial/ID (default: emulator-5554)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to droidrun config file (default: config.yaml in project root)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    
    args = parser.parse_args()
    
    # Load droidrun configuration from project local config file
    config = ConfigLoader.load(args.config)
    
    # Configuration is now managed in config.yaml
    # No need to modify settings here - edit config.yaml instead
    
    # Initialize the test runner
    runner = AndroidWorldTestRunner(
        config=config,
        device_id=args.device,
        timeout_per_task=args.timeout,
        verbose=args.verbose
    )
    
    success = True
    
    try:
        # Handle different modes
        if args.list_tasks:
            await list_tasks(runner)
        
        elif args.task:
            success = await run_single_task(runner, args.task, args.family, args.timeout)
        
        elif args.tasks:
            success = await run_multiple_tasks(runner, args.tasks, args.family, args.timeout)
        
        elif args.benchmark:
            success = await run_benchmark(runner, args.tasks, args.family, 
                                         args.combinations, args.timeout)
        
        else:
            # Default: show help
            parser.print_help()
            print("\nExamples:")
            print("  # List all available tasks")
            print("  python test.py --list-tasks")
            print("\n  # Run a single task")
            print("  python test.py --task ContactsAddContact")
            print("\n  # Run multiple tasks")
            print("  python test.py --tasks ContactsAddContact SimpleCalculatorEvaluate")
            print("\n  # Run full benchmark")
            print("  python test.py --benchmark")
    
    finally:
        # Cleanup
        runner.cleanup()
        print("\n✅ Completed!")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

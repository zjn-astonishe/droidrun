# Android World Evaluation for DroidRun

This module provides seamless integration between DroidRun and Android World benchmark testing framework, allowing you to run standardized Android automation tests using DroidRun agents.

## Overview

The integration consists of four main components:

- **AndroidWorldTaskLoader**: Loads and manages Android World tasks
- **AndroidWorldEvaluator**: Evaluates task completion using Android World's native logic
- **AndroidWorldTestRunner**: Orchestrates test execution with DroidRun agents
- **AndroidWorldResultReporter**: Generates comprehensive test reports and analysis

## Prerequisites

Before running the benchmark, ensure you have:

1. **Android World installed**:
   ```bash
   cd android_world
   pip install -e .
   ```

2. **Android Emulator running**:
   ```bash
   # Start emulator (if not already running)
   emulator -avd <your_avd_name>
   
   # Verify device connection
   adb devices
   ```

3. **DroidRun Portal installed** on the device:
   ```bash
   droidrun setup --device emulator-5554
   ```

4. **Required Android World apps** installed on the emulator (see Android World documentation)

## Quick Start

### 1. List Available Tasks

```bash
# List all available Android World tasks
python test.py --list-tasks
```

### 2. Run Single Task

```bash
# Run a specific task
python test.py --task ContactsAddContact

# Run with custom timeout
python test.py --task ContactsAddContact --timeout 600
```

### 3. Run Multiple Tasks

```bash
# Run specific tasks
python test.py --tasks ContactsAddContact SimpleCalculatorEvaluate

# Run from different task family
python test.py --tasks ContactsAddContact --family android_world
```

### 4. Run Benchmark Suite

```bash
# Run complete Android World benchmark (all tasks)
python test.py --benchmark

# Run benchmark with specific tasks
python test.py --benchmark --tasks ContactsAddContact SimpleCalculatorEvaluate

# Run with custom settings
python test.py --benchmark --family android_world --combinations 2 --timeout 600 --device emulator-5554
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--list-tasks` | List available tasks and exit | - |
| `--task TASK` | Run a specific task | - |
| `--tasks TASK1 TASK2 ...` | Run multiple specific tasks | - |
| `--benchmark` | Run full benchmark suite | - |
| `--family FAMILY` | Task family to use | `android_world` |
| `--combinations N` | Parameter combinations per task | `1` |
| `--timeout SECONDS` | Timeout per task in seconds | `300` |
| `--device DEVICE` | Device serial/ID | `emulator-5554` |
| `--config PATH` | Path to droidrun config file | None (uses default) |
| `--verbose` | Enable verbose output | `True` |

## Task Families

Available task families include:

- **android_world**: Main benchmark with 116 tasks across 20 apps
- **miniwob**: Web-based tasks from MiniWoB++ benchmark
- **android_family**: Android-specific tasks
- **information_retrieval**: Information gathering tasks

## Programmatic Usage

### Basic Example

```python
import asyncio
from droidrun.config_manager import ConfigLoader
from evaluator import AndroidWorldTestRunner

async def main():
    # Load DroidRun configuration
    config = ConfigLoader.load()
    
    # Create test runner
    runner = AndroidWorldTestRunner(
        config=config,
        device_id="emulator-5554",
        timeout_per_task=300,
        verbose=True
    )
    
    # Run single task
    result = await runner.run_single_task("ContactsAddContact")
    print(f"Success: {result['success']}")
    print(f"Score: {result['evaluation']['evaluation_score']:.2f}")
    
    # Cleanup
    runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

```python
import asyncio
from droidrun.config_manager import ConfigLoader
from evaluator import (
    AndroidWorldTaskLoader,
    AndroidWorldEvaluator,
    AndroidWorldTestRunner,
    AndroidWorldResultReporter
)

async def advanced_example():
    # Load configuration
    config = ConfigLoader.load()
    
    # Initialize components separately
    task_loader = AndroidWorldTaskLoader()
    evaluator = AndroidWorldEvaluator(device_id="emulator-5554")
    reporter = AndroidWorldResultReporter()
    
    # Load a task
    task_goal, task_instance, metadata = task_loader.get_task("ContactsAddContact")
    
    # Initialize task in Android World environment
    if evaluator.env and task_instance:
        task_instance.initialize_task(evaluator.env)
    
    # Create and run DroidRun agent
    from droidrun import DroidAgent
    agent = DroidAgent(goal=task_goal, config=config, timeout=300)
    
    handler = agent.run()
    actions = []
    async for event in handler.stream_events():
        if hasattr(event, 'action'):
            actions.append(str(event.action))
    
    result_event = await handler
    
    # Create agent result dict
    agent_result = {
        'finished': result_event.success,
        'actions': actions,
        'execution_time': 0,
        'success': result_event.success
    }
    
    # Evaluate
    evaluation = evaluator.evaluate_task(task_instance, agent_result, metadata)
    print(f"Evaluation: {evaluation}")
    
    # Generate report
    report = reporter.generate_task_report({
        'task_name': metadata['task_name'],
        'success': evaluation['success'],
        'evaluation': evaluation,
        'agent_result': agent_result,
        'metadata': metadata
    })
    
    # Cleanup
    evaluator.cleanup()

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

## Results and Analysis

### Output Structure

Test results are saved to timestamped directories with:

```
output/android_world_results/run_20240102_142530/
├── summary.json          # Overall results summary
├── tasks/                # Individual task results
│   ├── ContactsAddContact_001.json
│   └── ClockStopWatchRunning_001.json
├── reports/              # Generated reports
│   ├── benchmark_report.html
│   └── summary.csv
└── logs/                 # Execution logs
    └── test_runner.log
```

### Result Format

Each task result includes:

```json
{
  "task_name": "ContactsAddContact",
  "success": true,
  "execution_time": 45.2,
  "steps_taken": 8,
  "goal": "Add a new contact named John Doe",
  "agent_actions": [...],
  "evaluation_details": {...},
  "error_message": null
}
```

### Summary Statistics

Benchmark summaries provide:

- Success rate by task and overall
- Average execution time
- Step count analysis
- Error categorization
- Performance trends

## Integration Architecture

### Design Principles

1. **Non-invasive**: No changes to Open-AutoGLM's core `do()` format
2. **Modular**: Each component can be used independently
3. **Compatible**: Works with existing Android World tasks and evaluation logic
4. **Extensible**: Easy to add new task families or evaluation metrics

### Data Flow

```
Android World Task → Task Loader → Natural Language Goal
                                        ↓
Open-AutoGLM Agent ← Agent.run() ← Task Goal
                                        ↓
Agent Actions → Evaluator → Android World Evaluation → Results
```

### Environment Bridge

The integration automatically handles:

- Task parameter generation
- Natural language goal creation
- Environment state management
- Action format conversion
- Result evaluation and reporting

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `android_world` is installed
   ```bash
   cd android_world
   pip install -r requirements.txt
   python setup.py install
   ```

2. **No Tasks Found**: Check Android World installation
   ```bash
   python -c "from android_world import registry; print(registry.TaskRegistry().get_registry('android_world'))"
   ```

3. **Device Connection**: Ensure Android emulator is running
   ```bash
   adb devices
   # Should show connected device
   ```

4. **Timeout Issues**: Increase timeout for complex tasks
   ```bash
   python main.py --aw-task ComplexTask --aw-timeout 600
   ```

### Debug Mode

Enable verbose logging:

```bash
python main.py --aw-task ContactsAddContact --verbose
```

Or set in code:

```python
test_runner = AndroidWorldTestRunner(
    agent=agent,
    verbose=True  # Enable detailed logging
)
```

## Performance Tips

1. **Start Small**: Begin with single tasks before running full benchmark
2. **Use Checkpoints**: Results are automatically saved for analysis
3. **Monitor Resources**: Android World can be resource-intensive
4. **Optimize Timeouts**: Adjust based on task complexity

## Contributing

To add new task families or evaluation metrics:

1. Extend `AndroidWorldTaskLoader` for new task sources
2. Modify `AndroidWorldEvaluator` for custom evaluation logic
3. Update `AndroidWorldResultReporter` for new report formats
4. Add tests in `tests/` directory

## Examples

See `examples/android_world_example.py` for a complete working example demonstrating all integration features.

## Support

For issues related to:
- **Android World**: Check the [Android World repository](https://github.com/google-research/android_world)
- **Open-AutoGLM**: Check the main project documentation
- **Integration**: Review this README and example code

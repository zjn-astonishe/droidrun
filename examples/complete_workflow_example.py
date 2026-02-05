"""
Complete Workflow Example: From Trajectory to Executable Code

This example demonstrates the complete end-to-end workflow:
1. Extract skill from a trajectory
2. Generate executable Python code
3. Import and use the generated code
4. Execute the skill with parameters

完整流程示例：从轨迹到可执行代码
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from droidrun.skill import (
    Skill,
    SkillExtractor,
    SkillCodeGenerator,
    SkillLibrary,
    SkillExecutor,
)


def step_1_extract_skill():
    """Step 1: Extract skill from trajectory."""
    print("=" * 70)
    print("STEP 1: Extract Skill from Trajectory")
    print("=" * 70)
    
    trajectory_path = "output/trajectories/20260205_132221_770ad35a"
    
    # Check if trajectory exists
    if not os.path.exists(trajectory_path):
        print(f"⚠️  Trajectory not found: {trajectory_path}")
        print("   Please make sure the trajectory exists!")
        print("   Creating a mock trajectory for demonstration...")
        raise FileNotFoundError("Trajectory not found")
    
    # Extract skill
    print(f"\n📂 Extracting skill from: {trajectory_path}")
    
    extractor = SkillExtractor()
    
    # Let the system auto-generate name, description and tags
    skill = extractor.extract_from_trajectory(
        trajectory_path=trajectory_path,
        skill_name=None,  # Auto-generated from trajectory
        description=None,  # Auto-generated from actions
        auto_parameterize=True,
        tags=None,  # No tags by default
        analyze_complexity=True,
    )
    
    print(f"\n✅ Skill extracted successfully!")
    print(f"   Name: {skill.name}")
    print(f"   Description: {skill.description}")
    print(f"   Actions: {len(skill.actions)}")
    print(f"   Parameters: {len(skill.parameters)}")
    print(f"   Complexity: {skill.complexity.value}")
    print(f"   Reliability: {skill.reliability_score:.2f}")
    
    if skill.parameters:
        print(f"\n   📋 Parameters:")
        for param in skill.parameters:
            print(f"      - {param.name} ({param.param_type.value}): {param.description}")
    
    # Save skill
    skills_dir = "output/skills"
    os.makedirs(skills_dir, exist_ok=True)
    skill_path = f"{skills_dir}/{skill.name}.json"
    skill.save(skill_path)
    
    print(f"\n💾 Skill saved to: {skill_path}")
    
    return skill, skill_path


def step_2_generate_code(skill_path):
    """Step 2: Generate executable Python code from skill."""
    print("\n" + "=" * 70)
    print("STEP 2: Generate Executable Python Code")
    print("=" * 70)
    
    # Load skill
    print(f"\n📖 Loading skill from: {skill_path}")
    skill = Skill.load(skill_path)
    
    # Create code generator
    print("\n🔧 Initializing code generator...")
    generator = SkillCodeGenerator()
    
    # Generate code (without LLM for demonstration)
    print("   Generating Python function code...")
    code = generator.generate_function_code(skill, use_llm=False)
    
    print(f"\n✅ Code generated successfully!")
    print(f"   Function name: {skill.name}()")
    print(f"   Lines of code: {len(code.split(chr(10)))}")
    
    # Save code to file
    output_dir = "output/generated_skills"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = generator.save_skill_code(skill, output_dir, use_llm=False)
    print(f"\n💾 Code saved to: {file_path}")
    
    # Update skill library
    library_path = f"{output_dir}/skill_library.json"
    generator.update_skill_library(
        skill,
        library_path,
        workflow_tasks=[
            "Launch calculator",
            "Perform addition calculation",
            "Get result"
        ]
    )
    
    print(f"📚 Updated skill library: {library_path}")
    
    # Show code preview
    print(f"\n📄 Generated Code Preview:")
    print("-" * 70)
    lines = code.split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:3d} | {line}")
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    print("-" * 70)
    
    return file_path, library_path, code


def step_3_inspect_outputs(file_path, library_path):
    """Step 3: Inspect generated outputs."""
    print("\n" + "=" * 70)
    print("STEP 3: Inspect Generated Outputs")
    print("=" * 70)
    
    # Show file structure
    print("\n📁 Generated Files:")
    output_dir = os.path.dirname(file_path)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            size = os.path.getsize(os.path.join(root, file))
            print(f"{subindent}{file} ({size} bytes)")
    
    # Show skill_library.json content
    print(f"\n📄 Skill Library Content:")
    print("-" * 70)
    with open(library_path, 'r') as f:
        library_data = json.load(f)
    print(json.dumps(library_data, indent=2))
    print("-" * 70)


def step_4_use_generated_code(file_path):
    """Step 4: Import and use the generated code."""
    print("\n" + "=" * 70)
    print("STEP 4: Use Generated Code")
    print("=" * 70)
    
    # Add output directory to Python path
    output_dir = os.path.dirname(file_path)
    sys.path.insert(0, output_dir)
    
    # Import the generated function
    skill_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"\n📦 Importing generated function...")
    print(f"   from {os.path.basename(output_dir)}.{skill_name} import {skill_name}")
    
    try:
        # Dynamic import
        module = __import__(skill_name)
        skill_function = getattr(module, skill_name)
        
        print(f"✅ Successfully imported: {skill_name}()")
        
        # Call the function
        print(f"\n🔧 Calling function...")
        actions = skill_function()
        
        print(f"\n✅ Function executed successfully!")
        print(f"   Generated {len(actions)} actions")
        
        # Display actions
        print(f"\n📋 Generated Actions:")
        print("-" * 70)
        for i, action in enumerate(actions, 1):
            action_type = action.get('action', 'unknown')
            print(f"{i}. {action_type}")
            
            # Show action details
            for key, value in action.items():
                if key != 'action':
                    print(f"   {key}: {value}")
        print("-" * 70)
        
        return actions
        
    except Exception as e:
        print(f"❌ Error importing/using code: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_5_execute_with_executor(actions):
    """Step 5: Execute actions with SkillExecutor (simulation)."""
    print("\n" + "=" * 70)
    print("STEP 5: Execute Actions (Simulation)")
    print("=" * 70)
    
    if not actions:
        print("⚠️  No actions to execute")
        return
    
    print(f"\n🎯 Simulating execution of {len(actions)} actions...")
    print("-" * 70)
    
    for i, action in enumerate(actions, 1):
        action_type = action.get('action', 'unknown')
        print(f"\n{i}. Executing: {action_type}")
        
        if action_type == "Launch":
            app = action.get('app', 'unknown')
            print(f"   ▶ Launching app: {app}")
            
        elif action_type == "Tap":
            element = action.get('element', 'unknown')
            print(f"   👆 Tapping element: {element}")
            
        elif action_type == "Type":
            text = action.get('text', '')
            element = action.get('element', 'unknown')
            print(f"   ⌨️  Typing '{text}' into: {element}")
            
        else:
            print(f"   ⚡ Executing {action_type}")
        
        print(f"   ✅ Action completed")
    
    print("\n" + "-" * 70)
    print("✅ All actions executed successfully!")


def complete_workflow():
    """Run the complete workflow."""
    print("\n" + "=" * 70)
    print("🚀 COMPLETE WORKFLOW: Trajectory → Code → Execution")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Extracting a skill from a trajectory")
    print("  2. Generating executable Python code")
    print("  3. Inspecting the generated outputs")
    print("  4. Importing and using the generated code")
    print("  5. Executing the actions (simulation)")
    print()
    
    try:
        # Step 1: Extract skill
        skill, skill_path = step_1_extract_skill()
        
        # Step 2: Generate code
        file_path, library_path, code = step_2_generate_code(skill_path)
        
        # Step 3: Inspect outputs
        step_3_inspect_outputs(file_path, library_path)
        
        # Step 4: Use generated code
        actions = step_4_use_generated_code(file_path)
        
        # Step 5: Execute actions
        if actions:
            step_5_execute_with_executor(actions)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📂 Generated Files:")
        print(f"   Skill JSON: {skill_path}")
        print(f"   Python Code: {file_path}")
        print(f"   Skill Library: {library_path}")
        print("\n💡 Next Steps:")
        print("   1. Review the generated code")
        print("   2. Customize parameters if needed")
        print("   3. Import and use in your own code:")
        print(f"      from output.generated_skills.{skill.name} import {skill.name}")
        print(f"      actions = {skill.name}()")
        print()
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    complete_workflow()

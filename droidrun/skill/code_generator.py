"""
Code generator for converting skills to executable Python functions.

This module generates Python code from skills that can be directly imported and executed.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from droidrun.skill.skill import Skill

logger = logging.getLogger("droidrun")


class SkillCodeGenerator:
    """
    Generate executable Python code from skills.
    
    Converts Skill objects into Python functions that can be directly
    imported and used in code.
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_model: str = "gpt-4",
    ):
        """
        Initialize the code generator.
        
        Args:
            llm_api_key: Optional LLM API key for code generation
            llm_api_base: Optional LLM API base URL
            llm_model: LLM model name
        """
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.llm_model = llm_model
        self.llm_enabled = bool(llm_api_key and llm_api_base)
    
    def generate_function_code(
        self,
        skill: Skill,
        use_llm: bool = True,
    ) -> str:
        """
        Generate Python function code from a skill.
        
        Args:
            skill: The skill to convert
            use_llm: Whether to use LLM for generation (if configured)
            
        Returns:
            Python function code as string
        """
        if use_llm and self.llm_enabled:
            # Try LLM-based generation
            code = self._generate_with_llm(skill)
            if code:
                return code
        
        # Fallback to template-based generation
        return self._generate_with_template(skill)
    
    def _generate_with_llm(self, skill: Skill) -> Optional[str]:
        """Generate code using LLM."""
        try:
            from openai import OpenAI
            
            # Build action description
            actions_desc = []
            for i, action in enumerate(skill.actions[:10], 1):
                action_type = action.get("action_type", "unknown")
                desc = action.get("description", "")
                actions_desc.append(f"{i}. {action_type}: {desc}")
            
            if len(skill.actions) > 10:
                actions_desc.append(f"... and {len(skill.actions) - 10} more actions")
            
            # Build parameters description
            params_desc = []
            for param in skill.parameters:
                param_desc = f"- {param.name} ({param.param_type.value})"
                if param.required:
                    param_desc += " [required]"
                if param.default_value is not None:
                    param_desc += f" (default: {param.default_value})"
                param_desc += f": {param.description}"
                params_desc.append(param_desc)
            
            prompt = f"""Generate a Python function that implements this mobile automation skill:

Skill Name: {skill.name}
Description: {skill.description}

Parameters:
{chr(10).join(params_desc) if params_desc else "None"}

Actions ({len(skill.actions)} total):
{chr(10).join(actions_desc)}

Requirements:
1. Function name should be: {skill.name}
2. Include all parameters with proper type hints
3. Add comprehensive docstring explaining the function
4. Return a list of action dictionaries
5. Each action dict should have fields like "action", "element", "app", "text", etc.
6. Use the exact element IDs from the original actions
7. Include proper error handling and validation
8. Make the code clean, readable, and well-commented

Generate ONLY the Python function code, no markdown, no explanations.
Start with "def {skill.name}(" and end with "return actions".
"""
            
            client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_api_base)
            
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in mobile automation code generation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            
            code = completion.choices[0].message.content.strip()
            
            # Clean up code blocks
            code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
            code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
            
            logger.info(f"✨ Generated code for '{skill.name}' using LLM")
            return code.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate code with LLM: {e}")
            return None
    
    def _generate_with_template(self, skill: Skill) -> str:
        """Generate code using template."""
        # Build function signature
        params = []
        for param in skill.parameters:
            param_str = param.name
            if param.default_value is not None:
                param_str += f"={repr(param.default_value)}"
            params.append(param_str)
        
        signature = f"def {skill.name}({', '.join(params)}):"
        
        # Build docstring
        docstring_lines = [
            f'    """',
            f'    {skill.description}',
            ''
        ]
        
        if skill.parameters:
            docstring_lines.append('    Args:')
            for param in skill.parameters:
                req = " (required)" if param.required else ""
                default = f" (default: {param.default_value})" if param.default_value is not None else ""
                docstring_lines.append(f'        {param.name}: {param.description}{req}{default}')
            docstring_lines.append('')
        
        docstring_lines.extend([
            '    Returns:',
            '        list: A list of action dictionaries',
            '    """'
        ])
        
        # Build function body
        body_lines = [
            '    actions = []',
            ''
        ]
        
        # Add actions
        for i, action in enumerate(skill.actions):
            action_type = action.get("action_type", "unknown")
            body_lines.append(f'    # Step {i + 1}: {action.get("description", action_type)}')
            
            action_dict = {}
            
            if action_type in ["tap", "long_press"]:
                action_dict["action"] = "Tap" if action_type == "tap" else "LongPress"
                if "element" in action:
                    action_dict["element"] = action["element"]
            
            elif action_type in ["type", "input_text"]:
                action_dict["action"] = "Type"
                if "element" in action:
                    action_dict["element"] = action["element"]
                if "text" in action:
                    # Check if it's parameterized
                    text = action["text"]
                    if "{{" in str(text) and "}}" in str(text):
                        # Extract parameter name
                        param_name = re.search(r'\{\{(\w+)\}\}', str(text))
                        if param_name:
                            action_dict["text"] = f"f\"{{{param_name.group(1)}}}\""
                    else:
                        action_dict["text"] = text
            
            elif action_type == "swipe":
                action_dict["action"] = "Swipe"
                if "element" in action:
                    action_dict["element"] = action["element"]
                if "direction" in action:
                    action_dict["direction"] = action["direction"]
                if "dist" in action:
                    action_dict["dist"] = action["dist"]
            
            elif action_type in ["launch", "open_app"]:
                action_dict["action"] = "Launch"
                if "app" in action:
                    action_dict["app"] = action["app"]
            
            else:
                # Generic action
                action_dict = {"action": action_type}
                for key, value in action.items():
                    if key not in ["action_type", "timestamp", "description"]:
                        action_dict[key] = value
            
            # Format action dict
            action_str = json.dumps(action_dict, indent=8)
            body_lines.append(f'    actions.append({action_str})')
            body_lines.append('')
        
        body_lines.append('    return actions')
        
        # Combine all parts
        code = '\n'.join([
            signature,
            '\n'.join(docstring_lines),
            '\n'.join(body_lines)
        ])
        
        return code
    
    def save_skill_code(
        self,
        skill: Skill,
        output_dir: str,
        use_llm: bool = True,
    ) -> str:
        """
        Generate and save skill code to file.
        
        Args:
            skill: The skill to save
            output_dir: Directory to save the file
            use_llm: Whether to use LLM for generation
            
        Returns:
            Path to the saved file
        """
        code = self.generate_function_code(skill, use_llm=use_llm)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{skill.name}.py"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"💾 Saved skill code to: {file_path}")
        return str(file_path)
    
    def update_skill_library(
        self,
        skill: Skill,
        library_path: str,
        workflow_tasks: Optional[List[str]] = None,
    ):
        """
        Update skill library JSON file.
        
        Args:
            skill: The skill to add/update
            library_path: Path to skill_library.json
            workflow_tasks: Optional list of workflow tasks this skill handles
        """
        library_path = Path(library_path)
        
        # Load existing library
        if library_path.exists():
            with open(library_path, 'r', encoding='utf-8') as f:
                library_data = json.load(f)
        else:
            library_data = {
                "version": "1.0",
                "created_time": datetime.now().isoformat(),
                "updated_time": datetime.now().isoformat(),
                "skills": {}
            }
        
        # Extract function signature info
        params = []
        for param in skill.parameters:
            params.append({
                "name": param.name,
                "default": param.default_value
            })
        
        # Build skill info
        skill_info = {
            "function_name": skill.name,
            "tag": list(skill.tags)[0] if skill.tags else "",
            "description": skill.description,
            "parameters": params,
            "workflow_count": len(workflow_tasks) if workflow_tasks else 1,
            "workflow_tasks": workflow_tasks or [skill.description],
            "created_time": datetime.now().isoformat(),
            "file_path": f"{skill.name}.py"
        }
        
        # Update library
        library_data["skills"][skill.name] = skill_info
        library_data["updated_time"] = datetime.now().isoformat()
        
        # Save
        library_path.parent.mkdir(parents=True, exist_ok=True)
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Updated skill library: {library_path}")

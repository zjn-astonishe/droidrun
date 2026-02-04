"""
CLI commands for skill management.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import click

from droidrun.skill.skill_extractor import SkillExtractor
from droidrun.skill.skill_executor import SkillExecutor
from droidrun.skill.skill_library import SkillLibrary

logger = logging.getLogger("droidrun")


@click.group()
def skill():
    """Manage skills extracted from trajectories."""
    pass


@skill.command()
@click.argument("trajectory_path", type=click.Path(exists=True))
@click.option("--name", "-n", help="Name for the skill")
@click.option("--description", "-d", help="Description of the skill")
@click.option("--output", "-o", type=click.Path(), help="Output file path for the skill")
@click.option(
    "--no-auto-param",
    is_flag=True,
    help="Disable automatic parameter identification",
)
@click.option("--tags", "-t", multiple=True, help="Tags for the skill")
def extract(
    trajectory_path: str,
    name: Optional[str],
    description: Optional[str],
    output: Optional[str],
    no_auto_param: bool,
    tags: tuple,
):
    """Extract a skill from a trajectory."""
    click.echo(f"📂 Extracting skill from: {trajectory_path}")

    extractor = SkillExtractor()

    try:
        skill = extractor.extract_from_trajectory(
            trajectory_path=trajectory_path,
            skill_name=name,
            description=description,
            auto_parameterize=not no_auto_param,
            tags=list(tags) if tags else None,
        )

        click.echo(f"✨ Successfully extracted skill: {skill.name}")
        click.echo(f"   Description: {skill.description}")
        click.echo(f"   Actions: {len(skill.actions)}")
        click.echo(f"   Parameters: {len(skill.parameters)}")

        if skill.parameters:
            click.echo("\n📋 Parameters:")
            for param in skill.parameters:
                req = "required" if param.required else "optional"
                click.echo(f"   - {param.name} ({param.param_type}, {req})")
                click.echo(f"     {param.description}")

        # Save skill
        if output:
            output_path = Path(output)
        else:
            # Default output path
            output_path = Path("skills") / f"{skill.name}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        skill.save(str(output_path))
        click.echo(f"\n💾 Skill saved to: {output_path}")

    except Exception as e:
        click.echo(f"❌ Error extracting skill: {str(e)}", err=True)
        raise click.Abort()


@skill.command()
@click.argument("skill_path", type=click.Path(exists=True))
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Parameter in format name=value (can be specified multiple times)",
)
@click.option("--dry-run", is_flag=True, help="Show execution plan without executing")
def execute(skill_path: str, param: tuple, dry_run: bool):
    """Execute a skill with parameters."""
    from droidrun.skill.skill import Skill

    click.echo(f"📖 Loading skill from: {skill_path}")

    try:
        skill = Skill.load(skill_path)
        click.echo(f"✨ Loaded skill: {skill.name}")
        click.echo(f"   {skill.description}")

        # Parse parameters
        parameters = {}
        for p in param:
            if "=" not in p:
                click.echo(f"❌ Invalid parameter format: {p}", err=True)
                click.echo("   Use format: name=value", err=True)
                raise click.Abort()

            name, value = p.split("=", 1)
            # Try to convert to appropriate type
            try:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string

            parameters[name] = value

        if parameters:
            click.echo("\n📋 Parameters:")
            for name, value in parameters.items():
                click.echo(f"   {name} = {value}")

        # Execute skill
        executor = SkillExecutor()
        result = executor.execute(skill, parameters, dry_run=dry_run)

        if result["success"]:
            if dry_run:
                click.echo(f"\n✅ {result['message']}")
                click.echo(f"\n📝 Execution plan ({len(result['actions'])} actions):")
                for i, action in enumerate(result["actions"], 1):
                    click.echo(f"   {i}. {action.get('action_type', 'unknown')}")
            else:
                click.echo(
                    f"\n✅ Skill executed successfully "
                    f"({result['successful_actions']}/{result['total_actions']} actions)"
                )
        else:
            click.echo(f"\n❌ Execution failed: {result.get('error')}", err=True)
            if "details" in result:
                click.echo(f"   Details: {result['details']}")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@skill.command()
@click.argument("skill_path", type=click.Path(exists=True))
def info(skill_path: str):
    """Show detailed information about a skill."""
    from droidrun.skill.skill import Skill

    try:
        skill = Skill.load(skill_path)

        click.echo(f"📋 Skill Information")
        click.echo(f"=" * 50)
        click.echo(f"Name: {skill.name}")
        click.echo(f"Description: {skill.description}")
        click.echo(f"\nActions: {len(skill.actions)}")
        click.echo(f"Parameters: {len(skill.parameters)}")

        if skill.parameters:
            click.echo(f"\n📝 Parameters:")
            for param in skill.parameters:
                req = "✓ required" if param.required else "○ optional"
                default = f" (default: {param.default_value})" if param.default_value else ""
                click.echo(f"\n  {param.name} ({param.param_type}) [{req}]{default}")
                click.echo(f"    {param.description}")

        if skill.metadata:
            click.echo(f"\n📊 Metadata:")
            for key, value in skill.metadata.items():
                if key == "action_stats":
                    click.echo(f"  Action Statistics:")
                    for action_type, count in value.items():
                        click.echo(f"    - {action_type}: {count}")
                else:
                    click.echo(f"  {key}: {value}")

        click.echo(f"\n📄 Actions Preview (first 3):")
        for i, action in enumerate(skill.actions[:3], 1):
            click.echo(f"  {i}. {action.get('action_type', 'unknown')}")
            if len(skill.actions) > 3:
                click.echo(f"  ... and {len(skill.actions) - 3} more")
                break

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@skill.command()
@click.argument("library_path", type=click.Path(), default="skills")
def list(library_path: str):
    """List all skills in a library."""
    library = SkillLibrary(library_path)

    skills = library.list_skills()

    if not skills:
        click.echo(f"No skills found in {library_path}")
        return

    click.echo(f"📚 Skills in {library_path}:")
    click.echo("=" * 50)

    for skill_name in skills:
        skill = library.get_skill(skill_name)
        if skill:
            tags_str = f" [{', '.join(skill.metadata.get('tags', []))}]" if skill.metadata.get('tags') else ""
            click.echo(f"\n{skill_name}{tags_str}")
            click.echo(f"  {skill.description}")
            click.echo(
                f"  Actions: {len(skill.actions)}, Parameters: {len(skill.parameters)}"
            )


@skill.command()
@click.argument("skill_path", type=click.Path(exists=True))
def suggest(skill_path: str):
    """Suggest additional parameterization for a skill."""
    from droidrun.skill.skill import Skill

    try:
        skill = Skill.load(skill_path)
        extractor = SkillExtractor()

        suggestions = extractor.suggest_parameterization(skill)

        if not suggestions:
            click.echo("✅ No additional parameterization suggestions found.")
            return

        click.echo(f"💡 Parameterization Suggestions for '{skill.name}':")
        click.echo("=" * 50)

        for i, suggestion in enumerate(suggestions, 1):
            click.echo(f"\n{i}. {suggestion['suggestion']}")
            click.echo(f"   Value: '{suggestion['value']}'")
            click.echo(f"   Appears in actions: {suggestion['action_indices']}")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    skill()

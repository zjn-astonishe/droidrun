"""
Skill library for managing and organizing skills.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from droidrun.skill.skill import Skill

logger = logging.getLogger("droidrun")


class SkillLibrary:
    """
    Manages a collection of skills, providing functionality to:
    - Store and retrieve skills
    - Search for skills by name or tags
    - List available skills
    - Import/export skills
    """

    def __init__(self, library_path: str = "skills"):
        """
        Initialize the skill library.
        
        Args:
            library_path: Path to the directory where skills are stored
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self._skills: Dict[str, Skill] = {}
        self._load_all_skills()

    def _load_all_skills(self) -> None:
        """Load all skills from the library directory."""
        if not self.library_path.exists():
            return

        for skill_file in self.library_path.glob("*.json"):
            try:
                skill = Skill.load(str(skill_file))
                self._skills[skill.name] = skill
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")

        logger.info(f"📚 Loaded {len(self._skills)} skills from library")

    def add_skill(self, skill: Skill, overwrite: bool = False) -> None:
        """
        Add a skill to the library.
        
        Args:
            skill: The skill to add
            overwrite: Whether to overwrite if a skill with the same name exists
        """
        if skill.name in self._skills and not overwrite:
            raise ValueError(
                f"Skill '{skill.name}' already exists. Use overwrite=True to replace it."
            )

        # Save to file
        skill_file = self.library_path / f"{skill.name}.json"
        skill.save(str(skill_file))

        # Add to in-memory cache
        self._skills[skill.name] = skill
        logger.info(f"✅ Added skill '{skill.name}' to library")

    def get_skill(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.
        
        Args:
            name: The name of the skill
            
        Returns:
            The skill if found, None otherwise
        """
        return self._skills.get(name)

    def remove_skill(self, name: str) -> bool:
        """
        Remove a skill from the library.
        
        Args:
            name: The name of the skill to remove
            
        Returns:
            True if the skill was removed, False if it didn't exist
        """
        if name not in self._skills:
            return False

        # Remove from file
        skill_file = self.library_path / f"{name}.json"
        if skill_file.exists():
            skill_file.unlink()

        # Remove from in-memory cache
        del self._skills[name]
        logger.info(f"🗑️  Removed skill '{name}' from library")
        return True

    def list_skills(self) -> List[str]:
        """
        List all skill names in the library.
        
        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def search_skills(self, query: str) -> List[Skill]:
        """
        Search for skills by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
            ):
                results.append(skill)

        return results

    def get_skills_by_tag(self, tag: str) -> List[Skill]:
        """
        Get all skills with a specific tag.
        
        Args:
            tag: The tag to search for
            
        Returns:
            List of skills with the specified tag
        """
        results = []
        for skill in self._skills.values():
            tags = skill.metadata.get("tags", [])
            if tag in tags:
                results.append(skill)
        return results

    def export_library(self, output_path: str) -> None:
        """
        Export the entire library to a single JSON file.
        
        Args:
            output_path: Path to the output file
        """
        library_data = {
            "version": "1.0",
            "skills": {name: skill.to_dict() for name, skill in self._skills.items()},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(library_data, f, indent=2, ensure_ascii=False)

        logger.info(f"📦 Exported {len(self._skills)} skills to {output_path}")

    def import_library(self, input_path: str, overwrite: bool = False) -> int:
        """
        Import skills from a library JSON file.
        
        Args:
            input_path: Path to the library file
            overwrite: Whether to overwrite existing skills
            
        Returns:
            Number of skills imported
        """
        with open(input_path, "r", encoding="utf-8") as f:
            library_data = json.load(f)

        skills_data = library_data.get("skills", {})
        imported_count = 0

        for name, skill_data in skills_data.items():
            try:
                skill = Skill.from_dict(skill_data)
                self.add_skill(skill, overwrite=overwrite)
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import skill '{name}': {e}")

        logger.info(f"📥 Imported {imported_count} skills from {input_path}")
        return imported_count

    def get_summary(self) -> str:
        """Get a summary of the library."""
        summary = [
            f"Skill Library: {self.library_path}",
            f"Total skills: {len(self._skills)}",
            "",
            "Skills:",
        ]

        for skill in self._skills.values():
            tags = skill.metadata.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            summary.append(
                f"  - {skill.name}{tag_str}: {skill.description} ({len(skill.actions)} actions)"
            )

        return "\n".join(summary)

    def __str__(self) -> str:
        return self.get_summary()

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

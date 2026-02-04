from droidrun.skill.skill_extractor import SkillExtractor

extractor = SkillExtractor()

# 从轨迹提取技能
skill = extractor.extract_from_trajectory(
    trajectory_path="output/trajectories/20260204_193903_18e1ab05",
    skill_name="my_skill",
    description="My custom skill",
    auto_parameterize=True,  # 自动识别参数
    tags=["automation", "test"]
)

# 保存技能
skill.save("output/skills/my_skill.json")

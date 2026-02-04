# Skill System

将轨迹（trajectories）抽象为可复用、可参数化的技能（skills）。

## 概述

Skill 系统允许你：
- 从轨迹数据中提取可复用的技能
- 自动识别并参数化可变部分
- 管理技能库
- 执行技能并传入参数

## 快速开始

### 1. 从轨迹提取技能

```bash
# 基本用法
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05

# 指定技能名称和描述
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05 \
  --name "open_calculator" \
  --description "打开计算器应用" \
  --output skills/open_calculator.json

# 添加标签
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05 \
  --name "send_email" \
  --tags "email" --tags "communication"

# 禁用自动参数化
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05 \
  --no-auto-param
```

### 2. 查看技能信息

```bash
# 查看技能详情
python -m droidrun.skill.cli info skills/open_calculator.json

# 列出技能库中所有技能
python -m droidrun.skill.cli list skills/

# 获取参数化建议
python -m droidrun.skill.cli suggest skills/send_email.json
```

### 3. 执行技能

```bash
# 执行技能（dry-run 模式）
python -m droidrun.skill.cli execute skills/send_email.json \
  --param recipient=example@email.com \
  --param subject="Hello" \
  --dry-run

# 实际执行技能
python -m droidrun.skill.cli execute skills/send_email.json \
  --param recipient=example@email.com \
  --param subject="Hello World"
```

## 编程接口

### 提取技能

```python
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
skill.save("skills/my_skill.json")
```

### 使用技能库

```python
from droidrun.skill.skill_library import SkillLibrary

# 创建或加载技能库
library = SkillLibrary("skills/")

# 添加技能
library.add_skill(skill)

# 获取技能
my_skill = library.get_skill("my_skill")

# 列出所有技能
all_skills = library.list_skills()

# 按标签搜索
email_skills = library.search_by_tags(["email"])
```

### 执行技能

```python
from droidrun.skill.skill_executor import SkillExecutor
from droidrun.skill.skill import Skill

# 加载技能
skill = Skill.load("skills/send_email.json")

# 创建执行器
executor = SkillExecutor()

# 执行技能
result = executor.execute(
    skill=skill,
    parameters={
        "recipient": "user@example.com",
        "subject": "Test Email",
        "message": "Hello from skill system!"
    }
)

if result["success"]:
    print(f"✅ 执行成功！")
    print(f"执行了 {result['successful_actions']} 个动作")
else:
    print(f"❌ 执行失败: {result['error']}")
```

### 手动创建技能

```python
from droidrun.skill.skill import Skill, SkillParameter

# 创建技能
skill = Skill(
    name="custom_search",
    description="在应用中搜索内容",
    actions=[
        {"action_type": "tap", "x": 100, "y": 200},
        {"action_type": "input_text", "text": "{{query}}"},
        {"action_type": "tap", "x": 300, "y": 400}
    ],
    parameters=[
        SkillParameter(
            name="query",
            param_type="string",
            description="搜索查询文本",
            required=True
        )
    ],
    metadata={
        "created_by": "manual",
        "version": "1.0"
    }
)

# 保存
skill.save("skills/custom_search.json")
```

## 技能格式

技能以 JSON 格式存储：

```json
{
  "name": "send_email",
  "description": "发送邮件",
  "actions": [
    {"action_type": "tap", "x": 100, "y": 200},
    {"action_type": "input_text", "text": "{{recipient}}"},
    {"action_type": "tap", "x": 150, "y": 300},
    {"action_type": "input_text", "text": "{{subject}}"}
  ],
  "parameters": [
    {
      "name": "recipient",
      "type": "string",
      "description": "收件人邮箱地址",
      "required": true
    },
    {
      "name": "subject",
      "type": "string",
      "description": "邮件主题",
      "required": true,
      "default_value": "No Subject"
    }
  ],
  "metadata": {
    "created_at": "2026-02-04T21:40:00",
    "source_trajectory": "output/trajectories/20260204_193903_18e1ab05",
    "tags": ["email", "communication"]
  }
}
```

## 参数化

### 自动参数化

系统会自动识别轨迹中可参数化的部分：
- 文本输入（input_text）
- 坐标值（在一定阈值内变化）
- 延迟时间

### 参数类型

支持的参数类型：
- `string`: 字符串
- `integer`: 整数
- `float`: 浮点数
- `boolean`: 布尔值
- `coordinate`: 坐标 (x, y)

### 参数模板

在 action 中使用 `{{parameter_name}}` 语法引用参数：

```json
{
  "action_type": "input_text",
  "text": "{{username}}"
}
```

## 高级功能

### 技能组合

```python
from droidrun.skill.skill import Skill

# 加载多个技能
open_app = Skill.load("skills/open_app.json")
login = Skill.load("skills/login.json")

# 组合动作
combined_skill = Skill(
    name="open_and_login",
    description="打开应用并登录",
    actions=open_app.actions + login.actions,
    parameters=open_app.parameters + login.parameters
)
```

### 条件执行

```python
# 获取执行计划
plan = executor.get_execution_plan(skill, parameters)

# 检查计划
if plan["success"]:
    print(f"将执行 {len(plan['actions'])} 个动作")
    # 决定是否继续执行
    result = executor.execute(skill, parameters)
```

### 自定义 Action 执行器

```python
from droidrun.macro.replay import MacroPlayer

# 使用实际的 action 执行器
player = MacroPlayer(device)
executor = SkillExecutor(action_executor=player)

result = executor.execute(skill, parameters)
```

## 最佳实践

1. **命名规范**：使用描述性的技能名称，如 `open_settings`、`send_message`
2. **参数验证**：为参数设置合理的默认值和描述
3. **技能粒度**：保持技能专注于单一任务
4. **标签管理**：使用标签组织和检索技能
5. **版本控制**：在 metadata 中记录版本信息
6. **测试先行**：使用 dry-run 模式验证技能

## 故障排查

### 技能提取失败
- 检查轨迹路径是否正确
- 确认 macro.json 文件存在且格式正确

### 参数替换错误
- 验证参数名称与模板匹配
- 检查参数类型是否正确

### 执行失败
- 使用 `--dry-run` 查看执行计划
- 检查设备连接状态
- 验证动作序列的有效性

## 更多信息

查看源代码了解更多细节：
- `skill.py`: 核心 Skill 类
- `skill_extractor.py`: 轨迹提取逻辑
- `skill_executor.py`: 技能执行逻辑
- `skill_library.py`: 技能库管理

# Enhanced Skill System (v2.0)

将轨迹（trajectories）抽象为可复用、可参数化、具有 Claude-like agent 特性的智能技能（skills）。

## 🆕 新特性 (v2.0)

### Claude-like Agent 模式
- ✅ **Pre/Post-Conditions**: 执行前后条件验证
- ✅ **依赖管理**: 技能间依赖关系和版本约束
- ✅ **技能组合**: 将多个技能组合成复杂工作流
- ✅ **上下文感知**: 基于执行上下文的智能决策
- ✅ **重试机制**: 指数退避的自动重试
- ✅ **回滚支持**: 失败时自动回滚操作
- ✅ **执行遥测**: 性能监控和统计分析
- ✅ **复杂度分析**: 自动评估技能复杂度
- ✅ **可靠性评分**: 基于特征的可靠性评估

## 概述

增强的 Skill 系统允许你：
- 从轨迹数据中提取可复用的技能
- 自动识别并参数化可变部分
- 定义执行的前置和后置条件
- 管理技能间的依赖关系
- 组合多个技能形成复杂工作流
- 执行技能并传入参数，支持重试和回滚
- 监控执行性能和成功率

## 快速开始

### 1. 从轨迹提取技能

```bash
# 基本用法 - 现在会自动分析复杂度和可靠性
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05

# 指定技能名称和描述
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05 \
  --name "open_calculator" \
  --description "打开计算器应用" \
  --output skills/open_calculator.json

# 添加标签用于分类
python -m droidrun.skill.cli extract output/trajectories/20260204_193903_18e1ab05 \
  --name "send_email" \
  --tags "email" --tags "communication"
```

### 2. 查看技能信息

```bash
# 查看技能详情（包括复杂度、可靠性等新信息）
python -m droidrun.skill.cli info skills/open_calculator.json

# 列出技能库中所有技能
python -m droidrun.skill.cli list skills/

# 获取参数化建议
python -m droidrun.skill.cli suggest skills/send_email.json
```

### 3. 生成可执行代码（新功能）

```bash
# 从 skill JSON 生成 Python 函数代码
python -m droidrun.skill.cli generate-code skills/send_email.json

# 指定输出目录
python -m droidrun.skill.cli generate-code skills/send_email.json \
  --output generated_skills/

# 禁用 LLM，使用模板生成
python -m droidrun.skill.cli generate-code skills/send_email.json \
  --no-llm
```

### 4. 执行技能

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

### 基础用法

#### 提取技能

```python
from droidrun.skill import SkillExtractor

# 基础提取器
extractor = SkillExtractor()

# 从轨迹提取技能（包含自动复杂度分析）
skill = extractor.extract_from_trajectory(
    trajectory_path="output/trajectories/20260204_193903_18e1ab05",
    skill_name="my_skill",
    description="My custom skill",
    auto_parameterize=True,
    tags=["automation", "test"],
    analyze_complexity=True
)

# 保存技能
skill.save("skills/my_skill.json")

print(f"Complexity: {skill.complexity}")
print(f"Reliability: {skill.reliability_score}")
print(f"Estimated duration: {skill.estimated_duration}s")

# 可选：使用 LLM 增强描述生成
extractor_with_llm = SkillExtractor(
    llm_api_key="your-api-key",
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4"
)

skill_enhanced = extractor_with_llm.extract_from_trajectory(
    trajectory_path="output/trajectories/20260204_193903_18e1ab05",
    auto_parameterize=True
)
# LLM 会自动生成更准确的描述
```

#### 生成可执行代码（新功能）

```python
from droidrun.skill import Skill, SkillCodeGenerator

# 加载技能
skill = Skill.load("skills/my_skill.json")

# 创建代码生成器
generator = SkillCodeGenerator(
    llm_api_key="your-api-key",  # 可选
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4"
)

# 生成 Python 函数代码
code = generator.generate_function_code(skill, use_llm=True)
print(code)

# 保存到文件
file_path = generator.save_skill_code(skill, "generated_skills/")

# 更新 skill 库（生成 skill_library.json）
generator.update_skill_library(
    skill,
    "generated_skills/skill_library.json",
    workflow_tasks=["创建闹钟", "设置时间"]
)

# 生成的代码可以直接导入和使用
from generated_skills.my_skill import my_skill
actions = my_skill(param1="value1", param2="value2")
```

生成的 `skill_library.json` 格式：
```json
{
  "version": "1.0",
  "created_time": "2026-02-05T15:00:00",
  "updated_time": "2026-02-05T15:00:00",
  "skills": {
    "alarm_create": {
      "function_name": "alarm_create",
      "tag": "alarm.create",
      "description": "Creates an alarm in the Clock app with specified time...",
      "parameters": [
        {"name": "hour", "default": null},
        {"name": "minute", "default": null},
        {"name": "days", "default": null},
        {"name": "vibrate_enabled", "default": "True"}
      ],
      "workflow_count": 4,
      "workflow_tasks": [
        "set an alarm at 12:30 pm every Friday",
        "set an alarm at 09:15 am every Monday"
      ],
      "created_time": "2026-02-05T15:00:00",
      "file_path": "alarm_create.py"
    }
  }
}
```

生成的 `alarm_create.py` 示例：
```python
def alarm_create(hour, minute, days, vibrate_enabled=True):
    """
    Creates an alarm in the Clock app.
    
    Args:
        hour (int): The hour for the alarm (0-23 format)
        minute (int): The minute for the alarm (0-59)
        days (list): List of day abbreviations
        vibrate_enabled (bool): Whether vibration enabled
    
    Returns:
        list: A list of action dictionaries
    """
    actions = []
    
    # Launch the Clock app
    actions.append({
        "action": "Launch",
        "app": "com.google.android.deskclock"
    })
    
    # Tap the "+" button to add alarm
    actions.append({
        "action": "Tap",
        "element": "com.google.android.deskclock:id/fab|Add alarm"
    })
    
    # More actions...
    
    return actions
```

#### 执行技能

```python
from droidrun.skill import Skill, SkillExecutor, ExecutionContext

# 加载技能
skill = Skill.load("skills/send_email.json")

# 创建执行器（可选：传入技能库用于依赖解析）
executor = SkillExecutor(
    action_executor=None,  # 传入实际的 action executor
    enable_telemetry=True  # 启用性能遥测
)

# 创建执行上下文
context = ExecutionContext(
    device_state={"screen_on": True},
    app_state={"email_app_open": True}
)

# 执行技能
result = executor.execute(
    skill=skill,
    parameters={
        "recipient": "user@example.com",
        "subject": "Test Email"
    },
    context=context
)

if result.success:
    print(f"✅ 执行成功！用时 {result.execution_time:.2f}s")
    print(f"执行了 {result.successful_actions}/{result.total_actions} 个动作")
else:
    print(f"❌ 执行失败: {result.error}")
    print(f"警告: {result.warnings}")
```

### 高级功能

#### 1. 定义前置和后置条件

```python
from droidrun.skill import Skill, SkillParameter, Condition, ParameterType

# 定义条件检查函数
def check_app_open(context_dict):
    return context_dict.get("app_state", {}).get("email_app_open", False)

def check_email_sent(context_dict):
    return len(context_dict.get("previous_actions", [])) > 0

# 创建带条件的技能
skill = Skill(
    name="send_email",
    description="发送邮件",
    actions=[...],
    parameters=[
        SkillParameter(
            name="recipient",
            description="收件人邮箱",
            param_type=ParameterType.STRING,
            required=True,
            constraints={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"}  # 邮箱格式验证
        )
    ],
    preconditions=[
        Condition(
            name="app_open",
            description="邮件应用必须已打开",
            check=check_app_open,
            required=True,
            error_message="请先打开邮件应用"
        )
    ],
    postconditions=[
        Condition(
            name="email_sent",
            description="邮件应已发送",
            check=check_email_sent,
            required=True
        )
    ]
)
```

#### 2. 技能依赖管理

```python
from droidrun.skill import Skill, SkillDependency

# 定义带依赖的技能
skill = Skill(
    name="send_scheduled_email",
    description="定时发送邮件",
    actions=[...],
    dependencies=[
        SkillDependency(
            skill_name="open_email_app",
            version_constraint=">=1.0,<2.0",
            optional=False
        ),
        SkillDependency(
            skill_name="compose_email",
            version_constraint=">=1.5",
            optional=False
        )
    ]
)
```

#### 3. 技能组合

```python
from droidrun.skill import Skill

# 加载多个技能
open_app = Skill.load("skills/open_email_app.json")
compose = Skill.load("skills/compose_email.json")
send = Skill.load("skills/send_email.json")

# 组合成新技能
full_workflow = open_app.compose_with(compose).compose_with(send)
full_workflow.name = "complete_email_workflow"
full_workflow.description = "完整的邮件发送工作流"

# 保存组合技能
full_workflow.save("skills/complete_email_workflow.json")
```

#### 4. 技能链执行

```python
from droidrun.skill import SkillExecutor, ExecutionContext

executor = SkillExecutor(skill_library=library)

# 执行技能链，共享上下文
skills = [open_app, compose, send]
parameters_list = [
    {},  # open_app 无参数
    {"recipient": "user@example.com", "subject": "Hello"},  # compose 参数
    {}   # send 无参数
]

results = executor.execute_chain(
    skills=skills,
    parameters_list=parameters_list,
    context=ExecutionContext(),
    stop_on_failure=True  # 遇到失败时停止
)

for i, result in enumerate(results):
    print(f"Skill {i+1}: {'✅' if result.success else '❌'} {result.skill_name}")
```

#### 5. 重试和回滚

```python
from droidrun.skill import Skill, SkillComplexity

skill = Skill(
    name="unstable_operation",
    description="可能失败的操作",
    actions=[...],
    complexity=SkillComplexity.COMPLEX,
    max_retries=3,  # 最多重试3次
    retry_delay=1.0,  # 重试延迟（会指数增长）
    rollback_actions=[  # 失败时的回滚操作
        {"action_type": "tap", "x": 100, "y": 200},  # 返回按钮
        {"action_type": "wait", "duration": 1.0}
    ]
)

# 执行器会自动处理重试和回滚
result = executor.execute(skill, parameters={})
```

#### 6. 执行遥测

```python
from droidrun.skill import SkillExecutor

executor = SkillExecutor(enable_telemetry=True)

# 执行多个技能...
for skill in skills:
    executor.execute(skill, parameters={})

# 获取统计信息
telemetry = executor.get_telemetry()

print(f"总执行次数: {telemetry['total_executions']}")
print(f"成功率: {telemetry['success_rate']:.2%}")
print(f"平均执行时间: {telemetry['average_execution_time']:.2f}s")

# 每个技能的统计
for skill_name, stats in telemetry['skill_statistics'].items():
    print(f"\n{skill_name}:")
    print(f"  执行次数: {stats['executions']}")
    print(f"  成功: {stats['successes']}, 失败: {stats['failures']}")
    print(f"  平均时间: {stats['avg_time']:.2f}s")
```

### 参数高级特性

#### 参数验证和约束

```python
from droidrun.skill import SkillParameter, ParameterType

# 带约束的参数
param = SkillParameter(
    name="retry_count",
    description="重试次数",
    param_type=ParameterType.INTEGER,
    default_value=3,
    required=False,
    constraints={
        "min": 1,
        "max": 10
    }
)

# 带枚举的参数
param = SkillParameter(
    name="priority",
    description="优先级",
    param_type=ParameterType.STRING,
    default_value="medium",
    constraints={
        "enum": ["low", "medium", "high"]
    }
)

# 带正则验证的参数
param = SkillParameter(
    name="email",
    description="邮箱地址",
    param_type=ParameterType.STRING,
    constraints={
        "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"
    }
)

# 自定义验证函数
def validate_phone(value):
    return len(value) == 11 and value.isdigit()

param = SkillParameter(
    name="phone",
    description="手机号码",
    param_type=ParameterType.STRING,
    validator=validate_phone
)
```

## 技能格式

增强的技能 JSON 格式：

```json
{
  "name": "send_email",
  "description": "发送邮件",
  "version": "1.0",
  "complexity": "moderate",
  "estimated_duration": 5.2,
  "reliability_score": 0.95,
  "max_retries": 2,
  "retry_delay": 1.0,
  "tags": ["email", "communication"],
  "actions": [
    {"action_type": "tap", "x": 100, "y": 200},
    {"action_type": "input_text", "text": "{{recipient}}"},
    {"action_type": "tap", "x": 150, "y": 300},
    {"action_type": "input_text", "text": "{{subject}}"}
  ],
  "parameters": [
    {
      "name": "recipient",
      "description": "收件人邮箱地址",
      "param_type": "string",
      "required": true,
      "constraints": {
        "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
      }
    },
    {
      "name": "subject",
      "description": "邮件主题",
      "param_type": "string",
      "required": true,
      "default_value": "No Subject"
    }
  ],
  "rollback_actions": [
    {"action_type": "tap", "x": 50, "y": 50}
  ],
  "dependencies": [
    {
      "skill_name": "open_email_app",
      "version_constraint": ">=1.0",
      "optional": false
    }
  ],
  "metadata": {
    "created_at": "2026-02-05T14:00:00",
    "source_trajectory": "output/trajectories/20260204_193903_18e1ab05",
    "tags": ["email", "communication"],
    "action_stats": {
      "tap": 2,
      "input_text": 2
    },
    "patterns": {
      "sequential_taps": 2,
      "input_then_submit": true,
      "scroll_and_tap": false
    }
  }
}
```

## 最佳实践

### 1. 技能设计
- **单一职责**: 每个技能专注于一个明确的任务
- **适当粒度**: 不要太细（难以管理）也不要太粗（难以复用）
- **清晰命名**: 使用描述性名称，如 `open_settings`、`send_message`
- **完善文档**: 为技能和参数提供清晰的描述

### 2. 参数管理
- **合理默认值**: 为参数设置有意义的默认值
- **强类型**: 使用正确的参数类型而不是都用字符串
- **验证约束**: 添加适当的验证规则防止无效输入
- **最小必需**: 只将真正需要变化的部分参数化

### 3. 条件设计
- **明确条件**: 前置条件应清晰可验证
- **合理宽松**: 不要设置过于严格的条件
- **错误信息**: 提供有帮助的错误消息
- **可选条件**: 对非关键条件使用 `required=False`

### 4. 依赖管理
- **最小依赖**: 只声明真正需要的依赖
- **版本约束**: 使用合理的版本约束范围
- **可选依赖**: 适当使用可选依赖和后备方案
- **循环检测**: 避免循环依赖

### 5. 错误处理
- **适度重试**: 根据操作性质设置合理的重试次数
- **回滚机制**: 为有副作用的操作提供回滚逻辑
- **错误日志**: 记录详细的错误信息便于调试
- **优雅降级**: 失败时提供有意义的反馈

### 6. 性能优化
- **遥测监控**: 启用遥测了解性能瓶颈
- **估计时间**: 设置合理的预期执行时间
- **上下文复用**: 在技能链中复用执行上下文
- **并行执行**: 对独立技能考虑并行执行（未来版本）

### 7. 组合策略
- **层次化**: 从简单技能组合成复杂工作流
- **可测试性**: 确保组合后的技能可以独立测试
- **版本管理**: 为组合技能维护版本信息
- **元数据追踪**: 记录组合关系便于追溯

## 故障排查

### 技能提取失败
```bash
# 问题：找不到 macro.json
❌ FileNotFoundError: macro.json not found

# 解决：检查路径
ls output/trajectories/20260204_193903_18e1ab05/macro.json
```

### 参数验证错误
```python
# 问题：参数类型不匹配
❌ Parameter validation failed: Expected integer, got str

# 解决：使用正确的类型
result = executor.execute(skill, {"count": 5})  # 而不是 "5"
```

### 前置条件失败
```python
# 问题：前置条件不满足
❌ Preconditions not met: 请先打开应用

# 解决1：设置正确的上下文
context = ExecutionContext(app_state={"app_open": True})

# 解决2：使用 force=True 跳过检查（谨慎使用）
result = executor.execute(skill, parameters, force=True)
```

### 依赖解析失败
```python
# 问题：找不到依赖的技能
❌ Dependency resolution failed: Required dependency 'open_app' not found

# 解决：确保依赖技能在库中
library.add_skill(dependency_skill)
executor = SkillExecutor(skill_library=library)
```

### 执行超时
```python
# 问题：技能执行时间过长
⚠️ Execution time: 30.5s (estimated: 5.0s)

# 解决：调整预期时间或优化技能
skill.estimated_duration = 35.0
```

## API 参考

### 核心类

#### Skill
- `name`: 技能名称
- `description`: 技能描述
- `actions`: 动作列表
- `parameters`: 参数列表
- `complexity`: 复杂度级别 (SIMPLE/MODERATE/COMPLEX)
- `reliability_score`: 可靠性评分 (0.0-1.0)
- `estimated_duration`: 预计执行时间（秒）
- `preconditions`: 前置条件列表
- `postconditions`: 后置条件列表
- `dependencies`: 依赖技能列表
- `max_retries`: 最大重试次数
- `retry_delay`: 重试延迟
- `rollback_actions`: 回滚操作列表

方法：
- `validate_parameters(params)`: 验证参数
- `check_preconditions(context)`: 检查前置条件
- `check_postconditions(context)`: 检查后置条件
- `apply_parameters(params)`: 应用参数到动作
- `compose_with(other_skill)`: 与另一技能组合

#### SkillExecutor
方法：
- `execute(skill, parameters, context, dry_run, force)`: 执行技能
- `execute_chain(skills, parameters_list, context)`: 执行技能链
- `get_execution_plan(skill, parameters)`: 获取执行计划
- `get_telemetry()`: 获取遥测数据
- `clear_telemetry()`: 清除遥测历史

#### SkillExtractor
方法：
- `extract_from_trajectory(trajectory_path, ...)`: 从轨迹提取
- `extract_from_action_sequence(actions, ...)`: 从动作序列提取
- `suggest_parameterization(skill)`: 建议参数化机会

#### SkillLibrary
方法：
- `add_skill(skill, overwrite)`: 添加技能
- `get_skill(name)`: 获取技能
- `remove_skill(name)`: 移除技能
- `list_skills()`: 列出所有技能
- `search_skills(query)`: 搜索技能
- `get_skills_by_tag(tag)`: 按标签获取

## 迁移指南

### 从 v1.0 迁移到 v2.0

大多数 v1.0 代码无需修改即可运行，但建议利用新特性：

```python
# v1.0 代码
skill = Skill.load("skills/my_skill.json")
result = executor.execute(skill, {"param": "value"})

# v2.0 增强
skill = Skill.load("skills/my_skill.json")
# 现在自动包含 complexity, reliability_score 等

# 使用新的 ExecutionResult
result = executor.execute(skill, {"param": "value"})
print(f"执行时间: {result.execution_time}s")
print(f"可靠性: {skill.reliability_score}")

# 使用遥测
telemetry = executor.get_telemetry()
print(f"成功率: {telemetry['success_rate']}")
```

### 参数类型更新

```python
# v1.0: 字符串类型
param_type="string"

# v2.0: 使用枚举
from droidrun.skill import ParameterType
param_type=ParameterType.STRING  # 推荐
```

## 更多信息

### 源代码
- `skill.py`: 核心 Skill 类和相关数据结构
- `skill_executor.py`: 增强的执行器，支持重试、回滚、遥测
- `skill_extractor.py`: 智能提取器，自动分析复杂度和模式
- `skill_library.py`: 技能库管理

### 相关文档
- [DroidRun 主文档](../../README.md)
- [Macro 系统](../macro/README.md)
- [Agent 系统](../agent/README.md)

### 贡献
欢迎贡献代码、报告问题或提出改进建议！

---

**版本**: 2.0.0  
**更新日期**: 2026-02-05  
**遵循**: Claude-like Agent Skill 设计模式

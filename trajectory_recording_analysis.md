# DroidRun Trajectory 记录机制分析文档

## 概述

DroidRun 的 trajectory（轨迹）记录系统是一个**异步、非阻塞**的数据持久化机制，用于记录 Agent 执行任务时的完整过程，包括：
- 事件序列（Events）
- 截图（Screenshots）
- UI 状态（UI States）
- 宏操作（Macro Actions）

---

## 核心组件

### 1. Trajectory 数据结构 (`droidrun/agent/utils/trajectory.py`)

```python
class Trajectory:
    """存储 trajectory 数据的容器"""
    - goal: str                          # 任务目标
    - trajectory_folder: Path            # 存储目录
    - events: List[Event]                # 事件列表
    - macro: List[MacroEvent]            # 宏操作列表
    - screenshot_queue: List[bytes]      # 待写入的截图队列
    - screenshot_count: int              # 截图计数
    - ui_states: List[Dict]              # UI 状态列表
```

### 2. TrajectoryWriter (`droidrun/agent/trajectory/writer.py`)

**异步后台写入器**，通过队列机制实现非阻塞 I/O：

```python
class TrajectoryWriter:
    """
    核心功能：
    1. 管理一个异步工作线程（WriterWorker）
    2. 将写入任务放入队列（max_queue_size=300）
    3. 后台顺序处理写入任务
    4. 不阻塞 Agent 主线程
    """
```

#### 关键方法

- **`start()`**: 启动后台工作线程
- **`write(trajectory, stage)`**: 创建数据快照并提交写入任务
- **`write_final(trajectory)`**: 最终写入 + 生成 GIF
- **`stop(timeout=30)`**: 等待队列清空并停止工作线程

---

## WriteJob 任务类型

所有写入操作通过不可变的 `WriteJob` 实现：

### 1. EventsWriteJob
**写入 `trajectory.json`**
- 序列化所有事件（Events）
- 包含 LLM 调用、工具使用等完整日志

### 2. MacroWriteJob
**写入 `macro.json`**
- 记录可重放的宏操作序列
- 包含 tap、swipe、input_text 等原子操作

### 3. ScreenshotWriteJob
**写入截图 `screenshots/0000.png`**
- 从 screenshot_queue 批量写入
- 按顺序编号（0000, 0001, ...）

### 4. UIStateWriteJob
**写入 UI 状态 `ui_states/0000.json`**
- 包含 UI 层级、元素坐标等信息

### 5. GifWriteJob
**生成动画 GIF**
- 仅在 finalize 时执行
- 从所有截图生成 `trajectory.gif`

---

## 完整记录流程

### 阶段 1: 初始化 (DroidAgent.__init__)

```python
# 1. 创建 Trajectory 容器
self.trajectory = Trajectory(
    goal=self.shared_state.instruction,
    base_path=self.config.logging.trajectory_path
)

# 2. 创建 TrajectoryWriter
self.trajectory_writer = TrajectoryWriter(queue_size=300)
```

### 阶段 2: 启动后台写入器 (start_handler)

```python
await self.trajectory_writer.start()

# 写入初始状态
if self.config.logging.save_trajectory != "none":
    self.trajectory_writer.write(self.trajectory, stage="init")
```

### 阶段 3: 运行时事件捕获 (handle_stream_event)

```python
def handle_stream_event(self, ev: Event, ctx: Context):
    """捕获并分类存储各类事件"""
    
    if isinstance(ev, ScreenshotEvent):
        # 截图 -> 加入队列
        self.trajectory.screenshot_queue.append(ev.screenshot)
        self.trajectory.screenshot_count += 1
        
    elif isinstance(ev, MacroEvent):
        # 宏操作 -> 直接添加
        self.trajectory.macro.append(ev)
        
    elif isinstance(ev, RecordUIStateEvent):
        # UI 状态 -> 直接添加
        self.trajectory.ui_states.append(ev.ui_state)
        
    else:
        # 其他事件 -> 通用事件列表
        self.trajectory.events.append(ev)
```

**关键点**：
- 截图使用**队列缓存**，减少频繁 I/O
- 事件立即追加到内存列表
- 所有操作都是非阻塞的

### 阶段 4: 阶段性快照写入

在以下关键节点调用 `writer.write()`：

#### a) CodeAct 模式每步之后
```python
# execute_task 方法中
if self.config.logging.save_trajectory != "none":
    self.shared_state.step_number += 1
    self.trajectory_writer.write(
        self.trajectory,
        stage=f"codeact_step_{self.shared_state.step_number}"
    )
```

#### b) Manager-Executor 模式每步之后
```python
# handle_executor_result / handle_scripter_result 中
self.shared_state.step_number += 1

if self.config.logging.save_trajectory != "none":
    self.trajectory_writer.write(
        self.trajectory,
        stage=f"step_{self.shared_state.step_number}"
    )
```

**快照机制**：
```python
# writer.py 的 write() 方法
events_snapshot = list(trajectory.events)              # 复制事件列表
macro_snapshot = list(trajectory.macro)                # 复制宏列表
screenshot_queue_snapshot = list(trajectory.screenshot_queue)  # 复制截图队列
ui_states_snapshot = list(trajectory.ui_states)        # 复制 UI 状态

# 创建写入任务并提交到队列
jobs = [
    EventsWriteJob(...),
    MacroWriteJob(...),
    ScreenshotWriteJob(...),
    UIStateWriteJob(...)
]

for job in jobs:
    self.worker.submit(job)

# 清空已提交的截图队列
trajectory.screenshot_queue.clear()
```

### 阶段 5: 最终化 (finalize)

```python
async def finalize(self, ctx: Context, ev: FinalizeEvent) -> ResultEvent:
    # 1. 捕获最终截图
    screenshot_result = await self.tools_instance.take_screenshot()
    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
    
    # 2. 最终写入 + GIF 生成
    if self.config.logging.save_trajectory != "none":
        self.trajectory_writer.write_final(
            self.trajectory,
            self.config.logging.trajectory_gifs  # 是否生成 GIF
        )
        
        # 3. 等待所有写入任务完成
        await self.trajectory_writer.stop()
        
        logger.info(f"📁 Trajectory saved: {self.trajectory.trajectory_folder}")
```

---

## 数据序列化

### make_serializable 函数

递归地将对象转换为 JSON 可序列化格式：

```python
def make_serializable(obj):
    """处理各种 Python 对象"""
    
    # 1. ChatMessage 对象 -> 提取 role 和 content
    if obj.__class__.__name__ == "ChatMessage":
        return {"role": obj.role.value, "content": obj.content}
    
    # 2. 字典/列表 -> 递归处理
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    
    # 3. 对象 -> 转换 __dict__
    elif hasattr(obj, "__dict__"):
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    
    # 4. 基础类型 -> 直接返回
    else:
        return obj
```

---

## 输出文件结构

```
output/trajectories/
└── 20260205_193329_e2a18059/          # 时间戳_唯一ID
    ├── trajectory.json                 # 完整事件日志
    ├── macro.json                      # 可重放的宏操作
    ├── screenshots/
    │   ├── 0000.png                   # 按步骤编号的截图
    │   ├── 0001.png
    │   ├── ...
    │   └── trajectory.gif             # 动画演示（可选）
    └── ui_states/
        ├── 0000.json                  # UI 层级结构
        ├── 0001.json
        └── ...
```

### trajectory.json 示例

```json
[
  {
    "type": "LLMCallEvent",
    "agent": "codeact",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "response": "...",
    "tokens": {
      "prompt_tokens": 1234,
      "completion_tokens": 567
    }
  },
  {
    "type": "ToolCallEvent",
    "tool": "tap",
    "arguments": {"x": 100, "y": 200},
    "result": "success"
  }
]
```

### macro.json 示例

```json
{
  "version": "1.0",
  "description": "Open Gmail and send email",
  "timestamp": "20260205_193329",
  "total_actions": 5,
  "actions": [
    {
      "type": "TapEvent",
      "x": 540,
      "y": 960,
      "duration": 0.1
    },
    {
      "type": "InputTextEvent",
      "text": "Hello World",
      "clear": false
    }
  ]
}
```

---

## 关键设计优势

### 1. 异步非阻塞
- **Agent 永不等待 I/O**：所有写入在后台队列中完成
- **性能影响最小**：不影响 Agent 决策速度

### 2. 数据一致性
- **快照机制**：写入时创建数据副本，避免竞态条件
- **不可变任务**：WriteJob 使用 `@dataclass(frozen=True)`

### 3. 容错能力
- **队列满时丢弃**：避免内存溢出（记录警告）
- **写入失败继续**：单个任务失败不影响其他任务

### 4. 灵活配置
```yaml
# config.yaml
logging:
  save_trajectory: "all"      # all / errors_only / none
  trajectory_path: "./output/trajectories"
  trajectory_gifs: true       # 生成动画 GIF
```

---

## 从 main.py 的调用链

```
main.py (asyncio.run)
    ↓
evaluator/test_runner.py: AndroidWorldTestRunner.run_single_task()
    ↓
DroidAgent.__init__()
    → 创建 Trajectory & TrajectoryWriter
    ↓
DroidAgent.run() → start_handler()
    → await trajectory_writer.start()
    → trajectory_writer.write(trajectory, "init")
    ↓
[Agent 循环执行]
    → handle_stream_event() 捕获事件
    → 每步后: trajectory_writer.write(trajectory, f"step_{N}")
    ↓
    result = await handler
```

**trajectory 记录时机**:

1. **Agent 初始化** → 创建 Trajectory & TrajectoryWriter
2. **Agent.run()** → `start_handler()` 启动 writer，写入 init
3. **执行过程中** → 每个 step 完成后写入中间状态
4. **Agent 完成** → `finalize()` 最终保存 + 生成 GIF
5. **Writer 停止** → 等待队列清空，关闭 worker

## 性能优化设计

### 1. 非阻塞设计
- Agent 不等待 I/O 完成
- 数据写入在后台异步进行
- 队列满时会丢弃任务（记录警告）

### 2. 批量处理
- 一次 `write()` 调用创建多个 Job
- Worker 串行处理，避免文件竞争

### 3. 内存管理
- 使用队列限制内存占用（max 300 items）
- 写入后立即清空 screenshot_queue
- 使用弱引用避免循环引用

### 4. 错误处理
- 单个 Job 失败不影响其他 Job
- 记录错误但继续处理队列
- 超时机制防止无限等待

## 典型使用场景

### 场景 1: 正常任务执行

```
1. main.py 启动 → runner.run_single_task()
2. 创建 DroidAgent(goal="打开设置")
3. agent.run() 开始执行
   ├─ start_handler(): write(stage="init")
   ├─ execute_task(): 每步 write(stage="step_N")
   └─ finalize(): write_final() + stop()
4. 保存到 output/trajectories/20260207_xxx/
```

### 场景 2: 仅保存最终结果

```yaml
# config.yaml
logging:
  save_trajectory: "final"
```

- `start_handler()`: 跳过 init 写入
- `execute_task()`: 跳过中间写入
- `finalize()`: 执行完整保存

### 场景 3: 禁用 trajectory

```yaml
# config.yaml
logging:
  save_trajectory: "none"
```

- 所有 `write()` 调用被跳过
- Worker 仍然启动但不处理任何任务
- 事件仍然收集到内存（用于调试）

## 关键时间点总结

| 阶段 | 方法 | Trajectory 操作 | 文件写入 |
|------|------|----------------|----------|
| 初始化 | `__init__()` | 创建 Trajectory & Writer | 无 |
| 启动 | `start_handler()` | 启动 worker + write(init) | trajectory.json |
| 执行中 | `execute_task()` / `run_executor()` | write(step_N) | trajectory.json + screenshots |
| 完成 | `finalize()` | write_final() + stop() | 所有文件 + GIF |

## 调试技巧

### 查看 trajectory 内容

```python
import json

# 读取事件
with open("output/trajectories/xxx/trajectory.json") as f:
    events = json.load(f)
    
# 读取宏指令
with open("output/trajectories/xxx/macro.json") as f:
    macro = json.load(f)
```

### 监控写入状态

```python
# TrajectoryWriter 内部统计
print(f"写入成功: {writer.worker._write_count}")
print(f"写入失败: {writer.worker._error_count}")
print(f"队列剩余: {writer.worker.queue.qsize()}")
```

### 常见问题

1. **Trajectory 文件为空**: 检查 `save_trajectory` 配置
2. **缺少截图**: 检查 `ScreenshotEvent` 是否正确触发
3. **GIF 未生成**: 确保 `trajectory_gifs=True` 且有截图
4. **写入缓慢**: 增加 `queue_size` 或检查磁盘 I/O

## 总结

DroidRun 的 trajectory 记录系统通过以下机制实现高效、可靠的数据持久化：

1. **事件驱动**: 通过 `handle_stream_event()` 自动捕获所有事件
2. **异步非阻塞**: Worker 队列在后台处理，不影响 Agent 执行
3. **数据快照**: 写入时创建副本，避免并发修改
4. **分阶段保存**: init → 中间步骤 → final，可配置保存策略
5. **多种数据类型**: 支持事件、截图、UI 状态、宏指令、GIF
6. **错误容忍**: 单个写入失败不影响整体流程

这个设计使得 DroidRun 能够在不影响性能的前提下，完整记录 Agent 的执行过程，为调试、分析和复现提供强大支持。

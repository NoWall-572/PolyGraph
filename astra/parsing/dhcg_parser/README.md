# 动态异构因果图 (DHCG) 解析器

## 简介

本解析器用于解析 `Who&When` 数据集的 JSON 日志文件，并将其转换为动态异构因果图 (Dynamic Heterogeneous Causal Graph, DHCG)。该图结构能够显式捕捉智能体、工具、文件之间的交互和因果依赖关系。

## 功能特性

- ✅ 解析 JSON 格式的任务日志
- ✅ 自动识别节点类型（Agent、Tool、Artifact、Environment）
- ✅ 构建多种类型的边（Invoke、Return、Reference、Communicate、Affect）
- ✅ 提取节点的动态特征（包括文本嵌入）
- ✅ 支持时间序列的图结构构建
- ✅ **支持单个文件和目录批量处理**
- ✅ **支持结果保存到JSON文件**

## 依赖要求

### Python 版本
- Python 3.7+

### 必需的 Python 包
```bash
pip install sentence-transformers
```

注意：`sentence-transformers` 会自动安装其依赖项（如 `torch`、`transformers` 等），首次运行时会自动下载模型 `all-MiniLM-L6-v2`。

## 安装

1. 确保已安装 Python 3.7 或更高版本
2. 安装依赖包：
```bash
pip install sentence-transformers
```

## 使用方法

### 基本用法

#### 1. 处理单个文件

```bash
python parser.py <json_file_path>
```

**示例：**
```bash
# 从项目根目录运行
python parser.py "Who&When/Algorithm-Generated/1.json"
python parser.py "Who&When/Hand-Crafted/1.json"
```

#### 2. 批量处理整个目录 ⭐

```bash
python parser.py <directory_path>
```

**示例：**
```bash
# 处理 Algorithm-Generated 目录下的所有JSON文件（126个文件）
python parser.py "Who&When/Algorithm-Generated"

# 处理 Hand-Crafted 目录下的所有JSON文件（58个文件）
python parser.py "Who&When/Hand-Crafted"
```

#### 3. 批量处理并保存结果

```bash
python parser.py <directory_path> --save --output <output_directory>
```

**示例：**
```bash
# 处理并保存结果到 outputs/ 目录
python dhcg_parser/parser.py "Who&When/Algorithm-Generated" --save --output outputs/
python dhcg_parser/parser.py "Who&When/Hand-Crafted" --save --output outputs/
```
```

#### 4. 静默模式（只显示总结）

```bash
python parser.py <directory_path> --quiet
```

### 命令行参数

- `input_path`: 输入的JSON文件路径或目录路径（必需）
- `--save`: 保存解析结果到JSON文件
- `--output`: 输出目录（默认: `outputs`）
- `--quiet`: 静默模式，只显示总结信息

## 输出说明

### 单个文件处理

解析器会输出以下信息：

1. **图摘要**：节点总数、边总数、问题描述
2. **问题内容**：完整的任务描述
3. **节点样本**：前5个节点的详细信息，包括节点ID、类型、创建时间，以及最新时间步的特征
4. **边样本**：前5条边的详细信息，包括时间戳、源节点、目标节点、边类型和特征
5. **统计信息**：
   - 各类型节点的数量
   - 各类型边的数量

### 批量处理

批量处理时会显示：

1. **进度信息**：当前处理的文件编号和总数
2. **每个文件的处理结果**：成功/失败状态，节点数和边数
3. **总结信息**：
   - 总文件数
   - 成功处理的文件数
   - 失败的文件数

### 输出示例

#### 单个文件输出
```
Processing: 1.json...
  ✓ Success: 8 nodes, 12 edges

--- Graph Summary ---
DynamicGraph(
  Nodes: 8,
  Edges: 12,
  Question: 'This spreadsheet contains a list...'
)

--- Question ---
This spreadsheet contains a list of clients for a retractable awning company...

--- Statistics ---
Node types: {'Environment': 2, 'Agent': 3, 'Tool': 1, 'Artifact': 2}
Edge types: {'Reference': 1, 'Invoke': 2, 'Return': 2, 'Communicate': 4, 'Affect': 0}
```

#### 批量处理输出
```
Found 126 JSON files in: Who&When/Algorithm-Generated
============================================================

[1/126] Processing: 1.json...
  ✓ Success: 8 nodes, 12 edges

[2/126] Processing: 2.json...
  ✓ Success: 12 nodes, 18 edges

...

============================================================
Processing Complete!
  Total files: 126
  Successful: 126
  Failed: 0
============================================================
```

## 数据集目录结构

项目中的数据集位于以下目录：

```
Who&When/
├── Algorithm-Generated/    # 算法生成的数据集（126个JSON文件）
│   ├── 1.json
│   ├── 2.json
│   └── ...
└── Hand-Crafted/           # 手工制作的数据集（58个JSON文件）
    ├── 1.json
    ├── 2.json
    └── ...
```

## 一键处理所有数据集

### 方法1：分别处理两个目录

```bash
# 处理 Algorithm-Generated 数据集
python parser.py "Who&When/Algorithm-Generated"

# 处理 Hand-Crafted 数据集
python parser.py "Who&When/Hand-Crafted"
```

### 方法2：使用脚本批量处理（推荐）

创建 `process_all.sh` (Linux/Mac) 或 `process_all.bat` (Windows):

**process_all.bat (Windows):**
```batch
@echo off
echo Processing Algorithm-Generated dataset...
python parser.py "Who&When/Algorithm-Generated" --save --output outputs/algorithm-generated

echo.
echo Processing Hand-Crafted dataset...
python parser.py "Who&When/Hand-Crafted" --save --output outputs/hand-crafted

echo.
echo All datasets processed!
```

**process_all.sh (Linux/Mac):**
```bash
#!/bin/bash

echo "Processing Algorithm-Generated dataset..."
python parser.py "Who&When/Algorithm-Generated" --save --output outputs/algorithm-generated

echo ""
echo "Processing Hand-Crafted dataset..."
python parser.py "Who&When/Hand-Crafted" --save --output outputs/hand-crafted

echo ""
echo "All datasets processed!"
```

## 数据结构说明

### Node（节点）

- `id`: 节点唯一标识符
- `type`: 节点类型（Agent、Tool、Artifact、Environment）
- `created_at`: 节点创建的时间步
- `features`: 字典，键为时间步，值为该时间步的特征字典

### Edge（边）

- `source`: 源节点ID
- `target`: 目标节点ID
- `type`: 边类型（Invoke、Return、Reference、Communicate、Affect）
- `timestamp`: 边创建的时间步
- `features`: 边的特征字典

### DynamicGraph（动态图）

- `question`: 任务问题描述
- `ground_truth`: 真实标签（包括mistake_agent、mistake_step等）
- `nodes`: 节点字典（键为节点ID）
- `edges`: 边列表

## 节点类型识别规则

1. **Tool**: 
   - 节点名称为 "Computer_terminal"
   - 或者事件中包含 exitcode 信息

2. **Agent**:
   - 节点名称在 `system_prompt` 中定义
   - 或者节点名称匹配模式 `(Expert|Assistant|Orchestrator|Surfer|Planner)`
   - 默认类型为 Agent

3. **Artifact**: 
   - 通过文件路径或URL引用识别

4. **Environment**: 
   - 预定义的全局节点（Broadcast、Env）

## 边类型说明

1. **Invoke**: Agent 调用工具（当 Agent 的内容包含代码块时）
2. **Return**: 工具返回结果给调用者（当 Tool 执行完成时）
3. **Reference**: 引用文件或URL（从内容中提取路径）
4. **Communicate**: Agent 之间的通信（通过 @ 提及或时序推断）
5. **Affect**: 环境对节点的影响（当检测到错误关键词时）

## 节点特征

### 所有节点
- `content_embedding`: 使用 `all-MiniLM-L6-v2` 模型生成的文本嵌入向量（384维）

### Tool 节点
- `exitcode_status`: "success" 或 "failure"

### Agent 节点
- `is_terminate`: 布尔值，内容中是否包含 "TERMINATE"
- `plan_signal`: 布尔值，内容中是否包含 "Plan" 或 "Step"
- `active_ratio`: 浮点数，节点在历史中的活跃度（出现次数 / 总时间步数）

## 注意事项

1. **首次运行**：首次运行时，`sentence-transformers` 会自动下载模型文件（约 90MB），需要网络连接
2. **内存使用**：对于大型 JSON 文件，解析过程可能占用较多内存
3. **处理时间**：文本嵌入生成可能需要一些时间，特别是对于包含大量文本的事件
4. **批量处理**：处理大量文件时，建议使用 `--quiet` 模式以减少输出

## 错误处理

解析器包含完整的错误处理机制：
- 文件不存在错误
- JSON 格式错误
- 其他运行时错误

所有错误都会显示详细的错误信息，便于调试。批量处理时，即使某些文件处理失败，也会继续处理其他文件。

## 文件结构

```
dhcg_parser/
├── parser.py          # 主解析器代码
└── README.md          # 本说明文档
```

## 许可证

请参考项目根目录的 LICENSE 文件。

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。


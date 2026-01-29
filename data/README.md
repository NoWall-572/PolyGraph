# 数据集说明

本目录包含 ASTRA 项目使用的所有数据集。

## 目录结构

```
data/
├── raw/                    # 原始数据集
│   ├── whowhen/            # Who&When benchmark 原始数据
│   └── tracertraj/         # TracerTraj (AgenTracer) 原始数据
└── processed/              # 处理后的图数据
    ├── whowhen/
    │   └── graphs_whowhen_ollama/  # Who&When 转换后的图数据
    └── tracertraj/
        └── graphs_tracertraj/      # TracerTraj 转换后的图数据
```

## 数据集说明

### Who&When Benchmark

- **原始数据**: `raw/whowhen/`
  - 来源: Who&When benchmark 数据集
  - 格式: JSON 文件
  - 用途: 多智能体系统故障归因基准测试

- **处理后的图数据**: `processed/whowhen/graphs_whowhen_ollama/`
  - 格式: JSON 格式的 HeteroGraph
  - 生成方式: 使用 `astra/parsing/dhcg_parser/` 解析原始数据
  - 用途: GNN 训练和评估

### TracerTraj (AgenTracer)

- **原始数据**: `raw/tracertraj/`
  - 来源: AgenTracer 数据集
  - 格式: CSV/Parquet 文件
  - 用途: 代码生成任务的多智能体交互轨迹

- **处理后的图数据**: `processed/tracertraj/graphs_tracertraj/`
  - 格式: JSON 格式的 HeteroGraph
  - 生成方式: 使用 `scripts/preprocess_external.py` 预处理
  - 用途: GNN 训练和评估

## 使用说明

### 预处理外部数据集

```bash
# 预处理 TracerTraj 数据集
python scripts/preprocess_external.py \
    --input_file data/raw/tracertraj/train-00000-of-00001.csv \
    --output_dir data/processed/tracertraj/graphs_tracertraj \
    --domain "Code" \
    --split_type "train"
```

### 解析 Who&When 数据集

```bash
# 解析 Who&When 数据集
python scripts/parse_dataset.py \
    --input_dir data/raw/whowhen \
    --output_dir data/processed/whowhen/graphs_whowhen_ollama
```

## 注意事项

- 原始数据集文件较大，已通过 `.gitignore` 排除，不会上传到 GitHub
- 处理后的图数据文件较多，建议使用 Git LFS 管理（如果文件超过 100MB）
- 数据集仅供研究使用，请遵守相应的数据使用协议


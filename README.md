# 🕸️ PolyGraph: Autonomous System for Trace Analysis
*(Formerly named as ASTRA in the sourse code)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Framework-PolyGraph-blueviolet)](https://github.com/your-repo)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 💡 **Naming Convention Note**
>     In case of ambiguity
> 📌 **PolyGraph** is the name of our method as presented in the academic paper (previously named **ASTRA** in the code).
>
> 📌 **PolyGen** is our data generation engine (referenced in the codebase as **ASTRA-Gen 3.0**).
>
> ⚠️ *Please note: While the conceptual descriptions below use the new terminology (**PolyGraph/PolyGen**), the source code, file structures, and command lines retain the original `astra` namespace to ensure reproducibility.*

---

## 📖 Abstract

**PolyGraph** is a cutting-edge, end-to-end framework for multi-agent system fault attribution 🕵️‍♂️. It seamlessly combines **Graph Neural Networks (GNN)** 🕸️ and **Large Language Models (LLM)** 🤖 to achieve surgical precision in fault localization within complex multi-agent environments.

The system employs a smart **Coarse-to-Fine** 📉 two-stage strategy:
1.  🚀 **Stage 1:** Using GNN to rapidly identify top-K candidate agents.
2.  🔬 **Stage 2:** Using LLM for fine-grained, reasoning-based analysis.

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🎲 **Dynamic Causal Simulation** | **PolyGen** (ASTRA-Gen 3.0) generates highly realistic multi-agent interaction traces. |
| 🕸️ **DHCG** | **Dynamic Heterogeneous Causal Graph** captures intricate temporal ⏳ and causal 🔗 relationships. |
| 🧠 **PolyGraph-MoE Model** | STGAT-based GNN equipped with a **Mixture of Experts** for robust coarse-grained fault attribution. |
| 🔧 **LLM Fine-tuning** | Specialized **Qwen 8B** model fine-tuned for deep-dive fault analysis. |
| 🎯 **Coarse-to-Fine Eval** | A sophisticated two-stage evaluation system delivering high accuracy. |

## 📂 Project Structure

The codebase retains the `astra` package structure as follows:

```
ASTRA_Release/
├── astra/                      # 📦 Main source code package
│   ├── generation/             # 🏭 Stage 1: Data generation
│   │   └── generator.py        # PolyGen (ASTRA-Gen 3.0) generator
│   ├── parsing/                # 🧩 Stage 1: Graph parsing
│   │   └── dhcg_parser/        # DHCG parser implementation
│   ├── data/                   # 🔄 Stage 2: Data adapter
│   │   ├── adapter.py          # GraphDataConverter
│   │   └── graph_data.py       # HeteroGraph data structure
│   ├── model/                  # 🧠 Stage 3: Model architecture
│   │   ├── gnn.py              # PolyGraph-MoE (ASTRA-MoE) model
│   │   ├── stgat.py            # STGAT implementation
│   │   └── loss.py             # Loss functions
│   ├── training/               # 🏋️ Stage 3 & 4: Training scripts
│   │   ├── train_gnn.py        # GNN training script
│   │   └── prep_llm_data.py    # LLM data preparation
│   └── evaluation/             # 📊 Stage 5: Evaluation
│       ├── eval_pipeline.py    # Coarse-to-fine evaluation
│       └── eval_benchmark.py   # Benchmark evaluation
├── scripts/                    # 🛠️ Utility scripts
│   ├── parse_dataset.py        # Dataset parsing
│   └── preprocess_external.py  # External dataset preprocessing
├── examples/                   # 📝 Sample data
│   ├── golden_sample.json      # ✅ Golden trace example
│   ├── fatal_sample.json       # ❌ Fatal trace example
│   └── healed_sample.json      # 💊 Healed trace example
├── requirements.txt            # 📋 Python dependencies
└── README.md                   # 📄 This file
```

## ⚙️ Installation

### Prerequisites

*   🐍 Python >= 3.8
*   🎮 CUDA >= 11.8 (for GPU acceleration)
*   🧠 16GB+ RAM recommended
*   💾 20GB+ disk space for models and data

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ASTRA_Release
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download pre-trained models (optional):**
    *   📥 **Qwen 8B base model:** Download from HuggingFace or ModelScope.
    *   📍 Place models in the `models/` directory.

## 🚀 Usage

### 🏗️ Stage 1: Data Generation (PolyGen) and Graph Parsing

#### Generate PolyGen Dataset
Run the **PolyGen** (ASTRA-Gen 3.0) engine to create synthetic tasks:

```bash
python -m astra.generation.generator \
    --num_tasks 100 \
    --output_dir outputs/astra_v3 \
    --api_base <your-llm-api-endpoint>
```

#### Parse Traces to Graphs
Convert the raw logs into graph structures:

```bash
python scripts/parse_dataset.py \
    --input_dir outputs/astra_v3 \
    --output_dir processed_graphs/astra_v3
```

### 🔄 Stage 2: Data Conversion

The graph data is **automatically converted** during the training phase. 🧙‍♂️
The `GraphDataConverter` handles:
*   🔹 Node feature extraction and encoding
*   🔹 Edge feature extraction
*   🔹 HeteroGraph sequence construction

### 🧠 Stage 3: GNN Training (PolyGraph-MoE)

Train the coarse-grained expert model:

```bash
python -m astra.training.train_gnn \
    --data_dir processed_graphs/astra_v3 \
    --output_dir checkpoints/astra_moe \
    --epochs 50 \
    --batch_size 8 \
    --device cuda
```

### 🎓 Stage 4: LLM Data Preparation and Fine-tuning

#### Prepare LLM Training Data
Filter data using the GNN checkpoint to create focused samples for the LLM:

```bash
python -m astra.training.prep_llm_data \
    --graph_dir processed_graphs/astra_v3 \
    --gnn_checkpoint checkpoints/astra_moe/best_model.pt \
    --output_dir training_data/llm \
    --top_k 4
```

#### Fine-tune LLM
Use your preferred fine-tuning framework (e.g., PEFT) to train the **PolyGraph** reasoning module:

```bash
# Use your preferred LLM fine-tuning framework
# Example with PEFT:
python -m astra.training.finetune_llm \
    --base_model Qwen/Qwen3-8B \
    --data_dir training_data/llm \
    --output_dir adapters/qwen8b_astra
```

### 📊 Stage 5: Evaluation

#### Coarse-to-Fine Evaluation
Run the full **PolyGraph** pipeline:

```bash
python -m astra.evaluation.eval_pipeline \
    --test_data_dir processed_graphs/test \
    --gnn_checkpoint checkpoints/astra_moe/best_model.pt \
    --llm_adapter adapters/qwen8b_astra \
    --base_model_name Qwen/Qwen3-8B \
    --top_k 4 \
    --device cuda
```

#### Benchmark Evaluation (TracerTraj)

```bash
python -m astra.evaluation.eval_benchmark \
    --test_data_dir processed_graphs/tracertraj \
    --gnn_checkpoint checkpoints/astra_moe/best_model.pt \
    --llm_adapter adapters/qwen8b_astra \
    --base_model_name Qwen/Qwen3-8B \
    --device cuda
```

## 📝 Example Data

The `examples/` directory contains sample data files for quick testing:
*   ✅ `sample_golden.json`: A successful multi-agent interaction trace (no fault).
*   ❌ `sample_fatal.json`: A trace with injected fault.

You can use these to test the parsing and evaluation pipeline. See `examples/README.md` for more details.

## 🧩 Key Components

### PolyGraph-MoE Model (ASTRA-MoE)
The core GNN architecture consists of:
*   **MicroStateEncoder**: 📷 Multi-modal node feature encoder.
*   **STGAT**: 🕸️ Spatio-temporal graph attention network.
*   **TemporalReasoning**: ⏳ Causal temporal reasoning with RoPE.
*   **MoEHead**: 🚦 Uncertainty-aware expert routing.

### DHCG Parser
The parser extracts the **Dynamic Heterogeneous Causal Graph**:
*   **Nodes** 🟣: Agents, Tools, Artifacts, Environment.
*   **Edges** ➖: Invoke, Return, Reference, Communicate, Affect.
*   **Features** 📄: Text embeddings, metadata features.

### Coarse-to-Fine Strategy
1.  **Coarse Stage (GNN)** ⚡: Predicts top-K candidate agents.
2.  **Fine Stage (LLM)** 🔍: Analyzes candidate logs to identify exact fault agent and step.

## 📈 Performance

**PolyGraph** demonstrates state-of-the-art results:

*   🏆 **Agent Accuracy**: ~67.39% on Who&When benchmark, and 77.95% on TracerTraj-Code.
*   🎯 **Step Accuracy**: ~40.22% on Who&When benchmark, and 31.50% on TracerTraj-Code.
*   📉 **Token Efficiency**: Optimized prompt design significantly reduces LLM token usage.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

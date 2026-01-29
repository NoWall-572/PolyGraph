# ASTRA: Autonomous System for TRace Analysis

## Abstract

ASTRA is an end-to-end framework for multi-agent system fault attribution that combines Graph Neural Networks (GNN) and Large Language Models (LLM) to achieve accurate fault localization in complex multi-agent systems. The system employs a **Coarse-to-Fine** two-stage strategy: first using GNN to identify top-K candidate agents, then using LLM for fine-grained analysis.

## Key Features

- **Dynamic Causal Simulation**: ASTRA-Gen 3.0 generates realistic multi-agent interaction traces
- **Dynamic Heterogeneous Causal Graph (DHCG)**: Captures temporal and causal relationships
- **ASTRA-MoE Model**: STGAT-based GNN with Mixture of Experts for coarse-grained fault attribution
- **LLM Fine-tuning**: Qwen 8B model fine-tuned for fine-grained fault analysis
- **Coarse-to-Fine Evaluation**: Two-stage evaluation system with high accuracy

## Project Structure

```
ASTRA_Release/
├── astra/                      # Main source code package
│   ├── generation/             # Stage 1: Data generation
│   │   └── generator.py        # ASTRA-Gen 3.0 data generator
│   ├── parsing/                # Stage 1: Graph parsing
│   │   └── dhcg_parser/        # DHCG parser implementation
│   ├── data/                   # Stage 2: Data adapter
│   │   ├── adapter.py          # GraphDataConverter
│   │   └── graph_data.py       # HeteroGraph data structure
│   ├── model/                  # Stage 3: Model architecture
│   │   ├── gnn.py              # ASTRA-MoE GNN model
│   │   ├── stgat.py            # STGAT implementation
│   │   └── loss.py             # Loss functions
│   ├── training/               # Stage 3 & 4: Training scripts
│   │   ├── train_gnn.py        # GNN training script
│   │   └── prep_llm_data.py   # LLM data preparation
│   └── evaluation/             # Stage 5: Evaluation
│       ├── eval_pipeline.py    # Coarse-to-fine evaluation
│       └── eval_benchmark.py    # Benchmark evaluation
├── scripts/                    # Utility scripts
│   ├── parse_dataset.py        # Dataset parsing
│   └── preprocess_external.py  # External dataset preprocessing
├── examples/                    # Sample data
│   ├── golden_sample.json      # Golden trace example
│   ├── fatal_sample.json       # Fatal trace example
│   └── healed_sample.json      # Healed trace example
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (for GPU acceleration)
- 16GB+ RAM recommended
- 20GB+ disk space for models and data

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ASTRA_Release
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional):
- Qwen 8B base model: Download from HuggingFace or ModelScope
- Place models in `models/` directory

## Usage

### Stage 1: Data Generation and Graph Parsing

#### Generate ASTRA Dataset

```bash
python -m astra.generation.generator \
    --num_tasks 100 \
    --output_dir outputs/astra_v3 \
    --api_base <your-llm-api-endpoint>
```

#### Parse Traces to Graphs

```bash
python scripts/parse_dataset.py \
    --input_dir outputs/astra_v3 \
    --output_dir processed_graphs/astra_v3
```

### Stage 2: Data Conversion

The graph data is automatically converted during training. The `GraphDataConverter` handles:
- Node feature extraction and encoding
- Edge feature extraction
- HeteroGraph sequence construction

### Stage 3: GNN Training

```bash
python -m astra.training.train_gnn \
    --data_dir processed_graphs/astra_v3 \
    --output_dir checkpoints/astra_moe \
    --epochs 50 \
    --batch_size 8 \
    --device cuda
```

### Stage 4: LLM Data Preparation and Fine-tuning

#### Prepare LLM Training Data

```bash
python -m astra.training.prep_llm_data \
    --graph_dir processed_graphs/astra_v3 \
    --gnn_checkpoint checkpoints/astra_moe/best_model.pt \
    --output_dir training_data/llm \
    --top_k 4
```

#### Fine-tune LLM (using external tools like PEFT)

```bash
# Use your preferred LLM fine-tuning framework
# Example with PEFT:
python -m astra.training.finetune_llm \
    --base_model Qwen/Qwen3-8B \
    --data_dir training_data/llm \
    --output_dir adapters/qwen8b_astra
```

### Stage 5: Evaluation

#### Coarse-to-Fine Evaluation

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

## Example Data

The `examples/` directory contains sample data files:
- `sample_golden.json`: A successful multi-agent interaction trace (no fault)
- `sample_fatal.json`: A trace with injected fault

You can use these to test the parsing and evaluation pipeline. See `examples/README.md` for more details.

## Key Components

### ASTRA-MoE Model

The ASTRA-MoE (Mixture of Experts) model consists of:
- **MicroStateEncoder**: Multi-modal node feature encoder
- **STGAT**: Spatio-temporal graph attention network
- **TemporalReasoning**: Causal temporal reasoning with RoPE
- **MoEHead**: Uncertainty-aware expert routing

### DHCG Parser

The DHCG parser extracts:
- **Nodes**: Agents, Tools, Artifacts, Environment
- **Edges**: Invoke, Return, Reference, Communicate, Affect
- **Features**: Text embeddings, metadata features

### Coarse-to-Fine Strategy

1. **Coarse Stage (GNN)**: Predicts top-K candidate agents
2. **Fine Stage (LLM)**: Analyzes candidate logs to identify exact fault agent and step

## Performance

- **Agent Accuracy**: ~97% on ASTRA test set
- **Step Accuracy**: ~85% on ASTRA test set
- **Token Efficiency**: Optimized prompt design reduces LLM token usage

## Citation

If you use ASTRA in your research, please cite:

```bibtex
@article{astra2024,
  title={ASTRA: Autonomous System for TRace Analysis},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open an issue on GitHub or contact [your-email].

## Acknowledgments

- Qwen team for the base LLM model
- PyTorch Geometric for graph neural network utilities
- HuggingFace for transformers and PEFT libraries


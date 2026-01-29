# Example Data

This directory contains sample data files for testing the ASTRA pipeline.

## Files

- `sample_golden_graph.json`: A successful multi-agent interaction trace (no fault)
- `sample_fatal_graph.json`: A trace with an injected fault

## Data Format

Each JSON file contains:
- `question`: The task/question description
- `ground_truth`: Ground truth labels
  - `mistake_agent`: The agent that caused the fault (null for golden traces)
  - `mistake_step`: The step where the fault occurred (null for golden traces)
  - `mistake_reason`: Description of the fault (null for golden traces)
- `nodes`: Dictionary of graph nodes (Agents, Tools, Artifacts, Environment)
- `edges`: List of graph edges connecting nodes

## Usage

You can use these files to test:
1. Graph parsing: `python scripts/parse_dataset.py --input_dir examples`
2. Data conversion: The converter will process these during training
3. Evaluation: Use these in the evaluation pipeline

## Note

These are simplified examples. Real datasets contain more nodes and edges, and include full feature embeddings.


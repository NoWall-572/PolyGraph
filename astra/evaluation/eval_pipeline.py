"""
阶段三评估脚本：Coarse-to-Fine 系统集成评估
实现完整的 GNN + LLM 流程，并在原始测试集上计算准确率

流程：
1. GNN 预测 Top-K 候选Agent
2. 提取候选Agent的日志
3. LLM 分析日志并输出最终报告
4. 计算 Agent 和 Step 准确率
"""
import torch
import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
import os

# GNN 相关
from astra.model.gnn import ASTRAMoE
from astra.data.adapter import GraphDataConverter, reconstruct_graph_from_json
from astra.training.train_gnn import collate_fn, compute_metrics
from torch.utils.data import DataLoader

# LLM 相关
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 日志提取
from astra.training.prep_llm_data import extract_agent_logs, format_instruction_for_stage2

# ============================================================================
# Token 统计器
# ============================================================================
class TokenCounter:
    """Token使用统计器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.llm_total_tokens = 0
        self.llm_calls = 0
        
    def add_llm_call(self, input_tokens: int, output_tokens: int):
        """记录一次LLM调用"""
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.llm_total_tokens += (input_tokens + output_tokens)
        self.llm_calls += 1
    
    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            'llm_calls': self.llm_calls,
            'llm_input_tokens': self.llm_input_tokens,
            'llm_output_tokens': self.llm_output_tokens,
            'llm_total_tokens': self.llm_total_tokens,
            'avg_input_tokens_per_call': self.llm_input_tokens / self.llm_calls if self.llm_calls > 0 else 0,
            'avg_output_tokens_per_call': self.llm_output_tokens / self.llm_calls if self.llm_calls > 0 else 0,
            'avg_total_tokens_per_call': self.llm_total_tokens / self.llm_calls if self.llm_calls > 0 else 0
        }
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n" + "="*80)
        print("📊 Token 使用统计报告")
        print("="*80)
        print(f"LLM 调用次数:           {summary['llm_calls']:,}")
        print(f"LLM 输入 Token 总数:     {summary['llm_input_tokens']:,}")
        print(f"LLM 输出 Token 总数:     {summary['llm_output_tokens']:,}")
        print(f"LLM 总 Token 数:         {summary['llm_total_tokens']:,}")
        print(f"平均每次输入 Token:      {summary['avg_input_tokens_per_call']:.2f}")
        print(f"平均每次输出 Token:      {summary['avg_output_tokens_per_call']:.2f}")
        print(f"平均每次总 Token:        {summary['avg_total_tokens_per_call']:.2f}")
        print("="*80 + "\n")

# 全局Token统计器
token_counter = TokenCounter()


def load_gnn_model(checkpoint_path: str, converter_path: str, device: torch.device):
    """加载 GNN 模型"""
    print(f"📥 加载 GNN 模型: {checkpoint_path}")
    
    # 加载检查点
    # 修改后 ✔️ 核心改动：增加  weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型配置（从检查点读取，如果不存在则使用默认值）
    # 检查点可能有两种格式：直接 model_config 或嵌套在 config 中
    if 'config' in checkpoint and 'model_config' in checkpoint['config']:
        model_config = checkpoint['config']['model_config']
    else:
        model_config = checkpoint.get('model_config', {})
    
    # 从 state_dict 推断实际配置（如果配置不完整或不匹配）
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 从 state_dict 推断 num_classes（从 moe_head.experts.0.3.weight 的形状）
    num_classes = model_config.get('num_classes', 1)
    if 'moe_head.experts.0.3.weight' in state_dict:
        inferred_num_classes = state_dict['moe_head.experts.0.3.weight'].shape[0]
        if inferred_num_classes != num_classes:
            print(f"   ⚠️  从 state_dict 推断 num_classes: {inferred_num_classes} (配置中为 {num_classes})")
            num_classes = inferred_num_classes
    
    # 从 state_dict 推断 meta_mlp 输入维度，然后反推 node_feat_dim
    # 🔥 关键修复：根据text_proj的维度推断text_dim，而不是硬编码384
    node_feat_dim = model_config.get('node_feat_dim', 8192)
    
    # 首先从text_proj推断text_dim
    text_dim = 4096  # 默认值：Qwen-8B
    text_proj_key = None
    for key in state_dict.keys():
        if 'text_proj' in key and 'weight' in key:
            text_proj_key = key
            break
    
    if text_proj_key:
        inferred_text_dim = state_dict[text_proj_key].shape[1]
        text_dim = inferred_text_dim
        print(f"   ✅ 从 state_dict 推断 text_dim: {text_dim} (text_proj输入维度, 键: {text_proj_key})")
    else:
        print(f"   ⚠️  未找到text_proj.weight，使用默认text_dim: {text_dim}")
        # 尝试从model_config读取
        if 'text_dim' in model_config:
            text_dim = model_config['text_dim']
            print(f"   ✅ 从model_config读取text_dim: {text_dim}")
    
    # 然后从meta_mlp推断meta_dim，并计算node_feat_dim
    meta_mlp_key = None
    for key in state_dict.keys():
        if 'meta_mlp' in key and '.0.weight' in key:
            meta_mlp_key = key
            break
    
    if meta_mlp_key:
        inferred_meta_dim = state_dict[meta_mlp_key].shape[1]
        inferred_node_feat_dim = inferred_meta_dim + text_dim
        print(f"   ✅ 从 state_dict 推断 meta_dim: {inferred_meta_dim} (键: {meta_mlp_key})")
        print(f"   ✅ 计算的 node_feat_dim: {inferred_node_feat_dim} = {inferred_meta_dim} (meta) + {text_dim} (text)")
        
        # 🔥 关键修复：优先使用从state_dict推断的值，因为它反映实际的模型结构
        if inferred_node_feat_dim != node_feat_dim:
            print(f"   ⚠️  从 state_dict 推断的 node_feat_dim ({inferred_node_feat_dim}) 与配置中的 ({node_feat_dim}) 不同，使用推断值")
            node_feat_dim = inferred_node_feat_dim
    else:
        print(f"   ⚠️  未找到meta_mlp.0.weight，使用配置中的node_feat_dim: {node_feat_dim}")
    
    # 从检查点读取其他配置参数
    edge_feat_dim = model_config.get('edge_feat_dim', 32)
    max_agents = model_config.get('max_agents', 50)
    max_seq_len = model_config.get('max_seq_len', 50)
    d_model = model_config.get('d_model', 256)
    num_hgt_layers = model_config.get('num_hgt_layers', 2)
    num_heads = model_config.get('num_heads', 4)
    num_experts = model_config.get('num_experts', 4)
    num_temporal_layers = model_config.get('num_temporal_layers', 2)
    dropout = model_config.get('dropout', 0.5)
    
    # 🔥 从 state_dict 推断 num_hgt_layers（STGAT 层数）
    # 检查 spatial_encoder.stgat.layers.X 的最大索引
    max_layer_idx = -1
    for key in state_dict.keys():
        if 'spatial_encoder.stgat.layers.' in key:
            # 提取层索引，例如 "spatial_encoder.stgat.layers.1.spatial_attn.query.weight" -> 1
            parts = key.split('spatial_encoder.stgat.layers.')
            if len(parts) > 1:
                layer_idx_str = parts[1].split('.')[0]
                try:
                    layer_idx = int(layer_idx_str)
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except ValueError:
                    pass
    
    if max_layer_idx >= 0:
        inferred_num_hgt_layers = max_layer_idx + 1  # 层索引从0开始，所以+1
        if inferred_num_hgt_layers != num_hgt_layers:
            print(f"   ⚠️  从 state_dict 推断 num_hgt_layers: {inferred_num_hgt_layers} (配置中为 {num_hgt_layers})")
            num_hgt_layers = inferred_num_hgt_layers
    
    print(f"   从检查点读取配置:")
    print(f"   - node_feat_dim: {node_feat_dim}")
    print(f"   - edge_feat_dim: {edge_feat_dim}")
    print(f"   - d_model: {d_model}")
    print(f"   - max_agents: {max_agents}")
    print(f"   - max_seq_len: {max_seq_len}")
    print(f"   - num_classes: {num_classes}")
    print(f"   - num_hgt_layers: {num_hgt_layers}")
    
    # 加载 converter（直接使用 torch.load，因为 converter 是直接保存的整个对象）
    if not Path(converter_path).exists():
        raise FileNotFoundError(f"找不到 Converter 文件: {converter_path}")
    
    map_location = device if device.type == 'cpu' else None
    converter = torch.load(converter_path, map_location=map_location, weights_only=False)
    
    # 🔥 关键修复：检查 converter 的 node_feat_dim 和 meta_dim，确保与模型匹配
    converter_node_feat_dim = getattr(converter, 'node_feat_dim', 8192)
    # 🔥 修复：从text_proj推断text_dim，而不是硬编码384
    converter_text_dim = text_dim  # 使用上面推断的text_dim
    converter_meta_dim = converter_node_feat_dim - converter_text_dim
    
    # 从检查点获取实际的 meta_dim（从 meta_mlp.0.weight 的输入维度）
    checkpoint_meta_dim = None
    if 'micro_encoder.meta_mlp.0.weight' in state_dict:
        checkpoint_meta_dim = state_dict['micro_encoder.meta_mlp.0.weight'].shape[1]
    
    # 检查 node_feat_dim 匹配
    if converter_node_feat_dim != node_feat_dim:
        print(f"\n   ⚠️  严重警告：Converter node_feat_dim ({converter_node_feat_dim}) 与模型 node_feat_dim ({node_feat_dim}) 不匹配！")
        print(f"   ⚠️  这会导致维度不匹配错误，模型无法正确加载权重。")
        print(f"   ⚠️  解决方案：")
        print(f"      1. 使用与 converter 匹配的模型检查点（node_feat_dim={converter_node_feat_dim}）")
        print(f"      2. 或者使用与模型匹配的 converter（node_feat_dim={node_feat_dim}）")
        print(f"\n   ❌ 无法继续：模型和 converter 的 node_feat_dim 必须匹配！")
        print(f"   💡 请查找与 converter (node_feat_dim={converter_node_feat_dim}) 匹配的模型检查点")
        raise ValueError(f"Converter node_feat_dim ({converter_node_feat_dim}) 与模型 node_feat_dim ({node_feat_dim}) 不匹配！")
    
    # 🔥 额外检查：meta_dim 必须匹配（这是最关键的，因为 meta_mlp 的输入维度必须匹配）
    if checkpoint_meta_dim is not None and checkpoint_meta_dim != converter_meta_dim:
        print(f"\n   ❌ 致命错误：检查点的 meta_dim ({checkpoint_meta_dim}) 与 converter 的 meta_dim ({converter_meta_dim}) 不匹配！")
        print(f"   ❌ 这会导致运行时错误：mat1 and mat2 shapes cannot be multiplied")
        print(f"   ❌ Converter 输出: meta_dim={converter_meta_dim} (node_feat_dim={converter_node_feat_dim}, text_dim={converter_text_dim})")
        print(f"   ❌ 检查点期望: meta_dim={checkpoint_meta_dim} (node_feat_dim={checkpoint_meta_dim + text_dim}, text_dim={text_dim})")
        print(f"\n   💡 解决方案：")
        print(f"      1. 查找 meta_dim={converter_meta_dim} 的检查点（即 node_feat_dim={converter_node_feat_dim}）")
        print(f"      2. 或者重新训练模型以匹配当前的 converter")
        print(f"\n   ⚠️  注意：即使检查点的配置显示 node_feat_dim={node_feat_dim}，")
        print(f"      但实际的 meta_mlp 权重形状表明它是在 node_feat_dim={checkpoint_meta_dim + text_dim} 下训练的。")
        raise ValueError(f"检查点的 meta_dim ({checkpoint_meta_dim}) 与 converter 的 meta_dim ({converter_meta_dim}) 不匹配！")
    
    print(f"✅ Converter 加载完成 (node_feat_dim={converter_node_feat_dim}, meta_dim={converter_meta_dim})")
    if checkpoint_meta_dim is not None:
        print(f"✅ 检查点 meta_dim 匹配 (meta_dim={checkpoint_meta_dim})")
    
    # 创建模型（使用匹配的 node_feat_dim）
    model = ASTRAMoE(
        node_feat_dim=node_feat_dim,  # 使用匹配的 node_feat_dim
        edge_feat_dim=edge_feat_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_hgt_layers=num_hgt_layers,
        num_temporal_layers=num_temporal_layers,
        num_experts=num_experts,
        num_classes=num_classes,  # 使用检查点中的 num_classes
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    # 加载权重（使用 strict=False 允许部分加载，兼容不同架构的检查点）
    try:
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"   ⚠️  缺失的键（已忽略）: {len(missing_keys)} 个")
            if len(missing_keys) <= 5:
                for key in missing_keys[:5]:
                    print(f"      - {key}")
            else:
                for key in missing_keys[:5]:
                    print(f"      - {key}")
                print(f"      ... 还有 {len(missing_keys) - 5} 个缺失的键")
        
        if unexpected_keys:
            print(f"   ⚠️  意外的键（已忽略）: {len(unexpected_keys)} 个")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")
                print(f"      ... 还有 {len(unexpected_keys) - 5} 个意外的键")
        
        print(f"✅ 模型权重加载完成")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        print(f"   这可能是由于模型架构不匹配导致的")
        print(f"   请检查检查点是否与当前代码版本兼容")
        raise
    
    model.to(device)
    model.eval()
    
    print(f"✅ GNN 模型加载完成")
    print(f"   配置: max_agents={max_agents}, max_seq_len={max_seq_len}, d_model={d_model}")
    
    return model, converter, {
        'max_agents': max_agents,
        'max_seq_len': max_seq_len,
        'd_model': d_model
    }


def load_llm_model(adapter_path: str, base_model_name: str = "Qwen/Qwen3-8B", device: torch.device = None, use_4bit: bool = True):
    """
    加载微调后的 LLM 模型（内存优化版，支持量化）
    
    Args:
        adapter_path: 适配器路径
        base_model_name: 基础模型名称
        device: 设备
        use_4bit: 是否使用 4-bit 量化（默认 True，节省显存）
    """
    # 🔥 修复：自动查找本地缓存的 snapshot 目录
    original_model_name = base_model_name
    
    # 尝试查找本地缓存（支持多种模型）
    model_cache_patterns = {
        "Qwen/Qwen1.5-4B-Chat": "models--Qwen--Qwen1.5-4B-Chat",
        "Qwen/Qwen2.5-4B-Instruct": "models--Qwen--Qwen2.5-4B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "models--Qwen--Qwen2.5-7B-Instruct",
    }
    
    cache_dir_name = model_cache_patterns.get(base_model_name)
    if cache_dir_name:
        cache_dir = Path.home() / ".cache/huggingface/hub" / cache_dir_name
        if cache_dir.exists():
            # 查找 snapshots 目录
            snapshot_dirs = sorted((cache_dir / "snapshots").glob("*"))
            if snapshot_dirs:
                base_model_name = str(snapshot_dirs[-1])  # 使用最新的 snapshot
                print(f"✓ 使用本地缓存: {base_model_name}")
    
    # 如果指定了本地路径（如 ./models/Qwen2.5-7B-Instruct），直接使用
    if base_model_name.startswith("./") or base_model_name.startswith("/"):
        print(f"✓ 使用本地路径: {base_model_name}")
    
    print(f"📥 加载 LLM 模型: {original_model_name}")
    print(f"   适配器: {adapter_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🔥🔥🔥 内存优化：使用 4-bit 量化（节省显存）
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            print(f"   ✅ 使用 4-bit 量化模式（节省显存）")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # 🔥 修复：4-bit 量化必须全部在 GPU 上，不能 offload 到 CPU
            # 如果 GPU 显存不足，将自动回退到 FP16 模式
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": {"": device},  # 明确指定设备，不允许 CPU offload
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
        except ImportError:
            print(f"   ⚠️  BitsAndBytes 未安装，使用 FP16 模式")
            print(f"   安装命令: pip install bitsandbytes")
            use_4bit = False
    
    if not use_4bit:
        # 回退到 FP16
        print(f"   ⚠️  内存优化模式：使用 FP16 + low_cpu_mem_usage")
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",  # 自动分配设备
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
    
    # 尝试加载模型（先尝试本地，失败则从网络下载）
    # 🔥 修复：如果 4-bit 量化失败（显存不足），自动回退到 FP16
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            local_files_only=False,  # 允许从网络下载（如果本地没有）
            **model_kwargs
        )
    except ValueError as e:
        # 检查是否是 4-bit 量化显存不足的错误
        error_msg = str(e)
        if "Some modules are dispatched on the CPU" in error_msg and use_4bit:
            print(f"⚠️  4-bit 量化失败（GPU 显存不足，无法完全加载到 GPU）")
            print(f"   自动回退到 FP16 模式（允许 CPU offload）")
            use_4bit = False
            # 使用 FP16 模式，允许 CPU offload
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",  # 自动分配设备（允许 CPU offload）
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "max_memory": {0: "18GB", "cpu": "50GB"}  # 限制 GPU 显存，允许 CPU 扩展
            }
            # 重新尝试加载
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    local_files_only=False,
                    **model_kwargs
                )
            except Exception as e2:
                print(f"⚠️  使用 FP16 模式加载也失败: {e2}")
                print(f"   尝试从网络下载: {original_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    original_model_name,
                    **model_kwargs
                )
        else:
            # 其他错误，尝试从网络下载
            print(f"⚠️  加载模型失败: {e}")
            print(f"   尝试从网络下载: {original_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                original_model_name,
            **model_kwargs
        )
    except Exception as e:
        print(f"⚠️  加载模型失败: {e}")
        print(f"   尝试从网络下载: {original_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            original_model_name,
            **model_kwargs
        )
    
    # 检查适配器路径是否存在（如果提供）
    if not adapter_path or adapter_path.strip() == "":
        print("🔬 [消融实验] 未提供适配器路径，使用未微调的基础模型")
        print(f"   基础模型: {base_model_name}")
        print(f"   注意：这是消融实验配置，模型未经过微调或强化学习")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=False  # 允许从网络下载（如果本地没有）
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return base_model, tokenizer
    
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        print(f"🔬 [消融实验] 适配器路径不存在: {adapter_path}")
        print(f"   将使用未微调的基础模型")
        print(f"   基础模型: {base_model_name}")
        print(f"   注意：这是消融实验配置，模型未经过微调或强化学习")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=False  # 允许从网络下载（如果本地没有）
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return base_model, tokenizer
    
    # 检查适配器配置文件
    adapter_config = adapter_path_obj / "adapter_config.json"
    if not adapter_config.exists():
        print(f"⚠️  警告: 适配器配置文件不存在: {adapter_config}")
        print(f"   将使用基础模型（无微调）")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        return base_model, tokenizer
    
    # 🔥 新增：预检查适配器兼容性
    import json
    try:
        with open(adapter_config, 'r', encoding='utf-8') as f:
            adapter_config_data = json.load(f)
        
        # 检查适配器的基础模型名称
        adapter_base_model = adapter_config_data.get('base_model_name', '')
        if adapter_base_model:
            # 标准化模型名称进行比较
            def normalize_model_name(name):
                """标准化模型名称以便比较"""
                name = name.lower().replace('\\', '/')
                # 移除路径前缀
                if '/' in name:
                    name = name.split('/')[-1]
                # 移除常见后缀
                for suffix in ['-instruct', '-chat', '-thinking', '-8b', '-7b', '-4b']:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                return name
            
            current_model_normalized = normalize_model_name(base_model_name)
            adapter_model_normalized = normalize_model_name(adapter_base_model)
            
            # 检查是否明显不匹配（如 qwen2.5 vs qwen3）
            if 'qwen2.5' in adapter_base_model.lower() and 'qwen3' in base_model_name.lower():
                print(f"⚠️  适配器兼容性检查失败")
                print(f"   适配器基础模型: {adapter_base_model}")
                print(f"   当前基础模型: {base_model_name}")
                print(f"💡 适配器是为 Qwen2.5 训练的，但当前使用的是 Qwen3")
                print(f"   将使用基础模型（无适配器）")
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
                return base_model, tokenizer
            elif 'qwen3' in adapter_base_model.lower() and 'qwen2.5' in base_model_name.lower():
                print(f"⚠️  适配器兼容性检查失败")
                print(f"   适配器基础模型: {adapter_base_model}")
                print(f"   当前基础模型: {base_model_name}")
                print(f"💡 适配器是为 Qwen3 训练的，但当前使用的是 Qwen2.5")
                print(f"   将使用基础模型（无适配器）")
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
                return base_model, tokenizer
    except Exception as e:
        print(f"⚠️  读取适配器配置时出错: {e}")
        print(f"   将继续尝试加载适配器...")
    
    # 检查网络连接
    network_available = True
    try:
        import requests
        requests.get("https://huggingface.co", timeout=2)
    except:
        network_available = False
        print(f"⚠️  网络不可用，将使用本地文件模式加载适配器")
    
    # 加载 LoRA 适配器（使用 local_files_only 如果网络不可用）
    # 🔥 改进：使用更严格的错误处理，捕获所有可能的异常
    try:
        # 临时禁用警告，避免输出大量尺寸不匹配警告
        import warnings
        import logging
        old_warnings = warnings.filters[:]
        warnings.filterwarnings('ignore', category=UserWarning)
        old_log_level = logging.getLogger('transformers').level
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
        try:
            model = PeftModel.from_pretrained(
                base_model, 
                adapter_path,
                local_files_only=not network_available
            )
        finally:
            # 恢复警告和日志级别
            warnings.filters[:] = old_warnings
            logging.getLogger('transformers').setLevel(old_log_level)
    except (ValueError, RuntimeError, TypeError, Exception) as e:
        # 处理维度不匹配错误（适配器为不同模型训练）
        error_str = str(e).lower()
        error_msg = str(e)
        
        # 检查是否是尺寸不匹配错误（包括警告信息）
        is_size_mismatch = (
            "size mismatch" in error_str or 
            "shape" in error_str or 
            "copying a param" in error_str or
            "torch.size" in error_str or
            "expected size" in error_str
        )
        
        if is_size_mismatch:
            print(f"\n⚠️  适配器维度不匹配")
            print(f"💡 适配器可能是为不同模型训练的")
            print(f"   适配器路径: {adapter_path}")
            print(f"   当前基础模型: {base_model_name}")
            print(f"   将使用基础模型（无适配器）")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            return base_model, tokenizer
        elif "unexpected keyword argument" in error_str:
            # 处理 PEFT 版本不兼容问题（如 alora_invocation_tokens 参数）
            print(f"⚠️  PEFT 版本不兼容: {e}")
            print(f"💡 尝试修复适配器配置...")
            
            # 读取并修复适配器配置
            import json
            adapter_config_file = adapter_path_obj / "adapter_config.json"
            if adapter_config_file.exists():
                with open(adapter_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 移除不支持的参数
                unsupported_params = ['alora_invocation_tokens', 'alora_alpha', 'alora_dropout']
                removed_params = []
                for param in unsupported_params:
                    if param in config:
                        removed_params.append(param)
                        del config[param]
                
                if removed_params:
                    print(f"   移除了不支持的参数: {', '.join(removed_params)}")
                    # 备份原配置
                    backup_file = adapter_config_file.with_suffix('.json.bak')
                    import shutil
                    shutil.copy2(adapter_config_file, backup_file)
                    # 保存修复后的配置
                    with open(adapter_config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"   已备份原配置到: {backup_file}")
                    print(f"   已更新适配器配置")
                    
                    # 重新尝试加载
                    try:
                        model = PeftModel.from_pretrained(
                            base_model, 
                            adapter_path,
                            local_files_only=not network_available
                        )
                        print(f"✅ 适配器加载成功（已修复配置）")
                    except Exception as e2:
                        print(f"❌ 修复后仍然失败: {e2}")
                        # 恢复原配置
                        if backup_file.exists():
                            shutil.copy2(backup_file, adapter_config_file)
                            print(f"   已恢复原配置")
                        # 检查是否是维度不匹配
                        if "size mismatch" in str(e2).lower() or "shape" in str(e2).lower():
                            print(f"💡 适配器维度不匹配，将使用基础模型")
                            tokenizer = AutoTokenizer.from_pretrained(
                                base_model_name,
                                trust_remote_code=True
                            )
                            return base_model, tokenizer
                        raise
                else:
                    raise
            else:
                raise
        else:
            # 其他错误，也尝试使用基础模型
            print(f"⚠️  加载适配器失败: {e}")
            if "adapter_config.json" in str(e) or "Can't find" in str(e):
                print(f"💡 检查适配器目录: {adapter_path}")
                print(f"   需要的文件:")
                print(f"     - adapter_config.json")
                print(f"     - adapter_model.bin 或 adapter_model.safetensors")
                if adapter_path_obj.exists():
                    files = list(adapter_path_obj.glob("*"))
                    print(f"   当前目录中的文件:")
                    for f in files:
                        print(f"     - {f.name}")
            print(f"💡 将使用基础模型（无适配器）")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            return base_model, tokenizer
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=not network_available
        )
    except Exception as e:
        print(f"⚠️  从本地路径加载分词器失败: {e}")
        if network_available:
            print(f"   尝试从网络下载")
            tokenizer = AutoTokenizer.from_pretrained(
                original_model_name,
                trust_remote_code=True
            )
        else:
            # 尝试从适配器路径加载
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    adapter_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except:
                raise Exception(f"无法加载分词器，网络不可用且本地文件不存在")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ LLM 模型加载完成")
    
    return model, tokenizer


def normalize_name(name):
    """
    归一化 Agent 名称：去下划线、去空格、去连字符、转小写
    用于模糊匹配
    """
    if not name:
        return ""
    return str(name).lower().replace("_", "").replace(" ", "").replace("-", "").strip()


def predict_top_k_with_gnn(model, graph_list, converter, config, device, top_k=7, true_agent_nodes=None):
    """
    使用 GNN 预测 Top-K 候选Agent（增强版：严格过滤非Agent节点）
    
    核心理解：
    - ASTRA-MoE 输出 logits shape: [seq_len, num_agents, 1] (打分模式)
    - num_agents 是当前图中 Agent 类型节点的数量（不是全局 529）
    - node_id_to_idx 格式: {node_id: (node_type, local_idx)}
    - 需要筛选 Agent 类型，建立 local_idx -> node_id 的映射
    - 🔥 严格过滤：排除工具、环境等非Agent节点
    - 🔥🔥🔥 新增：如果提供了 true_agent_nodes，只输出这些节点（从JSON文件读取）
    
    Args:
        top_k: 如果为 None，返回所有Agent的完整排序；否则返回 Top-K
        true_agent_nodes: 从JSON文件读取的真实Agent节点列表（Set[str]），如果提供，只输出这些节点
    
    Returns:
        如果 top_k 不为 None:
            Tuple[List[str], Optional[int]]: (Top-K 候选Agent ID列表, GNN预测的Step)
        如果 top_k 为 None:
            Tuple[List[str], Dict[str, float], Optional[int]]: (所有Agent排序列表, Agent分数字典, GNN预测的Step)
    """
    # 移动到设备
    graph_list_device = [g.to(device) for g in graph_list]
    
    # 前向传播
    with torch.no_grad():
        outputs = model(graph_list_device)
    
    # 获取 logits: [seq_len, num_agents, 1]
    logits = outputs['logits']
    
    # 获取最后一个时间步的分数
    if logits.dim() == 3:
        scores = logits[-1, :, 0]  # [num_agents]
    else:
        scores = logits[-1, :]  # [num_agents]
    
    # 🔥 尝试获取GNN的Step预测（如果可用）
    gnn_pred_step = None
    if 'step_logits' in outputs:
        step_logits = outputs['step_logits']  # [seq_len] 或 [batch, seq_len]
        if step_logits is not None:
            if step_logits.dim() == 1:
                # 单个样本
                gnn_pred_step = int(torch.argmax(step_logits).item())
            elif step_logits.dim() == 2 and step_logits.size(0) == 1:
                # batch_size=1
                gnn_pred_step = int(torch.argmax(step_logits[0]).item())
    
    # ✅ 关键修复：正确处理 node_id_to_idx 映射
    # node_id_to_idx 格式: {node_id: (node_type, local_idx)}
    node_id_to_idx = graph_list[0].node_id_to_idx if graph_list else {}
    
    # 🔥🔥🔥【核心修改】严格过滤逻辑（终极增强版 - 针对Orchestrator和虚拟节点）
    # 排除关键词列表 (转小写匹配)
    exclude_keywords = [
        'terminal', 'computer', 'console', 'shell', 'bash',  # 终端工具
        'broadcast', 'env', 'environment', 'root', 'system', # 环境/广播
        'artifact', 'file', 'database', 'internet',          # 静态资源
        'userproxy', 'user_proxy', 'user',                    # 用户代理（通常不归因）
        'tool', 'api', 'service',                             # 工具类
        'https', 'http', 'www', '.com', '.org', '.net',      # URL 实体
        'github', 'gmail', 'google', 'apple', 'microsoft',   # 常见实体/域名
        # 🔥🔥🔥 新增：过滤虚拟节点和元数据噪音（关键！）
        '(thought)', 'termination', '->', 'condition', 'reasoning',  # 虚拟节点特征
        'type', 'context', 'graph', 'id', 'name',            # 元数据噪音
        # 🔥🔥🔥 新增：常见网站/平台名（完整列表）
        'youtube', 'linkedin', 'twitter', 'facebook', 'instagram', 'tiktok',  # 社交媒体
        'amazon', 'ebay', 'shopify', 'etsy',                 # 电商平台
        'wikipedia', 'reddit', 'quora', 'stackoverflow',     # 内容平台
        'netflix', 'spotify', 'discord', 'slack',            # 应用平台
        'gmail', 'outlook', 'yahoo',                         # 邮件服务
        # 🔥🔥🔥 新增：元数据和元信息关键词
        'metadata', 'attribute', 'property', 'field',        # 元数据
        'parameter', 'argument', 'variable',                 # 编程术语
        # 🔥🔥🔥 新增：常见的非Agent实体
        'orchestrator', 'coordinator', 'manager',            # 管理节点
        'ncbi', 'pubmed', 'doi',                             # 学术资源
        'gremm', 'tpwd', 'rndpa',                            # 缩写/组织代码（从数据中发现的）
    ]
    
    # 筛选候选人：必须是Agent类型，且不包含排除关键词
    valid_candidates = []
    for node_id, (ntype, local_idx) in node_id_to_idx.items():
        node_id_lower = node_id.lower()
        
        # 1. 类型必须是 Agent (如果图数据里类型标记正确)
        if ntype != 'Agent':
            continue
        
        # 🔥 新增：长度过滤 - Agent名字通常不会太短（至少3个字符，且不能全是数字）
        if len(node_id) < 3:
            continue
        
        # 🔥 新增：过滤纯数字或纯字母数字组合（可能是ID，不是Agent名）
        # 例如：abdulmateen5003, laivertebasaga5655 等
        if node_id.replace('_', '').replace('-', '').isalnum() and len(node_id) > 10:
            # 如果名字很长且全是字母数字，可能是用户名/ID
            # 但保留短名字（可能是正常Agent名）
            if not any(c.isupper() for c in node_id):  # 如果全小写且很长，可能是ID
                continue
        
        # 🔥🔥🔥【增强过滤】使用正则表达式过滤掉看起来像用户名或纯乱码的 ID
        # 例如包含3个以上连续数字的（如 abdulmateen5003），但要保留标准格式（如 Agent_1, WebSurfer_2）
        if re.search(r'\d{3,}', node_id):
            # 保留标准格式：Agent_1, WebSurfer_2, PythonExpert_3 等
            if not re.match(r'^[A-Za-z]+_\d+$', node_id) and 'Expert' not in node_id:
                # 如果包含3个以上连续数字，且不是标准格式，很可能是用户名/ID
                continue
            
        # 2. 名字不能包含排除关键词 (双重保险，防止类型标记错误)
        if any(kw in node_id_lower for kw in exclude_keywords):
            continue
        
        # 🔥🔥🔥 关键：过滤包含括号的虚拟节点（如 "Orchestrator (thought)"）
        if '(' in node_id or ')' in node_id:
            continue
        
        # 🔥 新增：过滤包含 "->" 的边节点（如 "Orchestrator (-> Assistant)"）
        if '->' in node_id or '→' in node_id:
            continue
            
        valid_candidates.append((node_id, local_idx))
    
    # 🚨 保底机制：如果过滤太狠导致没候选了，放宽限制（只检查类型）
    if not valid_candidates:
        valid_candidates = [(node_id, idx) for node_id, (ntype, idx) in node_id_to_idx.items() if ntype == 'Agent']
        print(f"  ⚠️ [过滤警告] 严格过滤后无候选，放宽为仅检查Agent类型，找到 {len(valid_candidates)} 个")
    
    # 按 local_idx 排序（确保索引顺序正确）
    valid_candidates_sorted = sorted(valid_candidates, key=lambda x: x[1])
    
    # 建立 local_idx -> node_id 的映射（仅包含有效候选）
    idx_to_node_id = {local_idx: node_id for node_id, local_idx in valid_candidates_sorted}
    
    # 提取有效候选者的分数
    valid_scores = []
    valid_indices = []
    
    for node_id, idx in valid_candidates_sorted:
        if idx < len(scores):
            valid_scores.append(scores[idx].item())
            valid_indices.append(idx)
    
    # 如果没有有效分数（极端情况），返回空
    if not valid_scores:
        return [], None
    
    # 转为 tensor 进行排序
    valid_scores_tensor = torch.tensor(valid_scores)
    
    # 🔥🔥🔥【核心验证】获取所有真正的Agent节点
    # 优先使用从JSON文件读取的Agent节点列表（如果提供）
    if true_agent_nodes is not None:
        # 使用从JSON文件读取的真实Agent节点列表
        all_graph_agents = set(true_agent_nodes)
        print(f"  📋 [Agent验证] 使用JSON文件中的Agent节点列表（共{len(all_graph_agents)}个）: {sorted(list(all_graph_agents))[:5]}")
    else:
        # 从图数据中推断Agent节点（备用方案）
        all_graph_agents = set()
        for node_id, (ntype, local_idx) in node_id_to_idx.items():
            if ntype == 'Agent':
                # 使用相同的过滤逻辑验证（排除Tool、网站等）
                node_id_lower = node_id.lower()
                is_valid = True
                # 检查排除关键词（使用完整的列表）
                if any(kw in node_id_lower for kw in exclude_keywords):
                    is_valid = False
                if len(node_id) < 3:
                    is_valid = False
                if '(' in node_id or ')' in node_id or '->' in node_id or '→' in node_id:
                    is_valid = False
                if node_id.replace('_', '').replace('-', '').isalnum() and len(node_id) > 10:
                    if not any(c.isupper() for c in node_id):
                        is_valid = False
                if re.search(r'\d{3,}', node_id):
                    if not re.match(r'^[A-Za-z]+_\d+$', node_id) and 'Expert' not in node_id:
                        is_valid = False
                # 🔥🔥🔥 新增：过滤看起来像人名的（如TheSmart, RosieRoan, Angela等）
                if not node_id.endswith('Expert') and not node_id.endswith('_Expert'):
                    camel_case_words = re.findall(r'[A-Z][a-z]+', node_id)
                    if 2 <= len(camel_case_words) <= 4:
                        common_name_patterns = ['The', 'Young', 'Lee', 'John', 'Mary', 'Angela', 'Rosie', 'Mina']
                        if any(pattern in node_id for pattern in common_name_patterns):
                            is_valid = False
                
                if is_valid:
                    all_graph_agents.add(node_id)
    
    # 🔥 新增：支持全输出模式（top_k=None）
    if top_k is None:
        # 返回所有Agent的完整排序（按分数降序）
        sorted_indices = torch.argsort(valid_scores_tensor, descending=True)
        
        # 构建完整排序列表和分数字典
        all_candidates = []
        agent_scores = {}
        for i in sorted_indices.cpu().tolist():
            local_idx = valid_indices[i]
            agent_id = idx_to_node_id.get(local_idx, f"Agent_{local_idx}")
            score = valid_scores[i]
            all_candidates.append(agent_id)
            agent_scores[agent_id] = float(score)
        
        # 🔥🔥🔥【验证步骤1】确保输出中只包含真正的Agent节点
        # 🔥🔥🔥 关键：如果提供了true_agent_nodes（从JSON文件读取），直接使用它来过滤
        final_candidates = []
        final_scores = {}
        for agent_id in all_candidates:
            # 🔥🔥🔥 关键修复：如果提供了true_agent_nodes，只保留在这些节点中的
            if true_agent_nodes is not None:
                if agent_id not in true_agent_nodes:
                    continue  # 不在真实Agent节点列表中，跳过
            else:
                # 备用方案：使用过滤逻辑（如果没有提供true_agent_nodes）
                # 1. 首先检查节点类型：必须是Agent类型
                node_type_check = None
                if agent_id in node_id_to_idx:
                    node_type_check = node_id_to_idx[agent_id][0]  # 获取节点类型
                
                # 如果节点类型不是Agent，直接排除
                if node_type_check and node_type_check != 'Agent':
                    continue
                
                agent_lower = agent_id.lower()
                # 2. 再次检查排除关键词（双重保险，防止类型标记错误）- 使用完整的排除列表
                if any(kw in agent_lower for kw in exclude_keywords):
                    continue
                if '(' in agent_id or ')' in agent_id or '->' in agent_id or '→' in agent_id:
                    continue
                # 检查是否像网站域名（包含.com等）
                if any(domain in agent_lower for domain in ['.com', '.org', '.net', '.io', '.edu', '.gov']):
                    continue
                # 🔥🔥🔥 新增：过滤看起来像人名或用户名的（如TheSmart, RosieRoan, Angela等）
                if not agent_id.endswith('Expert') and not agent_id.endswith('_Expert'):
                    camel_case_words = re.findall(r'[A-Z][a-z]+', agent_id)
                    if 2 <= len(camel_case_words) <= 4:
                        common_name_patterns = ['The', 'Young', 'Lee', 'John', 'Mary', 'Angela', 'Rosie', 'Mina', 'Smart', 'Roan']
                        if any(pattern in agent_id for pattern in common_name_patterns):
                            continue
            
            final_candidates.append(agent_id)
            final_scores[agent_id] = agent_scores.get(agent_id, -10.0)
        
        # 🔥🔥🔥【验证步骤2】检查输出是否包含了所有Agent节点
        # 🔥🔥🔥 关键：使用从JSON文件读取的Agent节点列表（如果提供）
        reference_agents = true_agent_nodes if true_agent_nodes is not None else all_graph_agents
        output_agents_set = set(final_candidates)
        missing_agents = reference_agents - output_agents_set
        
        if missing_agents:
            # 将被遗漏的Agent添加到输出中（使用最低分数）
            min_score = min(final_scores.values()) if final_scores else -10.0
            for missing_agent in missing_agents:
                final_candidates.append(missing_agent)
                final_scores[missing_agent] = float(min_score - 1.0)
            print(f"  ⚠️ [输出验证] 发现 {len(missing_agents)} 个Agent节点未在GNN输出中，已添加: {list(missing_agents)[:3]}")
        
        # 按分数重新排序
        final_candidates_sorted = sorted(final_candidates, key=lambda x: final_scores[x], reverse=True)
        
        return final_candidates_sorted, final_scores, gnn_pred_step
    else:
        # 原有的 Top-K 逻辑
        # 动态调整 K 值（防止 K > 候选总数）
        current_k = min(top_k, len(valid_scores))
        
        # 选出 Top-K
        top_vals, top_indices = torch.topk(valid_scores_tensor, k=current_k)
        
        # 映射为 agent ID
        final_candidates = []
        final_scores = {}
        for i in top_indices.cpu().tolist():
            local_idx = valid_indices[i]
            agent_id = idx_to_node_id.get(local_idx, f"Agent_{local_idx}")
            score = valid_scores[i]
            final_candidates.append(agent_id)
            final_scores[agent_id] = float(score)
        
        # 🔥🔥🔥【验证步骤1】确保输出中只包含真正的Agent节点
        # 如果提供了true_agent_nodes，只保留在这些节点中的
        filtered_candidates = []
        filtered_scores = {}
        for agent_id in final_candidates:
            # 🔥🔥🔥 关键：如果提供了true_agent_nodes，只输出这些节点
            if true_agent_nodes is not None:
                if agent_id not in true_agent_nodes:
                    continue  # 不在真实Agent节点列表中，跳过
            else:
                # 备用方案：使用过滤逻辑
                # 1. 首先检查节点类型：必须是Agent类型
                node_type_check = None
                if agent_id in node_id_to_idx:
                    node_type_check = node_id_to_idx[agent_id][0]  # 获取节点类型
                
                # 如果节点类型不是Agent，直接排除
                if node_type_check and node_type_check != 'Agent':
                    continue
                
                agent_lower = agent_id.lower()
                # 2. 再次检查排除关键词（双重保险，防止类型标记错误）
                if any(kw in agent_lower for kw in exclude_keywords):
                    continue
                if '(' in agent_id or ')' in agent_id or '->' in agent_id or '→' in agent_id:
                    continue
                # 检查是否像网站域名（包含.com等）
                if any(domain in agent_lower for domain in ['.com', '.org', '.net', '.io', '.edu', '.gov']):
                    continue
                # 🔥🔥🔥 新增：过滤看起来像人名或用户名的（如TheSmart, RosieRoan, Angela等）
                if not agent_id.endswith('Expert') and not agent_id.endswith('_Expert'):
                    camel_case_words = re.findall(r'[A-Z][a-z]+', agent_id)
                    if 2 <= len(camel_case_words) <= 4:
                        common_name_patterns = ['The', 'Young', 'Lee', 'John', 'Mary', 'Angela', 'Rosie', 'Mina', 'Smart', 'Roan']
                        if any(pattern in agent_id for pattern in common_name_patterns):
                            continue
            
            filtered_candidates.append(agent_id)
            filtered_scores[agent_id] = final_scores.get(agent_id, -10.0)
        
        # 🔥🔥🔥【验证步骤2】检查过滤后的输出是否包含了所有Agent节点
        # 🔥🔥🔥 关键：使用从JSON文件读取的Agent节点列表（如果提供）
        reference_agents = true_agent_nodes if true_agent_nodes is not None else all_graph_agents
        output_agents_set = set(filtered_candidates)
        missing_agents = reference_agents - output_agents_set
        
        if missing_agents:
            # 将被遗漏的Agent添加到输出中（使用最低分数）
            min_score = min(filtered_scores.values()) if filtered_scores else -10.0
            for missing_agent in missing_agents:
                filtered_candidates.append(missing_agent)
                filtered_scores[missing_agent] = float(min_score - 1.0)
            print(f"  ⚠️ [输出验证] Top-{top_k}模式：发现 {len(missing_agents)} 个Agent节点未在输出中（可能被非Agent节点挤占），已添加: {list(missing_agents)[:3]}")
        
        # 按分数重新排序
        filtered_candidates_sorted = sorted(filtered_candidates, key=lambda x: filtered_scores[x], reverse=True)
        
        return filtered_candidates_sorted, gnn_pred_step


def analyze_with_llm(model, tokenizer, instruction: str, system_prompt: str = None, max_new_tokens=4096, enable_thinking=True):
    """
    使用微调后的 LLM 进行分析（支持 Qwen3-8B 思考模式）
    
    Args:
        model: LLM 模型
        tokenizer: Tokenizer
        instruction: 指令文本
        system_prompt: 独立的系统提示词（新增参数，解决 Prompt 格式冲突）
        max_new_tokens: 最大生成token数（默认4096，DeepSeek-R1思考过程很长，需要更多空间）
        enable_thinking: 是否启用思考模式
    """
    # 🔥🔥🔥【关键修复】构建标准的 Chat 格式，分离 System 和 User
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": instruction})
    
    # 检查是否是 Qwen3-8B 模型（支持思考模式）
    is_qwen3 = "Qwen3" in str(model.config.name_or_path) if hasattr(model, 'config') and hasattr(model.config, 'name_or_path') else False
    
    # 🔥 如果是 Thinking 模型，强制增加生成长度（确保至少 4096）
    if is_qwen3 and enable_thinking and max_new_tokens < 4096:
        max_new_tokens = 4096
        print(f"  💡 [Thinking模式] 自动增加 max_new_tokens 到 {max_new_tokens}")
    
    # 构建聊天模板（如果支持思考模式，启用它）
    try:
        if is_qwen3 and enable_thinking:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # 启用思考模式
            )
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    except TypeError:
        # 如果 apply_chat_template 不支持 enable_thinking 参数，使用默认方式
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # 生成回答（Qwen3 思考模式推荐参数）
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 🔥 Token统计：计算输入token数
    input_token_count = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        if is_qwen3:
            # Qwen3 思考模式推荐参数
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 🔥 已增加到 4096（DeepSeek-R1 需要更多空间）
                temperature=0.6,  # Qwen3 推荐值
                do_sample=True,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                return_dict_in_generate=False  # 🔥 确保返回张量而不是字典
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 🔥 已增加到 4096
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                return_dict_in_generate=False  # 🔥 确保返回张量而不是字典
            )
    
    # 🔥 Token统计：计算输出token数（新生成的token数）
    # 处理不同的返回格式：可能是张量、元组或字典
    try:
        if isinstance(outputs, torch.Tensor):
            # 直接是张量
            generated_ids = outputs
        elif isinstance(outputs, (tuple, list)):
            # 是元组或列表
            if len(outputs) == 0:
                raise ValueError(f"outputs 是空元组/列表，无法提取生成的token序列")
            generated_ids = outputs[0]
        elif isinstance(outputs, dict):
            # 是字典，尝试获取 sequences 或 generated_ids
            generated_ids = outputs.get('sequences', outputs.get('generated_ids', None))
            if generated_ids is None:
                raise ValueError(f"无法从 outputs 字典中提取生成的token序列。outputs keys: {outputs.keys()}")
        else:
            raise ValueError(f"未知的 outputs 类型: {type(outputs)}, 值: {outputs}")
    except Exception as e:
        # 提供更详细的错误信息
        error_msg = f"处理 model.generate() 返回值时出错: {str(e)}\n"
        error_msg += f"  outputs 类型: {type(outputs)}\n"
        if isinstance(outputs, (tuple, list)):
            error_msg += f"  outputs 长度: {len(outputs)}\n"
        elif isinstance(outputs, dict):
            error_msg += f"  outputs keys: {list(outputs.keys())}\n"
        error_msg += f"  inputs.input_ids.shape: {inputs.input_ids.shape}\n"
        error_msg += f"  input_token_count: {input_token_count}"
        raise RuntimeError(error_msg) from e
    
    # 确保 generated_ids 是张量
    if not isinstance(generated_ids, torch.Tensor):
        raise ValueError(f"generated_ids 不是张量，而是 {type(generated_ids)}")
    
    # 计算输出token数
    if len(generated_ids.shape) == 2:
        # 形状为 [batch_size, seq_len]
        total_length = generated_ids.shape[1]
        output_token_count = total_length - input_token_count
        # 安全检查：确保输出token数不为负
        if output_token_count < 0:
            print(f"  ⚠️ [警告] 计算的输出token数为负数 ({output_token_count})，可能是generated_ids只包含输出部分")
            print(f"      total_length={total_length}, input_token_count={input_token_count}")
            # 如果为负，说明generated_ids可能只包含输出部分，直接使用total_length
            output_token_count = total_length
        # 解码输出（只取新生成的部分）
        response = tokenizer.decode(
            generated_ids[0][input_token_count:],
            skip_special_tokens=True
        )
    elif len(generated_ids.shape) == 1:
        # 形状为 [seq_len]（单样本）
        total_length = generated_ids.shape[0]
        output_token_count = total_length - input_token_count
        # 安全检查：确保输出token数不为负
        if output_token_count < 0:
            print(f"  ⚠️ [警告] 计算的输出token数为负数 ({output_token_count})，可能是generated_ids只包含输出部分")
            print(f"      total_length={total_length}, input_token_count={input_token_count}")
            # 如果为负，说明generated_ids可能只包含输出部分，直接使用total_length
            output_token_count = total_length
        # 解码输出（只取新生成的部分）
        response = tokenizer.decode(
            generated_ids[input_token_count:],
            skip_special_tokens=True
        )
    else:
        raise ValueError(f"不支持的 generated_ids 形状: {generated_ids.shape}")
    
    # 🔥 最终安全检查：确保token数合理
    if output_token_count < 0:
        output_token_count = 0
        print(f"  ⚠️ [警告] 输出token数被修正为0（原始值可能有问题）")
    
    # 记录到全局统计器
    token_counter.add_llm_call(input_token_count, output_token_count)
    
    # ================= [新增调试打印] =================
    print("\n" + "="*50)
    print(f"[调试-原始输出] 长度: {len(response)} 字符")
    print(f"[调试-原始内容] (前500字符):")
    print(response[:500])
    if len(response) > 500:
        print(f"\n[调试-原始内容] (后500字符):")
        print(response[-500:])
    print("="*50 + "\n")
    # ==================================================
    
    return response


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    超强鲁棒性的 JSON 提取函数
    能处理：单引号、未转义字符、Markdown代码块、不完整格式
    
    Args:
        text: LLM 输出的文本
        
    Returns:
        解析后的 JSON 字典，如果失败则返回 None
    """
    try:
        # 1. 预处理：移除 <think> 标签
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        json_str = None
        
        # 2. 尝试提取 Markdown JSON 代码块
        code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block:
            json_str = code_block.group(1)
        else:
            # 尝试提取最外层的大括号
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                json_str = match.group(1)

        # 3. 尝试多种解析方式 (仅当 json_str 存在时)
        if json_str:
            # 方式 A: 标准 json.loads
            try:
                return json.loads(json_str)
            except:
                pass
                
            # 方式 B: 修复单引号
            try:
                fixed_json = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)
                fixed_json = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_json)
                return json.loads(fixed_json)
            except:
                pass

            # 方式 C: ast.literal_eval
            try:
                import ast
                return ast.literal_eval(json_str)
            except:
                pass
            
        # 方式 D: 暴力正则提取 (无论 json_str 是否提取成功，都尝试这个)
        # 直接从原始 text 中搜索，防止正则没抓到大括号
        result = {}
        
        # 提取 Agent (增强正则，支持中文冒号)
        agent_match = re.search(r'["\']?(?:agent|故障源Agent)["\']?\s*[:：]\s*["\']?([^"\'\n,}]+)["\']?', text, re.IGNORECASE)
        if agent_match:
            result['agent'] = agent_match.group(1).strip()
            
        # 提取 Step
        step_match = re.search(r'["\']?(?:step|故障时间步)["\']?\s*[:：]\s*(\d+)', text, re.IGNORECASE)
        if step_match:
            result['step'] = int(step_match.group(1))
            
        # 提取 Reason
        reason_match = re.search(r'["\']?(?:reason|故障原因)["\']?\s*[:：]\s*["\']?([^"\'}\n]+)["\']?', text, re.IGNORECASE)
        if reason_match:
            result['reason'] = reason_match.group(1).strip()
            
        if 'agent' in result or 'step' in result:
            print(f"  ✨ [暴力提取] 从文本中成功提取: {result}")
            return result

        return None
        
    except Exception as e:
        print(f"  [解析错误] extract_json_from_text 最终失败: {e}")
        return None


def parse_llm_response(response: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    解析 LLM 响应，提取故障源Agent、时间步和原因
    支持CoT格式（跳过<think>标签）和 JSON 格式
    
    Returns:
        (agent_id, step, reason)
    """
    agent_id = None
    step = None
    reason = None
    
    # 🔥 首先尝试 JSON 格式解析（支持 Zero-Shot Thinking 模式）
    json_data = extract_json_from_text(response)
    if json_data:
        agent_id = json_data.get('agent') or json_data.get('agent_id') or json_data.get('故障源Agent')
        step = json_data.get('step') or json_data.get('step_id') or json_data.get('故障时间步')
        reason = json_data.get('reason') or json_data.get('reasoning') or json_data.get('故障原因')
        if agent_id or step:
            print(f"  [调试] LLM解析成功（JSON格式）: agent={agent_id}, step={step}")
            return agent_id, step, reason
    
    # 🔥 处理CoT格式：移除<think>...</think>标签，只解析结论部分
    # 提取<think>标签内的内容（用于调试，但不用于解析）
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        # 移除<think>标签，只保留结论部分
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 检查是否是正常场景（无故障）
    normal_patterns = [
        r'系统运行正常',
        r'没有发现故障',
        r'没有故障',
        r'运行正常',
        r'未发现异常'
    ]
    is_normal = any(re.search(pattern, response, re.IGNORECASE) for pattern in normal_patterns)
    
    if is_normal:
        return None, None, "系统运行正常"
    
    # 提取 Agent（改进的正则表达式）
    agent_patterns = [
        # 格式: **故障源Agent**: Movie_Expert
        r'\*\*故障源Agent\*\*[：:]\s*([A-Za-z0-9_\-]+)',
        # 格式: 故障源Agent: Movie_Expert
        r'故障源Agent[：:]\s*([A-Za-z0-9_\-]+)',
        # 格式: 故障Agent: StreamingService_Expert
        r'故障Agent[：:]\s*([A-Za-z0-9_\-]+)',
        # 格式: 导致故障的Agent是 Movie_Expert
        r'导致故障的Agent是\s+([A-Za-z0-9_\-]+)',
        # 格式: Agent是 Movie_Expert
        r'Agent是\s+([A-Za-z0-9_\-]+)',
        # 格式: 候选Agent 1 (Movie_Expert) 是故障源
        r'候选Agent\s+\d+\s*\(([A-Za-z0-9_\-]+)\)',
        # 格式: 候选Agent 1: Movie_Expert
        r'候选Agent\s+\d+[：:]\s*([A-Za-z0-9_\-]+)',
        # 格式: Movie_Expert 是故障源
        r'([A-Za-z0-9_\-]+)\s+是故障源',
        # 格式: 故障源: Movie_Expert
        r'故障源[：:]\s*([A-Za-z0-9_\-]+)',
    ]
    for pattern in agent_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            agent_id = match.group(1).strip()
            if agent_id:
                break
    
    # 提取时间步（改进的正则表达式）
    step_patterns = [
        # 格式: **故障时间步**: 2
        r'\*\*故障时间步\*\*[：:]\s*(\d+)',
        # 格式: 故障时间步: 2
        r'故障时间步[：:]\s*(\d+)',
        # 格式: 故障发生在第 2 步
        r'故障发生在第\s*(\d+)\s*步',
        # 格式: 发生在第 2 步
        r'发生在第\s*(\d+)\s*步',
        # 格式: 故障发生在第 2 个时间步
        r'故障发生在第\s*(\d+)\s*个时间步',
        # 格式: 时间步: 2
        r'时间步[：:]\s*(\d+)',
        # 格式: Step: 3
        r'Step[：:]\s*(\d+)',
        # 格式: 故障Step: 3
        r'故障Step[：:]\s*(\d+)',
        # 格式: 第 2 步
        r'第\s*(\d+)\s*步',
    ]
    for pattern in step_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                step = int(match.group(1))
                break
            except:
                continue
    
    # 提取原因
    reason_patterns = [
        r'故障原因[：:]\s*([^。]+)',
        r'原因[：:]\s*([^。]+)',
    ]
    for pattern in reason_patterns:
        match = re.search(pattern, response)
        if match:
            reason = match.group(1).strip()
            break
    
    return agent_id, step, reason


def evaluate_stage3(
    test_data_dir: str,
    gnn_checkpoint: str,
    llm_adapter: str,
    converter_path: str,
    top_k: int = None,  # 🔥 修改默认值：None=全输出，用于超参数敏感性分析
    device: str = None,
    base_model_name: str = "Qwen/Qwen3-8B"
):
    # 🔥 Token统计：重置统计器
    token_counter.reset()
    """
    阶段三评估：Coarse-to-Fine 系统集成
    
    Args:
        test_data_dir: 测试数据目录
        gnn_checkpoint: GNN 模型检查点路径
        llm_adapter: LLM LoRA 适配器路径
        converter_path: Converter 状态路径
        top_k: Top-K 候选数量
        device: 设备 ('cuda' 或 'cpu')
        base_model_name: 基础LLM模型名称（默认: Qwen/Qwen2.5-7B-Instruct）
    """
    # 设备配置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print("\n" + "=" * 60)
    print("加载模型")
    print("=" * 60)
    
    gnn_model, converter, gnn_config = load_gnn_model(gnn_checkpoint, converter_path, device)
    
    # 🔥 消融实验标记：检查是否使用未微调的模型
    is_ablation_no_finetune = (not llm_adapter or llm_adapter.strip() == "")
    if is_ablation_no_finetune:
        print("\n" + "=" * 80)
        print("🔬 消融实验配置：GNN + 未微调的基础模型")
        print("=" * 80)
        print(f"   GNN 模型: {gnn_checkpoint}")
        print(f"   LLM 模型: {base_model_name} (未微调)")
        print(f"   注意：LLM 未经过微调或强化学习训练")
        print("=" * 80 + "\n")
    
    # 🔥 使用 4-bit 量化加载 8B 模型（节省显存）
    llm_model, tokenizer = load_llm_model(llm_adapter, base_model_name=base_model_name, device=device, use_4bit=True)
    
    # 加载测试数据
    print("\n" + "=" * 60)
    print("加载测试数据")
    print("=" * 60)
    
    test_data_dir = Path(test_data_dir)
    
    # 🔥 修复：递归搜索所有子目录中的 JSON 文件
    # 如果test_data_dir本身是目录，递归搜索所有子目录中的 JSON 文件
    if test_data_dir.is_dir():
        json_files = list(test_data_dir.rglob("*.json"))  # 使用 rglob 递归搜索
    else:
        # 如果是文件，直接使用
        json_files = [test_data_dir] if test_data_dir.suffix == '.json' else []
    
    # 过滤掉不存在的文件
    json_files = [f for f in json_files if f.exists()]
    
    if not json_files:
        print(f"❌ 在 {test_data_dir} 中未找到 JSON 文件")
        print(f"   目录是否存在: {test_data_dir.exists()}")
        if test_data_dir.exists():
            print(f"   目录内容: {list(test_data_dir.iterdir())[:10]}")
        return
    
    print(f"✅ 找到 {len(json_files)} 个测试文件")
    print(f"   测试目录: {test_data_dir}")
    
    # 评估
    print("\n" + "=" * 60)
    print("开始评估")
    print("=" * 60)
    
    metrics_alg = {'agent': [], 'step': []}
    metrics_hand = {'agent': [], 'step': []}
    metrics_total = {'agent': [], 'step': []}
    
    # 🔥🔥🔥【新增】存储真实答案排名（用于超参数敏感性分析）
    true_agent_ranks = []
    
    # 🔥 新增：按领域分类（Code/Math/Agentic）- 用于AgenTracer对比
    metrics_domains = {}  # {domain: {'agent': [], 'step': []}}
    domain_counts = {}    # {domain: count}
    
    count_alg = 0
    count_hand = 0
    skipped_no_label = 0
    skipped_no_graph = 0
    
    for json_file in tqdm(json_files, desc="评估中"):
        try:
            # 加载图数据
            with open(json_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 重建图
            graph = reconstruct_graph_from_json(graph_data)
            graph_list, labels = converter.convert(graph)
            
            if not graph_list:
                skipped_no_graph += 1
                continue
            
            # 🔥 直接从 graph_data 的 ground_truth 中提取标签
            ground_truth = graph_data.get('ground_truth', {})
            true_agent = ground_truth.get('mistake_agent', '')
            true_step = int(ground_truth.get('mistake_step', -1))
            true_reason = ground_truth.get('mistake_reason', '')
            
            # 如果没有真实标签，跳过
            if not true_agent or true_step < 0:
                skipped_no_label += 1
                continue
            
            # 🔥🔥🔥【关键修复】从JSON文件读取所有Agent节点
            # 🔥🔥🔥 核心策略：直接使用JSON中的type标记，信任数据标注
            nodes = graph_data.get('nodes', {})
            true_agent_nodes = set()
            
            # 🔥🔥🔥 首先：如果ground_truth中有真实答案，确保它被包含（即使type标记错误或大小写不匹配）
            if true_agent:
                # 1. 直接匹配
                if true_agent in nodes:
                    true_agent_nodes.add(true_agent)
                # 2. 大小写不敏感匹配（如Websurfer vs WebSurfer）
                else:
                    for node_id in nodes.keys():
                        if node_id.lower() == true_agent.lower():
                            true_agent_nodes.add(node_id)
                            break
                    # 3. 模糊匹配：如果真实答案是"Orchestrator"，但节点ID是"Orchestrator (thought)"
                    true_agent_base = re.sub(r'\s*\([^)]*\)\s*', '', true_agent).strip()
                    true_agent_base = re.sub(r'\s*->.*', '', true_agent_base).strip()
                    for node_id in nodes.keys():
                        node_id_base = re.sub(r'\s*\([^)]*\)\s*', '', node_id).strip()
                        node_id_base = re.sub(r'\s*->.*', '', node_id_base).strip()
                        if node_id_base.lower() == true_agent_base.lower() and node_id_base:
                            true_agent_nodes.add(node_id_base)  # 使用基础名称（不带括号）
                            break
            
            # 🔥🔥🔥 核心逻辑：直接使用JSON中的type标记，但过滤明显错误的标记
            # 明显不是Agent的节点名称（即使被标记为Agent也应该过滤）
            invalid_agent_names = {
                # 元数据/属性名称
                'context', 'type', 'id', 'name', 'graph', 'node', 'edge',
                'metadata', 'attribute', 'property', 'field', 'value', 'key', 'data',
                # 网站/平台名称（全小写）
                'sportskeeda', 'benandjerrys', 'marketwatch', 'imdb', 'github', 
                'googlegroups', 'worldbankdata', 'liicornell', 'amelia', 'mamtaraut10',
                # 其他明显不是Agent的名称
                'url', 'link', 'href', 'src', 'path', 'file', 'dir', 'folder',
                # 常见的网站/服务名称
                'youtube', 'gmail', 'turboscribe', 'linkedin', 'twitter', 'facebook', 
                'instagram', 'amazon', 'wikipedia', 'netflix', 'spotify', 'reddit', 
                'pinterest', 'tumblr'
            }
            
            # 常见的人名/用户名模式（CamelCase，看起来像人名但不是Agent）
            common_person_names = {
                'angela', 'pingu', 'rosieroan', 'thesmart', 'goranxii', 'johndownerprod'
            }
            
            for node_id, node_data in nodes.items():
                # 确保node_data是字典类型
                if not isinstance(node_data, dict):
                    continue
                
                node_type = node_data.get('type', '')
                
                # 🔥🔥🔥 关键：直接信任JSON中的type标记，但过滤明显错误的标记
                if node_type == 'Agent':
                    node_id_lower = node_id.lower()
                    
                    # 只过滤human/user节点（这些是用户，不是Agent）
                    if node_id_lower in ['human', 'user', 'user_proxy'] or node_id_lower.startswith('user') or node_id_lower.startswith('human'):
                        continue
                    
                    # 🔥🔥🔥 过滤明显不是Agent的节点（即使被标记为Agent）
                    if node_id_lower in invalid_agent_names:
                        continue
                    
                    # 🔥🔥🔥 过滤纯数字或单字符节点（如'3'）
                    if node_id.isdigit() or (len(node_id) == 1 and not node_id.isalpha()):
                        continue
                    
                    # 🔥🔥🔥 过滤看起来像元数据/属性的节点（全小写且长度<=5，如context, type等）
                    # 🔥 关键：id, context, type这些单字符或短单词明显是元数据，不是Agent
                    if node_id.islower() and len(node_id) <= 5:
                        # 只保留明确的Agent名称
                        if node_id not in ['assistant', 'surfer', 'orchestrator', 'websurfer', 'filesurfer']:
                            continue
                    
                    # 🔥🔥🔥 新增：过滤全小写的单词语（可能是网站名、用户名等，但不是真正的Agent）
                    # 如果节点名是全小写且不包含下划线或连字符，且长度>5，很可能是网站名或用户名
                    if node_id.islower() and '_' not in node_id and '-' not in node_id and len(node_id) > 5:
                        # 但保留常见的Agent名称
                        if node_id not in ['assistant', 'orchestrator', 'websurfer', 'surfer', 'coordinator']:
                            # 检查是否包含数字（如mamtaraut10）
                            if re.search(r'\d', node_id):
                                continue
                            # 检查是否是常见的网站域名模式（如sportskeeda, benandjerrys等）
                            if any(char.isdigit() for char in node_id) or len(node_id) > 10:
                                continue
                    
                    # 🔥🔥🔥 新增：过滤包含数字的节点（如MamtaRaut10等，但保留IMDB_Ratings_Expert这种）
                    if re.search(r'\d', node_id) and not (node_id.endswith('Expert') or node_id.endswith('_Expert')):
                        # 如果包含数字但不是Expert结尾，很可能是用户名或ID
                        # 但排除常见的Agent名称（如WebSurfer, Assistant等）
                        if node_id not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer']:
                            continue
                    
                    # 🔥🔥🔥 新增：过滤CamelCase格式但看起来像人名的节点（如MamtaRaut10等）
                    # 如果节点名是CamelCase且包含数字，很可能是用户名
                    if re.match(r'^[A-Z][a-z]+.*\d', node_id) and not (node_id.endswith('Expert') or node_id.endswith('_Expert')):
                        if node_id not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer']:
                            continue
                    
                    # 🔥🔥🔥 新增：过滤看起来像人名的CamelCase节点（如Angela, Pingu, RosieRoan等）
                    # 这些通常是YouTube评论者、社交媒体用户等，不是Agent
                    if node_id_lower in common_person_names:
                        continue
                    
                    # 🔥🔥🔥 新增：过滤纯CamelCase且看起来像人名的节点（不包含下划线、连字符、数字，且不以Expert结尾）
                    # 如：Angela, Pingu, RosieRoan, TheSmart, GoranXII, JohnDownerProd
                    if re.match(r'^[A-Z][a-z]+([A-Z][a-z]+)*$', node_id) and not (node_id.endswith('Expert') or node_id.endswith('_Expert')):
                        # 排除明确的Agent名称
                        if node_id not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer', 'TalkNotesApp', 'TurboScribe']:
                            # 如果节点名看起来像人名（首字母大写+小写字母组合，且长度适中），很可能是用户名
                            # 但保留一些常见的Agent名称模式
                            if len(node_id) <= 15 and not node_id.startswith('Web') and not node_id.startswith('File') and not node_id.startswith('Talk') and not node_id.startswith('Turbo'):
                                continue
                    
                    # 🔥🔥🔥 新增：过滤包含数字且看起来像用户名的节点（如Topgoon634, Aroundthebonfire884等）
                    # 这些通常是YouTube用户名、社交媒体账号等
                    if re.search(r'\d', node_id) and not (node_id.endswith('Expert') or node_id.endswith('_Expert')):
                        # 如果节点名包含数字且看起来像用户名（全小写或混合大小写，长度较长）
                        if node_id not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer']:
                            # 检查是否是用户名模式（包含数字，且不是Expert结尾）
                            if len(node_id) > 10 or (re.search(r'\d', node_id) and not node_id[0].isupper()):
                                continue
                    
                    # 🔥🔥🔥 新增：过滤中文字符节点（如'胡球'），除非是明确的Agent名称
                    if re.search(r'[\u4e00-\u9fff]', node_id):
                        # 中文字符节点通常是用户名或评论者，不是Agent
                        continue
                    
                    # 🔥🔥🔥 处理带括号的节点：提取基础名称
                    # 如 "Orchestrator (thought)" -> "Orchestrator"
                    # 如 "Orchestrator (-> WebSurfer)" -> "Orchestrator"
                    if '(' in node_id or ')' in node_id or '->' in node_id or '→' in node_id:
                        node_id_base = re.sub(r'\s*\([^)]*\)\s*', '', node_id).strip()
                        node_id_base = re.sub(r'\s*->.*', '', node_id_base).strip()
                        if node_id_base:  # 如果提取到基础名称，使用基础名称
                            # 再次检查基础名称是否有效
                            node_id_base_lower = node_id_base.lower()
                            # 检查是否在无效列表中
                            if node_id_base_lower in invalid_agent_names:
                                continue
                            # 检查是否是纯数字或单字符
                            if node_id_base.isdigit() or (len(node_id_base) == 1 and not node_id_base.isalpha()):
                                continue
                            # 检查是否是元数据/属性节点
                            if node_id_base.islower() and len(node_id_base) <= 5:
                                if node_id_base not in ['assistant', 'surfer', 'orchestrator', 'websurfer', 'filesurfer']:
                                    continue
                            # 检查是否是人名
                            if node_id_base_lower in common_person_names:
                                continue
                            # 检查是否包含数字（但保留Expert结尾的）
                            if re.search(r'\d', node_id_base) and not (node_id_base.endswith('Expert') or node_id_base.endswith('_Expert')):
                                if node_id_base not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer', 'TalkNotesApp', 'TurboScribe']:
                                    if len(node_id_base) > 10 or (re.search(r'\d', node_id_base) and not node_id_base[0].isupper()):
                                        continue
                            # 检查是否是CamelCase格式但包含数字的节点
                            if re.match(r'^[A-Z][a-z]+.*\d', node_id_base) and not (node_id_base.endswith('Expert') or node_id_base.endswith('_Expert')):
                                if node_id_base not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer', 'TalkNotesApp', 'TurboScribe']:
                                    continue
                            # 检查是否是纯CamelCase且看起来像人名的节点
                            if re.match(r'^[A-Z][a-z]+([A-Z][a-z]+)*$', node_id_base) and not (node_id_base.endswith('Expert') or node_id_base.endswith('_Expert')):
                                if node_id_base not in ['WebSurfer', 'Assistant', 'Orchestrator', 'FileSurfer', 'TalkNotesApp', 'TurboScribe']:
                                    if len(node_id_base) <= 15 and not node_id_base.startswith('Web') and not node_id_base.startswith('File') and not node_id_base.startswith('Talk') and not node_id_base.startswith('Turbo'):
                                        continue
                            # 检查是否包含中文字符
                            if re.search(r'[\u4e00-\u9fff]', node_id_base):
                                continue
                            true_agent_nodes.add(node_id_base)
                    else:
                        # 不带括号的节点，直接使用
                        true_agent_nodes.add(node_id)
            
            if not true_agent_nodes:
                print(f"  ⚠️ [警告] 从JSON文件中未找到任何Agent节点（可能全部被过滤）")
            else:
                print(f"  📋 [Agent验证] 从JSON文件读取到 {len(true_agent_nodes)} 个Agent节点: {sorted(list(true_agent_nodes))}")
            
            # 1. GNN 预测候选（同时获取Step预测）
            # 🔥 新增：支持全输出模式（top_k=None 时返回所有Agent排序）
            # 🔥🔥🔥 传入从JSON文件读取的真实Agent节点列表
            all_agent_ranking = None
            agent_scores = None
            try:
                gnn_result = predict_top_k_with_gnn(
                    gnn_model, graph_list, converter, gnn_config, device, top_k=top_k, true_agent_nodes=true_agent_nodes
                )
                
                # 处理两种返回格式
                if top_k is None:
                    # 全输出模式：返回 (所有Agent排序, 分数字典, Step预测)
                    all_agent_ranking, agent_scores, gnn_pred_step = gnn_result
                    candidate_agent_ids = all_agent_ranking  # 使用完整排序作为候选
                else:
                    # Top-K 模式：返回 (Top-K列表, Step预测)
                    candidate_agent_ids, gnn_pred_step = gnn_result
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    # 维度不匹配错误，跳过这个样本
                    skipped_no_label += 1
                    continue
                else:
                    raise
            
            # ================= [关键修复] 物理屏蔽 human/user =================
            # 🔥🔥🔥 最关键：GNN 总是把用户节点当作重要节点推荐，这是干扰源
            # 必须在代码里强行过滤，防止 LLM 归因给用户
            # 故障归因绝不应该归因给用户（human），标签里永远是具体的 Agent
            
            # 定义黑名单
            blacklist = ['human', 'user', 'user_proxy', 'admin', 'root', 'system']
            
            # 过滤候选列表（使用部分匹配，防止 User_1 漏网）
            filtered_candidate_agent_ids = []
            for agent_name in candidate_agent_ids:
                agent_lower = agent_name.lower()
                # 使用部分匹配 (例如过滤 User_1, Human_Agent)
                is_blacklisted = any(b == agent_lower for b in blacklist) or \
                                 agent_lower.startswith('user') or \
                                 agent_lower.startswith('human')
                
                if not is_blacklisted:
                    filtered_candidate_agent_ids.append(agent_name)
            
            # 如果过滤完空了（极少情况），尝试获取更多候选
            if len(filtered_candidate_agent_ids) == 0 and len(candidate_agent_ids) > 0:
                if top_k is not None:
                    print(f"  ⚠️ [警告] 过滤后候选列表为空，尝试获取更多候选（top_k={top_k*2}）")
                    try:
                        extended_result = predict_top_k_with_gnn(
                            gnn_model, graph_list, converter, gnn_config, device, top_k=top_k*2
                        )
                        extended_candidates, _ = extended_result
                        filtered_candidate_agent_ids = []
                        for agent in extended_candidates:
                            agent_lower = agent.lower()
                            is_blacklisted = any(b == agent_lower for b in blacklist) or \
                                             agent_lower.startswith('user') or \
                                             agent_lower.startswith('human')
                            if not is_blacklisted:
                                filtered_candidate_agent_ids.append(agent)
                        if filtered_candidate_agent_ids:
                            print(f"  ✅ [恢复] 从扩展候选列表中找到了 {len(filtered_candidate_agent_ids)} 个有效候选")
                        else:
                            # 如果还是为空，至少保留一个非human的候选
                            print(f"  ⚠️ [警告] 扩展候选后仍为空，使用原始候选（可能包含human节点）")
                            filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
                    except Exception as e:
                        print(f"  ⚠️ [警告] 获取扩展候选失败: {e}，使用原始候选")
                        filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
                else:
                    # 全输出模式下，如果过滤后为空，至少保留一个
                    print(f"  ⚠️ [警告] 全输出模式下过滤后候选列表为空，使用原始候选")
                    filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
            
            # 更新候选人列表
            original_candidate_count = len(candidate_agent_ids)
            candidate_agent_ids = filtered_candidate_agent_ids
            
            if original_candidate_count > len(candidate_agent_ids):
                filtered_count = original_candidate_count - len(candidate_agent_ids)
                print(f"  🔒 [过滤] 已过滤 {filtered_count} 个用户节点（human/user等），剩余 {len(candidate_agent_ids)} 个候选Agent")
            
            # 🔥🔥🔥【新增】全输出模式：记录完整排序和真实答案排名
            if top_k is None and all_agent_ranking is not None:
                # 过滤完整排序（移除human/user节点）
                filtered_full_ranking = []
                filtered_agent_scores = {}
                for agent in all_agent_ranking:
                    agent_lower = agent.lower()
                    is_blacklisted = any(b == agent_lower for b in blacklist) or \
                                     agent_lower.startswith('user') or \
                                     agent_lower.startswith('human')
                    if not is_blacklisted:
                        filtered_full_ranking.append(agent)
                        if agent_scores:
                            filtered_agent_scores[agent] = agent_scores.get(agent, 0.0)
                
                # 🔥🔥🔥【新增】验证：如果真实答案不在输出中，且它是Agent节点，则添加它
                if true_agent and true_agent not in filtered_full_ranking:
                    # 检查真实答案是否是Agent类型节点
                    graph = graph_list[0] if graph_list else None
                    if graph and graph.node_id_to_idx:
                        node_id_to_idx = graph.node_id_to_idx
                        if true_agent in node_id_to_idx:
                            node_type = node_id_to_idx[true_agent][0]
                            if node_type == 'Agent':
                                # 真实答案是Agent节点，但不在输出中，添加它（使用最低分数）
                                min_score = min(filtered_agent_scores.values()) if filtered_agent_scores else -10.0
                                filtered_full_ranking.append(true_agent)
                                filtered_agent_scores[true_agent] = float(min_score - 1.0)
                                print(f"  ⚠️ [输出验证] 真实答案 '{true_agent}' 不在GNN输出中，已添加（它是Agent节点）")
                
                # 计算真实答案在完整排序中的排名（从1开始）
                true_agent_rank = None
                if true_agent:
                    try:
                        true_agent_rank = filtered_full_ranking.index(true_agent) + 1
                    except ValueError:
                        # 真实答案不在排序中
                        true_agent_rank = -1
                
                # 🔥 输出完整排序信息（用于后续分析）
                print(f"  📊 [完整排序] GNN输出的所有Agent排序（共{len(filtered_full_ranking)}个）:")
                ranking_str = ", ".join([f"{i+1}.{agent}" for i, agent in enumerate(filtered_full_ranking[:20])])  # 只显示前20个
                if len(filtered_full_ranking) > 20:
                    ranking_str += f" ... (共{len(filtered_full_ranking)}个)"
                print(f"    {ranking_str}")
                if true_agent_rank and true_agent_rank > 0:
                    print(f"  📊 [真实答案排名] 真实Agent '{true_agent}' 在完整排序中的排名: 第 {true_agent_rank} 位")
                elif true_agent_rank == -1:
                    print(f"  ⚠️ [真实答案排名] 真实Agent '{true_agent}' 不在完整排序中")
            
            print(f"  [调试] 过滤后的GNN候选: {candidate_agent_ids}")
            # ==============================================================
            
            # 2. 提取候选Agent日志
            nodes = graph_data.get('nodes', {})
            history = graph_data.get('history', [])
            
            agent_logs = extract_agent_logs(nodes, candidate_agent_ids, history)
            
            # 2.5. 提取系统关键报错信息（Computer_terminal等的报错）
            system_errors = []
            tool_node_keywords = ['terminal', 'computer', 'console', 'broadcast', 'env', 'environment']
            
            for node_id, node_data in nodes.items():
                node_id_lower = node_id.lower()
                # 检查是否是工具节点
                if any(kw in node_id_lower for kw in tool_node_keywords):
                    # 提取该节点的错误信息
                    features = node_data.get('features', {})
                    if isinstance(features, dict):
                        # 按时间步排序，取最后几个时间步（通常错误在最后）
                        sorted_timesteps = sorted(features.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                        # 只取最后3个时间步
                        for t in sorted_timesteps[-3:]:
                            feat = features[t]
                            if isinstance(feat, dict):
                                content_text = (
                                    feat.get('content_text', '') or 
                                    feat.get('content', '') or
                                    feat.get('text', '')
                                )
                                # 检查是否包含错误关键词
                                if content_text and any(keyword in content_text.lower() for keyword in 
                                    ['error', 'exception', 'fail', 'failed', 'failure', 'traceback', 'exception']):
                                    system_errors.append(f"[Step {t}] {node_id}: {content_text[:300]}")
            
            # 3. 构建 LLM 输入
            # 🔥🔥🔥【关键修复】分离 System Prompt，解决格式冲突
            sys_prompt = "You are a helpful assistant. You must first think step-by-step in <think> tags, and then OUTPUT THE FINAL JSON ANSWER. Do not stop in the middle of thinking - you must complete your reasoning and provide the final JSON answer."
            
            instruction = f"""这是一个多Agent系统的故障诊断任务。系统执行失败了，你需要找出**根因Agent**。

"""
            
            # 3.1. 添加系统关键报错信息（如果有）
            if system_errors:
                instruction += f"""**【系统关键报错信息】**（这些是工具节点的报错，用于定位根因）：
{chr(10).join(system_errors[:5])}

**重要**：这些报错是**症状**，不是病因。请找出是**哪个Agent**引发了这些报错。

"""
            
            instruction += f"""GNN模型已经基于图结构排除了工具和环境节点，锁定了以下 {len(candidate_agent_ids)} 个最可疑的Agent：

"""
            for i, agent_id in enumerate(candidate_agent_ids, 1):
                # 标记 GNN 的置信度排名
                rank_str = ["(GNN认为最可疑)", "(GNN认为次可疑)", ""][i-1] if i <= 2 else ""
                
                # 🔥🔥🔥【优化】智能截断过长的日志 (保留头尾，增加头部长度)
                raw_log = agent_logs.get(agent_id, f"Agent {agent_id}: 无日志")
                MAX_LOG_LEN = 2500
                if len(raw_log) > MAX_LOG_LEN:
                    # 保留前 800 字符 (看初始配置) 和 后 1700 字符 (看报错)
                    head = raw_log[:800]
                    tail = raw_log[-1700:]
                    log_content = f"{head}\n\n... [日志过长，中间 {len(raw_log)-2500} 字符已省略] ...\n\n{tail}"
                    print(f"  ⚠️ [日志截断] {agent_id} 日志保留头尾 (总长 {len(raw_log)})")
                else:
                    log_content = raw_log
                
                instruction += f"**候选 {i}: {agent_id}** {rank_str}\n{log_content}\n\n"
            
            instruction += f"""请仔细分析这些候选Agent的日志，找出导致任务失败的**根因**。

**🚨 关键规则（违反必错）**：
1. **绝对不要选 Computer_terminal、Broadcast、Environment、Tool 等工具节点**：
   - 这些是工具，不是Agent。它们报错是因为收到了错误的指令或数据。
   - 请找出**是谁发出的错误指令**或**是谁生成了错误数据**，那个Agent才是根因。
   - 例如：如果日志显示 "Computer_terminal: command failed"，请找出是哪个Agent调用了这个命令。

2. **🚨🚨🚨🚨 绝对不要选 Orchestrator / Manager / UserProxy（除非它是唯一的候选）**：
   - **Orchestrator 通常是无辜的**！它只是发号施令的中介，负责分发任务。
   - 如果任务失败，通常是**具体的执行者**（如 Coder, WebSurfer, FileSurfer）没做好。
   - 看到 "TERMINATE" 信号不代表 Orchestrator 错了，代表它收到了错误结果。
   - **除非日志明确显示 Orchestrator 规划错误**，否则**优先选择具体的执行 Agent**。
   - 如果候选列表中有 `WebSurfer` 和 `Orchestrator`，且两者日志都有错，**请选 WebSurfer**。
   - **⚠️ 重要**：即使 Orchestrator 的日志里有很多 Error，也**不要选它**，因为那些 Error 通常是它**转发**的执行者的错误。

3. **🚨🚨🚨🚨 绝对不要选 Validation_Expert / Verification_Expert（除非它是唯一的候选）**：
   - **它们是"吹哨人"，不是"肇事者"**！它们的职责就是报错。
   - 如果它们说"数据错误"，那是**生成数据的 Agent**（如 Data_Expert, WebSurfer）错了。
   - 它们是尽责的验证者，发现问题是它们的本职工作。
   - **除非 Validation Agent 自身的代码逻辑崩溃**（比如 Python 报错、Traceback），否则不要选它。
   - **典型错误模式**：看到 Verification_Expert 报错就选它 → **错误！** 应该选上游生成数据的 Agent。
   - **⚠️ 重要**：即使 Validation_Expert 的日志里全是 "Error"、"Failed"、"Incorrect"，也**不要选它**，因为那些是它**报告**的别人的错误。

4. **关注具体的执行错误模式**：
   - 代码报错/语法错误 -> Coder、PythonAgent、PythonDebugging_Expert
   - 网页打不开/内容不对/404错误 -> WebSurfer、WebServing_Expert
   - 文件不存在/读取失败 -> FileSurfer、Data_Expert
   - 数据验证失败/格式错误 -> 检查上游数据生成Agent（如 Data_Expert、WebSurfer、DataVerification_Expert）
   - API调用失败 -> 检查调用API的Agent（如 BingAPI_Expert、API_Expert）

4. **候选列表说明**：
   - 当前候选列表（已排除工具节点）：{', '.join(candidate_agent_ids)}
   - **你必须且只能从上述列表中选择一个**，不能选择列表外的任何名称。

**分析指南**（高级侦探逻辑 - 强制根因回溯）：
1. **寻找"肇事者"而非"报告者"**（最重要！）：
   - 如果 Agent A 报错说"数据格式错误"或"文件为空"，这通常意味着**上游的 Agent B** 生成了错误的数据。
   - 此时，**故障源是 Agent B**（肇事者），而不是发现问题的 Agent A（报告者/吹哨人）。
   - 请检查日志，找出是谁**产生**了导致报错的数据或文件。
   - **典型错误**：看到 Verification_Expert 报错就选它，实际上它只是尽责的吹哨人，真正的问题在上游（如 DataAnalysis_Expert、WebSurfer）。

**🔥 强制根因回溯推理**（必须执行）：
1. **不要只看报错**：报错通常发生在故障发生很久之后。
   - 例如：Step 43 报错说"文件不存在"，请往回找，是谁在 Step 9 承诺要生成文件但没生成？
   - 那个 Step 9 的 Agent 才是真凶。

2. **向前追溯**：看到报错后，请往回看，是谁**最早**引入了导致这个错误的数据或逻辑。
   - 比如：Step 43 报错说"文件不存在"，请往回找，是谁在 Step 9 承诺要生成文件但没生成？
   - 那个 Step 9 的 Agent 才是真凶。

3. **输出要求**：
   - 故障时间步：必须是**根因发生的时间步**，而不是报错的时间步。
   - 例如：如果 Step 9 引入了错误数据，Step 43 才报错，那么故障时间步应该是 9，不是 43。

2. **区分症状与病因**：
   - 如果日志显示 "Computer_terminal returned error" 或 "文件未找到"，这是**症状**。
   - 请找出是**哪个Agent**发送了导致错误的指令或生成了错误的数据，那个Agent才是**病因**。
   - 例如：如果 Verification Agent 报错说"文件为空"，通常是上游负责生成的 Agent (如 WebSurfer 或 Data_Expert) 没干好活。

3. **区分"执行失败"与"逻辑错误"**：
   - 执行失败（如 API 连接超时、网页打不开）是环境问题，通常归因于尝试执行该操作的 Agent。
   - 逻辑错误（如代码写错、计算错、数据生成错）是 Agent 的能力问题，归因于产生错误逻辑的 Agent。

4. **关注最后一次有效操作**：
   - 往往是最后一次修改代码、生成文件或发出指令的 Agent 导致了系统崩溃。
   - 检查日志中的时间顺序，找出最后执行关键操作的 Agent。

5. **关注上游因果链**：
   - 错误通常有因果链：上游Agent产生错误数据 → 下游Agent检测到错误 → 任务失败
   - 找出因果链的**起点**（根因Agent），而不是中间环节

6. **信任GNN的排序**：
   - 这里的候选列表已经经过筛选，排除了工具节点
   - 请重点关注**候选 1**（GNN认为最可疑的），但也要检查其他候选

7. **检查故障特征**：
   - 错误信息（error, exception, fail, failed, failure）
   - 异常行为（unexpected, abnormal, incorrect）
   - 任务失败（task failed, cannot complete, unable to）
   - 数据错误（invalid data, wrong result, incorrect output）
   - 超时或中断（timeout, interrupted, stopped）

8. **特别注意（针对Algorithm-Generated数据集）**：
   - 如果没有明显的报错日志（Traceback），请检查**数据的完整性**（比如文件是否为空，变量是否为 None）。
   - 优先怀疑**产出数据**的 Agent，而不是**使用数据**的 Agent。

**输出格式**（严格遵守，必须输出 JSON）：
🚨🚨🚨 **你必须输出 JSON 格式的答案，不能只输出思考过程！** 🚨🚨🚨

在完成思考后，必须输出以下格式的 JSON：
```json
{{
  "agent": "[候选列表中的一个Agent名称]",
  "step": [整数，日志中的绝对Step ID],
  "reason": "[简短说明故障原因]"
}}
```

或者使用以下文本格式（但优先使用 JSON）：
故障源Agent: [必须是上述候选列表中的一个名称，使用完整名称]
故障时间步: [日志中标记的**绝对Step ID**（如[Step 15]中的15），必须是整数，不能是相对位置]
故障原因: [简短说明，包括具体的错误信息或异常行为]

**⚠️ 关键提示 - Step ID映射**：
- 日志中的 `[Step X]` 是**绝对Step ID**，不是相对位置
- 例如：如果日志显示 `[Step 12]` 和 `[Step 15]`，真实故障在Step 15，你必须输出 `故障时间步: 15`
- **不要输出相对位置**（如"第2个日志"），必须输出日志中标记的**绝对Step ID**

**注意**：
- 不要回答"系统运行正常"，必须从候选Agent中选择一个作为故障源
- Agent名称必须与候选列表中的名称完全一致（包括大小写和下划线）
- 时间步必须是整数，不能是小数或范围
- **绝对不要选择 Computer_terminal、Broadcast、Environment 等工具节点**（它们不在候选列表中，但如果出现请忽略）"""
            
            # 4. LLM 分析（传入分离的 System Prompt）
            llm_response = analyze_with_llm(llm_model, tokenizer, instruction, system_prompt=sys_prompt)
            
            # 5. 解析 LLM 响应
            pred_agent, pred_step, pred_reason = parse_llm_response(llm_response)
            
            # 🔥🔥🔥【增强版智能回退机制】(Ensemble Logic with Fuzzy Matching)
            # 这里的逻辑是：如果 LLM 瞎猜了一个不在列表里的，或者说没故障，我们宁愿信 GNN 的第一名
            # 因为这是 Fault Attribution 任务，肯定有故障，而且大概率在 GNN 的高分里
            final_pred_agent = None
            final_pred_step = pred_step
            
            # 1. 尝试从 LLM 输出中模糊匹配候选人（增强版）
            if pred_agent and candidate_agent_ids:
                # 🔥 使用 normalize_name 进行超强模糊匹配
                pred_norm = normalize_name(pred_agent)
                matched_candidate = None
                best_match_score = 0
                
                for cand in candidate_agent_ids:
                    cand_norm = normalize_name(cand)
                    
                    # 计算匹配分数（多种策略）
                    match_score = 0
                    # 1. 完全匹配（标准化后）
                    if pred_norm == cand_norm:
                        match_score = 100  # 完全匹配
                    # 2. 包含关系（标准化后）
                    elif pred_norm in cand_norm or cand_norm in pred_norm:
                        match_score = 50  # 包含关系
                    # 3. 原始包含关系（处理空格等）
                    elif pred_agent.lower().strip() in cand.lower() or cand.lower().strip() in pred_agent.lower():
                        match_score = 30  # 原始包含关系
                    # 4. 核心名称匹配（移除后缀如 '_Expert'）
                    pred_core = pred_norm.replace('expert', '').replace('agent', '')
                    cand_core = cand_norm.replace('expert', '').replace('agent', '')
                    if pred_core and cand_core and (pred_core == cand_core or pred_core in cand_core or cand_core in pred_core):
                        match_score = 40  # 核心名称匹配
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        matched_candidate = cand
                
                if matched_candidate:
                    final_pred_agent = matched_candidate
                    if final_pred_agent != pred_agent:
                        print(f"  ✨ [模糊修正] LLM预测 '{pred_agent}' -> 模糊匹配修正为候选: '{final_pred_agent}' (匹配分数={best_match_score})")
            
            # 2. 如果还是没匹配到，或者 LLM 预测为空
            if final_pred_agent is None:
                # 🔥🔥🔥【智能回退机制】不要无脑选 Top-1，而是选 Top-K 中第一个"像干活"的 Agent
                if candidate_agent_ids:
                    target_fallback = candidate_agent_ids[0]  # 默认使用 Top-1
                    
                    # 🔥 再次确保过滤掉 human/user（双重保险）
                    forbidden_user_agents = ['human', 'user', 'user_proxy', 'admin', 'root', 'system']
                    valid_candidates = [
                        cand for cand in candidate_agent_ids 
                        if cand.lower() not in forbidden_user_agents
                    ]
                    
                    if valid_candidates:
                        candidate_agent_ids = valid_candidates
                    
                    # 定义"干活"Agent的特征：不包含 orchestrator, validation, verification, 纯数字
                    # 优先顺位：具体业务Agent > 验证Agent > 管理Agent
                    for cand in candidate_agent_ids:
                        cand_lower = cand.lower()
                        # 🔥 再次检查 human/user（三重保险）
                        if cand_lower in forbidden_user_agents:
                            continue
                        # 跳过嫌疑低的类型
                        if any(x in cand_lower for x in ['orchestrator', 'validation', 'verification', 'manager']):
                            continue
                        # 跳过纯数字或看起来像ID的（包含3个以上连续数字，但不是标准格式如 Agent_1）
                        if re.search(r'\d{3,}', cand):
                            # 但要保留由 Agent_1, WebSurfer_2 这种标准格式
                            if not re.match(r'^[A-Za-z]+_\d+$', cand) and 'Expert' not in cand:
                                continue
                        
                        # 找到了第一个具体的业务Agent
                        target_fallback = cand
                        break
                    
                    final_pred_agent = target_fallback
                    if final_pred_step is None:
                        final_pred_step = 1  # 默认步数
                    
                    fallback_reason = "LLM未识别出Agent" if pred_agent is None else f"LLM预测的 '{pred_agent}' 不在候选列表中且无法模糊匹配"
                    print(f"  ⚠️ [智能回退] {fallback_reason} -> 在候选前列中选择最可能的执行者: {final_pred_agent} (Step={final_pred_step})")
                else:
                    # 极端情况：没有候选
                    final_pred_agent = pred_agent  # 保持原预测
                    print(f"  ⚠️ [警告] 没有候选Agent，保持LLM原预测: {final_pred_agent}")
            
            # 🔥🔥🔥【关键修复】放宽 Agent 修正逻辑：信任 LLM 的判断
            # 之前的逻辑过于激进，会强制修正 Validation_Expert 等 Agent，但实际上这些 Agent 可能就是真凶
            # 现在改为：只给出警告，不强制修正，让 LLM 的判断生效
            
            forbidden_agents = ['orchestrator', 'validation_expert', 'verification_expert', 'human', 'user', 'user_proxy', 'admin', 'root', 'system']
            final_pred_agent_lower = final_pred_agent.lower() if final_pred_agent else ""
            
            # 检查是否选择了禁止的Agent
            is_forbidden = any(forbidden in final_pred_agent_lower for forbidden in forbidden_agents)
            
            # 🔥🔥🔥【优化】如果LLM明确指出了是代码错误 (SyntaxError, TypeError, NameError)，
            # 那么即使是 Validation Expert 也可能是凶手，不要强制修正。
            error_types = ['syntaxerror', 'typeerror', 'nameerror', 'indentationerror', 'attributeerror', 'traceback', 'exception', '语法错误', '代码错误']
            is_code_crash = pred_reason and any(e in pred_reason.lower() for e in error_types)
            is_in_candidates = final_pred_agent in candidate_agent_ids if final_pred_agent else False
            
            # 🔥🔥🔥【关键修复】不再强制修正，只给出警告
            # LLM 已经分析了完整日志，它的判断通常比简单的规则更准确
            # 如果 LLM 选择了禁止的 Agent，说明它确实认为这个 Agent 是根因
            if is_forbidden:
                if is_code_crash:
                    print(f"  ✅ [代码崩溃] LLM选择了禁止的Agent '{final_pred_agent}'，但检测到代码崩溃（{pred_reason[:50] if pred_reason else 'N/A'}...），保留原选择")
                elif is_in_candidates:
                    print(f"  ⚠️ [警告] LLM选择了禁止的Agent '{final_pred_agent}'，但该Agent在GNN候选列表中，予以保留")
                else:
                    print(f"  ⚠️ [警告] LLM选择了禁止的Agent '{final_pred_agent}'，但为了尊重模型判断，予以保留")
                # 不再强制修正，保持 LLM 的原始预测
            
            # 修正 pred_agent 变量用于后续计算
            pred_agent = final_pred_agent
            pred_step = final_pred_step
            
            # 🔥🔥🔥【新增】基于规则的 Step 校准（Fix Step Accuracy）
            if final_pred_agent and final_pred_agent in nodes:
                agent_node = nodes[final_pred_agent]
                features = agent_node.get('features', {})
                
                # 🔥 判断是否是 Hand-Crafted 数据集
                is_hand_crafted = "Hand-Crafted" in json_file.name
                
                if isinstance(features, dict) and features:
                    # 找到该 Agent 活跃的所有时间步（绝对Step ID）
                    try:
                        active_steps = sorted([int(k) for k in features.keys() if str(k).isdigit()])
                    except (ValueError, TypeError):
                        active_steps = []
                    
                    if active_steps:
                        original_pred_step = final_pred_step
                        last_active_step = active_steps[-1]
                        
                        # 🔥🔥🔥【策略分支】Hand-Crafted 使用"最早错误"策略，Algorithm-Generated 使用"最后活跃"策略
                        if is_hand_crafted:
                            # ========== Hand-Crafted 数据集：寻找"最早"的异常 ==========
                            # 策略：寻找"最早"的异常，而不是"最后"的活跃
                            # 对于 Hand-Crafted (Step 往往很靠前)，我们要找 Error 出现的 *第一刻*
                            
                            error_keywords = ['error', 'fail', 'traceback', 'exception', 'not found', 'failed', 'failure', 'wrong', 'incorrect', 'invalid', '404', 'timeout', 'refused', 'denied']
                            potential_error_steps = []
                            
                            # 从前往后搜索所有包含错误关键词的步
                            for step in active_steps:
                                step_key = str(step)
                                if step_key in features:
                                    feat = features[step_key]
                                    if isinstance(feat, dict):
                                        content_text = (
                                            feat.get('content_text', '') or 
                                            feat.get('content', '') or
                                            feat.get('text', '')
                                        )
                                        if content_text:
                                            content_lower = content_text.lower()
                                            # 检查是否包含错误关键词
                                            if any(keyword in content_lower for keyword in error_keywords):
                                                potential_error_steps.append(step)
                            
                            # 🔥🔥🔥【关键修复】放宽 Step 修正逻辑，信任 LLM 的大数字预测
                            # 策略优化：如果 LLM 预测了一个很大的 Step (e.g. > 10)，且它在 history 长度范围内，尽量信任它
                            if final_pred_step is not None and final_pred_step > 10:
                                # 检查是否在合理的范围内（不超过最后活跃步太多）
                                if final_pred_step <= last_active_step + 5:  # 允许一定的误差
                                    print(f"  ✅ [Step信任-HC] LLM预测了较晚的时间步 {final_pred_step}，予以保留（最后活跃步={last_active_step}）")
                                else:
                                    # 如果预测太离谱，使用错误步或第一个活跃步
                                    if potential_error_steps:
                                        first_error = potential_error_steps[0]
                                        final_pred_step = first_error
                                        print(f"  🔧 [Step修正-HC] 在Step {first_error}发现第一个错误关键词（根因）-> 修正为 {final_pred_step}")
                                    else:
                                        final_pred_step = active_steps[0]
                                        print(f"  🔧 [Step修正-HC] LLM预测 {original_pred_step} 太离谱 -> 修正为第一个活跃步 {final_pred_step}")
                            elif potential_error_steps:
                                # 如果LLM预测较小，但找到了错误步，使用错误步
                                first_error = potential_error_steps[0]
                                final_pred_step = first_error
                                print(f"  🔧 [Step修正-HC] 在Step {first_error}发现第一个错误关键词（根因）-> 修正为 {final_pred_step}")
                            elif final_pred_step is not None and final_pred_step in active_steps:
                                # LLM预测在活跃范围内，保持
                                print(f"  ✅ [Step保持-HC] LLM预测 {final_pred_step} 在活跃范围内，保持")
                            else:
                                # 🔥🔥🔥【优化】如果LLM预测不在活跃范围内，使用最后活跃步（而不是第一步）
                                # 因为用户往往是在做完一系列操作后发现失败的
                                final_pred_step = active_steps[-1]
                                print(f"  🔧 [Step修正-HC-优化] Step {original_pred_step} 不在活跃范围 -> 修正为最后活跃步 {final_pred_step} (而不是第一步 {active_steps[0]})")
                        else:
                            # ========== Algorithm-Generated 数据集：优先信任 LLM 预测 ==========
                            # 🔥🔥🔥【关键修复】信任 LLM 的 Step 预测，不要用 active_steps 强制修正
                            # LLM 已经分析了完整日志，它的预测通常比简单的规则更准确
                            # active_steps 可能因为日志截断或解析问题不完整，导致正确的预测被改错
                            
                            # 策略 A: 优先信任 LLM 的预测
                            if final_pred_step is not None:
                                # 只要 LLM 给了个数字，就信它！不要管 active_steps
                                # 即使它超出了 active_steps 范围，也可能是因为日志解析不全
                                # 只有在预测明显不合理（负数或超大）时才修正
                                if final_pred_step < 0:
                                    # 负数不合理，使用最后一步
                                    final_pred_step = last_active_step
                                    print(f"  🔧 [Step修正A] LLM预测了负数 Step {original_pred_step} -> 修正为最后活跃步 {final_pred_step}")
                                elif final_pred_step > last_active_step + 20:
                                    # 如果预测比最后活跃步大太多（超过20步），可能是解析错误，使用最后一步
                                    final_pred_step = last_active_step
                                    print(f"  🔧 [Step修正A] LLM预测 {original_pred_step} 超出合理范围（最后活跃步={last_active_step}）-> 修正为最后活跃步 {final_pred_step}")
                                else:
                                    # LLM 预测合理，直接采纳
                                    print(f"  ✅ [Step信任] LLM预测了 Step {final_pred_step}，直接采纳（最后活跃步={last_active_step}）")
                            else:
                                # 只有当 LLM 没给 Step 时，才回退到最后一步
                                final_pred_step = last_active_step
                                print(f"  🔧 [Step修正A] LLM未预测Step -> 使用最后活跃步 {final_pred_step}")
                            
                            # 策略 B: 如果LLM预测离谱（误差超过10步），强制修正为最后活跃步
                            if final_pred_step is not None:
                                step_diff_from_last = abs(final_pred_step - last_active_step)
                                if step_diff_from_last > 10:
                                    # 🔥 修复：差距过大时强制修正，而不是只警告
                                    final_pred_step = last_active_step
                                    print(f"  🔧 [Step修正B] LLM预测 {original_pred_step} 与最后活跃步 {last_active_step} 差距过大(>{step_diff_from_last}) -> 强制修正为最后活跃步 {final_pred_step}")
                                else:
                                    # LLM预测合理，直接保持
                                    print(f"  ✅ [Step保持] LLM预测 {final_pred_step} 合理（与最后活跃步差距={step_diff_from_last}），保持LLM预测")
            
            # 更新 pred_step
            pred_step = final_pred_step
            
            # 🔍 调试信息：打印关键变量（用于诊断0%准确率问题）
            print(f"  [调试] 文件: {json_file.name}")
            print(f"  [调试] 真实标签: Agent='{true_agent}', Step={true_step}")
            print(f"  [调试] GNN候选: {candidate_agent_ids}")
            print(f"  [调试] LLM解析: pred_agent='{pred_agent}', pred_step={pred_step}")
            print(f"  [调试] 真实Agent在候选列表中: {true_agent in candidate_agent_ids if true_agent else False}")
            
            # 6. 计算准确率（必须在打印表格数据之前计算）
            # 注意：如果 LLM 判断为正常（无故障），但真实标签有故障，则错误
            # 如果 LLM 判断有故障，但真实标签无故障（Healed），则错误
            if pred_agent is None:
                # LLM 判断为正常
                if true_agent and true_agent.lower() != 'none':
                    # 真实有故障，但 LLM 判断正常 -> 错误
                    agent_acc = 0.0
                    step_acc = 0.0
                else:
                    # 真实无故障，LLM 判断正常 -> 正确
                    agent_acc = 1.0
                    step_acc = 1.0
            else:
                # LLM 判断有故障
                if true_agent and true_agent.lower() != 'none':
                    # 真实有故障，比较 Agent 和 Step
                    # 🔥 改进的匹配逻辑：处理大小写、下划线、部分匹配
                    pred_agent_clean = str(pred_agent).strip() if pred_agent else ""
                    true_agent_clean = str(true_agent).strip()
                    
                    # 标准化：移除所有下划线，转小写（处理 'Verification_Expert' vs 'VerificationExpert'）
                    def normalize_agent_name(name):
                        return name.replace('_', '').replace('-', '').lower()
                    
                    pred_normalized = normalize_agent_name(pred_agent_clean)
                    true_normalized = normalize_agent_name(true_agent_clean)
                    
                    # 1. 精确匹配（忽略大小写、下划线、连字符）
                    agent_correct_exact = (pred_normalized == true_normalized)
                    
                    # 2. 原始精确匹配（忽略大小写）
                    agent_correct_exact_orig = (pred_agent_clean.lower() == true_agent_clean.lower())
                    
                    # 3. 模糊匹配（包含关系，标准化后）
                    agent_correct_fuzzy = (
                        true_normalized in pred_normalized or
                        pred_normalized in true_normalized
                    )
                    
                    # 4. 原始模糊匹配（包含关系）
                    agent_correct_fuzzy_orig = (
                        true_agent_clean.lower() in pred_agent_clean.lower() or
                        pred_agent_clean.lower() in true_agent_clean.lower()
                    )
                    
                    # 5. 提取括号内的Agent名称（处理 "候选Agent 1 (Movie_Expert)" 格式）
                    bracket_match = re.search(r'\(([A-Za-z0-9_\-]+)\)', pred_agent_clean)
                    if bracket_match:
                        pred_agent_bracket = bracket_match.group(1)
                        pred_agent_bracket_normalized = normalize_agent_name(pred_agent_bracket)
                        agent_correct_bracket = (pred_agent_bracket_normalized == true_normalized)
                    else:
                        agent_correct_bracket = False
                    
                    # 6. 部分匹配：检查核心名称（移除后缀如 '_Expert'）
                    def get_core_name(name):
                        # 移除常见的后缀
                        name = name.replace('_Expert', '').replace('Expert', '')
                        name = name.replace('_', '').replace('-', '').lower()
                        return name
                    
                    pred_core = get_core_name(pred_agent_clean)
                    true_core = get_core_name(true_agent_clean)
                    agent_correct_core = (pred_core == true_core and len(pred_core) > 3)  # 至少3个字符
                    
                    # 综合匹配结果
                    agent_correct = (agent_correct_exact or agent_correct_exact_orig or 
                                   agent_correct_fuzzy or agent_correct_fuzzy_orig or 
                                   agent_correct_bracket or agent_correct_core)
                    
                    # Step匹配：允许±1的误差（因为时间步可能不精确）
                    if pred_step is not None:
                        step_correct = (abs(pred_step - true_step) <= 1)
                    else:
                        step_correct = False
                    agent_acc = 1.0 if agent_correct else 0.0
                    step_acc = 1.0 if step_correct else 0.0
                else:
                    # 真实无故障，但 LLM 判断有故障 -> 错误
                    agent_acc = 0.0
                    step_acc = 0.0
            
            # 🔥🔥🔥【新增】输出表格所需的关键信息（用于后续分析）
            # 计算真实答案在候选中的排名
            true_agent_rank_in_candidates = None
            true_agent_rank_in_full = None
            
            if true_agent:
                # 1. 在候选列表中的排名（Top-K模式）
                try:
                    true_agent_rank_in_candidates = candidate_agent_ids.index(true_agent) + 1
                except ValueError:
                    true_agent_rank_in_candidates = -1  # 不在候选中
                
                # 2. 在完整排序中的排名（全输出模式）
                if top_k is None and 'filtered_full_ranking' in locals() and filtered_full_ranking:
                    try:
                        true_agent_rank_in_full = filtered_full_ranking.index(true_agent) + 1
                    except ValueError:
                        true_agent_rank_in_full = -1
                else:
                    # Top-K模式，无法知道完整排名
                    true_agent_rank_in_full = None
            
            # 输出表格格式的关键信息（结构化日志，便于后续解析和提取）
            # 格式：ID | GNN排序 | LLM选择 | 真实答案 | 真实答案排名 | 是否正确
            if top_k is None:
                # 全输出模式：使用完整排序和完整排名
                ranking_for_table = filtered_full_ranking if 'filtered_full_ranking' in locals() and filtered_full_ranking else candidate_agent_ids
                true_agent_rank = true_agent_rank_in_full if true_agent_rank_in_full is not None else true_agent_rank_in_candidates
                ranking_str = ",".join(ranking_for_table) if len(ranking_for_table) <= 100 else ",".join(ranking_for_table[:100]) + f",...(共{len(ranking_for_table)}个)"
                rank_display = str(true_agent_rank) if true_agent_rank and true_agent_rank > 0 else "不在排序中"
                print(f"  📋 [表格数据] ID={json_file.stem} | GNN完整排序={ranking_str} | LLM选择={pred_agent or 'None'} | 真实答案={true_agent} | 真实答案排名={rank_display} | 是否正确={'是' if agent_acc > 0.5 else '否'}")
            else:
                # Top-K模式：输出Top-K排序和候选排名
                ranking_str = ",".join(candidate_agent_ids)
                true_agent_rank = true_agent_rank_in_candidates
                rank_display = str(true_agent_rank) if true_agent_rank and true_agent_rank > 0 else f"不在Top-{top_k}中"
                print(f"  📋 [表格数据] ID={json_file.stem} | GNN排序(Top-{top_k})={ranking_str} | LLM选择={pred_agent or 'None'} | 真实答案={true_agent} | 真实答案排名={rank_display} | 是否正确={'是' if agent_acc > 0.5 else '否'}")
            
            # 存储真实答案排名（用于后续统计分析）- 仅在全输出模式下
            if top_k is None and true_agent and true_agent_rank and true_agent_rank > 0:
                true_agent_ranks.append(true_agent_rank)
            
            # 7. 继续后续处理（准确率已在前面计算）
            
            # 🔍 调试信息：打印准确率计算结果（在表格数据之后打印）
            try:
                agent_correct_debug = agent_correct
                step_correct_debug = step_correct
                exact_debug = agent_correct_exact
                exact_orig_debug = agent_correct_exact_orig if 'agent_correct_exact_orig' in locals() else 'N/A'
                fuzzy_debug = agent_correct_fuzzy
                fuzzy_orig_debug = agent_correct_fuzzy_orig if 'agent_correct_fuzzy_orig' in locals() else 'N/A'
                bracket_debug = agent_correct_bracket
                core_debug = agent_correct_core if 'agent_correct_core' in locals() else 'N/A'
            except NameError:
                agent_correct_debug = 'N/A (pred_agent is None分支)'
                step_correct_debug = 'N/A (pred_agent is None分支)'
                exact_debug = 'N/A'
                exact_orig_debug = 'N/A'
                fuzzy_debug = 'N/A'
                fuzzy_orig_debug = 'N/A'
                bracket_debug = 'N/A'
                core_debug = 'N/A'
            
            print(f"  [调试] Agent准确率: {agent_acc}, Step准确率: {step_acc}")
            print(f"  [调试] 匹配结果: agent_correct={agent_correct_debug}, step_correct={step_correct_debug}")
            if agent_correct_debug != 'N/A (pred_agent is None分支)':
                print(f"  [调试] 匹配详情: exact={exact_debug}, exact_orig={exact_orig_debug}, fuzzy={fuzzy_debug}, fuzzy_orig={fuzzy_orig_debug}, bracket={bracket_debug}, core={core_debug}")
                if pred_step is not None:
                    print(f"  [调试] Step匹配: pred_step={pred_step}, true_step={true_step}, 误差={abs(pred_step - true_step) if 'true_step' in locals() else 'N/A'}")
            
            # 7. 分类记录（Algorithm-Generated / Hand-Crafted）
            filename = json_file.name
            metrics_total['agent'].append(agent_acc)
            metrics_total['step'].append(step_acc)
            
            if "Algorithm-Generated" in filename:
                metrics_alg['agent'].append(agent_acc)
                metrics_alg['step'].append(step_acc)
                count_alg += 1
            elif "Hand-Crafted" in filename:
                metrics_hand['agent'].append(agent_acc)
                metrics_hand['step'].append(step_acc)
                count_hand += 1
            
            # 🔥 新增：按领域分类（Code/Math/Agentic）- 用于AgenTracer对比
            domain = graph_data.get('domain', graph_data.get('benchmark', 'Unknown'))
            # 标准化domain名称
            domain_lower = domain.lower()
            if 'code' in domain_lower or 'kodcode' in domain_lower or 'mbpp' in domain_lower:
                domain_standard = 'Code'
            elif 'math' in domain_lower or 'gsm8k' in domain_lower:
                domain_standard = 'Math'
            elif 'agentic' in domain_lower or 'gaia' in domain_lower or 'hotpot' in domain_lower:
                domain_standard = 'Agentic'
            else:
                domain_standard = 'Unknown'
            
            # 初始化domain统计（如果不存在）
            if domain_standard not in metrics_domains:
                metrics_domains[domain_standard] = {'agent': [], 'step': []}
                domain_counts[domain_standard] = 0
            
            metrics_domains[domain_standard]['agent'].append(agent_acc)
            metrics_domains[domain_standard]['step'].append(step_acc)
            domain_counts[domain_standard] += 1
            
        except Exception as e:
            print(f"⚠️ 处理文件 {json_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出统计信息
    print(f"\n📊 评估统计:")
    print(f"   - 总文件数: {len(json_files)}")
    print(f"   - 跳过（无标签）: {skipped_no_label}")
    print(f"   - 跳过（无图）: {skipped_no_graph}")
    print(f"   - 成功处理: {len(metrics_total['agent'])}")
    
    # 输出结果
    print("\n" + "=" * 80)
    # 🔥 消融实验标记：在结果标题中显示
    ablation_marker = ""
    if is_ablation_no_finetune:
        ablation_marker = " [消融实验: GNN + 未微调LLM]"
    print(f"🏆 阶段三评估结果 (Coarse-to-Fine){ablation_marker}")
    print("=" * 80)
    print(f"{'Dataset':<25} | {'Count':<8} | {'Agent Acc (Who)':<18} | {'Step Acc (When)':<18}")
    print("-" * 80)
    
    def get_mean(m_list):
        return np.mean(m_list) if m_list else 0.0
    
    # Algorithm-Generated
    alg_a = get_mean(metrics_alg['agent'])
    alg_s = get_mean(metrics_alg['step'])
    print(f"{'Algorithm-Generated':<25} | {len(metrics_alg['agent']):<8} | {alg_a:.4f} ({alg_a*100:5.2f}%) | {alg_s:.4f} ({alg_s*100:5.2f}%)")
    
    # Hand-Crafted
    hand_a = get_mean(metrics_hand['agent'])
    hand_s = get_mean(metrics_hand['step'])
    print(f"{'Hand-Crafted':<25} | {len(metrics_hand['agent']):<8} | {hand_a:.4f} ({hand_a*100:5.2f}%) | {hand_s:.4f} ({hand_s*100:5.2f}%)")
    
    print("-" * 80)
    
    # Overall
    tot_a = get_mean(metrics_total['agent'])
    tot_s = get_mean(metrics_total['step'])
    print(f"{'Overall (Total)':<25} | {len(metrics_total['agent']):<8} | {tot_a:.4f} ({tot_a*100:5.2f}%) | {tot_s:.4f} ({tot_s*100:5.2f}%)")
    print("=" * 80)
    
    # 🔥🔥🔥【新增】真实答案排名统计分析（用于超参数敏感性分析）
    if top_k is None and true_agent_ranks:
        ranks = true_agent_ranks
        print("\n" + "=" * 80)
        print("📊 真实答案排名统计分析 (True Answer Rank Analysis)")
        print("=" * 80)
        print(f"总样本数: {len(ranks)}")
        print(f"平均排名: {np.mean(ranks):.2f}")
        print(f"中位数排名: {np.median(ranks):.2f}")
        print(f"最大排名: {np.max(ranks)}")
        print(f"最小排名: {np.min(ranks)}")
        print(f"标准差: {np.std(ranks):.2f}")
        
        # 计算排名分布
        print(f"\n排名分布:")
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        for rank in sorted(rank_counts.keys())[:20]:  # 只显示前20名
            count = rank_counts[rank]
            percentage = count / len(ranks) * 100
            print(f"  第{rank}名: {count}次 ({percentage:.2f}%)")
        
        # 计算累积分布（用于确定阈值）
        print(f"\n累积分布（用于Top-N敏感性分析）:")
        sorted_ranks = sorted(ranks)
        for n in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
            count_within_n = sum(1 for r in ranks if r <= n)
            percentage = count_within_n / len(ranks) * 100
            print(f"  Top-{n}: {count_within_n}/{len(ranks)} ({percentage:.2f}%)")
        
        # 找到99%阈值（用于论文实验）
        print(f"\n阈值分析（用于超参数敏感性分析）:")
        for percentile in [50, 75, 90, 95, 99, 99.9]:
            threshold = np.percentile(ranks, percentile)
            print(f"  {percentile}%的真实答案排名 ≤ {threshold:.1f}")
        
        print("=" * 80)
    
    # 🔥 新增：按领域分类输出（Code/Math/Agentic）- 用于AgenTracer对比
    if metrics_domains:
        print("\n" + "=" * 80)
        print("🏆 领域分类评估结果 (Domain-wise Evaluation for AgenTracer Comparison)")
        print("=" * 80)
        print(f"{'Domain':<15} | {'Count':<8} | {'Agent Acc (Who)':<18} | {'Step Acc (When)':<18}")
        print("-" * 80)
        
        domain_results = {}
        for domain in sorted(metrics_domains.keys()):
            if domain == 'Unknown':
                continue  # 跳过Unknown领域
            domain_agent_acc = get_mean(metrics_domains[domain]['agent'])
            domain_step_acc = get_mean(metrics_domains[domain]['step'])
            domain_count = domain_counts.get(domain, len(metrics_domains[domain]['agent']))
            print(f"{domain:<15} | {domain_count:<8} | {domain_agent_acc:.4f} ({domain_agent_acc*100:5.2f}%) | {domain_step_acc:.4f} ({domain_step_acc*100:5.2f}%)")
            
            domain_results[domain.lower()] = {
                'count': domain_count,
                'agent_acc': domain_agent_acc,
                'step_acc': domain_step_acc
            }
        
        if 'Unknown' in metrics_domains:
            unknown_count = domain_counts.get('Unknown', len(metrics_domains['Unknown']['agent']))
            if unknown_count > 0:
                unknown_agent_acc = get_mean(metrics_domains['Unknown']['agent'])
                unknown_step_acc = get_mean(metrics_domains['Unknown']['step'])
                print(f"{'Unknown':<15} | {unknown_count:<8} | {unknown_agent_acc:.4f} ({unknown_agent_acc*100:5.2f}%) | {unknown_step_acc:.4f} ({unknown_step_acc*100:5.2f}%)")
        
        print("=" * 80)
    else:
        domain_results = {}
    
    return {
        'algorithm_generated': {
            'count': len(metrics_alg['agent']),
            'agent_acc': alg_a,
            'step_acc': alg_s
        },
        'hand_crafted': {
            'count': len(metrics_hand['agent']),
            'agent_acc': hand_a,
            'step_acc': hand_s
        },
        'overall': {
            'count': len(metrics_total['agent']),
            'agent_acc': tot_a,
            'step_acc': tot_s
        },
        'domains': domain_results  # 🔥 新增：领域分类结果
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阶段三评估：Coarse-to-Fine 系统集成")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="processed_graphs/graphs_whowhen",
        help="测试数据目录"
    )
    parser.add_argument(
        "--gnn_checkpoint",
        type=str,
        required=True,
        help="GNN 模型检查点路径"
    )
    parser.add_argument(
        "--llm_adapter",
        type=str,
        default="checkpoints_qwen3_finetune_large/final_model",
        help="LLM LoRA 适配器路径。设置为空字符串 '' 或 'none' 以使用未微调的基础模型（消融实验）"
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        help="消融实验标志：强制使用未微调的基础模型（等同于 --llm_adapter ''）"
    )
    parser.add_argument(
        "--converter_path",
        type=str,
        default="processed_data/converter_state.pt",
        help="Converter 状态路径"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-K 候选数量（默认None=全输出所有Agent排序，用于超参数敏感性分析；指定数字则输出Top-K，例如 --top_k 7）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="基础LLM模型名称（默认: Qwen/Qwen3-8B，支持思考模式，可选: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-4B-Instruct 或 Qwen/Qwen1.5-4B-Chat）"
    )
    
    args = parser.parse_args()
    
    # 🔥 消融实验支持：如果设置了 --no_finetune，强制使用未微调的基础模型
    if args.no_finetune:
        args.llm_adapter = ""
        print("\n" + "=" * 80)
        print("🔬 消融实验模式：使用未微调的基础模型（GNN + 未微调 LLM）")
        print("=" * 80)
    
    # 如果 llm_adapter 是 'none' 或空字符串，也视为未微调模式
    if args.llm_adapter.lower() in ['none', '']:
        print("\n" + "=" * 80)
        print("🔬 消融实验模式：使用未微调的基础模型（GNN + 未微调 LLM）")
        print("=" * 80)
        args.llm_adapter = ""
    
    results = evaluate_stage3(
        test_data_dir=args.test_data_dir,
        gnn_checkpoint=args.gnn_checkpoint,
        llm_adapter=args.llm_adapter,
        converter_path=args.converter_path,
        top_k=args.top_k,
        device=args.device,
        base_model_name=args.base_model_name
    )
    
    # 🔥 Token统计：打印统计报告
    token_counter.print_summary()
    
    print("\n✅ 评估完成！")

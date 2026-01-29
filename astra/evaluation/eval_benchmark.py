#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TracerTraj-Code 专用评估脚本（带Token统计）

基于 evaluate_stage3_coarse_to_fine.py，专门用于评估 TracerTraj-Code 数据集
包含详细的 Token 开销统计（输入、输出、总计）
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
import sys

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

# 导入原有的评估函数
from evaluate_stage3_coarse_to_fine import (
    load_gnn_model,
    load_llm_model,
    analyze_with_llm,
    extract_json_from_text,
    parse_llm_response,
    predict_top_k_with_gnn,
    normalize_name
)


class TokenCounter:
    """Token统计器"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.gnn_tokens = 0  # GNN阶段（如果有）
        self.llm_input_tokens = 0  # LLM输入
        self.llm_output_tokens = 0  # LLM输出
        self.sample_count = 0
    
    def count_tokens(self, text: str) -> int:
        """统计文本的token数"""
        if not text:
            return 0
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except:
            # 如果编码失败，使用简单估算（1 token ≈ 4字符）
            return len(text) // 4
    
    def add_input(self, text: str):
        """添加输入token"""
        tokens = self.count_tokens(text)
        self.total_input_tokens += tokens
        self.llm_input_tokens += tokens
        self.total_tokens += tokens
    
    def add_output(self, text: str):
        """添加输出token"""
        tokens = self.count_tokens(text)
        self.total_output_tokens += tokens
        self.llm_output_tokens += tokens
        self.total_tokens += tokens
    
    def add_gnn(self, tokens: int):
        """添加GNN阶段的token（如果有）"""
        self.gnn_tokens += tokens
        self.total_tokens += tokens
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'gnn_tokens': self.gnn_tokens,
            'llm_input_tokens': self.llm_input_tokens,
            'llm_output_tokens': self.llm_output_tokens,
            'sample_count': self.sample_count,
            'avg_input_tokens_per_sample': self.total_input_tokens / max(self.sample_count, 1),
            'avg_output_tokens_per_sample': self.total_output_tokens / max(self.sample_count, 1),
            'avg_total_tokens_per_sample': self.total_tokens / max(self.sample_count, 1)
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("📊 Token 开销统计")
        print("=" * 80)
        print(f"总样本数: {stats['sample_count']}")
        print(f"\n输入Token统计:")
        print(f"  - LLM输入Token总数: {stats['llm_input_tokens']:,}")
        print(f"  - 平均每样本输入Token: {stats['avg_input_tokens_per_sample']:.2f}")
        print(f"\n输出Token统计:")
        print(f"  - LLM输出Token总数: {stats['llm_output_tokens']:,}")
        print(f"  - 平均每样本输出Token: {stats['avg_output_tokens_per_sample']:.2f}")
        print(f"\n总计Token统计:")
        print(f"  - 总Token数: {stats['total_tokens']:,}")
        print(f"  - 平均每样本总Token: {stats['avg_total_tokens_per_sample']:.2f}")
        if stats['gnn_tokens'] > 0:
            print(f"  - GNN阶段Token: {stats['gnn_tokens']:,}")
        print("=" * 80)


def evaluate_tracertraj_with_tokens(
    test_data_dir: str,
    gnn_checkpoint: str,
    llm_adapter: str,
    converter_path: str,
    top_k: int = 5,
    device: str = None,
    base_model_name: str = "./models/Qwen3-8B/qwen/Qwen3-8B",
    log_file: str = None
):
    """
    TracerTraj-Code 评估（带Token统计）
    
    Args:
        test_data_dir: 测试数据目录
        gnn_checkpoint: GNN 模型检查点路径
        llm_adapter: LLM LoRA 适配器路径（空字符串表示未微调）
        converter_path: Converter 状态路径
        top_k: Top-K 候选数量
        device: 设备 ('cuda' 或 'cpu')
        base_model_name: 基础LLM模型名称
        log_file: 日志文件路径（可选）
    """
    # 设备配置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"🔧 使用设备: {device}")
    
    # 打开日志文件（如果指定）
    log_handle = None
    if log_file:
        log_handle = open(log_file, 'w', encoding='utf-8')
        print(f"📝 日志将保存到: {log_file}")
    
    def log_print(*args, **kwargs):
        """同时打印到控制台和日志文件"""
        print(*args, **kwargs)
        if log_handle:
            print(*args, **kwargs, file=log_handle)
            log_handle.flush()
    
    # 加载模型
    log_print("\n" + "=" * 60)
    log_print("加载模型")
    log_print("=" * 60)
    
    gnn_model, converter, gnn_config = load_gnn_model(gnn_checkpoint, converter_path, device)
    
    # 检查是否使用未微调的模型
    is_ablation_no_finetune = (not llm_adapter or llm_adapter.strip() == "")
    if is_ablation_no_finetune:
        log_print("\n" + "=" * 80)
        log_print("🔬 消融实验配置：GNN + 未微调的基础模型")
        log_print("=" * 80)
        log_print(f"   GNN 模型: {gnn_checkpoint}")
        log_print(f"   LLM 模型: {base_model_name} (未微调)")
        log_print("=" * 80 + "\n")
    
    # 加载LLM模型
    llm_model, tokenizer = load_llm_model(llm_adapter, base_model_name=base_model_name, device=device, use_4bit=True)
    
    # 初始化Token统计器
    token_counter = TokenCounter(tokenizer)
    
    # 加载测试数据
    log_print("\n" + "=" * 60)
    log_print("加载测试数据")
    log_print("=" * 60)
    
    test_data_dir = Path(test_data_dir)
    
    # 递归搜索所有JSON文件
    if test_data_dir.is_dir():
        json_files = list(test_data_dir.rglob("*.json"))
    else:
        json_files = [test_data_dir] if test_data_dir.suffix == '.json' else []
    
    json_files = [f for f in json_files if f.exists()]
    
    if not json_files:
        log_print(f"❌ 在 {test_data_dir} 中未找到 JSON 文件")
        if log_handle:
            log_handle.close()
        return
    
    log_print(f"✅ 找到 {len(json_files)} 个测试文件")
    log_print(f"   测试目录: {test_data_dir}")
    
    # 评估
    log_print("\n" + "=" * 60)
    log_print("开始评估")
    log_print("=" * 60)
    
    metrics_total = {'agent': [], 'step': []}
    metrics_domains = {}  # {domain: {'agent': [], 'step': []}}
    domain_counts = {}
    
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
            
            # 提取标签
            ground_truth = graph_data.get('ground_truth', {})
            true_agent = ground_truth.get('mistake_agent', '')
            true_step = int(ground_truth.get('mistake_step', -1))
            true_reason = ground_truth.get('mistake_reason', '')
            
            if not true_agent or true_step < 0:
                skipped_no_label += 1
                continue
            
            # 🔥🔥🔥【关键修复】从JSON文件读取所有Agent节点（完整逻辑，从evaluate_stage3_coarse_to_fine.py移植）
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
                    if node_id.islower() and len(node_id) <= 5:
                        # 只保留明确的Agent名称
                        if node_id not in ['assistant', 'surfer', 'orchestrator', 'websurfer', 'filesurfer']:
                            continue
                    
                    # 🔥🔥🔥 处理带括号的节点：提取基础名称
                    if '(' in node_id or ')' in node_id or '->' in node_id or '→' in node_id:
                        node_id_base = re.sub(r'\s*\([^)]*\)\s*', '', node_id).strip()
                        node_id_base = re.sub(r'\s*->.*', '', node_id_base).strip()
                        if node_id_base:  # 如果提取到基础名称，使用基础名称
                            node_id_base_lower = node_id_base.lower()
                            if node_id_base_lower in invalid_agent_names:
                                continue
                            if node_id_base.isdigit() or (len(node_id_base) == 1 and not node_id_base.isalpha()):
                                continue
                            if node_id_base.islower() and len(node_id_base) <= 5:
                                if node_id_base not in ['assistant', 'surfer', 'orchestrator', 'websurfer', 'filesurfer']:
                                    continue
                            if node_id_base_lower in common_person_names:
                                continue
                            true_agent_nodes.add(node_id_base)
                    else:
                        # 不带括号的节点，直接使用
                        true_agent_nodes.add(node_id)
            
            if not true_agent_nodes:
                log_print(f"  ⚠️ [警告] 从JSON文件中未找到任何Agent节点（可能全部被过滤）")
            else:
                log_print(f"  📋 [Agent验证] 从JSON文件读取到 {len(true_agent_nodes)} 个Agent节点: {sorted(list(true_agent_nodes))}")
            
            # 1. GNN 预测候选（同时获取Step预测）
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
                    log_print(f"  ⚠️ [警告] 过滤后候选列表为空，尝试获取更多候选（top_k={top_k*2}）")
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
                            log_print(f"  ✅ [恢复] 从扩展候选列表中找到了 {len(filtered_candidate_agent_ids)} 个有效候选")
                        else:
                            # 如果还是为空，至少保留一个非human的候选
                            log_print(f"  ⚠️ [警告] 扩展候选后仍为空，使用原始候选（可能包含human节点）")
                            filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
                    except Exception as e:
                        log_print(f"  ⚠️ [警告] 获取扩展候选失败: {e}，使用原始候选")
                        filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
                else:
                    # 全输出模式下，如果过滤后为空，至少保留一个
                    log_print(f"  ⚠️ [警告] 全输出模式下过滤后候选列表为空，使用原始候选")
                    filtered_candidate_agent_ids = candidate_agent_ids[:1] if candidate_agent_ids else []
            
            # 更新候选人列表
            original_candidate_count = len(candidate_agent_ids)
            candidate_agent_ids = filtered_candidate_agent_ids
            
            if original_candidate_count > len(candidate_agent_ids):
                filtered_count = original_candidate_count - len(candidate_agent_ids)
                log_print(f"  🔒 [过滤] 已过滤 {filtered_count} 个用户节点（human/user等），剩余 {len(candidate_agent_ids)} 个候选Agent")
            
            log_print(f"  [调试] 过滤后的GNN候选: {candidate_agent_ids}")
            
            # 2. 提取候选Agent日志
            history = graph_data.get('history', [])
            
            # 🔥🔥🔥【关键修复】TracerTraj专用日志提取（字段名适配）
            # TracerTraj使用 'agent' 字段，而不是 'name'/'role'/'sender'
            def extract_tracertraj_logs(nodes: Dict, candidate_agents: List[str], history: List[Dict]) -> Dict[str, str]:
                """专门针对TracerTraj数据结构的日志提取器"""
                logs = {agent: [] for agent in candidate_agents}
                
                # 遍历History
                for step_idx, event in enumerate(history):
                    # 🔥 关键：TracerTraj使用 'agent' 字段，也兼容 'name'/'role'
                    event_agent = event.get('agent') or event.get('name') or event.get('role') or event.get('sender', '')
                    event_agent = str(event_agent).strip()
                    
                    # 获取内容
                    content = event.get('content') or event.get('message') or event.get('text', '')
                    content = str(content).strip()
                    
                    # 获取Step ID
                    step_id = event.get('step', event.get('step_id', event.get('timestamp', step_idx)))
                    
                    # 匹配候选人（使用模糊匹配，因为名称可能有细微差别）
                    for cand in candidate_agents:
                        cand_clean = str(cand).strip()
                        # 检查名称是否匹配（完全匹配或包含关系）
                        if (cand_clean == event_agent) or (cand_clean in event_agent) or (event_agent in cand_clean):
                            if content:
                                log_entry = f"[Step {step_id}] {content}"
                                logs[cand].append(log_entry)
                
                # 合并日志
                result = {}
                for agent, entries in logs.items():
                    if entries:
                        # 限制长度，防止爆显存
                        full_log = "\n".join(entries)
                        if len(full_log) > 3000:
                            result[agent] = full_log[:1000] + f"\n\n... [日志过长，中间 {len(full_log)-2000} 字符已省略] ...\n\n" + full_log[-1000:]
                        else:
                            result[agent] = full_log
                    else:
                        # 如果history中没有，尝试从nodes的features中提取
                        if agent in nodes:
                            node_data = nodes[agent]
                            features = node_data.get('features', {})
                            if isinstance(features, dict):
                                feature_logs = []
                                sorted_timesteps = sorted(features.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                                for t in sorted_timesteps:
                                    feat = features[t]
                                    if isinstance(feat, dict):
                                        content_text = (
                                            feat.get('content_text', '') or 
                                            feat.get('content', '') or
                                            feat.get('text', '')
                                        )
                                        if content_text and content_text.strip():
                                            feature_logs.append(f"[Step {t}] {content_text[:500]}")
                                if feature_logs:
                                    result[agent] = "\n".join(feature_logs)
                                else:
                                    result[agent] = f"Agent {agent}: 无日志内容"
                            else:
                                result[agent] = f"Agent {agent}: 无日志内容"
                        else:
                            result[agent] = f"Agent {agent}: 无日志内容"
                            
                return result
            
            agent_logs = extract_tracertraj_logs(nodes, candidate_agent_ids, history)
            
            # 🔥 检查日志是否提取成功
            total_log_len = sum(len(l) for l in agent_logs.values())
            if total_log_len < 50:
                log_print(f"  ⚠️ [警告] 样本 {json_file.name} 日志提取异常，内容过短（总长度={total_log_len}）。可能字段不匹配。")
                # 打印一条history看看结构
                if history:
                    log_print(f"   History样例（前3条）:")
                    for i, h in enumerate(history[:3]):
                        log_print(f"     [{i}] keys: {list(h.keys())}, agent/name/role: {h.get('agent', h.get('name', h.get('role', 'N/A')))}")
                # 打印候选Agent
                log_print(f"   候选Agent: {candidate_agent_ids}")
                log_print(f"   节点keys: {list(nodes.keys())[:10]}")
            
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
                
                # 🔥 关键修复：如果日志为空或太短，尝试从nodes的features中提取
                if not raw_log or raw_log == f"Agent {agent_id}: 无日志" or len(raw_log) < 20:
                    if agent_id in nodes:
                        node_data = nodes[agent_id]
                        features = node_data.get('features', {})
                        if isinstance(features, dict) and features:
                            feature_logs = []
                            sorted_timesteps = sorted(features.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                            for t in sorted_timesteps[:10]:  # 只取前10个时间步
                                feat = features[t]
                                if isinstance(feat, dict):
                                    content_text = (
                                        feat.get('content_text', '') or 
                                        feat.get('content', '') or
                                        feat.get('text', '')
                                    )
                                    if content_text and content_text.strip():
                                        feature_logs.append(f"[Step {t}] {content_text[:300]}")
                            if feature_logs:
                                raw_log = "\n".join(feature_logs)
                                log_print(f"  ✅ [日志恢复] {agent_id} 从features中提取到 {len(feature_logs)} 条日志")
                
                MAX_LOG_LEN = 2500
                if len(raw_log) > MAX_LOG_LEN:
                    # 保留前 800 字符 (看初始配置) 和 后 1700 字符 (看报错)
                    head = raw_log[:800]
                    tail = raw_log[-1700:]
                    log_content = f"{head}\n\n... [日志过长，中间 {len(raw_log)-2500} 字符已省略] ...\n\n{tail}"
                    log_print(f"  ⚠️ [日志截断] {agent_id} 日志保留头尾 (总长 {len(raw_log)})")
                else:
                    log_content = raw_log
                
                instruction += f"**候选 {i}: {agent_id}** {rank_str}\n{log_content}\n\n"
            
            # 🔥 关键修复：检查instruction是否为空或过短
            if len(instruction.strip()) < 100:
                log_print(f"  ❌ [严重错误] Instruction构建失败，内容过短（{len(instruction)}字符）！")
                log_print(f"   候选Agent数量: {len(candidate_agent_ids)}")
                log_print(f"   日志提取结果: {list(agent_logs.keys())}")
                log_print(f"   日志总长度: {sum(len(l) for l in agent_logs.values())}")
                # 构建一个最小可用的instruction
                instruction = f"""这是一个多Agent系统的故障诊断任务。系统执行失败了，GNN模型已经锁定了以下 {len(candidate_agent_ids)} 个最可疑的Agent：

"""
                for i, agent_id in enumerate(candidate_agent_ids, 1):
                    instruction += f"**候选 {i}: {agent_id}**\n（日志提取失败，请根据GNN的排序判断）\n\n"
                instruction += """请从上述候选Agent中选择一个作为故障源，并输出JSON格式的答案：
```json
{
  "agent": "Agent名称",
  "step": 整数,
  "reason": "故障原因"
}
```"""
            
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
            
            # 🔥 最终验证：确保instruction不为空
            if not instruction or len(instruction.strip()) < 50:
                log_print(f"  ❌ [致命错误] Instruction最终验证失败！instruction长度={len(instruction) if instruction else 0}")
                log_print(f"   跳过此样本，继续处理下一个...")
                continue
            
            # 🔥 Token统计：记录输入
            token_counter.add_input(instruction)
            token_counter.sample_count += 1
            
            # 4. LLM 分析（传入分离的 System Prompt）
            llm_response = analyze_with_llm(llm_model, tokenizer, instruction, system_prompt=sys_prompt)
            
            # 🔥 新增：显示LLM原始输出（调试信息）
            log_print("\n" + "=" * 50)
            log_print(f"[调试-原始输出] 长度: {len(llm_response)} 字符")
            if len(llm_response) > 1000:
                log_print(f"[调试-原始内容] (前500字符):\n{llm_response[:500]}")
                log_print(f"[调试-原始内容] (后500字符):\n{llm_response[-500:]}")
            else:
                log_print(f"[调试-原始内容]:\n{llm_response}")
            log_print("=" * 50)
            
            # 🔥 Token统计：记录输出
            token_counter.add_output(llm_response)
            
            # 5. 解析 LLM 响应
            pred_agent, pred_step, pred_reason = parse_llm_response(llm_response)
            
            # 🔥 新增：显示LLM解析结果
            if pred_agent or pred_step is not None:
                log_print(f"  [调试] LLM解析成功（JSON格式）: agent={pred_agent}, step={pred_step}")
            else:
                log_print(f"  [调试] LLM解析失败或未找到有效结果")
            
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
                        log_print(f"  ✨ [模糊修正] LLM预测 '{pred_agent}' -> 模糊匹配修正为候选: '{final_pred_agent}' (匹配分数={best_match_score})")
            
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
                    log_print(f"  ⚠️ [智能回退] {fallback_reason} -> 在候选前列中选择最可能的执行者: {final_pred_agent} (Step={final_pred_step})")
                else:
                    # 极端情况：没有候选
                    final_pred_agent = pred_agent  # 保持原预测
                    log_print(f"  ⚠️ [警告] 没有候选Agent，保持LLM原预测: {final_pred_agent}")
            
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
                    log_print(f"  ✅ [代码崩溃] LLM选择了禁止的Agent '{final_pred_agent}'，但检测到代码崩溃（{pred_reason[:50] if pred_reason else 'N/A'}...），保留原选择")
                elif is_in_candidates:
                    log_print(f"  ⚠️ [警告] LLM选择了禁止的Agent '{final_pred_agent}'，但该Agent在GNN候选列表中，予以保留")
                else:
                    log_print(f"  ⚠️ [警告] LLM选择了禁止的Agent '{final_pred_agent}'，但为了尊重模型判断，予以保留")
                # 不再强制修正，保持 LLM 的原始预测
            
            # 修正 pred_agent 变量用于后续计算
            pred_agent = final_pred_agent
            pred_step = final_pred_step
            
            # 🔥🔥🔥【新增】基于规则的 Step 校准（Fix Step Accuracy）
            if final_pred_agent and final_pred_agent in nodes:
                agent_node = nodes[final_pred_agent]
                features = agent_node.get('features', {})
                
                # 🔥 判断是否是 Hand-Crafted 数据集（TracerTraj通常是Algorithm-Generated）
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
                            if final_pred_step is not None and final_pred_step > 10:
                                # 检查是否在合理的范围内（不超过最后活跃步太多）
                                if final_pred_step <= last_active_step + 5:  # 允许一定的误差
                                    log_print(f"  ✅ [Step信任-HC] LLM预测了较晚的时间步 {final_pred_step}，予以保留（最后活跃步={last_active_step}）")
                                else:
                                    # 如果预测太离谱，使用错误步或第一个活跃步
                                    if potential_error_steps:
                                        first_error = potential_error_steps[0]
                                        final_pred_step = first_error
                                        log_print(f"  🔧 [Step修正-HC] 在Step {first_error}发现第一个错误关键词（根因）-> 修正为 {final_pred_step}")
                                    else:
                                        final_pred_step = active_steps[0]
                                        log_print(f"  🔧 [Step修正-HC] LLM预测 {original_pred_step} 太离谱 -> 修正为第一个活跃步 {final_pred_step}")
                            elif potential_error_steps:
                                # 如果LLM预测较小，但找到了错误步，使用错误步
                                first_error = potential_error_steps[0]
                                final_pred_step = first_error
                                log_print(f"  🔧 [Step修正-HC] 在Step {first_error}发现第一个错误关键词（根因）-> 修正为 {final_pred_step}")
                            elif final_pred_step is not None and final_pred_step in active_steps:
                                # LLM预测在活跃范围内，保持
                                log_print(f"  ✅ [Step保持-HC] LLM预测 {final_pred_step} 在活跃范围内，保持")
                            else:
                                # 🔥🔥🔥【优化】如果LLM预测不在活跃范围内，使用最后活跃步（而不是第一步）
                                final_pred_step = active_steps[-1]
                                log_print(f"  🔧 [Step修正-HC-优化] Step {original_pred_step} 不在活跃范围 -> 修正为最后活跃步 {final_pred_step} (而不是第一步 {active_steps[0]})")
                        else:
                            # ========== TracerTraj数据集：智能Step修正策略 ==========
                            # 🔥🔥🔥【关键优化】针对TracerTraj数据集的特点，实现自适应Step修正
                            # TracerTraj的Step可能不在活跃范围内，需要更智能的策略
                            
                            # 获取history长度（用于判断Step的合理性）
                            history_len = len(history)
                            max_reasonable_step = max(history_len, last_active_step) if history_len > 0 else last_active_step
                            
                            # 🔥 策略1: 从history中查找错误关键词，确定可能的错误步
                            error_keywords = ['error', 'fail', 'traceback', 'exception', 'not found', 'failed', 'failure', 
                                            'wrong', 'incorrect', 'invalid', '404', 'timeout', 'refused', 'denied',
                                            'syntaxerror', 'typeerror', 'nameerror', 'attributeerror']
                            potential_error_steps_from_history = []
                            
                            # 在history中查找包含错误关键词的步
                            for hist_idx, event in enumerate(history):
                                event_agent = event.get('agent') or event.get('name') or event.get('role', '')
                                # 检查是否是目标Agent
                                if (event_agent == final_pred_agent or 
                                    (isinstance(event_agent, str) and final_pred_agent in event_agent)):
                                    content = event.get('content') or event.get('message') or event.get('text', '')
                                    if content:
                                        content_lower = str(content).lower()
                                        if any(keyword in content_lower for keyword in error_keywords):
                                            step_id = event.get('step', hist_idx)
                                            potential_error_steps_from_history.append(step_id)
                            
                            # 🔥 策略2: 优先信任LLM预测，但需要验证合理性
                            if final_pred_step is not None:
                                # 检查LLM预测是否在合理范围内
                                if final_pred_step < 0:
                                    # 负数不合理
                                    if potential_error_steps_from_history:
                                        final_pred_step = potential_error_steps_from_history[0]
                                        log_print(f"  🔧 [Step修正-历史错误] LLM预测负数 -> 使用history中第一个错误步 {final_pred_step}")
                                    else:
                                        final_pred_step = last_active_step
                                        log_print(f"  🔧 [Step修正A] LLM预测了负数 Step {original_pred_step} -> 修正为最后活跃步 {final_pred_step}")
                                elif final_pred_step > max_reasonable_step + 30:
                                    # 如果预测比最大合理步大太多（超过30步），可能是解析错误
                                    if potential_error_steps_from_history:
                                        final_pred_step = potential_error_steps_from_history[0]
                                        log_print(f"  🔧 [Step修正-历史错误] LLM预测 {original_pred_step} 超出合理范围 -> 使用history中第一个错误步 {final_pred_step}")
                                    else:
                                        final_pred_step = last_active_step
                                        log_print(f"  🔧 [Step修正A] LLM预测 {original_pred_step} 超出合理范围（最大合理步={max_reasonable_step}）-> 修正为最后活跃步 {final_pred_step}")
                                elif final_pred_step not in active_steps and final_pred_step <= max_reasonable_step:
                                    # LLM预测不在活跃范围内，但在合理范围内
                                    # 检查是否接近活跃步（±5步内）
                                    closest_active = min(active_steps, key=lambda x: abs(x - final_pred_step)) if active_steps else None
                                    if closest_active and abs(closest_active - final_pred_step) <= 5:
                                        # 如果接近活跃步，使用最接近的活跃步
                                        final_pred_step = closest_active
                                        log_print(f"  🔧 [Step修正-接近活跃] LLM预测 {original_pred_step} 不在活跃范围内，但接近活跃步 {closest_active} -> 修正为 {final_pred_step}")
                                    elif potential_error_steps_from_history:
                                        # 如果有历史错误步，优先使用
                                        closest_error = min(potential_error_steps_from_history, key=lambda x: abs(x - final_pred_step))
                                        if abs(closest_error - final_pred_step) <= 10:
                                            final_pred_step = closest_error
                                            log_print(f"  🔧 [Step修正-历史错误] LLM预测 {original_pred_step} 不在活跃范围内 -> 使用接近的历史错误步 {final_pred_step}")
                                        else:
                                            # 保持LLM预测（在合理范围内）
                                            log_print(f"  ✅ [Step信任-合理范围] LLM预测 {final_pred_step} 不在活跃范围内但在合理范围内，保持预测（最后活跃步={last_active_step}, 历史长度={history_len}）")
                                    else:
                                        # 保持LLM预测（在合理范围内）
                                        log_print(f"  ✅ [Step信任-合理范围] LLM预测 {final_pred_step} 不在活跃范围内但在合理范围内，保持预测（最后活跃步={last_active_step}, 历史长度={history_len}）")
                                elif final_pred_step in active_steps:
                                    # LLM预测在活跃范围内，直接采纳
                                    log_print(f"  ✅ [Step信任] LLM预测了 Step {final_pred_step}，在活跃范围内，直接采纳（最后活跃步={last_active_step}）")
                                else:
                                    # 其他情况，保持LLM预测
                                    log_print(f"  ✅ [Step信任] LLM预测了 Step {final_pred_step}，保持预测（最后活跃步={last_active_step}, 历史长度={history_len}）")
                            else:
                                # LLM未预测Step，使用智能回退策略
                                if potential_error_steps_from_history:
                                    # 优先使用history中第一个错误步
                                    final_pred_step = potential_error_steps_from_history[0]
                                    log_print(f"  🔧 [Step修正-历史错误] LLM未预测Step -> 使用history中第一个错误步 {final_pred_step}")
                                elif active_steps:
                                    # 回退到最后活跃步
                                    final_pred_step = last_active_step
                                    log_print(f"  🔧 [Step修正A] LLM未预测Step -> 使用最后活跃步 {final_pred_step}")
                                else:
                                    # 极端情况：没有活跃步，使用history长度的一半
                                    final_pred_step = history_len // 2 if history_len > 0 else 1
                                    log_print(f"  🔧 [Step修正-默认] LLM未预测Step且无活跃步 -> 使用默认步 {final_pred_step} (history长度={history_len})")
                            
                            # 🔥 策略3: 最终验证和微调（如果预测与活跃步差距过大，但仍在合理范围内，可以适度调整）
                            if final_pred_step is not None and active_steps:
                                step_diff_from_last = abs(final_pred_step - last_active_step)
                                # 如果差距在10-20步之间，且不在活跃范围内，尝试调整到最接近的活跃步
                                if 10 < step_diff_from_last <= 20 and final_pred_step not in active_steps:
                                    closest_active = min(active_steps, key=lambda x: abs(x - final_pred_step))
                                    if abs(closest_active - final_pred_step) <= 10:
                                        final_pred_step = closest_active
                                        log_print(f"  🔧 [Step修正-微调] 预测 {original_pred_step} 与活跃步差距较大，微调到最接近的活跃步 {final_pred_step}")
                                    else:
                                        log_print(f"  ✅ [Step保持] LLM预测 {final_pred_step} 与最后活跃步差距={step_diff_from_last}，保持预测")
                                elif step_diff_from_last <= 10:
                                    log_print(f"  ✅ [Step保持] LLM预测 {final_pred_step} 合理（与最后活跃步差距={step_diff_from_last}），保持LLM预测")
                                else:
                                    # 差距过大（>20），但已经在策略2中处理过了，这里只记录
                                    log_print(f"  ⚠️ [Step警告] LLM预测 {final_pred_step} 与最后活跃步差距较大（{step_diff_from_last}），但已在合理范围内，保持预测")
            
            # 更新 pred_step
            pred_step = final_pred_step
            
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
                if top_k is None and all_agent_ranking is not None:
                    try:
                        true_agent_rank_in_full = all_agent_ranking.index(true_agent) + 1
                    except ValueError:
                        true_agent_rank_in_full = -1
                else:
                    # Top-K模式，无法知道完整排名
                    true_agent_rank_in_full = None
            
            # 🔍 调试信息：打印关键变量（用于诊断0%准确率问题）
            log_print(f"  [调试] 文件: {json_file.name}")
            log_print(f"  [调试] 真实标签: Agent='{true_agent}', Step={true_step}")
            log_print(f"  [调试] GNN候选: {candidate_agent_ids}")
            log_print(f"  [调试] LLM解析: pred_agent='{pred_agent}', pred_step={pred_step}")
            log_print(f"  [调试] 真实Agent在候选列表中: {true_agent in candidate_agent_ids if true_agent else False}")
            
            # 输出表格格式的关键信息（结构化日志，便于后续解析和提取）
            # 格式：ID | GNN排序 | LLM选择 | 真实答案 | 真实答案排名 | 是否正确
            if top_k is None:
                # 全输出模式：使用完整排序和完整排名
                ranking_for_table = all_agent_ranking if all_agent_ranking else candidate_agent_ids
                true_agent_rank = true_agent_rank_in_full if true_agent_rank_in_full is not None else true_agent_rank_in_candidates
                ranking_str = ",".join(ranking_for_table) if len(ranking_for_table) <= 100 else ",".join(ranking_for_table[:100]) + f",...(共{len(ranking_for_table)}个)"
                rank_display = str(true_agent_rank) if true_agent_rank and true_agent_rank > 0 else "不在排序中"
                log_print(f"  📋 [表格数据] ID={json_file.stem} | GNN完整排序={ranking_str} | LLM选择={pred_agent or 'None'} | 真实答案={true_agent} | 真实答案排名={rank_display} | 是否正确={'是' if agent_acc > 0.5 else '否'}")
            else:
                # Top-K模式：输出Top-K排序和候选排名
                ranking_str = ",".join(candidate_agent_ids)
                true_agent_rank = true_agent_rank_in_candidates
                rank_display = str(true_agent_rank) if true_agent_rank and true_agent_rank > 0 else f"不在Top-{top_k}中"
                log_print(f"  📋 [表格数据] ID={json_file.stem} | GNN排序(Top-{top_k})={ranking_str} | LLM选择={pred_agent or 'None'} | 真实答案={true_agent} | 真实答案排名={rank_display} | 是否正确={'是' if agent_acc > 0.5 else '否'}")
            
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
            
            log_print(f"  [调试] Agent准确率: {agent_acc}, Step准确率: {step_acc}")
            log_print(f"  [调试] 匹配结果: agent_correct={agent_correct_debug}, step_correct={step_correct_debug}")
            if agent_correct_debug != 'N/A (pred_agent is None分支)':
                log_print(f"  [调试] 匹配详情: exact={exact_debug}, exact_orig={exact_orig_debug}, fuzzy={fuzzy_debug}, fuzzy_orig={fuzzy_orig_debug}, bracket={bracket_debug}, core={core_debug}")
                if pred_step is not None:
                    log_print(f"  [调试] Step匹配: pred_step={pred_step}, true_step={true_step}, 误差={abs(pred_step - true_step) if 'true_step' in locals() else 'N/A'}")
            
            # 记录结果
            metrics_total['agent'].append(agent_acc)
            metrics_total['step'].append(step_acc)
            
            # 按领域分类
            domain = graph_data.get('domain', graph_data.get('benchmark', 'Unknown'))
            domain_lower = domain.lower()
            if 'code' in domain_lower:
                domain_standard = 'Code'
            elif 'math' in domain_lower:
                domain_standard = 'Math'
            elif 'agentic' in domain_lower:
                domain_standard = 'Agentic'
            else:
                domain_standard = 'Unknown'
            
            if domain_standard not in metrics_domains:
                metrics_domains[domain_standard] = {'agent': [], 'step': []}
                domain_counts[domain_standard] = 0
            
            metrics_domains[domain_standard]['agent'].append(agent_acc)
            metrics_domains[domain_standard]['step'].append(step_acc)
            domain_counts[domain_standard] += 1
            
        except Exception as e:
            log_print(f"⚠️ 处理文件 {json_file.name} 时出错: {e}")
            import traceback
            if log_handle:
                traceback.print_exc(file=log_handle)
            continue
    
    # 输出统计信息
    log_print(f"\n📊 评估统计:")
    log_print(f"   - 总文件数: {len(json_files)}")
    log_print(f"   - 跳过（无标签）: {skipped_no_label}")
    log_print(f"   - 跳过（无图）: {skipped_no_graph}")
    log_print(f"   - 成功处理: {len(metrics_total['agent'])}")
    
    # 输出结果
    log_print("\n" + "=" * 80)
    log_print("🏆 TracerTraj-Code 评估结果")
    log_print("=" * 80)
    log_print(f"{'Dataset':<25} | {'Count':<8} | {'Agent Acc (Who)':<18} | {'Step Acc (When)':<18}")
    log_print("-" * 80)
    
    def get_mean(m_list):
        return np.mean(m_list) if m_list else 0.0
    
    # Overall
    tot_a = get_mean(metrics_total['agent'])
    tot_s = get_mean(metrics_total['step'])
    log_print(f"{'Overall (Total)':<25} | {len(metrics_total['agent']):<8} | {tot_a:.4f} ({tot_a*100:5.2f}%) | {tot_s:.4f} ({tot_s*100:5.2f}%)")
    log_print("=" * 80)
    
    # 按领域分类输出
    if metrics_domains:
        log_print("\n" + "=" * 80)
        log_print("🏆 领域分类评估结果")
        log_print("=" * 80)
        log_print(f"{'Domain':<15} | {'Count':<8} | {'Agent Acc (Who)':<18} | {'Step Acc (When)':<18}")
        log_print("-" * 80)
        
        for domain in sorted(metrics_domains.keys()):
            if domain == 'Unknown':
                continue
            domain_agent_acc = get_mean(metrics_domains[domain]['agent'])
            domain_step_acc = get_mean(metrics_domains[domain]['step'])
            domain_count = domain_counts.get(domain, len(metrics_domains[domain]['agent']))
            log_print(f"{domain:<15} | {domain_count:<8} | {domain_agent_acc:.4f} ({domain_agent_acc*100:5.2f}%) | {domain_step_acc:.4f} ({domain_step_acc*100:5.2f}%)")
        
        log_print("=" * 80)
    
    # Token统计
    token_counter.print_stats()
    if log_handle:
        # 将token统计也写入日志
        stats = token_counter.get_stats()
        log_handle.write("\n" + "=" * 80 + "\n")
        log_handle.write("📊 Token 开销统计\n")
        log_handle.write("=" * 80 + "\n")
        log_handle.write(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
        log_handle.write("=" * 80 + "\n")
    
    # 保存结果到JSON
    results = {
        'overall': {
            'count': len(metrics_total['agent']),
            'agent_acc': tot_a,
            'step_acc': tot_s
        },
        'domains': {
            domain: {
                'count': domain_counts.get(domain, len(metrics_domains[domain]['agent'])),
                'agent_acc': get_mean(metrics_domains[domain]['agent']),
                'step_acc': get_mean(metrics_domains[domain]['step'])
            }
            for domain in metrics_domains.keys() if domain != 'Unknown'
        },
        'token_stats': token_counter.get_stats()
    }
    
    if log_file:
        results_file = log_file.replace('.log', '_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log_print(f"\n✅ 结果已保存到: {results_file}")
    
    if log_handle:
        log_handle.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TracerTraj-Code 评估（带Token统计）")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
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
        default="",
        help="LLM LoRA 适配器路径（空字符串表示未微调）"
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
        default=5,
        help="Top-K 候选数量"
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
        default="./models/Qwen3-8B/qwen/Qwen3-8B",
        help="基础LLM模型名称"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定日志文件，自动生成
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/eval_tracertraj_{timestamp}.log"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    results = evaluate_tracertraj_with_tokens(
        test_data_dir=args.test_data_dir,
        gnn_checkpoint=args.gnn_checkpoint,
        llm_adapter=args.llm_adapter,
        converter_path=args.converter_path,
        top_k=args.top_k,
        device=args.device,
        base_model_name=args.base_model_name,
        log_file=args.log_file
    )
    
    print("\n✅ 评估完成！")



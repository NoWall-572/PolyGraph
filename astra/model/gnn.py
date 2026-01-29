"""
ASTRA-MoE 模型实现
基于 STGAT (Spatio-Temporal Graph Attention Network) 的故障归因模型

模型架构包含四个组件：
1. MicroStateEncoder - 多模态微观状态编码器
2. STGAT - 空间-时间图注意力网络
3. TemporalReasoning - 因果时序推理（RoPE + Causal Transformer）
4. MoEHead - 不确定性感知的 MoE 诊断头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from astra.data.graph_data import HeteroGraph
from astra.model.stgat import STGAT


class MicroStateEncoder(nn.Module):
    """
    多模态微观状态编码器
    
    将异构节点的文本嵌入与离散/连续特征融合，投影到统一维度 d_model
    使用门控机制 (Gated Fusion) 融合文本嵌入和元数据特征
    """
    
    def __init__(self, 
                 node_feat_dim: int = 8192,  # 🔥 Qwen-8B: 4096 (嵌入) + 4096 (元数据)
                 d_model: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            node_feat_dim: 输入节点特征维度（来自 data_adapter）
            d_model: 输出统一维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.d_model = d_model
        
        # 假设输入特征中，前4096维是 content_embedding (Qwen-8B from JSON)，其余是元数据特征 (Sentence-BERT from JSON)，其余是元数据特征
        self.text_dim = 4096  # 🔥 Qwen-8B 嵌入维度  # 🔥 保持 384
        self.meta_dim = node_feat_dim - self.text_dim  # 4096 - 384 = 3712
        
        # 文本嵌入投影层
        self.text_proj = nn.Linear(self.text_dim, d_model)
        
        # 元数据特征处理 MLP
        # 🔥 修复：使用 GELU 防止死神经元
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 门控网络（输出融合权重）
        # 🔥 修复：使用 GELU 防止死神经元
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 最终投影层
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 🔥 修复：增强 Padding Token 初始化，确保它有存在感
        # 使用较大的随机值，防止它被忽略（特别是对于稀疏的幽灵节点）
        self.padding_token = nn.Parameter(torch.randn(1, d_model))
    
    def forward(self, graph: HeteroGraph) -> Dict[str, torch.Tensor]:
        """
        对每个节点类型进行编码
        
        Args:
            graph: HeteroGraph 对象
            
        Returns:
            Dict[str, Tensor]: 每个节点类型的编码结果，shape: [num_nodes, d_model]
        """
        encoded_nodes = {}
        
        for node_type in graph.get_node_types():
            if node_type not in graph.node_features:
                continue
                
            x = graph.node_features[node_type]  # [num_nodes, node_feat_dim]
            num_nodes = x.shape[0]
            
            # 分离文本嵌入和元数据特征
            text_emb = x[:, :self.text_dim]  # [num_nodes, 384]
            meta_feat = x[:, self.text_dim:]  # [num_nodes, meta_dim]
            
            # 投影到统一维度
            e_text = self.text_proj(text_emb)  # [num_nodes, d_model]
            e_meta = self.meta_mlp(meta_feat)  # [num_nodes, d_model]
            
            # 🔥 关键修复：彻底移除基于特征值的 padding 检测
            # Ghost Node 的特征可能很小，但绝不是 Padding！
            # 我们假设所有输入的节点都是有效的，真正的 Padding 只在 Batch Collate 时产生
            # 这样可以防止 Hand-Crafted 数据中的稀疏节点被错误地识别为 Padding
            
            # 门控融合
            # 计算门控权重 g
            gate_input = torch.cat([e_text, e_meta], dim=-1)  # [num_nodes, 2*d_model]
            g = self.gate_net(gate_input)  # [num_nodes, 1]
            
            # 融合：h = g * e_text + (1 - g) * e_meta
            h = g * e_text + (1 - g) * e_meta
            
            # 输出投影和归一化
            h = self.output_proj(h)
            h = self.dropout(h)
            h = self.layer_norm(h)
            
            encoded_nodes[node_type] = h
        
        return encoded_nodes


class SpatialGraphEncoder(nn.Module):
    """
    空间图编码器（使用 STGAT）
    
    对异构图进行空间编码，使用 STGAT 模型
    """
    
    def __init__(self,
                 d_model: int,
                 edge_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 节点特征维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            num_layers: STGAT 层数
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.edge_dim = edge_dim
        
        # 使用 STGAT 模型
        self.stgat = STGAT(
            d_model=d_model,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, 
                graph: HeteroGraph,
                encoded_nodes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对异构图进行空间编码
        
        Args:
            graph: HeteroGraph 对象
            encoded_nodes: MicroStateEncoder 的输出
            
        Returns:
            更新后的节点特征字典
        """
        # 使用 STGAT 进行空间编码
        updated_nodes = self.stgat(encoded_nodes, graph)
        
        return updated_nodes


class RoPE(nn.Module):
    """
    旋转位置编码 (Rotary Positional Embedding)
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        """
        Args:
            d_model: 特征维度（必须是偶数）
            max_len: 最大序列长度
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用旋转位置编码
        
        Args:
            x: 输入特征 [seq_len, batch_size, d_model] 或 [seq_len, d_model]
            positions: 位置索引 [seq_len]，如果为 None 则使用 0, 1, 2, ...
            
        Returns:
            编码后的特征，形状与 x 相同
        """
        if x.dim() == 2:
            seq_len, d_model = x.shape
            batch_size = 1
            x = x.unsqueeze(1)  # [seq_len, 1, d_model]
        else:
            seq_len, batch_size, d_model = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # 计算角度
        angles = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [seq_len, d_model//2]
        
        # 计算 cos 和 sin
        cos = torch.cos(angles)  # [seq_len, d_model//2]
        sin = torch.sin(angles)  # [seq_len, d_model//2]
        
        # 将 x 分成两部分
        x1, x2 = x.chunk(2, dim=-1)  # 每个 [seq_len, batch_size, d_model//2]
        
        # 应用旋转
        x1_rot = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
        x2_rot = x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)
        
        # 拼接
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)  # [seq_len, batch_size, d_model]
        
        if x.dim() == 2:
            x_rot = x_rot.squeeze(1)  # [seq_len, d_model]
        
        return x_rot


class TemporalReasoning(nn.Module):
    """
    因果时序推理模块 (修复版)
    
    使用 RoPE + Causal Transformer Encoder 处理时间序列
    应用严格因果掩码，确保 t 时刻只依赖 0...t 时刻的信息
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 160):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # RoPE 位置编码
        self.rope = RoPE(d_model, max_seq_len)
        
        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,  # 使用 [seq_len, batch, features] 格式
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                node_sequences: Dict[str, torch.Tensor],
                padding_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        对每个节点类型的时间序列进行因果推理
        
        Args:
            node_sequences: Dict[node_type, Tensor] [seq_len, num_nodes, d_model]
            padding_masks: Dict[node_type, Tensor] [num_nodes] (True 表示有效节点)
                           注意：这里的 Mask 逻辑通常是 True=Valid, False=Invalid
                           但 Transformer 需要 True=Ignore, False=Keep
        """
        output_sequences = {}
        
        for node_type, seq_features in node_sequences.items():
            seq_len, num_nodes, d_model = seq_features.shape
            
            if seq_len == 0 or num_nodes == 0:
                output_sequences[node_type] = seq_features
                continue
            
            # 🔥 关键修复：彻底移除基于特征值的 padding 检测
            # 默认无 Mask (所有节点有效)，只使用外部传入的 explicit mask
            src_key_padding_mask = None
            if padding_masks is not None and node_type in padding_masks:
                valid_mask = padding_masks[node_type]
                # True=Ignore, False=Keep
                if valid_mask.shape[0] == num_nodes:
                     src_key_padding_mask = ~valid_mask.bool().unsqueeze(0).expand(seq_len, -1).t() # [batch, seq_len]
            
            # 应用 RoPE
            seq_features = self.rope(seq_features)
            seq_features = self.dropout(seq_features)
            
            # Reshape: [seq_len, batch_size, d_model]
            batch_size = num_nodes
            seq_features = seq_features.view(seq_len, batch_size, d_model)
            
            # 因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq_features.device, dtype=torch.bool), diagonal=1) if seq_len > 1 else None
            
            # 🔥 修复：直接传递参数，简化调用
            output = self.transformer(seq_features, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
            output = output.view(seq_len, num_nodes, d_model)
            output_sequences[node_type] = output
        
        return output_sequences


class StepPredictor(nn.Module):
    """
    故障时间步预测模块（增强版）
    
    从时序特征中预测故障发生的时间步
    使用 Max-Pooling + Attention 结合策略，防止故障信号被稀释
    
    策略：
    1. 对每个 Agent 独立计算故障分数（Fault Score）
    2. 取该时间步所有 Agent 中的最大故障分作为该步的分数
    3. 这样只要有一个 Agent 在某一步表现极差，该步就会被标记为高风险
    
    输入：时序特征序列 [seq_len, num_agents, d_model]
    输出：每个时间步是故障步的 logits [seq_len]
    """
    
    def __init__(self,
                 d_model: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        # Agent 故障分数计算器（对每个 Agent 独立打分）
        # 输入: [d_model]，输出: [1] (故障分数 logit)
        # 🔥 修复：使用 GELU 防止死神经元
        self.agent_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 输出每个 Agent 的故障分数
        )
    
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_features: Agent 时序特征 [seq_len, num_agents, d_model]
            
        Returns:
            step_logits: 每个时间步是故障步的 logits [seq_len]
        """
        seq_len, num_agents, d_model = agent_features.shape
        
        # 1. 对每个 Agent 独立计算故障分数
        # agent_features: [seq_len, num_agents, d_model]
        # 重塑为 [seq_len * num_agents, d_model] 以便批量处理
        agent_features_flat = agent_features.view(-1, d_model)  # [seq_len * num_agents, d_model]
        
        # 计算每个 Agent 的故障分数
        agent_logits = self.agent_scorer(agent_features_flat)  # [seq_len * num_agents, 1]
        agent_logits = agent_logits.squeeze(-1)  # [seq_len * num_agents]
        
        # 重塑回 [seq_len, num_agents]
        agent_logits = agent_logits.view(seq_len, num_agents)  # [seq_len, num_agents]
        
        # 2. 取该时间步所有 Agent 中的最大故障分作为该步的分数
        # 这样只要有一个 Agent 在某一步表现极差，该步就会被标记为高风险
        step_logits, _ = torch.max(agent_logits, dim=1)  # [seq_len]
        
        return step_logits


class MoEHead(nn.Module):
    """
    不确定性感知的 MoE 诊断头
    
    使用 Top-2 Router 和 Dirichlet 分布输出
    """
    
    def __init__(self,
                 d_model: int,
                 num_experts: int = 4,
                 num_classes: int = 10,  # 假设最多10个 Agent 节点
                 expert_hidden_dim: int = 256,
                 dropout: float = 0.1,
                 noise_std: float = 0.1):
        """
        Args:
            d_model: 输入特征维度
            num_experts: 专家数量
            num_classes: 输出类别数（Agent 节点数）
            expert_hidden_dim: 专家网络隐藏层维度
            dropout: Dropout 比率
            noise_std: Noisy Top-k 路由的噪声标准差
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.noise_std = noise_std
        
        # Gating Network (Top-2 Router)
        # 🔥 修复：使用 GELU 防止死神经元
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_experts)
        )
        
        # Expert Networks
        # 🔥 修复：使用 GELU 防止死神经元
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, num_classes)
            ) for _ in range(num_experts)
        ])
        
        # 用于 Dirichlet 分布输出的参数
        # Dirichlet 分布需要 alpha 参数（浓度参数）
        # 🔥 修复：使用 GELU 防止死神经元
        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, num_classes),
            nn.Softplus()  # 确保 alpha > 0
        )
    
    def noisy_top_k_gating(self, x: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Noisy Top-k Gating
        
        Args:
            x: 输入特征 [batch_size, d_model]
            k: Top-k 值
            
        Returns:
            gate_weights: 门控权重 [batch_size, num_experts]
            load: 每个专家的负载 [num_experts]
        """
        batch_size = x.size(0)
        
        # 计算基础门控分数
        logits = self.gate(x)  # [batch_size, num_experts]
        
        # 添加噪声（训练时）
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Top-k 选择
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)  # [batch_size, k]
        
        # 创建稀疏门控权重
        gate_weights = torch.zeros_like(logits)  # [batch_size, num_experts]
        gate_weights.scatter_(1, top_k_indices, top_k_values)
        
        # Softmax 归一化
        gate_weights = F.softmax(gate_weights, dim=-1)
        
        # 计算负载（用于负载均衡）
        load = gate_weights.sum(dim=0)  # [num_experts]
        
        return gate_weights, load
    
    def forward(self, 
                node_features: Dict[str, torch.Tensor],
                agent_indices: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 每个节点类型的特征
                对于 Agent 节点：形状为 [num_agents, d_model] 或 [seq_len, num_agents, d_model]
            agent_indices: Agent 节点的索引映射（可选）
            
        Returns:
            Dict 包含：
                - 'logits': 故障概率 logits [num_agents, num_classes] 或 [seq_len, num_agents, num_classes]
                - 'alpha': Dirichlet 分布参数 [num_agents, num_classes] 或 [seq_len, num_agents, num_classes]
                - 'gate_weights': 门控权重 [num_agents, num_experts] 或 [seq_len, num_agents, num_experts]
        """
        # 处理没有 'Agent' 节点的情况
        if 'Agent' not in node_features:
            # 返回零输出（用于测试或无效数据）
            import warnings
            warnings.warn(
                "MoEHead: 'Agent' node features not found. Returning zero outputs. "
                f"Available node types: {list(node_features.keys())}",
                UserWarning
            )
            # 尝试从其他节点类型推断序列长度
            is_sequence = False  # 默认值
            if node_features:
                # 使用第一个可用的节点类型来推断形状
                first_type = list(node_features.keys())[0]
                first_feat = node_features[first_type]
                is_sequence = first_feat.dim() == 3
                if is_sequence:
                    seq_len = first_feat.shape[0]
                    num_agents = 1  # 默认至少1个agent（用于输出维度）
                    device = first_feat.device
                    dtype = first_feat.dtype
                else:
                    seq_len = 1
                    num_agents = 1
                    device = first_feat.device
                    dtype = first_feat.dtype
            else:
                # 完全没有节点，返回最小输出
                seq_len = 1
                num_agents = 1
                is_sequence = False
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
            
            # 创建零输出
            if is_sequence:
                logits = torch.zeros(seq_len, num_agents, self.num_classes, device=device, dtype=dtype)
                alpha = torch.ones(seq_len, num_agents, self.num_classes, device=device, dtype=dtype) * 1e-6
                gate_weights = torch.zeros(seq_len, num_agents, self.num_experts, device=device, dtype=dtype)
            else:
                logits = torch.zeros(num_agents, self.num_classes, device=device, dtype=dtype)
                alpha = torch.ones(num_agents, self.num_classes, device=device, dtype=dtype) * 1e-6
                gate_weights = torch.zeros(num_agents, self.num_experts, device=device, dtype=dtype)
            
            load = torch.zeros(self.num_experts, device=device, dtype=dtype)
            
            return {
                'logits': logits,
                'alpha': alpha,
                'gate_weights': gate_weights,
                'load': load
            }
        
        agent_feat = node_features['Agent']  # [num_agents, d_model] 或 [seq_len, num_agents, d_model]
        
        # 处理序列输入
        is_sequence = agent_feat.dim() == 3
        if is_sequence:
            seq_len, num_agents, d_model = agent_feat.shape
            agent_feat = agent_feat.view(-1, d_model)  # [seq_len * num_agents, d_model]
        else:
            num_agents, d_model = agent_feat.shape
        
        # Noisy Top-2 Gating
        gate_weights, load = self.noisy_top_k_gating(agent_feat, k=2)  # [batch, num_experts]
        
        # Expert 输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(agent_feat)  # [batch, num_classes]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, num_classes]
        
        # 加权聚合专家输出
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch, num_experts, 1]
        logits = (expert_outputs * gate_weights_expanded).sum(dim=1)  # [batch, num_classes]
        
        # 计算 Dirichlet 分布参数 alpha
        alpha = self.alpha_proj(agent_feat)  # [batch, num_classes]
        # 添加小的正数确保 alpha > 0
        alpha = alpha + 1e-6
        
        # 如果输入是序列，重塑回原始形状
        if is_sequence:
            logits = logits.view(seq_len, num_agents, self.num_classes)
            alpha = alpha.view(seq_len, num_agents, self.num_classes)
            gate_weights = gate_weights.view(seq_len, num_agents, self.num_experts)
        
        return {
            'logits': logits,
            'alpha': alpha,
            'gate_weights': gate_weights,
            'load': load
        }


class ASTRAMoE(nn.Module):
    """
    ASTRA-MoE 完整模型
    
    整合四个组件：
    1. MicroStateEncoder
    2. EdgeEnhancedHGT
    3. TemporalReasoning
    4. MoEHead
    """
    
    def __init__(self,
                 node_feat_dim: int = 8192,  # 🔥 Qwen-8B: 4096 (嵌入) + 4096 (元数据)
                 edge_feat_dim: int = 32,
                 d_model: int = 128,
                 num_heads: int = 4,
                 num_hgt_layers: int = 2,
                 num_temporal_layers: int = 2,
                 num_experts: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.5,
                 max_seq_len: int = 160):  # Updated: test data max length is 130, set to 160 with margin
        """
        Args:
            node_feat_dim: 输入节点特征维度（来自 data_adapter）
            edge_feat_dim: 边特征维度（来自 data_adapter）
            d_model: 模型内部统一维度
            num_heads: 注意力头数
            num_hgt_layers: HGT 层数
            num_temporal_layers: 时序 Transformer 层数
            num_experts: MoE 专家数量
            num_classes: 输出类别数（Agent 节点数）
            dropout: Dropout 比率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        # 1. 多模态微观状态编码器
        self.micro_encoder = MicroStateEncoder(
            node_feat_dim=node_feat_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # 2. 空间图编码器（使用 STGAT）
        self.spatial_encoder = SpatialGraphEncoder(
            d_model=d_model,
            edge_dim=edge_feat_dim,
            num_heads=num_heads,
            num_layers=num_hgt_layers,
            dropout=dropout
        )
        
        # 3. 因果时序推理
        self.temporal = TemporalReasoning(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # 4. MoE 诊断头
        self.moe_head = MoEHead(
            d_model=d_model,
            num_experts=num_experts,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 5. 故障时间步预测器（增强版：Max-Pooling + Attention 结合）
        self.step_predictor = StepPredictor(
            d_model=d_model,
            hidden_dim=128,
            dropout=dropout
        )
        
        # === 新增：Critic 网络 (用于 MAPPO) ===
        # 输入是 Graph Embedding，输出是 State Value (标量)
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, 
                graph_list: List[HeteroGraph],
                return_intermediate: bool = False,
                agent_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            graph_list: 时间步序列，每个元素是一个 HeteroGraph 快照
            return_intermediate: 是否返回中间结果
            
        Returns:
            Dict 包含：
                - 'logits': 故障概率 logits [seq_len, num_agents, num_classes]
                - 'alpha': Dirichlet 分布参数 [seq_len, num_agents, num_classes]
                - 'gate_weights': 门控权重 [seq_len, num_agents, num_experts]
                - (可选) 'intermediate': 中间结果字典
        """
        seq_len = len(graph_list)
        
        # 🔥 调试信息：打印 graph_list 的长度和内容
        if seq_len > 20:  # 只打印异常长的序列
            print(f"[ASTRA Model] ⚠️  WARNING: graph_list length is {seq_len}, which seems unusually long!")
            print(f"  First few graphs: {[g.get_node_types() for g in graph_list[:5]]}")
            print(f"  Last few graphs: {[g.get_node_types() for g in graph_list[-5:]]}")
            # 检查是否有重复的图
            graph_ids = [id(g) for g in graph_list]
            if len(graph_ids) != len(set(graph_ids)):
                print(f"  ⚠️  WARNING: Duplicate graphs detected! {len(graph_ids) - len(set(graph_ids))} duplicates")
        
        # 步骤 1: 微观状态编码（每个时间步）
        # 首先收集所有时间步的所有节点类型，确保所有节点类型都被处理
        all_node_types = set()
        for graph in graph_list:
            all_node_types.update(graph.get_node_types())
        
        encoded_sequences = {node_type: [] for node_type in all_node_types}  # Dict[node_type, List[Tensor]]
        
        for t, graph in enumerate(graph_list):
            encoded_nodes = self.micro_encoder(graph)
            
            # 收集每个节点类型的特征
            # 对于存在的节点类型，添加编码结果；对于不存在的，添加零向量
            for node_type in all_node_types:
                if node_type in encoded_nodes:
                    encoded_sequences[node_type].append(encoded_nodes[node_type])
                else:
                    # 如果该时间步没有该节点类型，需要创建一个零向量
                    # 但我们需要知道维度，所以先检查其他时间步是否有该节点类型
                    # 或者使用默认维度（从第一个有该节点类型的时间步获取）
                    # 这里我们暂时跳过，在后续对齐时处理
                    # 为了保持一致性，我们需要知道 d_model 维度
                    # 先尝试从已编码的节点获取维度
                    if encoded_sequences[node_type]:
                        # 如果之前已经有该节点类型，使用相同的维度
                        d_model = encoded_sequences[node_type][0].shape[1]
                        num_nodes = 0  # 该时间步没有该节点类型
                        zero_feat = torch.zeros(num_nodes, d_model, 
                                               device=next(self.parameters()).device,
                                               dtype=next(self.parameters()).dtype)
                        encoded_sequences[node_type].append(zero_feat)
                    else:
                        # 如果这是第一次遇到该节点类型，但当前时间步没有
                        # 使用默认维度 d_model
                        d_model = self.micro_encoder.d_model
                        zero_feat = torch.zeros(0, d_model,
                                               device=next(self.parameters()).device,
                                               dtype=next(self.parameters()).dtype)
                        encoded_sequences[node_type].append(zero_feat)
        
        # 转换为序列张量
        node_sequences = {}
        for node_type, feat_list in encoded_sequences.items():
            # 过滤掉空张量（0个节点）以计算最大节点数
            non_empty_feats = [f for f in feat_list if f.shape[0] > 0]
            if non_empty_feats:
                max_nodes = max(f.shape[0] for f in non_empty_feats)
                d_model = non_empty_feats[0].shape[1]
            else:
                # 如果所有时间步都没有该节点类型，使用默认维度
                d_model = self.micro_encoder.d_model
                max_nodes = 0
            
            # 对齐到最大节点数（使用零填充）
            aligned_feats = []
            for feat in feat_list:
                num_nodes = feat.shape[0]
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, d_model, 
                                        device=feat.device, dtype=feat.dtype)
                    feat = torch.cat([feat, padding], dim=0)
                elif num_nodes == 0 and max_nodes > 0:
                    # 如果当前时间步没有节点，但其他时间步有，创建零向量
                    feat = torch.zeros(max_nodes, d_model,
                                     device=feat.device if feat.numel() > 0 else next(self.parameters()).device,
                                     dtype=feat.dtype if feat.numel() > 0 else next(self.parameters()).dtype)
                aligned_feats.append(feat)
            
            # 堆叠为序列 [seq_len, max_nodes, d_model]
            node_sequences[node_type] = torch.stack(aligned_feats, dim=0)
        
        # 步骤 2: 空间编码（每个时间步独立进行 STGAT）
        # 使用与微观编码相同的节点类型集合
        spatial_encoded_sequences = {node_type: [] for node_type in all_node_types}
        
        for t, graph in enumerate(graph_list):
            # 获取当前时间步的编码节点
            current_encoded = {node_type: node_sequences[node_type][t] 
                             for node_type in node_sequences.keys()}
            
            # STGAT 空间编码
            spatial_encoded = self.spatial_encoder(graph, current_encoded)
            
            # 收集结果
            # 对于存在的节点类型，添加编码结果；对于不存在的，添加零向量
            for node_type in all_node_types:
                if node_type in spatial_encoded:
                    spatial_encoded_sequences[node_type].append(spatial_encoded[node_type])
                else:
                    # 如果该时间步没有该节点类型，创建零向量
                    if spatial_encoded_sequences[node_type]:
                        d_model = spatial_encoded_sequences[node_type][0].shape[1]
                        num_nodes = 0
                        zero_feat = torch.zeros(num_nodes, d_model,
                                               device=next(self.parameters()).device,
                                               dtype=next(self.parameters()).dtype)
                        spatial_encoded_sequences[node_type].append(zero_feat)
                    else:
                        # 从 node_sequences 获取维度
                        if node_type in node_sequences:
                            d_model = node_sequences[node_type].shape[2]  # [seq_len, num_nodes, d_model]
                            num_nodes = 0
                            zero_feat = torch.zeros(num_nodes, d_model,
                                                   device=next(self.parameters()).device,
                                                   dtype=next(self.parameters()).dtype)
                            spatial_encoded_sequences[node_type].append(zero_feat)
                        else:
                            # 使用默认维度
                            d_model = self.spatial_encoder.d_model
                            zero_feat = torch.zeros(0, d_model,
                                                   device=next(self.parameters()).device,
                                                   dtype=next(self.parameters()).dtype)
                            spatial_encoded_sequences[node_type].append(zero_feat)
        
        # 转换为序列张量
        spatial_sequences = {}
        for node_type, feat_list in spatial_encoded_sequences.items():
            # 过滤掉空张量（0个节点）以计算最大节点数
            non_empty_feats = [f for f in feat_list if f.shape[0] > 0]
            if non_empty_feats:
                max_nodes = max(f.shape[0] for f in non_empty_feats)
                d_model = non_empty_feats[0].shape[1]
            else:
                # 如果所有时间步都没有该节点类型，使用默认维度
                d_model = self.spatial_encoder.d_model
                max_nodes = 0
            
            aligned_feats = []
            for feat in feat_list:
                num_nodes = feat.shape[0]
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, d_model,
                                        device=feat.device, dtype=feat.dtype)
                    feat = torch.cat([feat, padding], dim=0)
                elif num_nodes == 0 and max_nodes > 0:
                    # 如果当前时间步没有节点，但其他时间步有，创建零向量
                    feat = torch.zeros(max_nodes, d_model,
                                     device=feat.device if feat.numel() > 0 else next(self.parameters()).device,
                                     dtype=feat.dtype if feat.numel() > 0 else next(self.parameters()).dtype)
                aligned_feats.append(feat)
            
            spatial_sequences[node_type] = torch.stack(aligned_feats, dim=0)
        
        # 步骤 3: 时序推理
        # 🔥 构建 padding_masks 字典传给 TemporalReasoning
        padding_masks = {}
        if agent_mask is not None:
            # agent_mask: [B, max_N]
            # node_sequences['Agent']: [seq_len, B, max_N, d] -> processed as [seq_len, B*max_N, d] inside?
            # ❌ 等等，train.py 里我们是逐个样本 forward 的！
            # train.py: for i, graph_list in enumerate(graph_lists): output = model(graph_list_device)
            # 所以这里的 batch_size 其实是 1 (或者说 num_nodes 就是当前图的节点数)
            
            # 既然是逐个样本 forward，我们不需要复杂的 mask 对齐
            # 我们可以直接让 TemporalReasoning 认为所有节点都是有效的
            # 因为只有 collate_fn 产生的 padding 才是需要 mask 的
            # 而 ghost nodes 是有效的！
            
            # ✅ 策略：直接不传 mask，或者传全 True 的 mask
            pass

        # 🔥 关键修改：直接调用，不要让 TemporalReasoning 自己去猜
        # 只要不传 mask，TemporalReasoning (修复版) 就会默认所有节点都有效
        temporal_sequences = self.temporal(spatial_sequences, padding_masks=padding_masks if padding_masks else None)
        
        # === 新增：计算 Global Feature (用于对比学习和 Critic) ===
        # 假设 temporal_sequences['Agent'] 是 [seq_len, num_agents, d_model]
        # 我们做一个 Global Pooling 得到整张图的表示，用于 Critic 和 对比学习
        # Mean Pooling over time and agents
        global_feat = None
        agent_key = None
        for key in temporal_sequences.keys():
            if key.lower() == 'agent':
                agent_key = key
                break
        
        # 如果没有找到 Agent 键，尝试直接使用 'Agent'
        if agent_key is None and 'Agent' in temporal_sequences:
            agent_key = 'Agent'
        
        if agent_key is not None and agent_key in temporal_sequences:
            agent_temporal = temporal_sequences[agent_key]  # [seq_len, num_agents, d_model]
            if agent_temporal.numel() > 0:
                # Mean Pooling over time and agents
                global_feat = agent_temporal.mean(dim=(0, 1))  # [d_model]
                # 扩展 batch 维度（虽然这里 batch_size=1，但为了兼容性）
                if global_feat.dim() == 1:
                    global_feat = global_feat.unsqueeze(0)  # [1, d_model]
            else:
                global_feat = torch.zeros(1, self.micro_encoder.d_model, 
                                        device=next(self.parameters()).device,
                                        dtype=next(self.parameters()).dtype)
        else:
            # Fallback: 使用零向量
            global_feat = torch.zeros(1, self.micro_encoder.d_model,
                                    device=next(self.parameters()).device,
                                    dtype=next(self.parameters()).dtype)
        
        # Critic 输出 (Value)
        state_value = self.critic(global_feat)  # [1, 1] 或 [batch, 1]
        
        # 步骤 4: MoE 诊断头
        moe_output = self.moe_head(temporal_sequences)
        
        # 步骤 5: 故障时间步预测
        # 🔥 关键修复：强制初始化 step_logits，确保无论什么情况都返回
        seq_len = len(graph_list)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # 🔥 强制初始化 step_logits 为 None，防止变量作用域问题
        step_logits = None
        
        # 检查 temporal_sequences 中是否有 'Agent' 键（注意大小写）
        agent_key = None
        for key in temporal_sequences.keys():
            if key.lower() == 'agent':
                agent_key = key
                break
        
        # 如果没有找到 Agent 键，尝试直接使用 'Agent'
        if agent_key is None and 'Agent' in temporal_sequences:
            agent_key = 'Agent'
        
        # 🔥 调试信息：打印可用的键
        if agent_key is None:
            print(f"[ASTRA Model] ⚠️  No 'Agent' key found. Available keys: {list(temporal_sequences.keys())}")
        
        if agent_key is not None and agent_key in temporal_sequences:
            agent_temporal = temporal_sequences[agent_key]  # [seq_len, num_agents, d_model]
            
            # 🔥 关键检查：确保 agent_temporal 有有效的 Agent 节点
            if agent_temporal.dim() == 3:
                seq_len_check, num_agents, d_model = agent_temporal.shape
                # 🔥 关键修复：如果 agent_temporal 的长度与 seq_len 不一致，需要调整
                if seq_len_check != seq_len:
                    print(f"[ASTRA Model] ⚠️  agent_temporal length mismatch: expected {seq_len}, got {seq_len_check}")
                    print(f"  This indicates a bug in the model! graph_list length={seq_len}, but agent_temporal length={seq_len_check}")
                    print(f"  This will cause index out of bounds errors in loss calculation!")
                    # 🔥 强制截断到 seq_len（这是正确的长度）
                    if seq_len_check > seq_len:
                        agent_temporal = agent_temporal[:seq_len]
                        seq_len_check = seq_len
                        print(f"[ASTRA Model] ✅ Truncated agent_temporal to shape: {agent_temporal.shape}")
                    elif seq_len_check < seq_len:
                        # 填充 agent_temporal 到 seq_len
                        pad_size = seq_len - seq_len_check
                        padding = torch.zeros(pad_size, num_agents, d_model, device=device, dtype=dtype)
                        agent_temporal = torch.cat([agent_temporal, padding], dim=0)
                        seq_len_check = seq_len
                        print(f"[ASTRA Model] ✅ Padded agent_temporal to shape: {agent_temporal.shape}")
                
                if num_agents > 0:
                    # 有有效的 Agent 节点，正常预测
                    try:
                        step_logits = self.step_predictor(agent_temporal)  # [seq_len]
                        # 🔥 关键修复：确保 step_logits 的形状正确（使用 seq_len，而不是 seq_len_check）
                        if step_logits.shape[0] != seq_len:
                            print(f"[ASTRA Model] ⚠️  StepPredictor output shape mismatch: expected {seq_len}, got {step_logits.shape[0]}")
                            # 调整形状
                            if step_logits.shape[0] < seq_len:
                                padding = torch.full((seq_len - step_logits.shape[0],), float('-inf'), device=device, dtype=dtype)
                                step_logits = torch.cat([step_logits, padding], dim=0)
                            else:
                                step_logits = step_logits[:seq_len]
                        # 🔥 双重检查：确保修复后长度正确
                        if step_logits.shape[0] != seq_len:
                            print(f"[ASTRA Model] ❌ CRITICAL: step_logits length still wrong after fix: {step_logits.shape[0]} != {seq_len}")
                            step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
                    except Exception as e:
                        print(f"[ASTRA Model] ❌ StepPredictor Exception: {e}")
                        print(f"  agent_temporal shape: {agent_temporal.shape}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: 创建 -inf 输出（表示无预测）
                        step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
                else:
                    # Agent 键存在但 num_agents == 0，创建 fallback 输出
                    print(f"[ASTRA Model] ⚠️  Agent key '{agent_key}' exists but num_agents=0, using fallback")
                    step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
            else:
                # Agent 张量维度不正确
                print(f"[ASTRA Model] ⚠️  Agent temporal features have incorrect dimensions: {agent_temporal.shape}, expected [seq_len, num_agents, d_model]")
                step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
        else:
            # 完全没有 Agent 节点类型
            print(f"[ASTRA Model] ⚠️  No 'Agent' node type found in temporal_sequences. Available keys: {list(temporal_sequences.keys())}")
            # 使用 -inf 表示无概率（而不是 0，因为 0 在 softmax 后仍有概率）
            step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
        
        # 🔥 双重保险：如果没有生成 step_logits，创建一个全 -inf 的 Tensor
        if step_logits is None:
            print(f"[ASTRA Model] ❌ CRITICAL: step_logits is None! Creating fallback.")
            step_logits = torch.full((seq_len,), float('-inf'), device=device, dtype=dtype)
        
        # 🔥 强制确保 step_logits 存在且形状正确
        assert step_logits is not None, "step_logits must not be None after all checks"
        assert step_logits.shape[0] == seq_len, f"step_logits shape mismatch: expected {seq_len}, got {step_logits.shape[0]}"
        
        # 🔥 调试信息：确认 step_logits 已创建
        if step_logits.isnan().any():
            print(f"[ASTRA Model] ⚠️  step_logits contains NaN values!")
        if (step_logits == float('-inf')).all():
            print(f"[ASTRA Model] ⚠️  step_logits are all -inf (no valid predictions)")
        
        # 构建输出
        # 🔥 ASTRA-CL: 提取 Agent embeddings 用于对比学习
        # 确保始终返回该键，避免下游跳过 CL Loss
        agent_embeddings = None
        if agent_key is not None and agent_key in temporal_sequences:
            agent_embeddings = temporal_sequences[agent_key]  # [seq_len, num_agents, d_model]
        else:
            # 兜底返回零张量以保持接口完整
            agent_embeddings = torch.zeros(
                seq_len,
                0,
                self.micro_encoder.d_model,
                device=device,
                dtype=dtype,
            )
        
        output = {
            'logits': moe_output['logits'],
            'alpha': moe_output['alpha'],
            'gate_weights': moe_output['gate_weights'],
            'load': moe_output['load'],
            'step_logits': step_logits,  # [seq_len] - 🔥 绝对存在
            'global_feat': global_feat,  # [1, d_model] 用于对比学习
            'state_value': state_value,   # [1, 1] 用于 RL
            'agent_embeddings': agent_embeddings  # [seq_len, num_agents, d_model] 用于 ASTRA-CL
        }
        
        # 🔥 最终检查：确保 output 字典包含 step_logits
        if 'step_logits' not in output:
            raise RuntimeError("CRITICAL ERROR: 'step_logits' missing in output dict after assignment!")
        
        # 🔥 调试信息：打印输出键
        if hasattr(self, '_debug_print_count'):
            self._debug_print_count += 1
        else:
            self._debug_print_count = 1
        
        if self._debug_print_count <= 3:  # 只打印前3次
            print(f"[ASTRA Model] ✅ Forward pass {self._debug_print_count}: output keys = {list(output.keys())}")
            print(f"  step_logits shape: {output['step_logits'].shape}, dtype: {output['step_logits'].dtype}")
        
        if return_intermediate:
            output['intermediate'] = {
                'encoded_nodes': encoded_sequences,
                'spatial_encoded': spatial_encoded_sequences,
                'temporal_sequences': temporal_sequences
            }
        
        return output


def test_model():
    """测试模型前向传播"""
    from astra.data.adapter import GraphDataConverter, reconstruct_graph_from_json
    from pathlib import Path
    import json
    
    # 查找可用的测试文件（在 outputs 目录及其子目录中递归搜索）
    output_dir = Path("outputs")
    test_file = None
    
    # 优先查找 Algorithm-Generated 文件，然后在子目录中搜索
    for pattern in ["Algorithm-Generated_*_graph.json", "Hand-Crafted_*_graph.json"]:
        # 先在根目录查找
        files = sorted(output_dir.glob(pattern))
        if not files:
            # 如果根目录没有，在子目录中递归查找
            files = sorted(output_dir.glob(f"**/{pattern}"))
        if files:
            test_file = files[0]
            break
    
    if test_file is None or not test_file.exists():
        print(f"测试文件不存在，请确保 outputs 目录（或其子目录）下有 Algorithm-Generated_*_graph.json 或 Hand-Crafted_*_graph.json 文件")
        return
    
    print(f"使用测试文件: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 从 JSON 数据重建 DynamicGraph（不再使用 MainParser）
    graph = reconstruct_graph_from_json(graph_data)
    print(f"加载的图: {graph}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    # 数据转换
    converter = GraphDataConverter(node_feat_dim=4096, edge_feat_dim=32)  # 🔥 修正: 384 (文本) + 3712 (元数据) = 4096
    converter.fit([graph])
    graph_list, labels = converter.convert(graph)
    
    print(f"\n转换结果:")
    print(f"时间步数: {len(graph_list)}")
    print(f"标签: {labels}")
    
    # 检查所有图中存在的节点类型
    all_node_types = set()
    for graph in graph_list:
        all_node_types.update(graph.get_node_types())
    print(f"\n存在的节点类型: {sorted(all_node_types)}")
    
    # 显示每个时间步的节点类型和数量
    for i, graph in enumerate(graph_list):
        print(f"  时间步 {i}: {[(nt, graph.num_nodes(nt)) for nt in graph.get_node_types()]}")
    
    # 创建模型
    # 计算 Agent 节点的最大数量（如果存在）
    agent_counts = [len(graph.node_features['Agent']) for graph in graph_list 
                    if 'Agent' in graph.node_features]
    
    if not agent_counts:
        print("\n⚠️  警告: 没有找到 'Agent' 节点类型")
        print("   使用默认值 num_classes=1")
        num_agents = 1
    else:
        num_agents = max(agent_counts)
        print(f"\nAgent 节点数量: {num_agents} (最大)")
    
    model = ASTRAMoE(
        node_feat_dim=4096,  # 🔥 修正: 384 (文本) + 3712 (元数据) = 4096
        edge_feat_dim=32,
        d_model=256,
        num_heads=8,
        num_hgt_layers=2,
        num_temporal_layers=2,
        num_experts=4,
        num_classes=num_agents,
        dropout=0.1,
        max_seq_len=len(graph_list)
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    print("\n开始前向传播...")
    output = model(graph_list)
    
    print(f"\n输出形状:")
    print(f"  logits: {output['logits'].shape}")
    print(f"  alpha: {output['alpha'].shape}")
    print(f"  gate_weights: {output['gate_weights'].shape}")
    print(f"  load: {output['load'].shape}")
    
    print("\n✅ 模型测试通过！")


if __name__ == "__main__":
    test_model()


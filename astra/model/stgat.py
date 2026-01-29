"""
STGAT (Spatio-Temporal Graph Attention Network) 实现

STGAT 结合了空间和时间的注意力机制，用于处理动态异构图数据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from astra.data.graph_data import HeteroGraph


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    
    用于在图的每个时间步内，计算节点之间的空间关系
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 edge_dim: int = 32,
                 dropout: float = 0.1):
        """
        Args:
            in_channels: 输入节点特征维度
            out_channels: 输出节点特征维度
            num_heads: 注意力头数
            edge_dim: 边特征维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.edge_dim = edge_dim
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Query, Key, Value 投影
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        
        # 边特征投影
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # 输出投影
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        空间注意力前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]，第一行是源节点，第二行是目标节点
            edge_attr: 边特征 [num_edges, edge_dim]，可选
        
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        residual = x
        
        # 投影到 Q, K, V
        q = self.query(x)  # [num_nodes, out_channels]
        k = self.key(x)    # [num_nodes, out_channels]
        v = self.value(x) # [num_nodes, out_channels]
        
        # 重塑为多头形式
        q = q.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)
        
        # 计算 attention score
        # q[edge_index[1]]: 目标节点的 Query [num_edges, num_heads, head_dim]
        # k[edge_index[0]]: 源节点的 Key [num_edges, num_heads, head_dim]
        q_i = q[edge_index[1]]  # [num_edges, num_heads, head_dim]
        k_j = k[edge_index[0]]  # [num_edges, num_heads, head_dim]
        v_j = v[edge_index[0]]  # [num_edges, num_heads, head_dim]
        
        # 计算基础 attention score
        attn_score = (q_i * k_j).sum(dim=-1)  # [num_edges, num_heads]
        attn_score = attn_score / math.sqrt(self.head_dim)
        
        # 添加边特征偏置
        if edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # [num_edges, num_heads]
            attn_score = attn_score + edge_bias
        
        # 对每个目标节点的邻居进行 softmax 归一化
        # edge_index[1] 是目标节点索引
        index = edge_index[1]  # [num_edges]
        num_edges, num_heads = attn_score.shape
        
        # 分组 softmax（数值稳定）
        max_per_node = attn_score.new_full((num_nodes, num_heads), float('-inf'))
        max_per_node = max_per_node.index_copy_(0, index, attn_score)
        attn_score_stable = attn_score - max_per_node[index]
        exp_score = attn_score_stable.exp()
        
        # 计算归一化因子
        sum_per_node = exp_score.new_zeros((num_nodes, num_heads))
        sum_per_node = sum_per_node.index_add_(0, index, exp_score)
        attn_score = exp_score / (sum_per_node[index] + 1e-8)  # [num_edges, num_heads]
        
        attn_score = self.dropout(attn_score)
        attn_score = attn_score.unsqueeze(-1)  # [num_edges, num_heads, 1]
        
        # 应用 attention 权重到 value
        msg = attn_score * v_j  # [num_edges, num_heads, head_dim]
        
        # 聚合消息（按目标节点）
        out = msg.new_zeros((num_nodes, self.num_heads, self.head_dim))
        out = out.index_add_(0, index, msg)
        
        # 重塑回原始形状
        out = out.view(num_nodes, self.out_channels)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(out + residual)
        
        return out


class TemporalAttention(nn.Module):
    """
    时间注意力机制
    
    用于在时间序列上计算节点的时间依赖关系
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 特征维度
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Query, Key, Value 投影
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时间注意力前向传播
        
        Args:
            x: 时间序列特征 [seq_len, num_nodes, d_model]
            mask: 因果掩码 [seq_len, seq_len]，True 表示不能 attend
        
        Returns:
            更新后的时间序列特征 [seq_len, num_nodes, d_model]
        """
        seq_len, num_nodes, d_model = x.shape
        residual = x
        
        # 重塑为 [seq_len, batch, d_model]，其中 batch = num_nodes
        x_reshaped = x.view(seq_len, num_nodes, d_model)
        
        # 投影到 Q, K, V
        q = self.query(x_reshaped)  # [seq_len, num_nodes, d_model]
        k = self.key(x_reshaped)
        v = self.value(x_reshaped)
        
        # 重塑为多头形式
        q = q.view(seq_len, num_nodes, self.num_heads, self.head_dim)
        k = k.view(seq_len, num_nodes, self.num_heads, self.head_dim)
        v = v.view(seq_len, num_nodes, self.num_heads, self.head_dim)
        
        # 转置为 [num_nodes, num_heads, seq_len, head_dim]
        q = q.transpose(0, 1).transpose(1, 2)  # [num_nodes, num_heads, seq_len, head_dim]
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        
        # 计算 attention score
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [num_nodes, num_heads, seq_len, seq_len]
        
        # 应用因果掩码
        if mask is not None:
            attn_score = attn_score.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax 归一化
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        
        # 应用 attention 权重
        out = torch.matmul(attn_score, v)  # [num_nodes, num_heads, seq_len, head_dim]
        
        # 转置回 [seq_len, num_nodes, num_heads, head_dim]
        out = out.transpose(1, 2).transpose(0, 1)
        
        # 重塑为 [seq_len, num_nodes, d_model]
        out = out.contiguous().view(seq_len, num_nodes, d_model)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(out + residual)
        
        return out


class STGATLayer(nn.Module):
    """
    STGAT 层
    
    结合空间和时间注意力机制
    """
    
    def __init__(self,
                 d_model: int,
                 edge_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 特征维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()
        self.spatial_attn = SpatialAttention(
            in_channels=d_model,
            out_channels=d_model,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.temporal_attn = TemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self,
                node_features: Dict[str, torch.Tensor],
                graph: HeteroGraph,
                temporal_sequences: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            node_features: 当前时间步的节点特征 Dict[node_type, Tensor] -> [num_nodes, d_model]
            graph: 当前时间步的图结构
            temporal_sequences: 时间序列特征 Dict[node_type, Tensor] -> [seq_len, num_nodes, d_model]，可选
        
        Returns:
            (spatial_output, temporal_output)
            - spatial_output: 空间编码后的节点特征
            - temporal_output: 时间编码后的序列特征（如果提供了 temporal_sequences）
        """
        # 空间注意力（对每个边类型）
        spatial_output = {node_type: feat.clone() for node_type, feat in node_features.items()}
        
        # 对每个边类型进行空间注意力
        for edge_type_tuple, edge_index in graph.edge_indices.items():
            src_type, edge_type, dst_type = edge_type_tuple
            
            # 获取源节点和目标节点特征
            if src_type not in node_features or dst_type not in node_features:
                continue
            
            src_feat = node_features[src_type]  # [num_src_nodes, d_model]
            dst_feat = node_features[dst_type]  # [num_dst_nodes, d_model]
            
            # 获取边特征
            edge_attr = graph.edge_features.get(edge_type_tuple, None)
            
            # 合并源节点和目标节点特征（用于空间注意力）
            # 注意：edge_index 中的索引是相对于各自节点类型的局部索引
            # 我们需要创建一个包含所有相关节点的统一特征矩阵
            
            # 方案：分别对源节点和目标节点应用空间注意力
            # 对于目标节点，使用空间注意力聚合来自源节点的信息
            num_src_nodes = src_feat.size(0)
            num_dst_nodes = dst_feat.size(0)
            
            # 创建扩展的特征矩阵，包含源节点和目标节点
            # 索引映射：源节点 [0, num_src_nodes-1]，目标节点 [num_src_nodes, num_src_nodes+num_dst_nodes-1]
            combined_feat = torch.cat([src_feat, dst_feat], dim=0)  # [num_src_nodes + num_dst_nodes, d_model]
            
            # 调整 edge_index：源节点索引不变，目标节点索引需要加上 num_src_nodes
            adjusted_edge_index = edge_index.clone()
            adjusted_edge_index[1] = adjusted_edge_index[1] + num_src_nodes
            
            # 应用空间注意力
            updated_combined = self.spatial_attn(combined_feat, adjusted_edge_index, edge_attr)
            
            # 提取更新后的目标节点特征
            updated_dst_feat = updated_combined[num_src_nodes:]  # [num_dst_nodes, d_model]
            
            # 更新目标节点特征（使用残差连接）
            spatial_output[dst_type] = spatial_output[dst_type] + updated_dst_feat
        
        # 时间注意力（如果有时间序列）
        temporal_output = None
        if temporal_sequences is not None:
            temporal_output = {}
            for node_type, seq_feat in temporal_sequences.items():
                # 创建因果掩码
                seq_len = seq_feat.size(0)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, 
                                                   device=seq_feat.device, 
                                                   dtype=torch.bool), diagonal=1)
                
                # 时间注意力
                temporal_output[node_type] = self.temporal_attn(seq_feat, causal_mask)
        
        return spatial_output, temporal_output


class STGAT(nn.Module):
    """
    STGAT (Spatio-Temporal Graph Attention Network)
    
    完整的 STGAT 模型，包含多层空间-时间注意力
    """
    
    def __init__(self,
                 d_model: int,
                 edge_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 特征维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            num_layers: STGAT 层数
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            STGATLayer(
                d_model=d_model,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self,
                node_features: Dict[str, torch.Tensor],
                graph: HeteroGraph) -> Dict[str, torch.Tensor]:
        """
        前向传播（单时间步）
        
        Args:
            node_features: 节点特征 Dict[node_type, Tensor] -> [num_nodes, d_model]
            graph: 图结构
        
        Returns:
            更新后的节点特征
        """
        x = node_features
        
        for layer in self.layers:
            spatial_out, _ = layer(x, graph, temporal_sequences=None)
            x = spatial_out
        
        return x


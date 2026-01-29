"""
自定义图数据结构，替代 PyTorch Geometric 的 HeteroData

用于表示异构图数据，包含：
- 节点特征（按类型分组）
- 边索引和边特征（按类型分组）
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class HeteroGraph:
    """
    异构图数据结构
    
    替代 PyG 的 HeteroData，用于表示包含多种节点类型和边类型的图
    """
    
    # 节点特征：Dict[node_type, Tensor] -> [num_nodes, feat_dim]
    node_features: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 边索引：Dict[edge_type_tuple, Tensor] -> [2, num_edges]
    # edge_type_tuple 格式: (src_type, edge_type, dst_type)
    edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    
    # 边特征：Dict[edge_type_tuple, Tensor] -> [num_edges, feat_dim]
    edge_features: Dict[Tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    
    # 节点ID到索引的映射（可选，用于调试）
    node_id_to_idx: Optional[Dict[str, Tuple[str, int]]] = None
    
    def get_node_types(self) -> List[str]:
        """获取所有节点类型"""
        return list(self.node_features.keys())
    
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """获取所有边类型"""
        return list(self.edge_indices.keys())
    
    def num_nodes(self, node_type: str) -> int:
        """获取指定类型的节点数量"""
        if node_type in self.node_features:
            return self.node_features[node_type].shape[0]
        return 0
    
    def num_edges(self, edge_type: Tuple[str, str, str]) -> int:
        """获取指定类型的边数量"""
        if edge_type in self.edge_indices:
            return self.edge_indices[edge_type].shape[1]
        return 0
    
    def to(self, device: torch.device):
        """将图数据移动到指定设备"""
        new_graph = HeteroGraph()
        new_graph.node_features = {
            k: v.to(device) for k, v in self.node_features.items()
        }
        new_graph.edge_indices = {
            k: v.to(device) for k, v in self.edge_indices.items()
        }
        new_graph.edge_features = {
            k: v.to(device) for k, v in self.edge_features.items()
        }
        new_graph.node_id_to_idx = self.node_id_to_idx
        return new_graph
    
    def __repr__(self) -> str:
        node_info = ", ".join([f"{nt}: {self.num_nodes(nt)}" 
                              for nt in self.get_node_types()])
        edge_info = ", ".join([f"{et}: {self.num_edges(et)}" 
                               for et in self.get_edge_types()])
        return f"HeteroGraph(nodes=[{node_info}], edges=[{edge_info}])"





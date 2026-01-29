"""
数据适配层：将 DynamicGraph 转换为自定义图格式

本模块实现 GraphDataConverter 类，负责：
1. 将 parser.py 产生的 DynamicGraph 对象转换为 HeteroGraph 序列
2. 处理节点和边的特征编码（离散特征 LabelEncoder，连续特征标准化）
3. 构建标签（y_agent: 故障源节点分类，y_step: 故障时间步回归/分类）
"""

import torch
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from astra.data.graph_data import HeteroGraph

# 导入 parser 模块中的类型
import sys
import os
import importlib.util
from pathlib import Path

# 获取当前文件的目录并构建 parser.py 的路径
# parser 现在在 astra/parsing/dhcg_parser/ 目录下
current_dir = Path(__file__).parent.absolute()
# 从 astra/data/ 回到 astra/，然后到 astra/parsing/dhcg_parser/
astra_dir = current_dir.parent
parser_path = astra_dir / "parsing" / "dhcg_parser" / "parser.py"

# 使用 importlib 直接加载模块
if not parser_path.exists():
    raise ImportError(f"Cannot find parser.py at {parser_path}")

spec = importlib.util.spec_from_file_location("dhcg_parser.parser", parser_path)
parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_module)

# 从模块中提取需要的类
DynamicGraph = parser_module.DynamicGraph
Node = parser_module.Node
Edge = parser_module.Edge


def robust_clean_agent_name(name: str) -> str:
    """
    终极清洗逻辑：将名字压缩为最简形式以便匹配
    例如: "Verification_Expert_0" -> "verificationexpert"
    """
    if not name:
        return ""
    
    # 转字符串并转小写
    name = str(name).lower()
    
    # 移除括号及内容
    name = re.sub(r'\s*\(.*?\)', '', name)
    
    # 移除末尾的数字后缀 (例如 _0, .1)
    name = re.sub(r'[_\.]\d+$', '', name)
    
    # 移除所有非字母数字字符 (下划线、空格、点)
    # 这一步是关键：让 "Verification_Expert" 和 "VerificationExpert" 变成一样
    name = re.sub(r'[^a-z0-9]', '', name)
    
    return name


class GraphDataConverter:
    """
    将 DynamicGraph 转换为 HeteroGraph 序列的数据适配器
    修复版：增加了 Agent Name Encoding 以解决 Hand-Crafted 数据集的泛化问题
    
    每个时间步 t 对应一个 HeteroGraph 快照，包含：
    - 节点特征矩阵 (node_features)
    - 边索引 (edge_indices)
    - 边特征 (edge_features)
    - 标签 (y_agent, y_step)
    
    核心修复：显式编码 Agent 名字，使得模型能够区分不同的 Agent（如 WebSurfer vs PythonExpert），
    解决 Zero-Shot 场景下的 Domain Gap 问题。
    """
    
    # 节点类型
    NODE_TYPES = ['Agent', 'Tool', 'Artifact', 'Environment']
    
    # 边类型（元组格式：(source_type, edge_type, target_type)）
    EDGE_TYPES = [
        ('Agent', 'Invoke', 'Tool'),
        ('Tool', 'Return', 'Agent'),
        ('Agent', 'Reference', 'Artifact'),
        ('Agent', 'Communicate', 'Agent'),
        ('Environment', 'Affect', 'Agent'),
        ('Environment', 'Affect', 'Tool'),
        ('Environment', 'Affect', 'Artifact'),
    ]
    
    def __init__(self, 
                 node_feat_dim: int = 8192,  # 🔥 Qwen-8B: 4096 (嵌入) + 4096 (元数据)
                 edge_feat_dim: int = 32,
                 normalize_features: bool = True):
        """
        初始化转换器
        
        Args:
            node_feat_dim: 节点特征统一维度
            edge_feat_dim: 边特征维度
            normalize_features: 是否对连续特征进行标准化
        """
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.normalize_features = normalize_features
        
        # 节点特征编码器
        self.node_type_encoder = LabelEncoder()
        # === [混合特征编码] 双重 Agent 编码器 ===
        # ID Encoder: 用于区分同一图中不同的 Agent 实例 (如 WebSurfer_0 vs WebSurfer_1)
        self.agent_id_encoder = LabelEncoder()
        # Role Encoder: 用于跨数据集泛化 (如 WebSurfer 表示同一角色)
        self.agent_role_encoder = LabelEncoder()
        # ======================================
        self.artifact_type_encoder = LabelEncoder()
        self.exitcode_status_encoder = LabelEncoder()
        self.env_event_type_encoder = LabelEncoder()
        
        # 边特征编码器
        self.edge_intent_encoder = LabelEncoder()
        self.edge_status_encoder = LabelEncoder()
        
        # 标准化器（用于连续特征）
        self.scalers = {
            'active_ratio': StandardScaler(),
            'mention_count': StandardScaler(),
        }
        
        # 存储所有可能的值（用于 fit）
        self._node_type_values = set()
        self._agent_id_values = set()      # 原始 ID (WebSurfer_0)
        self._agent_role_values = set()    # 清洗后的 Role (WebSurfer)
        self._artifact_type_values = set()
        self._exitcode_status_values = set()
        self._env_event_type_values = set()
        self._edge_intent_values = set()
        self._edge_status_values = set()
        
        # 是否已拟合编码器
        self._fitted = False
    
    def fit(self, graphs: List[DynamicGraph]):
        """
        在所有图上拟合编码器和标准化器
        
        Args:
            graphs: DynamicGraph 对象列表
        """
        print("正在收集所有可能的值以拟合编码器 (Hybrid ID + Role)...")
        
        # 收集所有离散特征的可能值
        for graph in graphs:
            # 节点特征
            for node in graph.nodes.values():
                self._node_type_values.add(node.type)
                
                # === [混合特征编码] 同时收集 ID 和 Role ===
                if node.type == 'Agent':
                    # 1. ID: 原始 node_id (用于区分实例，如 WebSurfer_0)
                    self._agent_id_values.add(node.id)
                    # 2. Role: 清洗后的名字 (用于泛化，如 WebSurfer)
                    role = self._clean_agent_name(node.id)
                    self._agent_role_values.add(role)
                # ==========================================
                
                if node.type == 'Artifact':
                    if node.artifact_type:
                        self._artifact_type_values.add(node.artifact_type)
                    # 也从动态特征中收集
                    for t, features in node.features.items():
                        if 'artifact_type' in features:
                            self._artifact_type_values.add(features['artifact_type'])
                
                # 收集 Tool 的 exitcode_status
                for t, features in node.features.items():
                    if node.type == 'Tool' and 'exitcode_status' in features:
                        self._exitcode_status_values.add(features['exitcode_status'])
                    
                    if node.type == 'Environment' and 'env_event_type' in features:
                        self._env_event_type_values.add(features['env_event_type'])
            
            # 边特征
            for edge in graph.edges:
                if 'intent' in edge.features:
                    self._edge_intent_values.add(edge.features['intent'])
                if 'status' in edge.features:
                    self._edge_status_values.add(edge.features['status'])
        
        # 拟合编码器
        if self._node_type_values:
            self.node_type_encoder.fit(list(self._node_type_values))
        
        # === [混合特征编码] 拟合双重 Agent 编码器 ===
        # ID Encoder (需包含 unknown)
        agent_ids = list(self._agent_id_values)
        if 'unknown' not in agent_ids:
            agent_ids.append('unknown')
        if agent_ids:
            self.agent_id_encoder.fit(agent_ids)
        
        # Role Encoder (需包含 unknown)
        agent_roles = list(self._agent_role_values)
        if 'unknown' not in agent_roles:
            agent_roles.append('unknown')
        if agent_roles:
            self.agent_role_encoder.fit(agent_roles)
        # ============================================
        
        if self._artifact_type_values:
            self.artifact_type_encoder.fit(list(self._artifact_type_values))
        if self._exitcode_status_values:
            self.exitcode_status_encoder.fit(list(self._exitcode_status_values))
        if self._env_event_type_values:
            self.env_event_type_encoder.fit(list(self._env_event_type_values))
        if self._edge_intent_values:
            self.edge_intent_encoder.fit(list(self._edge_intent_values))
        if self._edge_status_values:
            self.edge_status_encoder.fit(list(self._edge_status_values))
        
        # 收集连续特征用于标准化
        if self.normalize_features:
            active_ratios = []
            mention_counts = []
            
            for graph in graphs:
                for node in graph.nodes.values():
                    for t, features in node.features.items():
                        if node.type == 'Agent' and 'active_ratio' in features:
                            active_ratios.append(features['active_ratio'])
                        if node.type == 'Artifact' and 'mention_count' in features:
                            mention_counts.append(features['mention_count'])
            
            # 拟合 scaler，如果没有数据则用默认值 [0.0] 初始化
            if active_ratios:
                self.scalers['active_ratio'].fit(np.array(active_ratios).reshape(-1, 1))
            else:
                # 如果没有数据，用默认值初始化，避免后续使用时出错
                self.scalers['active_ratio'].fit(np.array([0.0]).reshape(-1, 1))
            
            if mention_counts:
                self.scalers['mention_count'].fit(np.array(mention_counts).reshape(-1, 1))
            else:
                # 如果没有数据，用默认值初始化，避免后续使用时出错
                self.scalers['mention_count'].fit(np.array([0.0]).reshape(-1, 1))
        
        self._fitted = True
        id_count = 0
        role_count = 0
        if hasattr(self.agent_id_encoder, 'classes_') and self.agent_id_encoder.classes_ is not None:
            id_count = len(self.agent_id_encoder.classes_)
        if hasattr(self.agent_role_encoder, 'classes_') and self.agent_role_encoder.classes_ is not None:
            role_count = len(self.agent_role_encoder.classes_)
        print(f"✓ 编码器拟合完成。ID数: {id_count}, Role数: {role_count}")
    
    def _get_encoder_num_classes(self, encoder, default: int) -> int:
        """
        安全地获取编码器的类别数量
        
        Args:
            encoder: LabelEncoder 对象
            default: 如果编码器未拟合，返回的默认值
            
        Returns:
            类别数量
        """
        if self._fitted and hasattr(encoder, 'classes_') and encoder.classes_ is not None:
            return len(encoder.classes_)
        return default
    
    def _is_value_in_encoder(self, encoder, value: str) -> bool:
        """
        安全地检查值是否在编码器中
        
        Args:
            encoder: LabelEncoder 对象
            value: 要检查的值
            
        Returns:
            是否在编码器中
        """
        if self._fitted and hasattr(encoder, 'classes_') and encoder.classes_ is not None:
            return value in encoder.classes_
        return False
    
    def _is_scaler_fitted(self, scaler_name: str) -> bool:
        """
        检查 scaler 是否已拟合
        
        Args:
            scaler_name: scaler 的名称（如 'active_ratio', 'mention_count'）
            
        Returns:
            scaler 是否已拟合
        """
        if scaler_name not in self.scalers:
            return False
        scaler = self.scalers[scaler_name]
        # 检查 scaler 是否有 mean_ 属性（StandardScaler 拟合后会有的属性）
        return hasattr(scaler, 'mean_') and scaler.mean_ is not None
    
    def _get_one_hot(self, encoder, value: str, unknown_val: str = 'unknown') -> torch.Tensor:
        """
        辅助方法：生成 One-Hot 编码向量
        
        Args:
            encoder: LabelEncoder 对象
            value: 要编码的值
            unknown_val: 未知值的替代值
            
        Returns:
            One-Hot 编码向量
        """
        num_classes = self._get_encoder_num_classes(encoder, 10)
        vec = torch.zeros(num_classes)
        
        # 确定目标值：如果在编码器中则直接使用，否则使用 unknown
        target = value if self._is_value_in_encoder(encoder, value) else unknown_val
        
        if self._is_value_in_encoder(encoder, target):
            idx = encoder.transform([target])[0]
            vec[idx] = 1.0
        elif self._fitted and hasattr(encoder, 'classes_') and encoder.classes_ is not None:
            # Fallback: 如果 unknown 在编码器中，使用 unknown
            if unknown_val in encoder.classes_:
                idx = encoder.transform([unknown_val])[0]
                vec[idx] = 1.0
        
        return vec
    
    def _clean_agent_name(self, raw_name: str) -> str:
        """
        清洗 Agent 名字，提取真正的身份标识（增强版）
        
        🔥 核心修复：
        1. 去掉数字后缀: WebSurfer_0 -> WebSurfer
        2. 去除括号说明: Orchestrator (-> WebSurfer) -> Orchestrator
        3. 处理括号中的箭头指向: 提取括号内指向的 Agent 名字（如果有）
        
        Args:
            raw_name: 原始的 Agent 名字（可能是 node_id）
            
        Returns:
            清洗后的 Agent 名字
        """
        if not isinstance(raw_name, str) or not raw_name:
            return str(raw_name) if raw_name else raw_name
        
        name = raw_name
        
        # 1. 处理括号 (针对 Orchestrator (-> WebSurfer) 这种情况)
        # 直接去掉括号及里面的所有内容
        if "(" in name:
            name = name.split("(")[0].strip()
        
        # 2. 去掉 "_DIGIT" 后缀（如 "WebSurfer_1" -> "WebSurfer"）
        # 但保留 "_" 开头的情况（如 "_system"）
        if "_" in name:
            parts = name.split("_")
            # 检查最后一个部分是否为纯数字
            if len(parts) > 1 and parts[-1].isdigit():
                # 去掉最后一个数字部分
                clean_name = "_".join(parts[:-1])
                name = clean_name if clean_name else name  # 防止全部被去掉
        
        return name.strip()
    
    def _extract_node_features(self, node: Node, t: int) -> torch.Tensor:
        """
        提取节点在时间步 t 的特征向量
        
        Args:
            node: 节点对象
            t: 时间步
        
        Returns:
            特征向量 (node_feat_dim,)
        """
        # 🔥 关键修复：向前填充（Forward Fill）策略
        # 如果节点在时间步 t 没有 features，使用最后一个有效时间步的 features
        features = node.features.get(t, {})
        if not features and node.features:
            # 找到最后一个有效时间步（<= t）
            valid_timesteps = [ts for ts in node.features.keys() if ts <= t]
            if valid_timesteps:
                last_valid_t = max(valid_timesteps)
                features = node.features[last_valid_t]
                # 记录向前填充的情况（用于调试）
                import os
                debug_file = os.environ.get('HC_EMB_DEBUG_FILE', None)
                if debug_file and node.type == 'Agent':
                    try:
                        with open(debug_file, 'a', encoding='utf-8') as f:
                            f.write(f"FORWARD_FILL: node_id='{node.id}', t={t}, using t={last_valid_t}\n")
                    except:
                        pass
        
        # === [Embedding 回退] 直接使用 JSON 自带的 384 维 Embedding (Sentence-BERT) ===
        # 不再使用 Ollama API，直接使用 parser.py 在解析时生成的 384 维 Sentence-BERT embedding
        # JSON 文件中的 content_embedding 字段已经包含了完整的文本嵌入向量
        emb_list = features.get('content_embedding', [])
        if not emb_list:
            # 如果没有 embedding，使用零向量
            content_embedding = torch.zeros(384)
            # 🔥 关键修复：记录 Hand-Crafted 数据的零 embedding 情况
            import os
            debug_file = os.environ.get('HC_EMB_DEBUG_FILE', None)
            if debug_file and node.type == 'Agent':
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"ZERO_EMB: node_id='{node.id}', node_type='{node.type}', t={t}\n")
                except:
                    pass
        else:
            # 将列表转换为 tensor
            if isinstance(emb_list, list):
                content_embedding = torch.tensor(emb_list, dtype=torch.float32)
            else:
                content_embedding = torch.tensor(emb_list, dtype=torch.float32)
            
            # 确保是 384 维（Sentence-BERT 标准维度）
            if content_embedding.shape[0] != 384:
                if content_embedding.shape[0] < 384:
                    # 如果维度不足，用零填充
                    padding = torch.zeros(384 - content_embedding.shape[0])
                    content_embedding = torch.cat([content_embedding, padding])
                else:
                    # 如果维度过多，截断到 384
                    content_embedding = content_embedding[:384]
            
            # 🔥 关键修复：检查 embedding 是否全为 0（Hand-Crafted 数据特征提取失败）
            emb_sum = content_embedding.abs().sum().item()
            is_handcrafted_likely = any(
                keyword in node.id.lower() 
                for keyword in ['surfer', 'orchestrator', 'excel', 'researcher', 'analyst', 'planner', 'executor']
            )
            
            if emb_sum < 1e-6 and node.type == 'Agent':
                import os
                debug_file = os.environ.get('HC_EMB_DEBUG_FILE', None)
                if debug_file:
                    try:
                        with open(debug_file, 'a', encoding='utf-8') as f:
                            f.write(f"NEAR_ZERO_EMB: node_id='{node.id}', emb_sum={emb_sum:.6f}, is_hc_likely={is_handcrafted_likely}\n")
                    except:
                        pass
            
            # 🔥 关键修复：对于 Hand-Crafted 数据，额外检查 embedding 质量
            if is_handcrafted_likely and emb_sum < 0.1:
                # Hand-Crafted 数据的 embedding 异常小，可能是特征提取失败
                import os
                debug_file = os.environ.get('HC_EMB_DEBUG_FILE', None)
                if debug_file:
                    try:
                        with open(debug_file, 'a', encoding='utf-8') as f:
                            f.write(f"HC_LOW_EMB: node_id='{node.id}', emb_sum={emb_sum:.6f}, emb_mean={content_embedding.mean().item():.6f}\n")
                    except:
                        pass
        # ==========================================
        
        # 2. 节点类型编码 (one-hot)
        num_node_types = self._get_encoder_num_classes(self.node_type_encoder, 4)
        node_type_encoded = torch.zeros(num_node_types)
        if self._is_value_in_encoder(self.node_type_encoder, node.type):
            idx = self.node_type_encoder.transform([node.type])[0]
            node_type_encoded[idx] = 1.0
        
        # 3. 类型特定特征
        type_specific_features = []
        
        if node.type == 'Agent':
            # === [混合特征编码] 双重编码：ID + Role ===
            raw_name = node.id
            
            # A. Role Feature (泛化语义): 清洗后的名字
            role = self._clean_agent_name(raw_name)
            role_vec = self._get_one_hot(self.agent_role_encoder, role)
            
            # B. ID Feature (区分实例): 原始 node_id
            id_vec = self._get_one_hot(self.agent_id_encoder, raw_name)
            # ===========================================

            # is_terminate (bool -> float)
            is_terminate = float(features.get('is_terminate', False))
            # plan_signal (bool -> float)
            plan_signal = float(features.get('plan_signal', False))
            # active_ratio (float)
            active_ratio = features.get('active_ratio', 0.0)
            if self.normalize_features and self._is_scaler_fitted('active_ratio'):
                active_ratio = self.scalers['active_ratio'].transform([[active_ratio]])[0, 0]
            
            # 拼接: [常规特征] + [Role] + [ID]
            type_specific_features = [is_terminate, plan_signal, active_ratio] + role_vec.tolist() + id_vec.tolist()
        
        elif node.type == 'Tool':
            # exitcode_status (one-hot)
            num_exitcode_status = self._get_encoder_num_classes(self.exitcode_status_encoder, 3)
            exitcode_status_encoded = torch.zeros(num_exitcode_status)
            exitcode_status = features.get('exitcode_status', 'unknown')
            if self._is_value_in_encoder(self.exitcode_status_encoder, exitcode_status):
                idx = self.exitcode_status_encoder.transform([exitcode_status])[0]
                exitcode_status_encoded[idx] = 1.0
            type_specific_features = exitcode_status_encoded.tolist()
        
        elif node.type == 'Artifact':
            # artifact_type (one-hot)
            num_artifact_types = self._get_encoder_num_classes(self.artifact_type_encoder, 2)
            artifact_type_encoded = torch.zeros(num_artifact_types)
            artifact_type = features.get('artifact_type', node.artifact_type or 'file')
            if self._is_value_in_encoder(self.artifact_type_encoder, artifact_type):
                idx = self.artifact_type_encoder.transform([artifact_type])[0]
                artifact_type_encoded[idx] = 1.0
            # mention_count (int -> float, normalized)
            mention_count = float(features.get('mention_count', 0))
            if self.normalize_features and self._is_scaler_fitted('mention_count'):
                mention_count = self.scalers['mention_count'].transform([[mention_count]])[0, 0]
            type_specific_features = artifact_type_encoded.tolist() + [mention_count]
        
        elif node.type == 'Environment':
            # env_event_type (one-hot)
            num_env_event_types = self._get_encoder_num_classes(self.env_event_type_encoder, 5)
            env_event_type_encoded = torch.zeros(num_env_event_types)
            env_event_type = features.get('env_event_type', 'none')
            if self._is_value_in_encoder(self.env_event_type_encoder, env_event_type):
                idx = self.env_event_type_encoder.transform([env_event_type])[0]
                env_event_type_encoded[idx] = 1.0
            type_specific_features = env_event_type_encoded.tolist()
        
        # 拼接所有特征
        type_specific_tensor = torch.tensor(type_specific_features, dtype=torch.float32)
        all_features = torch.cat([
            content_embedding,  # 384 (Sentence-BERT from JSON)
            node_type_encoded,  # ~4
            type_specific_tensor  # 可变
        ])
        
        # 如果特征维度小于目标维度，用零填充
        if all_features.shape[0] < self.node_feat_dim:
            padding = torch.zeros(self.node_feat_dim - all_features.shape[0])
            all_features = torch.cat([all_features, padding])
        # 如果大于目标维度，截断
        elif all_features.shape[0] > self.node_feat_dim:
            all_features = all_features[:self.node_feat_dim]
        
        return all_features
    
    def _extract_edge_features(self, edge: Edge) -> torch.Tensor:
        """
        提取边特征向量
        
        Args:
            edge: 边对象
        
        Returns:
            特征向量 (edge_feat_dim,)
        """
        features = edge.features
        
        # 1. intent 编码 (one-hot)
        num_intents = self._get_encoder_num_classes(self.edge_intent_encoder, 5)
        intent_encoded = torch.zeros(num_intents)
        intent = features.get('intent', 'Inform')
        if self._is_value_in_encoder(self.edge_intent_encoder, intent):
            idx = self.edge_intent_encoder.transform([intent])[0]
            intent_encoded[idx] = 1.0
        
        # 2. status 编码 (one-hot)
        num_statuses = self._get_encoder_num_classes(self.edge_status_encoder, 3)
        status_encoded = torch.zeros(num_statuses)
        status = features.get('status', 'unknown')
        if self._is_value_in_encoder(self.edge_status_encoder, status):
            idx = self.edge_status_encoder.transform([status])[0]
            status_encoded[idx] = 1.0
        
        # 拼接特征
        edge_features = torch.cat([intent_encoded, status_encoded])
        
        # 调整维度
        if edge_features.shape[0] < self.edge_feat_dim:
            padding = torch.zeros(self.edge_feat_dim - edge_features.shape[0])
            edge_features = torch.cat([edge_features, padding])
        elif edge_features.shape[0] > self.edge_feat_dim:
            edge_features = edge_features[:self.edge_feat_dim]
        
        return edge_features
    
    def convert(self, graph: DynamicGraph) -> Tuple[List[HeteroGraph], Dict[str, Any]]:
        """
        将单个 DynamicGraph 转换为 HeteroGraph 序列
        
        Args:
            graph: DynamicGraph 对象
        
        Returns:
            (hetero_graph_list, labels_dict)
            - hetero_graph_list: List[HeteroGraph]，每个元素对应一个时间步的快照
            - labels_dict: 包含 y_agent 和 y_step 的字典
        """
        if not self._fitted:
            raise RuntimeError("转换器尚未拟合，请先调用 fit() 方法")
        
        # 确定时间步范围
        # max_actual_t: 实际日志中的最大时间步（有数据的时间步）
        max_actual_t = 0
        for edge in graph.edges:
            max_actual_t = max(max_actual_t, edge.timestamp)
        for node in graph.nodes.values():
            if node.features:
                max_actual_t = max(max_actual_t, max(node.features.keys()))
        
        # 修复：检查 mistake_step 是否越界，如果越界则扩展 num_timesteps
        # mistake_step 可能基于对话历史索引，而图的时间戳可能更短
        gt = graph.ground_truth
        mistake_step_str = gt.get('mistake_step', '')
        max_t = max_actual_t  # 初始化为实际最大时间步
        if mistake_step_str:
            try:
                mistake_step_int = int(mistake_step_str)
                # 如果 mistake_step 超出当前范围，扩展 num_timesteps
                if mistake_step_int >= 0 and mistake_step_int >= max_actual_t + 1:
                    max_t = mistake_step_int
            except (ValueError, TypeError):
                pass  # 如果转换失败，使用原来的 max_t
        
        num_timesteps = max_t + 1
        
        # 构建节点ID到索引的映射（按类型分组）
        node_id_to_idx = {}
        node_idx_to_id = {}
        node_type_groups = {nt: [] for nt in self.NODE_TYPES}
        
        for node_id, node in graph.nodes.items():
            node_type = node.type
            if node_type in node_type_groups:
                idx = len(node_type_groups[node_type])
                node_id_to_idx[node_id] = (node_type, idx)
                node_idx_to_id[(node_type, idx)] = node_id
                node_type_groups[node_type].append(node_id)
        
        # 为每个时间步创建 HeteroGraph
        hetero_graph_list = []
        # 存储最后一个有效时间步的特征（用于填充未来时间步）
        last_valid_features = {}  # Dict[node_type, Tensor]
        last_valid_edges = {}  # Dict[edge_type_tuple, (edge_index, edge_attr)]
        
        for t in range(num_timesteps):
            hetero_graph = HeteroGraph()
            
            # 🔥 关键修复：如果 t > max_actual_t，说明这是填充的未来时间步
            # 复制最后一个有效时间步的特征，而不是使用全零
            is_padding_step = t > max_actual_t
            
            if is_padding_step:
                # 使用最后一个有效时间步的特征（模拟"系统卡死"或"状态延续"）
                # 从 last_valid_features 复制节点特征
                for node_type in self.NODE_TYPES:
                    if node_type in last_valid_features:
                        # 深拷贝特征，避免引用问题
                        hetero_graph.node_features[node_type] = last_valid_features[node_type].clone()
                    else:
                        # 如果没有该节点类型的历史特征，使用空张量
                        hetero_graph.node_features[node_type] = torch.zeros(0, self.node_feat_dim)
                
                # 从 last_valid_edges 复制边（但边的时间戳信息不复制，因为这是填充步）
                # 注意：填充步通常没有新的边，所以这里可以选择不复制边，或者复制但标记为历史边
                # 为了简化，填充步不复制边（表示系统状态延续但没有新交互）
                # 如果需要，可以取消下面的注释来复制边
                # for edge_type_tuple in self.EDGE_TYPES:
                #     if edge_type_tuple in last_valid_edges:
                #         edge_index, edge_attr = last_valid_edges[edge_type_tuple]
                #         hetero_graph.edge_indices[edge_type_tuple] = edge_index.clone()
                #         hetero_graph.edge_features[edge_type_tuple] = edge_attr.clone()
            else:
                # 正常时间步：提取实际特征
                # 1. 添加节点特征
                # 确保所有节点类型都被添加，即使节点数为0（使用空张量）
                for node_type in self.NODE_TYPES:
                    node_ids = node_type_groups[node_type]
                    
                    # 为每个节点提取特征
                    node_features_list = []
                    for node_id in node_ids:
                        node = graph.nodes[node_id]
                        # 如果节点在时间步 t 存在，提取特征；否则用零向量
                        if node.created_at <= t:
                            feat = self._extract_node_features(node, t)
                        else:
                            feat = torch.zeros(self.node_feat_dim)
                        node_features_list.append(feat)
                    
                    # 即使节点列表为空，也添加该节点类型（使用空张量）
                    # 这样确保所有时间步都包含所有节点类型
                    if node_features_list:
                        node_features_tensor = torch.stack(node_features_list)
                        hetero_graph.node_features[node_type] = node_features_tensor
                        # 保存最后一个有效时间步的特征
                        last_valid_features[node_type] = node_features_tensor.clone()
                    else:
                        # 创建空张量 [0, node_feat_dim]，保持维度一致性
                        hetero_graph.node_features[node_type] = torch.zeros(0, self.node_feat_dim)
                        # 空张量也保存（虽然为空，但保持一致性）
                        last_valid_features[node_type] = torch.zeros(0, self.node_feat_dim)
                
                # 2. 添加边
                for edge_type_tuple in self.EDGE_TYPES:
                    src_type, edge_type, dst_type = edge_type_tuple
                    
                    # 筛选当前时间步的边
                    edge_indices = []
                    edge_attrs = []
                    
                    for edge in graph.edges:
                        if edge.type == edge_type and edge.timestamp == t:
                            src_node = graph.nodes.get(edge.source)
                            dst_node = graph.nodes.get(edge.target)
                            
                            if src_node and dst_node:
                                src_type_actual = src_node.type
                                dst_type_actual = dst_node.type
                                
                                # 检查类型是否匹配
                                if src_type_actual == src_type and dst_type_actual == dst_type:
                                    src_idx = node_id_to_idx[edge.source][1]
                                    dst_idx = node_id_to_idx[edge.target][1]
                                    
                                    edge_indices.append([src_idx, dst_idx])
                                    edge_attrs.append(self._extract_edge_features(edge))
                    
                    if edge_indices:
                        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                        edge_attr = torch.stack(edge_attrs)
                        hetero_graph.edge_indices[edge_type_tuple] = edge_index
                        hetero_graph.edge_features[edge_type_tuple] = edge_attr
                        # 保存最后一个有效时间步的边
                        last_valid_edges[edge_type_tuple] = (edge_index.clone(), edge_attr.clone())
            
            hetero_graph.node_id_to_idx = node_id_to_idx
            hetero_graph_list.append(hetero_graph)
        
        # 3. 构建标签
        labels = self._build_labels(graph, node_id_to_idx, num_timesteps)
        
        return hetero_graph_list, labels
    
    def _build_labels(self, 
                     graph: DynamicGraph, 
                     node_id_to_idx: Dict[str, Tuple[str, int]], 
                     num_timesteps: int) -> Dict[str, Any]:
        """
        构建标签 (修复版 V4：无视类型的强制匹配)
        """
        gt = graph.ground_truth
        mistake_agent_str = gt.get('mistake_agent')
        mistake_step_str = gt.get('mistake_step')
        
        # 1. 检查 Healed
        is_healed = (mistake_agent_str is None or 
                    (isinstance(mistake_agent_str, str) and not mistake_agent_str.strip()))
        
        if is_healed:
            return {
                'y_agent': -100,
                'y_step': -100,
                'mistake_agent_name': '',
                'mistake_step_str': '',
            }
        
        # 2. Agent 匹配 (终极修复：不再过滤 node_type)
        y_agent = -1
        target_clean = robust_clean_agent_name(mistake_agent_str)
        
        # 策略 A: 在所有节点中寻找（不仅仅是 Agent 类型的节点）
        # 因为有时候 Tool 或 Environment 也会被误标为 Agent
        best_match_idx = -1
        best_match_score = 0
        
        for node_id, (node_type, idx) in node_id_to_idx.items():
            # 只考虑 Agent 类型的节点作为候选，除非完全匹配失败
            is_agent_type = (node_type == 'Agent')
            
            current_clean = robust_clean_agent_name(node_id)
            
            # 1. 精确匹配 (最高优先级)
            if current_clean == target_clean:
                if is_agent_type:
                    y_agent = idx
                    break # 找到完美的 Agent，直接退出
                else:
                    # 找到了名字对的，但类型不是 Agent，暂存
                    if best_match_score < 3:
                        best_match_score = 3
                        best_match_idx = idx
            
            # 2. 包含匹配 (User in UserProxy)
            elif (target_clean in current_clean or current_clean in target_clean) and is_agent_type:
                if best_match_score < 2:
                    best_match_score = 2
                    best_match_idx = idx
        
        # 如果没有找到完美匹配 (y_agent == -1)，使用最佳备选
        if y_agent == -1 and best_match_idx != -1:
            y_agent = best_match_idx
            # print(f"⚠️ [Fuzzy Match] '{mistake_agent_str}' -> Node Index {y_agent} (Score: {best_match_score})")

        # 3. Step 匹配 (保持不变)
        y_step = -1
        if mistake_step_str is not None:
            try:
                # 有些 step 是 "Step 13"，需要清洗
                step_val_str = str(mistake_step_str).lower().replace('step', '').strip()
                val = int(float(step_val_str)) # 处理 "13.0"
                if 0 <= val < num_timesteps:
                    y_step = val
                else:
                    # 如果越界，截断到最后一步 (防止 Loss NaN)
                    y_step = min(val, num_timesteps - 1)
            except (ValueError, TypeError):
                y_step = -1

        # 🚨 最终保底检查：如果是 Fatal 样本但没匹配到 Agent，记录详细信息
        if not is_healed and y_agent == -1:
            # 🔥 关键修复：记录 Hand-Crafted 数据的匹配失败信息（用于诊断）
            # 检查是否是 Hand-Crafted 数据（通过 ground_truth 或其他方式判断）
            # 这里我们记录所有匹配失败的情况，后续可以通过日志分析
            import os
            debug_file = os.environ.get('HC_DEBUG_FILE', None)
            if debug_file:
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"MATCH_FAILED: mistake_agent='{mistake_agent_str}', "
                               f"target_clean='{target_clean}', "
                               f"available_nodes={list(node_id_to_idx.keys())[:10]}\n")
                except:
                    pass
            # 这种情况非常危险，会导致 Loss 计算被忽略
            # 但为了避免日志过多，我们只在特定条件下打印

        return {
            'y_agent': y_agent if y_agent != -1 else -100, # 如果实在没找到，设为 -100 忽略
            'y_step': y_step,
            'mistake_agent_name': mistake_agent_str if mistake_agent_str else '',
            'mistake_step_str': mistake_step_str if mistake_step_str else '',
            'matched_node_idx': y_agent
        }


def reconstruct_graph_from_json(json_data: Dict[str, Any]) -> DynamicGraph:
    """
    从 JSON 数据重建 DynamicGraph 对象 (修复版 V3：终极关键词覆盖)
    """
    # 🔥 核心修复 A: 扩充强制转正的关键词列表（定义在函数开头，供整个函数使用）
    # 只要名字里包含这些词，一律视为 Agent，确保能被标签匹配到
    agent_keywords = [
        "expert", "orchestrator", "user", "agent", # 原有词
        "terminal", "coder", "analyst", "surfer", "assistant", "planner", "executor" # 新增词
    ]
    
    # 1. 创建 DynamicGraph 对象
    question = json_data.get('question', '')
    ground_truth_raw = json_data.get('ground_truth', {})
    
    # 🔥 修复：处理 ground_truth 可能是字符串的情况
    # 如果 ground_truth 是字符串，则从 JSON 顶层提取字段构建字典
    if isinstance(ground_truth_raw, str):
        ground_truth = {
            'mistake_agent': json_data.get('mistake_agent', ''),
            'mistake_step': json_data.get('mistake_step', ''),
            'mistake_reason': json_data.get('mistake_reason', ''),
            'ground_truth': ground_truth_raw
        }
    elif isinstance(ground_truth_raw, dict):
        # 如果已经是字典，确保包含所有必要字段
        ground_truth = {
            'mistake_agent': ground_truth_raw.get('mistake_agent', json_data.get('mistake_agent', '')),
            'mistake_step': ground_truth_raw.get('mistake_step', json_data.get('mistake_step', '')),
            'mistake_reason': ground_truth_raw.get('mistake_reason', json_data.get('mistake_reason', '')),
            'ground_truth': ground_truth_raw.get('ground_truth', ground_truth_raw)
        }
    else:
        # 如果既不是字符串也不是字典，尝试从顶层提取
        ground_truth = {
            'mistake_agent': json_data.get('mistake_agent', ''),
            'mistake_step': json_data.get('mistake_step', ''),
            'mistake_reason': json_data.get('mistake_reason', ''),
            'ground_truth': str(ground_truth_raw) if ground_truth_raw else ''
        }
    
    graph = DynamicGraph(question=question, ground_truth=ground_truth)
    
    # 2. 重建节点 (从 nodes 列表)
    nodes_data = json_data.get('nodes', {})
    for node_id, node_data in nodes_data.items():
        original_type = node_data.get('type', 'Agent')
        created_at = node_data.get('created_at', 0)
        artifact_type = node_data.get('artifact_type', None)
        
        lower_id = node_id.lower()
        
        if any(keyword in lower_id for keyword in agent_keywords):
            final_type = 'Agent'
        else:
            final_type = original_type
            
        node = Node(
            node_id=node_id,
            node_type=final_type,
            creation_time=created_at,
            artifact_type=artifact_type
        )
        
        # 重建 features
        features_data = node_data.get('features', {})
        for t_str, feat_dict in features_data.items():
            try:
                t = int(t_str)
                node.features[t] = feat_dict
            except (ValueError, TypeError):
                continue
        
        graph.add_node(node)
    
    # 3. 重建边 & 自动复活缺失的幽灵节点
    edges_data = json_data.get('edges', [])
    for edge_data in edges_data:
        source = edge_data.get('source', '')
        target = edge_data.get('target', '')
        edge_type = edge_data.get('type', 'Communicate')
        timestamp = edge_data.get('timestamp', 0)
        features = edge_data.get('features', {})
        
        # 🔥 核心修复 B: 检查 source/target 是否已存在，不存在则自动创建
        # 同样应用强制修正逻辑
        if source and source not in graph.nodes:
            lower_s = source.lower()
            ghost_type = 'Agent' if any(k in lower_s for k in agent_keywords) else 'Agent' # 默认幽灵由于也是GT指控对象，倾向于设为Agent
            graph.add_node(Node(node_id=source, node_type=ghost_type, creation_time=0))
            
        if target and target not in graph.nodes:
            lower_t = target.lower()
            ghost_type = 'Agent' if any(k in lower_t for k in agent_keywords) else 'Agent'
            graph.add_node(Node(node_id=target, node_type=ghost_type, creation_time=0))

        # 创建边
        edge = Edge(
            source_id=source,
            target_id=target,
            edge_type=edge_type,
            timestamp=timestamp,
            features=features
        )
        graph.add_edge(edge)
    
    # 4. GT 复活保底
    mistake_agent = ground_truth.get('mistake_agent')
    if mistake_agent and mistake_agent not in graph.nodes:
        # GT 指控的一定是 Agent
        graph.add_node(Node(node_id=mistake_agent, node_type='Agent', creation_time=0))

    return graph


def test_data_adapter():
    """测试数据适配器"""
    import json
    from pathlib import Path
    
    # 查找可用的测试文件（使用新的命名格式）
    output_dir = Path("outputs")
    test_file = None
    
    # 优先查找 Algorithm-Generated 文件
    for pattern in ["Algorithm-Generated_*_graph.json", "Hand-Crafted_*_graph.json"]:
        files = sorted(output_dir.glob(pattern))
        if files:
            test_file = files[0]
            break
    
    if test_file is None or not test_file.exists():
        print(f"测试文件不存在，请确保 outputs 目录下有 Algorithm-Generated_*_graph.json 或 Hand-Crafted_*_graph.json 文件")
        return
    
    print(f"使用测试文件: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 从 JSON 数据重建 DynamicGraph（不再使用 MainParser）
    graph = reconstruct_graph_from_json(graph_data)
    
    print(f"加载的图: {graph}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    # 创建转换器
    converter = GraphDataConverter(node_feat_dim=4096, edge_feat_dim=32)  # 🔥 必须改成 4096！(涵盖 3635 需求)
    
    # 拟合（使用单个图）
    converter.fit([graph])
    
    # 转换
    hetero_graph_list, labels = converter.convert(graph)
    
    print(f"\n转换结果:")
    print(f"时间步数: {len(hetero_graph_list)}")
    print(f"标签: {labels}")
    
    # 检查第一个时间步的数据
    if hetero_graph_list:
        first_snapshot = hetero_graph_list[0]
        print(f"\n第一个时间步的快照:")
        print(f"节点类型: {first_snapshot.get_node_types()}")
        print(f"边类型: {first_snapshot.get_edge_types()}")
        for node_type in first_snapshot.get_node_types():
            if node_type in first_snapshot.node_features:
                print(f"  {node_type}: {first_snapshot.node_features[node_type].shape}")


if __name__ == "__main__":
    test_data_adapter()

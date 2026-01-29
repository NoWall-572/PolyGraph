"""
动态异构因果图 (DHCG) 解析器 - 语义增强版
用于解析 Who&When 数据集的 JSON 日志并构建动态异构图
包含 Agent 身份嵌入 (Identity Embedding) 以提升 Sim-to-Real 泛化能力
支持单个文件和目录批量处理
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# 在导入torch相关库之前禁用CUDA，避免兼容性问题
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用CUDA

from sentence_transformers import SentenceTransformer

# 全局模型实例（延迟加载）
_model = None


def get_embedding_model():
    """获取或初始化sentence-transformers模型（强制使用CPU）"""
    global _model
    if _model is None:
        model_name = 'sentence-transformers/bert-base-nli-mean-tokens'  # 默认模型名称
        try:
            # 🔥 修正 1: 尝试设置环境变量或使用镜像 (如果您的环境支持)
            # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 示例：使用国内镜像
            
            # 🔥 修正 2: 优先使用本地缓存路径
            local_cache_path = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--bert-base-nli-mean-tokens'
            if local_cache_path.exists():
                # 尝试找到实际的模型文件路径
                snapshot_dirs = list(local_cache_path.glob('snapshots/*'))
                if snapshot_dirs:
                    model_name = str(snapshot_dirs[0])
                    print(f"✅ 使用本地模型缓存: {model_name}")
                else:
                    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
            else:
                model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
            
            _model = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            # 如果上面的模型加载失败，尝试使用更基础的模型
            print(f"警告: 无法加载 {model_name}，尝试使用 all-MiniLM-L6-v2: {e}")
            try:
                local_cache_path_mini = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
                if local_cache_path_mini.exists():
                    snapshot_dirs_mini = list(local_cache_path_mini.glob('snapshots/*'))
                    if snapshot_dirs_mini:
                        model_name_mini = str(snapshot_dirs_mini[0])
                        print(f"✅ 使用本地模型缓存: {model_name_mini}")
                    else:
                        model_name_mini = 'all-MiniLM-L6-v2'
                else:
                    model_name_mini = 'all-MiniLM-L6-v2'
                     
                _model = SentenceTransformer(model_name_mini, device='cpu')
            except Exception as e2:
                print(f"错误: 无法加载任何模型: {e2}")
                raise
    return _model


class Node:
    """图节点类"""
    def __init__(self, node_id: str, node_type: str, creation_time: int, artifact_type: Optional[str] = None):
        self.id: str = node_id
        self.type: str = node_type  # e.g., "Agent", "Tool", "Artifact", "Environment"
        self.created_at: int = creation_time
        # features[t] 存储节点在时间步 t 的动态特征
        self.features: Dict[int, Dict[str, Any]] = {}
        # artifact_type: 仅对Artifact节点有效，值为"file"或"url"
        self.artifact_type: Optional[str] = artifact_type

    def __repr__(self):
        artifact_info = f", artifact_type='{self.artifact_type}'" if self.artifact_type else ""
        return f"Node(id='{self.id}', type='{self.type}', created_at={self.created_at}{artifact_info})"


class Edge:
    """图边类"""
    def __init__(self, source_id: str, target_id: str, edge_type: str, timestamp: int, features: Dict[str, Any]):
        self.source: str = source_id
        self.target: str = target_id
        self.type: str = edge_type  # e.g., "Communicate", "Invoke", "Return", "Reference", "Affect"
        self.timestamp: int = timestamp
        self.features: Dict[str, Any] = features

    def __repr__(self):
        features_str = f", Features: {self.features}" if self.features else ""
        return f"Edge(t={self.timestamp}): {self.source} -> {self.target} (Type: {self.type}{features_str})"


class DynamicGraph:
    """动态异构图类"""
    def __init__(self, question: str, ground_truth: Dict[str, Any]):
        self.question: str = question
        self.ground_truth: Dict[str, Any] = ground_truth
        self.nodes: Dict[str, Node] = {}  # Key: node_id
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        """添加节点到图中"""
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """添加边到图中"""
        self.edges.append(edge)

    def __repr__(self):
        question_preview = self.question[:30] + "..." if len(self.question) > 30 else self.question
        return (f"DynamicGraph(\n"
                f"  Nodes: {len(self.nodes)},\n"
                f"  Edges: {len(self.edges)},\n"
                f"  Question: '{question_preview}'\n)")


def GetOrCreateNode(name: str, node_type: str, t: int, node_registry: Dict[str, Node], artifact_type: Optional[str] = None) -> Node:
    """
    获取或创建节点
    """
    if name not in node_registry:
        # 如果是Artifact节点，根据name判断artifact_type
        if node_type == "Artifact" and artifact_type is None:
            if "http" in name.lower():
                artifact_type = "url"
            else:
                artifact_type = "file"
        node = Node(name, node_type, t, artifact_type=artifact_type)
        node_registry[name] = node
    return node_registry[name]


def DetermineNodeType(actor: str, system_prompt: Dict[str, Any], event: Dict[str, Any]) -> str:
    """
    确定节点的类型
    
    🔥 ASTRA-Gen 3.0 增强：
    1. 严格匹配 Computer_terminal（下划线）为 Tool
    2. 匹配 Coder、Surfer 等 Agent 类型
    """
    # 强制匹配 ASTRA-Gen 3.0 的命名：Computer_terminal（下划线）
    if actor == "Computer_terminal":
        return "Tool"

    # 如果event中包含exitcode字段，或者content中包含"exitcode:"，返回"Tool"
    if "exitcode" in event:
        return "Tool"
    content = event.get('content', '')
    if re.search(r'exitcode:\s*\d+', content, re.IGNORECASE):
        return "Tool"

    # 如果actor在system_prompt的键中，返回"Agent"
    if system_prompt and actor in system_prompt:
        return "Agent"

    # 如果actor的名字匹配正则表达式，返回"Agent"
    # 🔥 增强：匹配 ASTRA-Gen 3.0 的 Agent 命名（Coder、Surfer、Orchestrator等）
    if re.search(r'(Expert|Assistant|Orchestrator|Surfer|Coder|Planner)', actor):
        return "Agent"

    # 默认返回"Agent" (宽松策略，防止真实数据中的未知Agent被漏判)
    return "Agent"


def FindCallerAgent(tool_node: Node, t: int, history: List[Dict[str, Any]],
                   system_prompt: Dict[str, Any]) -> str:
    """
    查找调用工具的Agent
    """
    # 从时间步 t-1 开始向上回溯
    for tau in range(t - 1, -1, -1):
        if tau < len(history):
            prev_event = history[tau]
            prev_actor = prev_event.get('name') or prev_event.get('role', '')
            prev_content = prev_event.get('content', '')

            # 检查prev_actor是否是Agent类型
            actor_type = DetermineNodeType(prev_actor, system_prompt, prev_event)
            if actor_type == "Agent":
                # 检查prev_content是否包含代码块（表示调用了工具）
                code_block_pattern = r'```(?:python|sh|bash|javascript|js|java|cpp|c\+\+|c|go|rust|sql|html|css|xml|json|yaml|yml|markdown|md|text|plaintext)[\s\S]*?```'
                if re.search(code_block_pattern, prev_content, re.IGNORECASE):
                    return prev_actor

    return "Broadcast"


def ParseEdges(source_node: Node, content: str, t: int, history: List[Dict[str, Any]],
               node_registry: Dict[str, Node], event: Dict[str, Any], system_prompt: Dict[str, Any],
               mention_counter: Dict[str, int]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    解析边
    """
    interactions = []

    # 1. Invoke: 如果source_node.type == "Agent" 并且content包含代码块
    if source_node.type == "Agent":
        # 检查是否包含代码块
        code_block_pattern = r'```(?:python|sh|bash|javascript|js|java|cpp|c\+\+|c|go|rust|sql|html|css|xml|json|yaml|yml|markdown|md|text|plaintext)[\s\S]*?```'
        if re.search(code_block_pattern, content, re.IGNORECASE):
            interactions.append(("Computer_terminal", "Tool", "Invoke", {"intent": "Command"}))

    # 2. Return: 如果source_node.type == "Tool"
    if source_node.type == "Tool":
        exitcode = None
        if "exitcode" in event:
            exitcode = event.get("exitcode")
        else:
            exitcode_match = re.search(r'exitcode:\s*(\d+)', content, re.IGNORECASE)
            if exitcode_match:
                exitcode = exitcode_match.group(1)

        status = "unknown"
        if exitcode is not None:
            # 尝试转换为整数判断
            try:
                status = "success" if int(str(exitcode)) == 0 else "failure"
            except:
                pass

        # Return边的意图固定为"Inform"
        caller_agent = FindCallerAgent(source_node, t, history, system_prompt)
        interactions.append((caller_agent, "Agent", "Return", {"status": status, "intent": "Inform"}))

    # 3. Reference: 提取文件路径和URL
    reference_pattern = r'(\.\./[\w/.-]+|https?://[\w/.-]+|filename:\s*[\w.-]+)'
    references = re.findall(reference_pattern, content)
    references = list(set(references))

    for ref in references:
        ref_clean = ref.strip().rstrip('.,;:')
        if ref_clean.startswith("filename:"):
            ref_clean = ref_clean.replace("filename:", "").strip().rstrip('.,;:')

        mention_counter[ref_clean] += 1
        interactions.append((ref_clean, "Artifact", "Reference", {"intent": "Inform"}))

    # 4. Communicate
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, content)

    intent = "Inform"
    if mentions:
        if '?' in content or content.strip().startswith('@'):
            intent = "Query"
        for mention in mentions:
            interactions.append((mention, "Agent", "Communicate", {"intent": intent}))
    else:
        if t > 0 and t - 1 < len(history):
            prev_event = history[t - 1]
            prev_actor = prev_event.get('name') or prev_event.get('role', '')
            if prev_actor != source_node.id:
                if not re.search(r'```[\s\S]*?```', content):
                    if '?' in content: intent = "Query"
                    interactions.append((prev_actor, "Agent", "Communicate", {"intent": intent}))
                else:
                    intent = "Broadcast"
                    interactions.append(("Broadcast", "Environment", "Communicate", {"intent": intent}))
            else:
                intent = "Broadcast"
                interactions.append(("Broadcast", "Environment", "Communicate", {"intent": intent}))
        else:
            intent = "Broadcast"
            interactions.append(("Broadcast", "Environment", "Communicate", {"intent": intent}))

    # 5. Affect
    error_pattern = r'(Timeout|OOM|Permission Denied|Network Error)'
    error_match = re.search(error_pattern, content, re.IGNORECASE)
    if error_match:
        event_type = error_match.group(1).lower()
        interactions.append(("__AFFECT__", "Environment", "Affect", {"intent": "Reject", "event_type": event_type}))

    return interactions


def ExtractNodeFeatures(source_node: Node, event: Dict[str, Any], t: int, history: List[Dict[str, Any]],
                        mention_counter: Dict[str, int], system_prompt: Dict[str, Any], env_event_type: Optional[str] = None) -> Dict[str, Any]:
    """
    提取节点特征 (语义增强版)

    🔥 核心改进：
    1. 接收 system_prompt 参数
    2. 如果是 Agent 节点，将 Agent Name 和 Role Description 拼接到 Content 前面
    3. 生成富语义 Embedding
    """
    features = {}
    content = event.get('content', '')

    model = get_embedding_model()

    # --- 身份嵌入逻辑开始 ---
    if source_node.type == "Agent":
        agent_name = source_node.id
        # 尝试从 system_prompt 获取角色描述
        # 有些数据集 system_prompt 可能是 dict，有些可能是 list，这里做个防御
        role_desc = ""
        if isinstance(system_prompt, dict):
            role_desc = system_prompt.get(agent_name, "")

        # 构造富文本：[身份] + [描述] + [当前动作]
        rich_text = f"Agent: {agent_name}. Role: {role_desc}.\nAction: {content}"

        # 截断防止过长 (BERT通常限制512 tokens，这里按字符截断做个大致限制)
        if len(rich_text) > 2000:
            rich_text = rich_text[:2000]
    else:
        # 对于 Tool 或 Artifact，只看内容
        rich_text = content
    # --- 身份嵌入逻辑结束 ---

    # 使用富文本生成 Embedding
    embedding = model.encode(rich_text, convert_to_numpy=True).tolist()
    features['content_embedding'] = embedding

    # [新增] 结构化 Ledger 解析（ASTRA-Gen 3.0 增强）
    # 针对 Orchestrator (thought) 的 JSON Ledger 提取显式特征
    # 节点名称格式：Orchestrator (thought)
    if source_node.type == "Agent" and "thought" in source_node.id.lower() and "{" in content:
        try:
            # 尝试提取 JSON Ledger
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                ledger = json.loads(json_match.group(0))
                # 提取特征 1: Agent 自认为是否完成任务 (0.0/1.0)
                is_satisfied = ledger.get("is_request_satisfied", {})
                if isinstance(is_satisfied, dict):
                    features['satisfied_signal'] = 1.0 if is_satisfied.get("answer") else 0.0
                else:
                    features['satisfied_signal'] = 1.0 if is_satisfied else 0.0
                
                # 提取特征 2: 当前计划的步数 (反映任务复杂度)
                plan = ledger.get("plan", [])
                features['plan_length'] = float(len(plan))
        except (json.JSONDecodeError, KeyError, AttributeError):
            # 解析失败就忽略，不影响主流程
            pass

    # 2. Tool特征（ASTRA-Gen 3.0 增强：使用数值状态）
    if source_node.type == "Tool":
        exitcode = None
        if "exitcode" in event:
            exitcode = event.get("exitcode")
        else:
            # 注意下划线匹配 Computer_terminal
            if "Computer_terminal" in source_node.id or "exitcode:" in content:
                exitcode_match = re.search(r'exitcode:\s*(\d+)', content, re.IGNORECASE)
                if exitcode_match:
                    exitcode = exitcode_match.group(1)

        # 🔥 优化：使用数值状态（1.0=Success, -1.0=Fail, 0.0=Unknown）
        if exitcode is not None:
            try:
                exitcode_int = int(str(exitcode))
                if exitcode_int == 0:
                    features['tool_status'] = 1.0  # Success
                else:
                    features['tool_status'] = -1.0  # Fail
            except (ValueError, TypeError):
                features['tool_status'] = 0.0  # Unknown
        else:
            features['tool_status'] = 0.0  # Unknown
        
        # 保留原有的 exitcode_status 字符串格式（向后兼容）
        status = "unknown"
        if exitcode is not None:
            try:
                status = "success" if int(str(exitcode)) == 0 else "failure"
            except:
                pass
        features['exitcode_status'] = status

    # 3. Agent特征
    if source_node.type == "Agent":
        features['is_terminate'] = "TERMINATE" in content.upper()
        features['plan_signal'] = bool(re.search(r'\b(Plan|Step)\b', content, re.IGNORECASE))

        count = 0
        for i in range(t + 1):
            if i < len(history):
                event_i = history[i]
                actor_name = event_i.get('name') or event_i.get('role', '')
                if actor_name == source_node.id:
                    count += 1
        features['active_ratio'] = count / (t + 1) if t >= 0 else 0.0

    # 4. Artifact特征
    if source_node.type == "Artifact":
        if source_node.artifact_type:
            features['artifact_type'] = source_node.artifact_type
        else:
            if "http" in source_node.id.lower():
                features['artifact_type'] = "url"
            else:
                features['artifact_type'] = "file"
        features['mention_count'] = mention_counter.get(source_node.id, 0)

    # 5. Environment特征
    if source_node.type == "Environment":
        if env_event_type:
            features['env_event_type'] = env_event_type

    return features


def MainParser(json_data: Dict[str, Any]) -> DynamicGraph:
    """
    主解析函数

    🔥 修改：在调用 ExtractNodeFeatures 时传入 system_prompt
    """
    question = json_data.get('question', '')
    ground_truth = {
        'mistake_agent': json_data.get('mistake_agent', ''),
        'mistake_step': json_data.get('mistake_step', ''),
        'mistake_reason': json_data.get('mistake_reason', ''),
        'ground_truth': json_data.get('ground_truth', '')
    }
    graph = DynamicGraph(question, ground_truth)
    node_registry = graph.nodes

    GetOrCreateNode("Broadcast", "Environment", -1, node_registry)
    GetOrCreateNode("Env", "Environment", -1, node_registry)

    history = json_data.get('history', [])
    system_prompt = json_data.get('system_prompt', {})

    mention_counter = defaultdict(int)

    for t in range(len(history)):
        event = history[t]
        actor_name = event.get('name') or event.get('role', '')

        actor_type = DetermineNodeType(actor_name, system_prompt, event)
        source_node = GetOrCreateNode(actor_name, actor_type, t, node_registry)

        content = event.get('content', '')
        interactions = ParseEdges(source_node, content, t, history, node_registry, event, system_prompt, mention_counter)

        current_env_event_type = None

        for target_name, target_type, edge_type, edge_features in interactions:
            if edge_type == "Affect" and target_name == "__AFFECT__":
                env_node = GetOrCreateNode("Env", "Environment", -1, node_registry)
                if "event_type" in edge_features:
                    current_env_event_type = edge_features["event_type"]
                edge = Edge(env_node.id, source_node.id, edge_type, t, edge_features)
                graph.add_edge(edge)
            else:
                artifact_type = None
                if target_type == "Artifact":
                    if "http" in target_name.lower():
                        artifact_type = "url"
                    else:
                        artifact_type = "file"
                target_node = GetOrCreateNode(target_name, target_type, t, node_registry, artifact_type=artifact_type)
                edge = Edge(source_node.id, target_node.id, edge_type, t, edge_features)
                graph.add_edge(edge)

        env_event_type_for_features = current_env_event_type if source_node.id == "Env" and current_env_event_type else None

        # 🔥 这里传入了 system_prompt
        node_features = ExtractNodeFeatures(
            source_node, event, t, history, mention_counter,
            system_prompt, # 新增参数
            env_event_type_for_features
        )
        source_node.features[t] = node_features

    return graph


def process_single_file(json_file: Path, verbose: bool = True, save_result: bool = False, output_dir: Optional[Path] = None, source_dir_name: Optional[str] = None) -> Optional[DynamicGraph]:
    """处理单个JSON文件"""
    try:
        if verbose:
            print(f"Processing: {json_file.name}...")

        # 检查文件是否存在
        if not json_file.exists():
            if verbose:
                print(f"  ✗ Error: File '{json_file}' does not exist.")
            return None

        # 检查是否为文件（而不是目录）
        if not json_file.is_file():
            if verbose:
                print(f"  ✗ Error: '{json_file}' is not a file.")
            return None

        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        graph = MainParser(json_data)

        if verbose:
            print(f"  ✓ Success: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        if save_result and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            if source_dir_name:
                safe_dir_name = source_dir_name.replace('/', '_').replace('\\', '_').replace('&', '_')
                output_file = output_dir / f"{safe_dir_name}_{json_file.stem}_graph.json"
            else:
                output_file = output_dir / f"{json_file.stem}_graph.json"

            graph_dict = {
                'question': graph.question,
                'ground_truth': graph.ground_truth,
                'nodes': {},
                'edges': []
            }

            for node_id, node in graph.nodes.items():
                graph_dict['nodes'][node_id] = {
                    'id': node.id,
                    'type': node.type,
                    'created_at': node.created_at,
                    'features': {str(t): features for t, features in node.features.items()}
                }

            for edge in graph.edges:
                graph_dict['edges'].append({
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.type,
                    'timestamp': edge.timestamp,
                    'features': edge.features
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_dict, f, ensure_ascii=False, indent=2)

            if verbose:
                print(f"  ✓ Saved to: {output_file}")

        return graph

    except FileNotFoundError:
        if verbose: print(f"  ✗ Error: File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        if verbose: print(f"  ✗ Error: Invalid JSON format in '{json_file}': {e}")
        return None
    except Exception as e:
        if verbose: print(f"  ✗ Error: {type(e).__name__}: {e}")
        return None


def process_directory(directory: Path, verbose: bool = True, save_result: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """批量处理目录中的所有JSON文件，支持跳过已解析的文件"""
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'files': []}

    # 🔥 检查已解析的文件 (用于跳过)
    processed_count = 0
    files_to_process = []
    source_dir_name = directory.name

    if save_result and output_dir:
        # 推断输出文件名格式（与 process_single_file 中的保存逻辑同步）
        safe_dir_name = source_dir_name.replace('/', '_').replace('\\', '_').replace('&', '_')
        
        for json_file in json_files:
            output_file_name = f"{safe_dir_name}_{json_file.stem}_graph.json"
            output_file_path = output_dir / output_file_name
            
            if output_file_path.exists():
                processed_count += 1
                continue
            
            files_to_process.append(json_file)
    else:
        # 如果没有输出目录或不需要保存，处理所有文件
        files_to_process = json_files

    print(f"\nFound {len(json_files)} JSON files in: {directory}")
    if processed_count > 0:
        print(f"⏭️  Skipping {processed_count} already parsed files.")
    if len(files_to_process) > 0:
        print(f"📝 Processing {len(files_to_process)} files...")
    print("=" * 60)

    results = {
        'total': len(json_files),
        'success': 0,
        'failed': 0,
        'skipped': processed_count,
        'files': []
    }

    # 处理需要解析的文件
    for i, json_file in enumerate(files_to_process, 1):
        if verbose:
            print(f"\n[{i}/{len(files_to_process)}] ", end="")

        graph = process_single_file(json_file, verbose=verbose, save_result=save_result, output_dir=output_dir, source_dir_name=source_dir_name)

        if graph is not None:
            results['success'] += 1
            results['files'].append({'file': str(json_file), 'status': 'success'})
        else:
            results['failed'] += 1
            results['files'].append({'file': str(json_file), 'status': 'failed'})

    print("\n" + "=" * 60)
    if results['skipped'] > 0:
        print(f"Processing Complete! Total: {results['total']}, Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
    else:
        print(f"Processing Complete! Total: {results['total']}, Success: {results['success']}, Failed: {results['failed']}")
    print("=" * 60)

    return results


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='DHCG Parser')
    parser.add_argument('input_path', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        alternative_path = project_root / args.input_path
        if alternative_path.exists():
            input_path = alternative_path
        else:
            print(f"Error: Path '{args.input_path}' does not exist.")
            sys.exit(1)

    if input_path.is_file():
        if not input_path.suffix == '.json':
            sys.exit(1)
        output_dir = Path(args.output) if args.save else None
        source_dir_name = input_path.parent.name if input_path.parent.name else None
        process_single_file(input_path, verbose=not args.quiet, save_result=args.save, output_dir=output_dir, source_dir_name=source_dir_name)

    elif input_path.is_dir():
        output_dir = Path(args.output) if args.save else None
        process_directory(input_path, verbose=not args.quiet, save_result=args.save, output_dir=output_dir)

    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
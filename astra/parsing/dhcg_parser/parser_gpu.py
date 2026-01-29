"""
动态异构因果图 (DHCG) 解析器 - GPU版本
用于解析 Who&When 数据集的 JSON 日志并构建动态异构图
支持单个文件和目录批量处理
优先使用GPU加速，如果GPU不可用则自动回退到CPU
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# 尝试导入torch并检查CUDA可用性
try:
    import torch
    import warnings
    # 抑制sm_120兼容性警告（如果GPU计算测试成功，警告可以忽略）
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    
    CUDA_AVAILABLE = torch.cuda.is_available()
    BLACKWELL_DETECTED = False
    SM120_UNSUPPORTED = False
    
    if CUDA_AVAILABLE:
        print(f"[GPU模式] 检测到CUDA可用，使用GPU加速")
        print(f"[GPU模式] GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"[GPU模式] CUDA版本: {torch.version.cuda}")
        print(f"[GPU模式] PyTorch版本: {torch.__version__}")
        
        # 检查GPU计算能力
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 12:
            BLACKWELL_DETECTED = True
            print(f"[GPU模式] 检测到Blackwell架构 (sm_{capability[0]}{capability[1]})")
            # 检查PyTorch版本是否支持sm_120
            pytorch_version = torch.__version__
            version_parts = pytorch_version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            # PyTorch 2.6+ 或 nightly 版本才支持 sm_120
            if major < 2 or (major == 2 and minor < 6):
                if 'dev' not in pytorch_version and 'nightly' not in pytorch_version.lower():
                    SM120_UNSUPPORTED = True
                    print(f"[警告] 当前PyTorch版本 ({pytorch_version}) 不支持sm_120架构")
                    print(f"[提示] 需要PyTorch 2.6+ 或 Nightly版本")
        
        # 测试GPU是否真的可用（即使有警告）
        try:
            test_tensor = torch.randn(2, 2).cuda()
            _ = test_tensor @ test_tensor
            print(f"[GPU模式] GPU计算测试: ✓ 成功")
            DEVICE = 'cuda'
        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'no kernel image' in error_msg or 'kernel image is available' in error_msg:
                if BLACKWELL_DETECTED:
                    print(f"[错误] GPU计算测试失败: PyTorch不支持sm_120架构")
                    print(f"[解决方案] 请运行以下命令安装支持sm_120的PyTorch版本:")
                    print(f"  python install_pytorch_sm120.py")
                    print(f"  或手动安装: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
                    SM120_UNSUPPORTED = True
                else:
                    print(f"[错误] GPU计算测试失败: {e}")
            else:
                print(f"[警告] GPU计算测试失败: {e}")
            print("[回退] 将使用CPU模式")
            CUDA_AVAILABLE = False
            DEVICE = 'cpu'
        except Exception as e:
            print(f"[警告] GPU计算测试失败: {e}")
            print("[回退] 将使用CPU模式")
            CUDA_AVAILABLE = False
            DEVICE = 'cpu'
    else:
        print("[警告] CUDA不可用，将使用CPU模式")
        print("[提示] 如需使用GPU，请检查:")
        print("  1. PyTorch是否安装了CUDA版本")
        print("  2. GPU驱动是否正确安装")
        print("  3. CUDA工具包是否与PyTorch版本匹配")
        DEVICE = 'cpu'
except ImportError:
    print("[错误] PyTorch未安装，将使用CPU模式")
    CUDA_AVAILABLE = False
    DEVICE = 'cpu'
    BLACKWELL_DETECTED = False
    SM120_UNSUPPORTED = False

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# 全局模型实例（延迟加载）
_model = None
_tokenizer = None


def get_embedding_model(use_8bit: bool = True, force_cpu: bool = False):
    """
    获取或初始化 Qwen-8B 嵌入模型
    
    Args:
        use_8bit: 是否使用8-bit量化（节省显存，默认True）
        force_cpu: 是否强制使用CPU模式（默认False）
    """
    global _model, _tokenizer
    if _model is None:
        try:
            # 🔥 使用 Qwen-8B 模型（可以是本地路径或 HuggingFace 模型名）
            model_name = os.getenv("QWEN_MODEL_PATH", "models/Qwen3-8B/qwen/Qwen3-8B")
            if not os.path.exists(model_name):
                # 如果没有本地模型，使用默认路径
                model_name = "models/Qwen3-8B/qwen/Qwen3-8B"
            
            # 确定设备
            if force_cpu:
                device = 'cpu'
                print(f"[强制CPU模式] 正在加载 Qwen-8B 嵌入模型: {model_name}...")
            elif torch.cuda.is_available() and not force_cpu:
                device = 'cuda'
                print(f"[GPU模式] 正在加载 Qwen-8B 嵌入模型: {model_name}...")
            else:
                device = 'cpu'
                print(f"[CPU模式] GPU不可用，使用CPU加载模型: {model_name}...")
            
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # 🔥 关键修复：强制CPU模式时，确保不使用device_map="auto"（会尝试GPU）
            if force_cpu or device == 'cpu':
                # CPU模式：直接加载到CPU，不使用device_map
                print(f"[{device.upper()}模式] 强制CPU模式，不使用GPU...")
                print(f"[提示] 正在加载模型权重（可能需要1-2分钟）...")
                _model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
                _model = _model.to('cpu')
                _model.eval()
                print(f"[{device.upper()}模式] ✓ 模型权重加载完成！")
            # 🔥 关键修复：使用8-bit量化或CPU模式节省显存
            elif device == 'cuda' and use_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    print(f"[GPU模式] 使用8-bit量化加载模型（节省显存）...")
                    _model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        quantization_config=quantization_config
                    )
                    print(f"[GPU模式] ✓ 8-bit量化模型加载成功")
                except ImportError:
                    print(f"[警告] BitsAndBytes未安装，尝试使用FP16模式...")
                    _model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    print(f"[GPU模式] ✓ FP16模型加载成功")
                except Exception as e:
                    print(f"[警告] 8-bit量化加载失败: {e}")
                    print(f"[回退] 尝试使用FP16模式...")
                    _model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
            else:
                # 其他情况（不应该执行到这里，但为了安全保留）
                # 🔥 关键修复：GPU模式但不使用8-bit量化时，使用FP16
                print(f"[GPU模式] 使用FP16加载模型...")
                _model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            
            # 🔥 注意：CPU模式的eval()在上面已经调用了，这里只处理GPU模式
            if not (force_cpu or device == 'cpu'):
                _model.eval()  # 设置为评估模式
            
            # 测试编码（验证模型可用性）
            print(f"[{device.upper()}模式] 正在测试模型（计算第一个embedding）...")
            if force_cpu or device == 'cpu':
                print(f"[提示] CPU模式下embedding计算可能较慢（10-30秒），请耐心等待...")
            with torch.no_grad():
                test_text = "test"
                if device == 'cuda':
                    inputs = _tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                else:
                    inputs = _tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
                
                outputs = _model(**inputs)
                # 使用 mean pooling: 对所有 token 的 hidden states 取平均
                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    # 有些模型返回的是tuple
                    embedding = outputs[0].mean(dim=1).cpu().numpy()
                
                print(f"[{device.upper()}模式] ✓ Qwen-8B 模型成功初始化，嵌入维度: {embedding.shape[1]}")
            
        except Exception as e:
            print(f"[错误] Qwen-8B 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果GPU失败，尝试CPU
            if device == 'cuda' and not force_cpu:
                print(f"[回退] 尝试使用 CPU 模式...")
                try:
                    device = 'cpu'
                    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    _model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    _model.eval()
                    print(f"[CPU模式] ✓ Qwen-8B 模型成功初始化（回退模式）")
                except Exception as e2:
                    print(f"[严重错误] CPU 初始化也失败: {e2}")
                    raise
            else:
                raise
    
    return _model, _tokenizer

def encode_text(text: str, use_8bit: bool = None, force_cpu: bool = None) -> np.ndarray:
    """
    使用 Qwen-8B 编码文本（Mean Pooling）
    
    Args:
        text: 要编码的文本
        use_8bit: 是否使用8-bit量化（节省显存）。如果为None，从环境变量读取或默认True
        force_cpu: 是否强制使用CPU模式。如果为None，从环境变量读取或默认False
    """
    # 🔥 关键修复：从环境变量读取配置
    if use_8bit is None:
        use_8bit = os.getenv("USE_8BIT", "true").lower() in ["true", "1", "yes"]
    if force_cpu is None:
        force_cpu = os.getenv("FORCE_CPU", "").lower() in ["true", "1", "yes"]
    
    model, tokenizer = get_embedding_model(use_8bit=use_8bit, force_cpu=force_cpu)
    
    # 🔥 关键修复：获取模型所在的设备（兼容8-bit量化模型）
    device = None
    try:
        # 方法1: 检查是否有hf_device_map（8-bit量化模型的设备映射）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # 8-bit量化模型，设备映射可能分布在多个设备上
            # 获取第一个参数的设备（通常是embedding层所在的设备）
            try:
                device = next(model.parameters()).device
            except:
                # 如果无法获取，尝试从embed_tokens获取
                if hasattr(model, 'embed_tokens'):
                    device = next(model.embed_tokens.parameters()).device
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
        # 方法2: 直接获取device属性
        elif hasattr(model, 'device'):
            device = model.device
        # 方法3: 从第一个参数获取设备
        else:
            try:
                device = next(model.parameters()).device
            except:
                # 如果无法获取，尝试从embed_tokens获取
                if hasattr(model, 'embed_tokens'):
                    device = next(model.embed_tokens.parameters()).device
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    except:
        # 默认使用CUDA（如果可用且不是强制CPU）
        device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    
    # 🔥 添加进度提示（CPU模式很慢）
    if force_cpu or (device is not None and device.type == 'cpu'):
        # CPU模式：静默处理，不打印（避免输出过多）
        pass
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # 🔥 关键修复：确保输入数据在正确的设备上
        # 对于8-bit量化模型，embed_tokens可能在CUDA上，必须将inputs移到相同设备
        if device is not None and device.type == 'cuda':
            # GPU模式：将输入移到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
        # CPU模式：inputs已经在CPU上，不需要移动
        
        # 🔥 CPU模式下计算embedding会很慢（可能需要几秒到几十秒）
        outputs = model(**inputs)
        
        # Mean pooling: 对所有 token 的 hidden states 取平均
        if hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state.mean(dim=1)
        else:
            # 有些模型返回的是tuple
            embedding = outputs[0].mean(dim=1)
        
        # 确保结果在CPU上
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
    
    return embedding[0] if embedding.ndim > 1 else embedding  # 返回 [dim] 形状的数组


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
    
    Args:
        name: 节点名称
        node_type: 节点类型
        t: 时间步
        node_registry: 节点注册表
        artifact_type: 仅对Artifact节点有效，值为"file"或"url"
    
    Returns:
        Node对象
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
    
    Args:
        actor: 事件发起者名称
        system_prompt: 系统提示字典
        event: 事件字典
    
    Returns:
        节点类型字符串
    """
    # 如果actor是Computer_terminal，返回"Tool"
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
    if re.search(r'(Expert|Assistant|Orchestrator|Surfer|Planner)', actor):
        return "Agent"
    
    # 默认返回"Agent"
    return "Agent"


def FindCallerAgent(tool_node: Node, t: int, history: List[Dict[str, Any]], 
                   system_prompt: Dict[str, Any]) -> str:
    """
    查找调用工具的Agent（增强版：直接使用history而非edges）
    
    Args:
        tool_node: 工具节点
        t: 当前时间步
        history: 历史事件列表
        system_prompt: 系统提示字典（用于DetermineNodeType）
    
    Returns:
        调用者Agent的名称，如果找不到则返回"Broadcast"
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
    解析边，返回交互元组列表 (target_name, target_type, edge_type, edge_features)
    
    Args:
        source_node: 源节点
        content: 事件内容
        t: 当前时间步
        history: 历史事件列表
        node_registry: 节点注册表
        event: 当前事件字典
        system_prompt: 系统提示字典（用于FindCallerAgent）
        mention_counter: 引用计数器（用于统计Artifact被引用次数）
    
    Returns:
        交互元组列表
    """
    interactions = []
    
    # 1. Invoke: 如果source_node.type == "Agent" 并且content包含代码块
    if source_node.type == "Agent":
        # 检查是否包含代码块 (```python...``` 或 ```sh...```)
        code_block_pattern = r'```(?:python|sh|bash|javascript|js|java|cpp|c\+\+|c|go|rust|sql|html|css|xml|json|yaml|yml|markdown|md|text|plaintext)[\s\S]*?```'
        if re.search(code_block_pattern, content, re.IGNORECASE):
            # Invoke边的意图固定为"Command"
            interactions.append(("Computer_terminal", "Tool", "Invoke", {"intent": "Command"}))
    
    # 2. Return: 如果source_node.type == "Tool" 并且event中有exitcode
    if source_node.type == "Tool":
        exitcode = None
        # 首先检查event中是否有exitcode字段
        if "exitcode" in event:
            exitcode = event.get("exitcode")
        else:
            # 从content中提取exitcode
            exitcode_match = re.search(r'exitcode:\s*(\d+)', content, re.IGNORECASE)
            if exitcode_match:
                exitcode = exitcode_match.group(1)
        
        if exitcode is not None:
            # 处理exitcode值
            if isinstance(exitcode, str):
                exitcode_match = re.search(r'(\d+)', exitcode)
                if exitcode_match:
                    exitcode = int(exitcode_match.group(1))
                else:
                    exitcode = 0
            elif not isinstance(exitcode, int):
                exitcode = 0
            
            status = "success" if exitcode == 0 else "failure"
            # Return边的意图固定为"Inform"
            caller_agent = FindCallerAgent(source_node, t, history, system_prompt)
            interactions.append((caller_agent, "Agent", "Return", {"status": status, "intent": "Inform"}))
    
    # 3. Reference: 提取文件路径和URL（去重和规范化）
    # 正则表达式: (\.\./[\w/.-]+|https?://[\w/.-]+|filename:\s*[\w.-]+)
    reference_pattern = r'(\.\./[\w/.-]+|https?://[\w/.-]+|filename:\s*[\w.-]+)'
    references = re.findall(reference_pattern, content)
    # 去重：将列表转换为set再转回list
    references = list(set(references))
    
    for ref in references:
        # 规范化：清理引用字符串，去除末尾标点符号
        ref_clean = ref.strip().rstrip('.,;:')
        if ref_clean.startswith("filename:"):
            ref_clean = ref_clean.replace("filename:", "").strip().rstrip('.,;:')
        
        # 更新mention_counter
        mention_counter[ref_clean] += 1
        
        # Reference边的意图固定为"Inform"
        interactions.append((ref_clean, "Artifact", "Reference", {"intent": "Inform"}))
    
    # 4. Communicate（需要判断意图）
    # 优先使用正则表达式 @\w+ 查找直接@提及
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, content)
    
    # 判断Communicate边的意图
    intent = "Inform"  # 默认意图
    if mentions:
        # 如果content包含问号或@开头，意图为"Query"
        if '?' in content or content.strip().startswith('@'):
            intent = "Query"
        for mention in mentions:
            interactions.append((mention, "Agent", "Communicate", {"intent": intent}))
    else:
        # 如果没有提及，且content是自然语言，且t > 0，且history[t-1]的actor不是自己
        if t > 0 and t - 1 < len(history):
            prev_event = history[t - 1]
            prev_actor = prev_event.get('name') or prev_event.get('role', '')
            if prev_actor != source_node.id:
                # 检查content是否是自然语言（不包含代码块）
                if not re.search(r'```[\s\S]*?```', content):
                    # 判断是否是回复：如果包含问号，意图为"Query"，否则为"Inform"
                    if '?' in content:
                        intent = "Query"
                    else:
                        intent = "Inform"
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
    
    # 5. Affect: 查找环境错误关键词
    # 注意：Affect边的方向是从Env指向source_node
    # 这里我们返回一个特殊标记，主函数会处理
    error_pattern = r'(Timeout|OOM|Permission Denied|Network Error)'
    error_match = re.search(error_pattern, content, re.IGNORECASE)
    if error_match:
        # 提取匹配到的错误类型（转为小写）
        event_type = error_match.group(1).lower()
        # Affect边的意图固定为"Reject"
        interactions.append(("__AFFECT__", "Environment", "Affect", {"intent": "Reject", "event_type": event_type}))
    
    return interactions


def ExtractNodeFeatures(source_node: Node, event: Dict[str, Any], t: int, history: List[Dict[str, Any]],
                        mention_counter: Dict[str, int], env_event_type: Optional[str] = None) -> Dict[str, Any]:
    """
    提取节点特征
    
    Args:
        source_node: 源节点
        event: 当前事件字典
        t: 当前时间步
        history: 历史事件列表
        mention_counter: 引用计数器（用于Artifact节点的mention_count特征）
        env_event_type: 环境事件类型（用于Environment节点的env_event_type特征）
    
    Returns:
        特征字典
    """
    features = {}
    content = event.get('content', '')
    
    # 1. content_embedding: 使用 Qwen-8B 编码（优先使用GPU）
    # 使用 encode_text 函数进行编码
    embedding = encode_text(content).tolist()
    features['content_embedding'] = embedding
    # 🔥 存储原始文本，供 Ollama embedding 使用
    features['content_text'] = content
    
    # 2. Tool特征
    if source_node.type == "Tool":
        exitcode = None
        # 首先检查event中是否有exitcode字段
        if "exitcode" in event:
            exitcode = event.get("exitcode")
        else:
            # 从content中提取exitcode
            exitcode_match = re.search(r'exitcode:\s*(\d+)', content, re.IGNORECASE)
            if exitcode_match:
                exitcode = exitcode_match.group(1)
        
        if exitcode is not None:
            # 处理exitcode值
            if isinstance(exitcode, str):
                exitcode_match = re.search(r'(\d+)', exitcode)
                if exitcode_match:
                    exitcode = int(exitcode_match.group(1))
                else:
                    exitcode = 0
            elif not isinstance(exitcode, int):
                exitcode = 0
            features['exitcode_status'] = "success" if exitcode == 0 else "failure"
        else:
            features['exitcode_status'] = "unknown"
    
    # 3. Agent特征
    if source_node.type == "Agent":
        # is_terminate: content中是否包含"TERMINATE"
        features['is_terminate'] = "TERMINATE" in content.upper()
        
        # plan_signal: content中是否包含"Plan"或"Step"
        features['plan_signal'] = bool(re.search(r'\b(Plan|Step)\b', content, re.IGNORECASE))
        
        # active_ratio: 计算source_node.id在history的0到t步中出现的频率
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
        # artifact_type: 从节点的静态属性中读取
        if source_node.artifact_type:
            features['artifact_type'] = source_node.artifact_type
        else:
            # 如果节点没有artifact_type属性，根据ID判断
            if "http" in source_node.id.lower():
                features['artifact_type'] = "url"
            else:
                features['artifact_type'] = "file"
        
        # mention_count: 从mention_counter中读取当前计数
        features['mention_count'] = mention_counter.get(source_node.id, 0)
    
    # 5. Environment特征
    if source_node.type == "Environment":
        # env_event_type: 如果提供了env_event_type参数，添加到特征中
        if env_event_type:
            features['env_event_type'] = env_event_type
    
    return features


def MainParser(json_data: Dict[str, Any]) -> DynamicGraph:
    """
    主解析函数
    
    Args:
        json_data: JSON数据字典
    
    Returns:
        DynamicGraph对象
    """
    # 1. 初始化
    question = json_data.get('question', '')
    ground_truth = {
        'mistake_agent': json_data.get('mistake_agent', ''),
        'mistake_step': json_data.get('mistake_step', ''),
        'mistake_reason': json_data.get('mistake_reason', ''),
        'ground_truth': json_data.get('ground_truth', '')
    }
    graph = DynamicGraph(question, ground_truth)
    node_registry = graph.nodes
    
    # 预创建全局节点
    GetOrCreateNode("Broadcast", "Environment", -1, node_registry)
    GetOrCreateNode("Env", "Environment", -1, node_registry)
    
    history = json_data.get('history', [])
    system_prompt = json_data.get('system_prompt', {})
    
    # 初始化mention_counter（用于统计Artifact被引用次数）
    mention_counter = defaultdict(int)
    
    # 2. 遍历历史事件
    for t in range(len(history)):
        event = history[t]
        actor_name = event.get('name') or event.get('role', '')
        
        # 3. 识别并获取源节点
        actor_type = DetermineNodeType(actor_name, system_prompt, event)
        source_node = GetOrCreateNode(actor_name, actor_type, t, node_registry)
        
        # 4. 解析内容以创建边和目标节点
        content = event.get('content', '')
        interactions = ParseEdges(source_node, content, t, history, node_registry, event, system_prompt, mention_counter)
        
        # 用于存储当前时间步的env_event_type（如果有Affect边）
        current_env_event_type = None
        
        for target_name, target_type, edge_type, edge_features in interactions:
            # 特殊处理Affect边：方向是从Env指向source_node
            if edge_type == "Affect" and target_name == "__AFFECT__":
                # Affect边是从Env指向source_node
                env_node = GetOrCreateNode("Env", "Environment", -1, node_registry)
                # 从edge_features中提取event_type（如果存在）
                if "event_type" in edge_features:
                    current_env_event_type = edge_features["event_type"]
                edge = Edge(env_node.id, source_node.id, edge_type, t, edge_features)
                graph.add_edge(edge)
            else:
                # 对于Artifact节点，需要传递artifact_type参数
                artifact_type = None
                if target_type == "Artifact":
                    # 根据target_name判断artifact_type
                    if "http" in target_name.lower():
                        artifact_type = "url"
                    else:
                        artifact_type = "file"
                target_node = GetOrCreateNode(target_name, target_type, t, node_registry, artifact_type=artifact_type)
                edge = Edge(source_node.id, target_node.id, edge_type, t, edge_features)
                graph.add_edge(edge)
        
        # 5. 提取并存储当前节点的动态特征
        # 如果是Env节点且有Affect边，传递env_event_type
        env_event_type_for_features = current_env_event_type if source_node.id == "Env" and current_env_event_type else None
        node_features = ExtractNodeFeatures(source_node, event, t, history, mention_counter, env_event_type_for_features)
        source_node.features[t] = node_features
    
    return graph


def process_single_file(json_file: Path, verbose: bool = True, save_result: bool = False, output_dir: Optional[Path] = None, source_dir_name: Optional[str] = None) -> Optional[DynamicGraph]:
    """
    处理单个JSON文件
    
    Args:
        json_file: JSON文件路径
        verbose: 是否显示详细信息
        save_result: 是否保存结果
        output_dir: 输出目录
        source_dir_name: 源目录名称（用于区分不同目录的同名文件）
    
    Returns:
        DynamicGraph对象，如果失败则返回None
    """
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
        
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 调用MainParser构建图
        graph = MainParser(json_data)
        
        if verbose:
            print(f"  ✓ Success: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # 保存结果（如果需要）
        if save_result and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            # 如果提供了源目录名，在文件名中包含它以避免不同目录的同名文件冲突
            if source_dir_name:
                # 清理目录名，移除特殊字符，使其适合作为文件名的一部分
                safe_dir_name = source_dir_name.replace('/', '_').replace('\\', '_').replace('&', '_')
                output_file = output_dir / f"{safe_dir_name}_{json_file.stem}_graph.json"
            else:
                output_file = output_dir / f"{json_file.stem}_graph.json"
            
            # 将图转换为可序列化的字典
            graph_dict = {
                'question': graph.question,
                'ground_truth': graph.ground_truth,
                'nodes': {},
                'edges': []
            }
            
            # 序列化节点
            for node_id, node in graph.nodes.items():
                graph_dict['nodes'][node_id] = {
                    'id': node.id,
                    'type': node.type,
                    'created_at': node.created_at,
                    'features': {str(t): features for t, features in node.features.items()}
                }
            
            # 序列化边
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
        if verbose:
            print(f"  ✗ Error: File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  ✗ Error: Invalid JSON format in '{json_file}': {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
        return None


def process_directory(directory: Path, verbose: bool = True, save_result: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    批量处理目录中的所有JSON文件
    
    Args:
        directory: 目录路径
        verbose: 是否显示详细信息
        save_result: 是否保存结果
        output_dir: 输出目录
    
    Returns:
        处理结果统计字典
    """
    # 查找所有JSON文件
    json_files = sorted(directory.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory: {directory}")
        return {'total': 0, 'success': 0, 'failed': 0, 'files': []}
    
    print(f"\nFound {len(json_files)} JSON files in: {directory}")
    print("=" * 60)
    
    results = {
        'total': len(json_files),
        'success': 0,
        'failed': 0,
        'files': []
    }
    
    # 获取目录名称（用于区分不同目录的同名文件）
    source_dir_name = directory.name
    
    for i, json_file in enumerate(json_files, 1):
        if verbose:
            print(f"\n[{i}/{len(json_files)}] ", end="")
        
        graph = process_single_file(json_file, verbose=verbose, save_result=save_result, output_dir=output_dir, source_dir_name=source_dir_name)
        
        if graph is not None:
            results['success'] += 1
            results['files'].append({
                'file': str(json_file),
                'status': 'success',
                'nodes': len(graph.nodes),
                'edges': len(graph.edges)
            })
        else:
            results['failed'] += 1
            results['files'].append({
                'file': str(json_file),
                'status': 'failed'
            })
    
    # 打印总结
    print("\n" + "=" * 60)
    print(f"Processing Complete!")
    print(f"  Total files: {results['total']}")
    print(f"  Successful: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print("=" * 60)
    
    return results


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description='解析Who&When JSON日志并构建动态异构图（GPU加速版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个文件
  python parser_gpu.py "Who&When/Algorithm-Generated/1.json"
  
  # 批量处理整个目录
  python parser_gpu.py "Who&When/Algorithm-Generated"
  python parser_gpu.py "Who&When/Hand-Crafted"
  
  # 批量处理并保存结果
  python parser_gpu.py "Who&When/Algorithm-Generated" --save --output results/
        """
    )
    parser.add_argument('input_path', type=str, help='输入的JSON文件路径或目录路径')
    parser.add_argument('--save', action='store_true', help='保存解析结果到JSON文件')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录（默认: outputs）')
    parser.add_argument('--quiet', action='store_true', help='静默模式，只显示总结信息')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # 如果路径不存在，尝试从项目根目录查找
    if not input_path.exists():
        # 获取脚本所在目录（dhcg_parser）
        script_dir = Path(__file__).parent
        # 尝试从项目根目录查找（上一级目录）
        project_root = script_dir.parent
        alternative_path = project_root / args.input_path
        
        if alternative_path.exists():
            input_path = alternative_path
            if not args.quiet:
                print(f"Note: Using path from project root: {input_path}")
        else:
            print(f"Error: Path '{args.input_path}' does not exist.")
            print(f"  Tried: {Path(args.input_path).absolute()}")
            print(f"  Also tried: {alternative_path.absolute()}")
            sys.exit(1)
    
    # 判断是文件还是目录
    if input_path.is_file():
        # 处理单个文件
        if not input_path.suffix == '.json':
            print(f"Error: '{input_path}' is not a JSON file.")
            sys.exit(1)
        
        output_dir = Path(args.output) if args.save else None
        # 对于单个文件，使用父目录名作为标识（如果父目录名有意义）
        source_dir_name = input_path.parent.name if input_path.parent.name else None
        graph = process_single_file(input_path, verbose=not args.quiet, save_result=args.save, output_dir=output_dir, source_dir_name=source_dir_name)
        
        if graph is None:
            sys.exit(1)
        
        # 显示详细信息（如果不是静默模式）
        if not args.quiet:
            print("\n--- Graph Summary ---")
            print(graph)
            print(f"\n--- Question ---")
            print(f"{graph.question}")
            
            # 打印统计信息
            print("\n--- Statistics ---")
            node_types = {}
            for node in graph.nodes.values():
                node_types[node.type] = node_types.get(node.type, 0) + 1
            print(f"Node types: {node_types}")
            
            edge_types = {}
            for edge in graph.edges:
                edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
            print(f"Edge types: {edge_types}")
    
    elif input_path.is_dir():
        # 批量处理目录
        output_dir = Path(args.output) if args.save else None
        results = process_directory(input_path, verbose=not args.quiet, save_result=args.save, output_dir=output_dir)
        
        if results['failed'] > 0:
            sys.exit(1)
    
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
ASTRA-MoE 训练脚本

实现完整的训练流程，包括：
1. WhoWhenDataset - 数据集类（处理变长序列和Agent）
2. 自定义 collate_fn - 批处理对齐
3. 训练主循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import sys
from datetime import datetime

from astra.data.adapter import GraphDataConverter, reconstruct_graph_from_json
from astra.model.gnn import ASTRAMoE
from astra.model.loss import ASTRALoss, SupConLoss, ASTRAContrastiveLoss
from astra.data.graph_data import HeteroGraph
import random

# --- 强制修复 GPU 环境变量 ---
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    print("⚠️ 检测到 CUDA_VISIBLE_DEVICES 为空，正在强制清除以恢复 GPU...")
    del os.environ["CUDA_VISIBLE_DEVICES"]
print("\n" + "="*60)
print("🔍 深度环境诊断 (Deep Diagnostic)")
print("="*60)
print(f"Python 路径: {sys.executable}")
print(f"当前工作目录: {os.getcwd()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置 (Not Set)')}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否编译: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前设备索引: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("❌ torch.cuda.is_available() 返回 False")
print("="*60 + "\n")
# ========================


class TrainingLogger:
    """训练日志管理器：将详细日志保存到文件，终端只显示关键信息"""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.txt"
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        
    def log(self, message: str, to_terminal: bool = False):
        """记录日志到文件，可选是否同时输出到终端"""
        # 始终写入文件
        self.file_handle.write(message + '\n')
        self.file_handle.flush()
        
        # 根据标志决定是否输出到终端
        if to_terminal:
            print(message, flush=True)
    
    def log_epoch_metrics(self, epoch: int, total_epochs: int, 
                          train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float],
                          lr: float = None):
        """记录 epoch 评估指标（终端显示简洁版，文件保存详细版）"""
        # 终端输出：简洁格式，只显示关键指标
        terminal_msg = f"\n{'='*80}"
        terminal_msg += f"\nEpoch {epoch+1}/{total_epochs}"
        if lr is not None:
            terminal_msg += f" | LR: {lr:.2e}"
        terminal_msg += f"\n{'='*80}"
        terminal_msg += f"\n训练集 - Loss: {train_metrics['loss']:.6f}"
        terminal_msg += f"\n         Agent Acc: {train_metrics['agent_accuracy']:.4f} ({train_metrics['agent_accuracy']*100:.2f}%)"
        terminal_msg += f"\n         Step Acc:  {train_metrics['step_accuracy']:.4f} ({train_metrics['step_accuracy']*100:.2f}%)"
        if 'agent_loss' in train_metrics:
            terminal_msg += f"\n         Agent Loss: {train_metrics['agent_loss']:.6f}"
        if 'step_loss' in train_metrics:
            terminal_msg += f"\n         Step Loss:  {train_metrics['step_loss']:.6f}"
        if 'cl_loss' in train_metrics:
            terminal_msg += f"\n         CL Loss:    {train_metrics['cl_loss']:.6f}"
        if 'rl_loss' in train_metrics:
            terminal_msg += f"\n         RL Loss:    {train_metrics['rl_loss']:.6f}"
        terminal_msg += f"\n验证集 - Loss: {val_metrics['loss']:.6f}"
        terminal_msg += f"\n         Agent Acc: {val_metrics['agent_accuracy']:.4f} ({val_metrics['agent_accuracy']*100:.2f}%)"
        terminal_msg += f"\n         Step Acc:  {val_metrics['step_accuracy']:.4f} ({val_metrics['step_accuracy']*100:.2f}%)"
        terminal_msg += f"\n{'='*80}\n"
        
        # 文件输出：详细格式（包含所有指标）
        file_msg = f"\n{'='*80}"
        file_msg += f"\nEpoch {epoch+1}/{total_epochs}"
        if lr is not None:
            file_msg += f" | LR: {lr:.2e}"
        file_msg += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        file_msg += f"\n{'='*80}"
        file_msg += f"\n训练集指标:"
        for key, value in sorted(train_metrics.items()):
            file_msg += f"\n  {key}: {value:.6f}"
        file_msg += f"\n验证集指标:"
        for key, value in sorted(val_metrics.items()):
            file_msg += f"\n  {key}: {value:.6f}"
        file_msg += f"\n{'='*80}\n"
        
        # 输出到终端（简洁版）
        print(terminal_msg, flush=True)
        # 写入文件（详细版）
        self.file_handle.write(file_msg)
        self.file_handle.flush()
    
    def close(self):
        """关闭日志文件"""
        if self.file_handle:
            self.file_handle.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()


def seed_everything(seed: int = 42):
    """
    固定所有随机种子，确保实验可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class WhoWhenDataset(Dataset):
    """
    多智能体故障归因数据集 (优化版)
    1. 修复 JSON Extra data 错误
    2. 采用分片缓存策略 (避免生成 40GB 的巨型 .pt 文件)
    """

    def __init__(self,
                 data_dir: str = "outputs",
                 max_seq_len: int = 160,
                 max_agents: int = 50,
                 processed_dir: Optional[str] = None,
                 force_reprocess: bool = False,
                 enable_pairing: bool = True):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.max_agents = max_agents
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.force_reprocess = force_reprocess
        self.enable_pairing = enable_pairing

        # 查找所有 JSON 文件
        graph_files = list(self.data_dir.rglob("*_graph.json"))
        new_format_files = list(self.data_dir.rglob("*.json"))
        all_files = set(graph_files + new_format_files)
        self.json_files = sorted(list(all_files))

        if not self.json_files:
            raise ValueError(f"在 {data_dir} 及其子目录中未找到 JSON 文件")

        print(f"找到 {len(self.json_files)} 个数据文件")

        # 数据转换器
        self.converter = GraphDataConverter(node_feat_dim=8192, edge_feat_dim=32)  # 🔥 Qwen3-8B: 4096 (嵌入) + 4096 (元数据)

        # 内存中的数据列表 (仅存储已加载的数据索引，为了节省内存)
        # 实际的大 Tensor 建议在 __getitem__ 时加载，或者如果内存够大(64G+)也可以存内存
        # 这里为了稳妥，我们将数据缓存在内存中 (self.data_cache)，但通过分片写入磁盘防止 crash
        self.data_cache = [None] * len(self.json_files)
        
        # 初始化处理
        self._init_processing()
        
        # 配对逻辑
        if self.enable_pairing:
            self._pair_files()

    def _init_processing(self):
        """初始化处理流程：加载转换器，并进行分片处理"""
        
        # 1. 尝试加载或拟合 Converter
        global_converter_path = Path("processed_data/converter_state.pt")
        
        if global_converter_path.exists():
            print(f"\n🔥 [Dataset] 发现全局 Converter: {global_converter_path}")
            self.converter = torch.load(global_converter_path, weights_only=False)
            print("   ✅ 全局 Converter 加载成功")
        else:
            print("\n⚠️ [Dataset] 未找到全局 Converter，正在现场拟合...")
            all_graphs = []
            for json_file in tqdm(self.json_files, desc="拟合 Converter"):
                graph = self._safe_load_json(json_file) # 使用安全加载
                if graph: all_graphs.append(graph)
            self.converter.fit(all_graphs)
            # 保存拟合好的 converter
            if self.processed_dir:
                self.processed_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.converter, self.processed_dir / "converter_state.pt")

        # 2. 分片处理数据 (不再生成巨型 .pt 文件)
        print(f"\n🚀 开始数据加载与缓存 (分片模式)...")
        # 创建缓存子目录
        cache_dir = self.processed_dir / "cache" if self.processed_dir else Path("processed_data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
            
        success_count = 0
        
        # 使用 tqdm 显示进度
        import sys
        for idx, json_file in enumerate(tqdm(self.json_files, desc="Processing")):
            # 🔥 添加详细日志（每100个文件打印一次，37%附近详细打印）
            if idx > 0 and idx % 100 == 0:
                print(f"\n[进度] 已处理 {idx}/{len(self.json_files)} 个文件 ({idx*100//len(self.json_files)}%)", flush=True)
                print(f"  成功: {success_count}, 失败: {idx - success_count}", flush=True)
            # 37%附近（3500-3650）特别关注
            if 3500 <= idx <= 3650 and idx % 10 == 0:
                print(f"\n[37%区域] 处理文件 {idx}/{len(self.json_files)}: {json_file.name}", flush=True)
                sys.stdout.flush()
            
            # 计算该文件的缓存路径: processed_data/cache/{filename}.pt
            cache_name = f"{json_file.stem}.pt"
            cache_path = cache_dir / cache_name
            
            # 🔥 关键调试：3600附近强制打印（在cache_path定义后）
            if 3595 <= idx <= 3605:
                print(f"\n[关键调试] idx={idx}, file={json_file.name}, cache_path={cache_path}, exists={cache_path.exists()}", flush=True)
                sys.stdout.flush()
            
            # A. 尝试从分片缓存加载
            if cache_path.exists() and not self.force_reprocess:
                try:
                    # 🔥 关键修复：不加载到内存，只存储缓存路径（节省内存）
                    # 数据将在 __getitem__ 时按需加载
                    self.data_cache[idx] = cache_path  # 存储路径而不是数据
                    success_count += 1
                    continue
                except Exception as e:
                    # 🔥 修复：记录缓存加载失败的原因（但不中断）
                    if idx < 10 or (3500 <= idx <= 3650):
                        print(f"  ⚠️  缓存加载失败 {json_file.name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                    pass # 加载失败则重新处理

            # B. 重新处理
            try:
                # 1. 加载 JSON
                graph = self._safe_load_json(json_file)
                if not graph: 
                    if idx < 5:  # 只打印前5个失败的文件
                        print(f"⚠️  跳过文件（加载失败）: {json_file.name}")
                    continue

                # 2. 转换（在转换前再次验证 _fitted 状态）
                if not hasattr(self.converter, '_fitted') or not self.converter._fitted:
                    raise RuntimeError(f"Converter 未拟合！请在加载后调用 fit() 方法。_fitted={getattr(self.converter, '_fitted', 'N/A')}")
                graph_list, labels = self.converter.convert(graph)
                
                # 检查转换结果是否有效
                if not graph_list or len(graph_list) == 0:
                    if idx < 5:
                        print(f"⚠️  跳过文件（转换后为空）: {json_file.name}")
                    continue
                
                sample_data = {
                    'graph_list': graph_list,
                    'labels': labels,
                    'source_file': str(json_file)
                }
                
                # 3. 立即写入分片缓存 (防止程序崩溃导致数据丢失)
                torch.save(sample_data, cache_path)
                
                # 🔥 关键修复：不存入内存，只存储缓存路径（节省内存）
                # 数据将在 __getitem__ 时按需加载
                self.data_cache[idx] = cache_path  # 存储路径而不是数据
                success_count += 1
                    
            except Exception as e:
                # 🔥 修复：打印错误信息以便调试
                error_msg = str(e)
                error_type = type(e).__name__
                
                # 🔥 关键修复：37%附近（3600左右）的文件错误要详细打印
                should_print_detail = idx < 10 or (3500 <= idx <= 3650)
                
                if should_print_detail:
                    print(f"\n❌ [文件 {idx}/{len(self.json_files)}] 处理文件失败: {json_file.name}")
                    print(f"   错误类型: {error_type}")
                    print(f"   错误信息: {error_msg[:500]}")
                    # 如果是关键错误，打印完整堆栈
                    if idx < 3 or (3500 <= idx <= 3650):
                        import traceback
                        print(f"   堆栈跟踪:")
                        traceback.print_exc()
                elif idx == 10:
                    print(f"\n   ... (后续错误将不再显示详细信息，但37%附近会详细显示)")
                elif idx == 3651:
                    print(f"\n   ... (37%区域检查完成)")
                
                # 继续处理下一个文件
                continue

        # 清理 None 值 (处理失败的样本)
        # 🔥 修复：检查缓存路径是否存在（如果存储的是路径）
        from pathlib import Path as PathType
        self.valid_indices = []
        for i, d in enumerate(self.data_cache):
            if d is not None:
                # 如果是Path对象，检查文件是否存在
                if isinstance(d, (Path, PathType)):
                    if d.exists():
                        self.valid_indices.append(i)
                    else:
                        # 缓存文件不存在，标记为无效
                        self.data_cache[i] = None
                else:
                    # 是数据对象，直接添加
                    self.valid_indices.append(i)
        
        # 🔥 添加详细统计
        failed_count = len(self.json_files) - len(self.valid_indices)
        if failed_count > 0:
            print(f"\n⚠️  警告: {failed_count} 个文件处理失败")
            if failed_count == len(self.json_files):
                print(f"❌ 严重错误: 所有文件都处理失败！")
                print(f"   可能的原因:")
                print(f"   1. 数据格式问题")
                print(f"   2. 内存不足")
                print(f"   3. Converter 未正确拟合")
                print(f"   请检查前几个文件的错误信息")
        
        # 🔥 添加详细统计
        print(f"\n{'='*60}")
        print(f"数据加载完成统计")
        print(f"{'='*60}")
        print(f"总文件数: {len(self.json_files)}")
        print(f"成功处理: {len(self.valid_indices)}")
        failed_count = len(self.json_files) - len(self.valid_indices)
        print(f"处理失败: {failed_count}")
        
        if failed_count > 0:
            print(f"\n⚠️  警告: {failed_count} 个文件处理失败")
            if failed_count == len(self.json_files):
                print(f"❌ 严重错误: 所有文件都处理失败！")
                print(f"   可能的原因:")
                print(f"   1. 数据格式问题")
                print(f"   2. 内存不足")
                print(f"   3. Converter 未正确拟合")
                print(f"   请检查前几个文件的错误信息")
        else:
            print(f"✅ 所有文件处理成功！")
        
        print(f"{'='*60}\n")
        
        # 🔥 关键修复：确认数据加载完成
        if len(self.valid_indices) == 0:
            raise RuntimeError("❌ 严重错误：没有有效的训练样本！所有文件都处理失败。")
        
        print(f"✅ 数据加载完成！有效样本数: {len(self.valid_indices)}", flush=True)
        print(f"   已处理文件数: {success_count}/{len(self.json_files)}", flush=True)
        print(f"   将使用这些样本进行训练...", flush=True)
        import sys
        sys.stdout.flush()
        
        # 🔥 关键修复：检查是否所有文件都处理了
        if success_count < len(self.json_files) * 0.9:
            print(f"⚠️  警告：只处理了 {success_count}/{len(self.json_files)} 个文件（{success_count*100//len(self.json_files)}%）", flush=True)
            print(f"   如果程序在数据加载阶段停止，可能是内存不足或文件处理失败", flush=True)
        else:
            print(f"✅ 已处理 {success_count}/{len(self.json_files)} 个文件（{success_count*100//len(self.json_files)}%）", flush=True)

    def _safe_load_json(self, json_path: Path):
        """🔥 修复 'Extra data' 错误的健壮 JSON 加载器"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                # 尝试标准加载
                data = json.loads(content)
            except json.JSONDecodeError:
                # 🔥 修复：如果包含额外数据，只读取第一个 JSON 对象
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(content)
            
            return reconstruct_graph_from_json(data)
        except Exception as e:
            # print(f"加载失败 {json_path.name}: {e}")
            return None

    def _pair_files(self):
        """[最终修复版] 自动配对 Mutated 和 Healed 文件"""
        pairs = []
        # 建立 filename -> cache_index 的映射
        # 注意：这里只映射有效的、已加载的数据
        name_to_idx = {}
        for list_idx in self.valid_indices:
            # 🔥 关键修复：处理Path对象（缓存文件路径）
            cache_item = self.data_cache[list_idx]
            
            # 如果是Path对象，从缓存文件加载数据
            if isinstance(cache_item, Path):
                try:
                    data = torch.load(cache_item, weights_only=False)
                    source_file = data.get('source_file', str(cache_item))
                except Exception as e:
                    print(f"⚠️  配对时加载缓存失败 {cache_item}: {e}", flush=True)
                    continue
            else:
                # 如果已经是数据字典，直接使用
                source_file = cache_item.get('source_file', '')
            
            file_path = Path(source_file)
            name_to_idx[file_path.name] = list_idx

        # 找出所有的 fatal 文件
        fatal_files = [f for f in self.json_files if "_fatal_" in f.name]
        
        count_paired = 0
        count_unpaired = 0
        
        for mut_path in fatal_files:
            mut_name = mut_path.name
            
            # 如果这个 fatal 文件本身没加载成功，跳过
            if mut_name not in name_to_idx:
                continue
                
            mut_idx = name_to_idx[mut_name]
            pos_idx = None
            
            # 策略：直接字符串替换查找
            healed_name = mut_name.replace("_fatal_", "_healed_")
            golden_name = mut_name.replace("_fatal_", "_golden_")
            
            if healed_name in name_to_idx:
                pos_idx = name_to_idx[healed_name]
            elif golden_name in name_to_idx:
                pos_idx = name_to_idx[golden_name]
            
            if pos_idx is not None:
                pairs.append((mut_idx, pos_idx))
                count_paired += 1
            else:
                pairs.append((mut_idx, None))
                count_unpaired += 1

        print("="*60)
        print(f"🔥 ASTRA-Gen 数据配对统计:")
        print(f"   ✅ 成功配对: {count_paired}")
        print(f"   ⚠️ 无配对: {count_unpaired}")
        print("="*60)
        
        self.pairs = pairs

    def __len__(self):
        if self.enable_pairing and hasattr(self, 'pairs') and self.pairs:
            return len(self.pairs)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 🔥 关键修复：从缓存路径加载数据（如果存储的是路径）
        from pathlib import Path as PathType
        def _load_data(cache_item):
            if cache_item is None:
                return None
            # 如果是Path对象，从文件加载
            if isinstance(cache_item, (Path, PathType)):
                try:
                    return torch.load(cache_item, weights_only=False)
                except Exception as e:
                    print(f"⚠️  加载缓存失败 {cache_item}: {e}", flush=True)
                    return None
            # 如果已经是数据，直接返回
            return cache_item
        
        if self.enable_pairing and hasattr(self, 'pairs') and self.pairs:
            mut_idx, healed_idx = self.pairs[idx]
            
            # 从缓存加载数据（可能是路径或数据）
            data_mut = _load_data(self.data_cache[mut_idx])
            if data_mut is None:
                raise RuntimeError(f"无法加载mutated数据: index={mut_idx}")
            
            data_healed = None
            if healed_idx is not None:
                # 从缓存加载healed数据
                data_healed_raw = _load_data(self.data_cache[healed_idx])
                if data_healed_raw is not None:
                    # 浅拷贝避免修改原始缓存
                    import copy
                    data_healed = copy.deepcopy(data_healed_raw)
                    data_healed['labels'] = {
                        'y_agent': -100,
                        'y_step': -100,
                        'mistake_agent_name': '',
                        'mistake_step_str': ''
                    }
            
            return {'mutated': data_mut, 'healed': data_healed}
        else:
            # 单样本模式
            real_idx = self.valid_indices[idx]
            data = _load_data(self.data_cache[real_idx])
            if data is None:
                raise RuntimeError(f"无法加载数据: index={real_idx}")
            return data


def collate_fn(batch: List[Dict[str, Any]],
                max_seq_len: int = 160,  # Updated: test data max length is 130, set to 160 with margin
                max_agents: int = 10,
                is_paired: bool = False) -> Dict[str, torch.Tensor]:
    """
    自定义批处理函数

    将变长的图序列和变长的 Agent 数量对齐到固定维度

    Args:
        batch: 批次数据列表，每个元素是 Dataset.__getitem__ 的返回值
        max_seq_len: 最大序列长度
        max_agents: 最大 Agent 数量
        is_paired: 是否为配对数据模式

    Returns:
        批处理后的数据字典，包含：
            - 'graph_list': List[List[HeteroGraph]] 原始图列表（用于模型输入）
            - 'y_agent': [B, max_agents] Agent 故障标签（0 或 1）
            - 'y_step': [B] 故障时间步
            - 'agent_mask': [B, max_agents] Agent 掩码
            - 'seq_mask': [B, max_seq_len] 序列掩码
            - (如果 is_paired=True) 'healed_graph_list': List[List[HeteroGraph]] Healed 图列表
            - (如果 is_paired=True) 'healed_y_agent': [B, max_agents] Healed 标签（全为 -100）
            - (如果 is_paired=True) 'healed_y_step': [B] Healed 时间步（全为 -100）
            - (如果 is_paired=True) 'healed_agent_mask': [B, max_agents] Healed Agent 掩码
            - (如果 is_paired=True) 'healed_seq_mask': [B, max_seq_len] Healed 序列掩码
    """
    batch_size = len(batch)
    
    # 如果是配对模式，需要分别处理 mutated 和 healed
    if is_paired:
        # 🔥 修复：检查batch中的样本是否真的是配对格式
        if batch and isinstance(batch[0], dict) and 'mutated' in batch[0]:
            # 提取 mutated 和 healed 数据
            mutated_batch = [item['mutated'] for item in batch]
            healed_batch = [item['healed'] for item in batch if item.get('healed') is not None]
        else:
            # 如果没有配对数据，降级为单样本模式
            return _collate_single(batch, max_seq_len, max_agents)
        
        # 处理 mutated 数据（使用原有逻辑）
        mutated_collated = _collate_single(mutated_batch, max_seq_len, max_agents)
        
        # 处理 healed 数据（如果有）
        if healed_batch:
            healed_collated = _collate_single(healed_batch, max_seq_len, max_agents)
            # 合并结果
            result = mutated_collated.copy()
            result['healed_graph_list'] = healed_collated['graph_list']
            result['healed_y_agent'] = healed_collated['y_agent']
            result['healed_y_step'] = healed_collated['y_step']
            result['healed_agent_mask'] = healed_collated['agent_mask']
            result['healed_seq_mask'] = healed_collated['seq_mask']
            return result
        else:
            # 没有 healed 数据，只返回 mutated
            return mutated_collated
    else:
        # 单样本模式（原有逻辑）
        return _collate_single(batch, max_seq_len, max_agents)


def _collate_single(batch: List[Dict[str, Any]],
                    max_seq_len: int = 160,  # Updated: test data max length is 130, set to 160 with margin
                    max_agents: int = 10) -> Dict[str, torch.Tensor]:
    """
    单样本批处理函数（原有逻辑）
    """
    batch_size = len(batch)

    # 🔥 修正 4: 移除 max_seq_len 的动态扩展逻辑
    # max_seq_len 必须是一个固定值，不能依赖批次中的最大标签
    # 如果标签越界，则算作 -1 (无效)

    # 初始化输出（使用固定的 max_seq_len）
    y_agent_batch = torch.zeros(batch_size, max_agents, dtype=torch.long)
    y_step_batch = torch.zeros(batch_size, dtype=torch.long)
    agent_mask_batch = torch.zeros(batch_size, max_agents, dtype=torch.bool)
    seq_mask_batch = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    # 存储原始图列表（用于模型输入）
    graph_lists = []

    for i, sample in enumerate(batch):
        graph_list = sample['graph_list']
        labels = sample['labels']

        # 🔥 调试信息：检查原始 graph_list
        # Debug prints removed - 详细日志已保存到文件

        # 获取实际序列长度
        actual_seq_len = len(graph_list)
        seq_len = min(actual_seq_len, max_seq_len)

        # 设置序列掩码
        seq_mask_batch[i, :seq_len] = True

        # 如果序列超过最大长度，截断
        if actual_seq_len > max_seq_len:
            graph_list = graph_list[:max_seq_len]
            # Debug prints removed
        # 如果序列不足，保持原样（模型会处理）

        # Debug prints removed

        graph_lists.append(graph_list)

        # 处理 Agent 标签
        y_agent_idx = labels.get('y_agent', -1)
        
        # 🔥 诊断：打印 Hand-Crafted 数据的详细信息
        filename = sample.get('filename', '')
        if 'Hand-Crafted' in filename:
            import os
            hc_debug_file = os.path.join('checkpoints_large', 'hc_collate_debug.txt')
            os.makedirs('checkpoints_large', exist_ok=True)
            
            # 获取 Agent 节点数量
            num_agents = 0
            if graph_list and hasattr(graph_list[0], 'node_id_to_idx') and graph_list[0].node_id_to_idx:
                num_agents = sum(1 for (node_type, _) in graph_list[0].node_id_to_idx.values() 
                               if node_type == 'Agent')
            
            with open(hc_debug_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[Collate Debug] {filename}\n")
                f.write(f"  labels keys: {list(labels.keys())}\n")
                f.write(f"  y_agent_idx: {y_agent_idx}\n")
                f.write(f"  mistake_agent_name: {labels.get('mistake_agent_name', 'N/A')}\n")
                f.write(f"  num_agents in graph: {num_agents}\n")
                f.write(f"  max_agents: {max_agents}\n")
                f.write(f"  condition (y_agent_idx >= 0 and y_agent_idx < max_agents): {y_agent_idx >= 0 and y_agent_idx < max_agents}\n")
                f.write(f"  Will set y_agent_batch[{i}, {y_agent_idx}] = 1: {y_agent_idx >= 0 and y_agent_idx < max_agents}\n")
        
        if y_agent_idx >= 0 and y_agent_idx < max_agents:
            y_agent_batch[i, y_agent_idx] = 1  # 二分类：1 表示故障
            
            # 🔥 诊断：验证设置是否成功
            if 'Hand-Crafted' in filename:
                with open(hc_debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"  ✅ Successfully set y_agent_batch[{i}, {y_agent_idx}] = 1\n")
                    f.write(f"  y_agent_batch sum: {y_agent_batch[i].sum().item()}\n")

        # 🔥 修正 6: Agent Mask 应该基于整个序列中出现的所有 Agent
        # 策略1: 从 node_id_to_idx 中统计 Agent 类型的节点数量（最准确）
        num_agents = 0
        if graph_list:
            first_graph = graph_list[0]
            # 优先使用 node_id_to_idx 统计（如果存在）
            if hasattr(first_graph, 'node_id_to_idx') and first_graph.node_id_to_idx is not None:
                # 统计所有 Agent 类型的节点数量
                agent_count_from_mapping = sum(
                    1 for (node_type, _) in first_graph.node_id_to_idx.values() 
                    if node_type == 'Agent'
                )
                if agent_count_from_mapping > 0:
                    num_agents = agent_count_from_mapping
            else:
                # 策略2: 遍历整个序列，找到最大的 Agent 节点数量
                max_agent_count = 0
                for graph in graph_list:
                    if 'Agent' in graph.node_features:
                        agent_count = graph.node_features['Agent'].shape[0]
                        max_agent_count = max(max_agent_count, agent_count)
                if max_agent_count > 0:
                    num_agents = max_agent_count
                else:
                    # 策略3: 如果都没有，使用 max_agents 作为后备
                    # 但需要确保 y_agent_batch 中为 1 的 Agent 对应的列必须是 True
                    num_agents = max_agents
            
            # 限制在 max_agents 范围内
            num_agents = min(num_agents, max_agents)
            
            # 设置 agent_mask：至少标记所有有效的 Agent
            if num_agents > 0:
                agent_mask_batch[i, :num_agents] = True
            
            # 额外检查：确保 y_agent_batch 中为 1 的 Agent 对应的列必须是 True
            # 即使该 Agent 的索引超出了 num_agents（不应该发生，但为了安全）
            if y_agent_idx >= 0 and y_agent_idx < max_agents:
                agent_mask_batch[i, y_agent_idx] = True

        # 🔥 修正 5: 确保 y_step 在有效序列范围内才有效
        y_step = labels.get('y_step', -1)
        
        # 检查 y_step 是否在有效范围内：
        # 1. 必须 >= 0（非负数）
        # 2. 必须 < actual_seq_len（不能超出原始序列长度）
        # 3. 必须 < max_seq_len（不能超出最大允许长度）
        # 4. 如果序列被截断，必须 < seq_len（不能超出截断后的长度）
        if y_step >= 0:
            # 检查是否在有效序列范围内
            if y_step >= actual_seq_len:
                # 标签超出原始序列长度，无效
                y_step = -1
            elif y_step >= max_seq_len:
                # 标签超出最大允许长度，无效
                y_step = -1
            elif actual_seq_len > max_seq_len and y_step >= seq_len:
                # 序列被截断，且标签超出截断后的长度，无效
                y_step = -1
        
        # 设置标签（-1 表示无效标签）
        y_step_batch[i] = y_step

    return {
        'graph_list': graph_lists,  # List[List[HeteroGraph]]
        'y_agent': y_agent_batch,  # [B, max_agents]
        'y_step': y_step_batch,  # [B]
        'agent_mask': agent_mask_batch,  # [B, max_agents]
        'seq_mask': seq_mask_batch,  # [B, max_seq_len]
    }


def compute_metrics(outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    计算评估指标 (修复版：适配 num_classes=1 的打分模式)
    
    🔥 关键修复：根据每个样本的实际序列长度提取 scores，而不是统一取 [:, -1, :]
    """
    logits = outputs['logits']  # [B, T, N, 1]
    y_agent = targets['y_agent']  # [B, max_agents]
    agent_mask = masks['agent_mask']  # [B, max_agents]
    seq_mask = masks['seq_mask']  # [B, T] - 🔥 关键：用于找到每个样本的实际最后一个时间步
    
    # 🔥 关键修复：根据每个样本的实际序列长度提取 scores
    # 问题：之前使用 logits[:, -1, :, 0]统一取最后一个时间步，但不同样本的实际序列长度不同
    # 结果：短序列样本取到了padding位置（全零），导致 15/16 样本scores为0，模型坍缩
    B, T, N, _ = logits.shape
    scores = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
    
    for i in range(B):
        # 找到第 i 个样本的最后一个有效时间步
        valid_steps = seq_mask[i].nonzero(as_tuple=True)[0]  # 有效时间步的索引
        if valid_steps.numel() > 0:
            last_step = valid_steps[-1].item()  # 最后一个有效时间步
            scores[i] = logits[i, last_step, :, 0]  # ✅ 从正确的时间步提取
        else:
            # 如果没有有效时间步（不应该发生），使用全零
            scores[i] = 0.0
    
    # 对齐维度
    target_N = y_agent.shape[1]
    valid_N = min(N, target_N)
    
    scores = scores[:, :valid_N]
    y_agent_aligned = y_agent[:, :valid_N]
    mask = agent_mask[:, :valid_N]
    
    # 屏蔽无效节点 (将无效Agent的分数设为极小)
    scores_masked = scores.clone()
    if mask.shape == scores.shape:
        scores_masked[~mask.bool()] = -1e9
    
    # 预测：分数最高的 Agent
    pred_idx = scores_masked.argmax(dim=1)  # [B]
    true_idx = y_agent_aligned.argmax(dim=1)  # [B]
    
    # 仅计算有有效标签的样本
    has_label = y_agent_aligned.sum(dim=1) > 0
    if has_label.sum() > 0:
        correct = (pred_idx[has_label] == true_idx[has_label]).float()
        agent_acc = correct.mean().item()
    else:
        # 🔥 关键修复：如果所有样本都没有有效标签，返回 0.0
        # 这通常发生在 Hand-Crafted 数据中，mistake_agent 无法匹配到图中的节点
        agent_acc = 0.0

    # Step Accuracy (保持不变)
    step_acc = 0.0
    if 'step_logits' in outputs:
        step_logits = outputs['step_logits']
        y_step = targets['y_step']
        T_step = step_logits.shape[1]
        valid_step_mask = (y_step >= 0) & (y_step < T_step)
        if valid_step_mask.any():
            pred_step = step_logits.argmax(dim=1)
            step_acc = (pred_step[valid_step_mask] == y_step[valid_step_mask]).float().mean().item()

    return {
        'agent_accuracy': agent_acc,
        'step_accuracy': step_acc
    }


def train_epoch(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: ASTRALoss,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               logger: Optional[TrainingLogger] = None,
               w_sup: float = 1.0,
               w_cl: float = 0.1,
               w_rl: float = 0.0,
               gradient_accumulation_steps: int = 1) -> Dict[str, float]:
    """训练一个 epoch（混合模式：监督学习 + 对比学习 + MAPPO）"""
    model.train()
    
    # 初始化对比损失
    contrastive_criterion = SupConLoss(temperature=0.07).to(device)
    astra_cl_criterion = ASTRAContrastiveLoss(margin=1.0, alpha=0.7).to(device)
    
    # 权重超参数（从函数参数传入）
    W_SUP = w_sup   # 监督损失权重
    W_CL = w_cl     # ASTRA-CL 对比损失权重（建议从 0.1 开始，必须 > 0 才能启用对比学习）
    W_RL = w_rl     # ⛔ 暂时禁用 RL，直到监督学习稳定
    
    # 初始化 ASTRA-CL 对比损失
    astra_cl_criterion = ASTRAContrastiveLoss(margin=1.0, alpha=0.7).to(device)
    
    # 🔥 修改：暂时完全禁用 RL，不再使用 Warm-up 策略
    # 原因：根据 IMPLEMENTATION_PLAN.md，需要暂时禁用 RL 以避免不稳定
    if epoch == 0:
        rl_msg = f"\n[配置] 强化学习已禁用 (W_RL=0.0)\n  原因: 暂时禁用 RL，直到监督学习稳定\n  策略: 专注于监督学习和对比学习\n"
        if logger:
            logger.log(rl_msg, to_terminal=False)
        else:
            print(rl_msg)
    
    total_loss = 0.0
    total_agent_loss = 0.0
    total_step_loss = 0.0
    total_aux_loss = 0.0
    total_cl_loss = 0.0
    total_rl_loss = 0.0

    all_metrics = {'agent_accuracy': [], 'step_accuracy': []}

    # 简化进度条输出，不显示详细信息
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, ncols=80)
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        graph_lists = batch['graph_list']  # List[List[HeteroGraph]]
        y_agent = batch['y_agent'].to(device)  # [B, max_agents]
        y_step = batch['y_step'].to(device)  # [B]
        agent_mask = batch['agent_mask'].to(device)  # [B, max_agents]
        seq_mask = batch['seq_mask'].to(device)  # [B, max_seq_len]
        
        # 🔥 ASTRA-CL: 检查是否有配对数据
        has_healed = 'healed_graph_list' in batch and batch['healed_graph_list'] is not None
        
        # 🔥 调试信息：打印第一个 batch 的配对状态
        if batch_idx == 0:
            debug_msg = f"\n[DEBUG] Batch {batch_idx}: has_healed = {has_healed}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Batch keys: {list(batch.keys())}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            if has_healed:
                debug_msg = f"  healed_graph_list type: {type(batch['healed_graph_list'])}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                debug_msg = f"  healed_graph_list length: {len(batch['healed_graph_list']) if batch['healed_graph_list'] else 0}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
            debug_msg = f"  Loss weights: W_SUP={W_SUP}, W_CL={W_CL}, W_RL={W_RL}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)

        # 前向传播
        # 注意：模型期望 List[HeteroGraph]，但批处理中每个样本是 List[HeteroGraph]
        # 我们需要逐个处理每个样本，或者修改模型以支持批处理
        # 这里先使用逐个处理的方式

        batch_outputs = []
        for i, graph_list in enumerate(graph_lists):
            # 🔥 调试信息：打印每个样本的 graph_list 长度
            # Debug prints removed - 详细日志已保存到文件
            
            # 将图数据移动到设备
            graph_list_device = [graph.to(device) for graph in graph_list]
            
            # 🔥 再次检查移动到设备后的长度
            # Debug prints removed
            
            output = model(graph_list_device)
            batch_outputs.append(output)

        # 合并批处理输出
        # 由于不同样本可能有不同的序列长度和 Agent 数量，需要找到最大值并 padding
        B = len(graph_lists)

        # 找到批次中的最大序列长度和 Agent 数量
        max_T = max(out['logits'].shape[0] for out in batch_outputs)
        max_N = max(out['logits'].shape[1] for out in batch_outputs)
        num_classes = batch_outputs[0]['logits'].shape[2]
        num_experts = batch_outputs[0]['gate_weights'].shape[2]

        # 🔥 关键修复：在创建掩码之前，先检查 y_step 是否需要扩展 max_T
        # 这确保 output_seq_mask 能够覆盖所有有效的 y_step 位置
        y_step_cpu = batch['y_step']  # 还在 CPU 上
        max_y_step = y_step_cpu.max().item() if y_step_cpu.numel() > 0 and y_step_cpu.max() >= 0 else -1
        if max_y_step >= 0 and max_y_step >= max_T:
            # 需要扩展 max_T 以包含越界的 y_step
            max_T = max_y_step + 1

        # 初始化批处理张量（使用扩展后的 max_T）
        logits_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['logits'].dtype)
        alpha_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['alpha'].dtype)
        gate_weights_batch = torch.zeros(B, max_T, max_N, num_experts, device=device, dtype=batch_outputs[0]['gate_weights'].dtype)

        # 创建输出掩码（用于损失计算时忽略 padding）
        # 🔥 关键修复：使用扩展后的 max_T 创建掩码
        output_seq_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        output_agent_mask = torch.zeros(B, max_T, max_N, dtype=torch.bool, device=device)

        # 填充每个样本的输出
        for i, out in enumerate(batch_outputs):
            T_i = out['logits'].shape[0]
            N_i = out['logits'].shape[1]

            # 复制实际数据
            logits_batch[i, :T_i, :N_i, :] = out['logits']
            alpha_batch[i, :T_i, :N_i, :] = out['alpha']
            gate_weights_batch[i, :T_i, :N_i, :] = out['gate_weights']

            # 设置掩码：标记模型输出的有效时间步
            output_seq_mask[i, :T_i] = True
            output_agent_mask[i, :T_i, :N_i] = True
            
            # 🔥 关键修复：确保 y_step 所在的位置也被标记为有效
            # 即使 y_step 超出了模型输出长度（由于 max_T 扩展），也要标记为有效
            y_step_i = y_step[i].item() if y_step.numel() > i else -1
            if y_step_i >= 0 and y_step_i < max_T:
                output_seq_mask[i, y_step_i] = True  # 强制激活标签位置的掩码

        # load 需要特殊处理（可能是 [num_experts] 或 [T, num_experts]）
        load_list = [out['load'] for out in batch_outputs]
        if load_list[0].dim() == 1:
            # [num_experts] -> [B, num_experts]
            load_batch = torch.stack(load_list, dim=0)
        elif load_list[0].dim() == 2:
            # [T, num_experts] -> [B, max_T, num_experts]
            max_T_load = max(load.shape[0] for load in load_list)
            load_batch = torch.zeros(B, max_T_load, num_experts, device=device, dtype=load_list[0].dtype)
            for i, load in enumerate(load_list):
                T_load = load.shape[0]
                load_batch[i, :T_load, :] = load
        else:
            load_batch = torch.stack(load_list, dim=0)

        # global_feat 和 state_value 处理（如果存在）
        has_global_feat = all('global_feat' in out for out in batch_outputs)
        has_state_value = all('state_value' in out for out in batch_outputs)
        
        if has_global_feat:
            global_feat_list = [out['global_feat'] for out in batch_outputs]
            # global_feat 形状是 [1, d_model] 或 [d_model]，需要堆叠成 [B, d_model]
            global_feat_batch = torch.stack([feat.squeeze(0) if feat.dim() > 1 and feat.shape[0] == 1 else feat 
                                            for feat in global_feat_list], dim=0)  # [B, d_model]
        else:
            global_feat_batch = None
        
        if has_state_value:
            state_value_list = [out['state_value'] for out in batch_outputs]
            # state_value 形状是 [1, 1] 或 [1]，需要堆叠成 [B, 1]
            state_value_batch = torch.stack([val.squeeze() if val.dim() > 0 else val.unsqueeze(0) 
                                            for val in state_value_list], dim=0)  # [B, 1]
            if state_value_batch.dim() == 1:
                state_value_batch = state_value_batch.unsqueeze(-1)  # [B, 1]
        else:
            state_value_batch = None
        
        # step_logits 处理（如果存在）
        # 🔥 关键修复：检查所有输出是否都有 step_logits，并给出明确警告
        has_step_logits = all('step_logits' in out for out in batch_outputs)
        
        # 🔥 增强调试：在第一个 batch 打印详细信息
        if batch_idx == 0 and epoch == 0:
            debug_msg = f"\n[DEBUG] Batch Alignment - Checking step_logits:"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Batch size: {B}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Has step_logits in all outputs: {has_step_logits}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            for i, out in enumerate(batch_outputs):
                keys = list(out.keys())
                has_sl = 'step_logits' in out
                debug_msg = f"  Sample {i}: keys={keys}, has_step_logits={has_sl}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                if has_sl:
                    sl_shape = out['step_logits'].shape
                    debug_msg = f"    step_logits shape: {sl_shape}"
                    if logger:
                        logger.log(debug_msg, to_terminal=True)
                    else:
                        print(debug_msg)
        
        if has_step_logits:
            step_logits_list = [out['step_logits'] for out in batch_outputs]
            # step_logits 形状是 [T]，需要对齐到扩展后的 max_T（已经在上面扩展过了）
            # 使用 -inf 填充越界时间步，表示这些时间步不可预测
            step_logits_batch = torch.full((B, max_T), float('-inf'), device=device, dtype=step_logits_list[0].dtype)
            for i, step_logits in enumerate(step_logits_list):
                T_i = step_logits.shape[0]
                y_step_i = y_step[i].item() if y_step.numel() > i else -1
                # 🔥 关键修复：限制复制长度为 min(T_i, max_T)，防止索引越界
                # 额外检查：确保 step_logits 不为空且索引有效
                if T_i > 0:
                    copy_len = min(T_i, max_T, step_logits.shape[0])
                    if copy_len > 0 and i < step_logits_batch.shape[0]:
                        step_logits_batch[i, :copy_len] = step_logits[:copy_len]
                
                # 🔥 调试信息：检查 y_step 是否在有效范围内
                if y_step_i >= 0:
                    if y_step_i >= max_T:
                        error_msg = f"[ERROR] Step Alignment: Sample {i} has y_step={y_step_i} >= max_T={max_T}!"
                        if logger:
                            logger.log(error_msg, to_terminal=True)
                        else:
                            print(error_msg)
                        error_msg = f"  step_logits shape: {step_logits.shape}, T_i={T_i}"
                        if logger:
                            logger.log(error_msg, to_terminal=True)
                        else:
                            print(error_msg)
                        error_msg = f"  This should have been caught earlier - max_T should have been extended!"
                        if logger:
                            logger.log(error_msg, to_terminal=True)
                        else:
                            print(error_msg)
                    elif y_step_i >= T_i:
                        # y_step 超出了实际的 step_logits 长度，但仍在 max_T 范围内
                        # 这是正常的，因为 max_T 可能被扩展了
                        # Debug prints removed - 详细日志已保存到文件
                        pass
                # 注意：如果 y_step 越界，step_logits 在越界位置保持 -inf
                # 但 output_seq_mask 已经标记为 True，损失函数会正确处理
        else:
            # 🔥 严重警告：模型没有返回 step_logits
            missing_count = sum(1 for out in batch_outputs if 'step_logits' not in out)
            if missing_count > 0:
                error_msg = f"\n[ERROR] ⚠️  {missing_count}/{B} samples missing 'step_logits' in model output!"
                if logger:
                    logger.log(error_msg, to_terminal=True)
                else:
                    print(error_msg)
                error_msg = f"  Available keys in first output: {list(batch_outputs[0].keys())}"
                if logger:
                    logger.log(error_msg, to_terminal=True)
                else:
                    print(error_msg)
                error_msg = f"  This is a CRITICAL error - the model's forward() method must return 'step_logits'!"
                if logger:
                    logger.log(error_msg, to_terminal=True)
                else:
                    print(error_msg)
                # 创建 fallback step_logits（全 -inf，表示无预测）
                step_logits_batch = torch.full((B, max_T), float('-inf'), device=device, dtype=torch.float32)
            else:
                step_logits_batch = None

        # 🔥 ASTRA-CL: 提取 agent_embeddings（如果存在）- 在截断之前提取
        has_agent_embeddings = all('agent_embeddings' in out for out in batch_outputs)
        agent_emb_batch = None  # 初始化为 None，确保变量在作用域内
        if has_agent_embeddings:
            agent_emb_list = [out['agent_embeddings'] for out in batch_outputs]
            # agent_embeddings 形状是 [T, N, D]，需要对齐到 [B, max_T, max_N, D]
            agent_emb_batch = torch.zeros(B, max_T, max_N, agent_emb_list[0].shape[2], 
                                         device=device, dtype=agent_emb_list[0].dtype)
            for i, emb in enumerate(agent_emb_list):
                T_i, N_i, D_i = emb.shape
                agent_emb_batch[i, :min(T_i, max_T), :min(N_i, max_N), :] = emb[:min(T_i, max_T), :min(N_i, max_N), :]

        # 构建目标字典
        targets = {
            'y_agent': y_agent,
            'y_step': y_step
        }

        # 构建掩码字典
        # 注意：损失函数期望 agent_mask 是 [B, N]，其中 N 是 Agent 数量
        # 由于输出 logits 是 [B, max_T, max_N, num_classes]，而标签是 [B, max_agents]
        # 我们需要确保 max_N >= max_agents，或者对输出进行截断/对齐

        # 获取实际的 Agent 数量（从输出中）
        actual_max_N = max_N  # 输出中的最大 Agent 数

        # 如果输出的 Agent 数量与输入不匹配，需要对齐
        # 这里我们假设 max_N >= max_agents（因为模型输出可能包含更多 Agent）
        # 如果 max_N < max_agents，我们需要扩展输出（但这种情况不应该发生）

        # 对于损失计算，我们使用输入掩码（因为标签是基于输入的）
        # 但需要确保输出 logits 的 Agent 维度与标签匹配
        target_agent_dim = agent_mask.shape[1]  # 目标 Agent 维度（来自标签）

        # 🔥 关键修复：同步截断 agent_embeddings（如果存在）
        if max_N > target_agent_dim:
            # 输出有更多 Agent，截断到输入的数量
            logits_batch = logits_batch[:, :, :target_agent_dim, :]
            alpha_batch = alpha_batch[:, :, :target_agent_dim, :]
            gate_weights_batch = gate_weights_batch[:, :, :target_agent_dim, :]
            output_agent_mask = output_agent_mask[:, :, :target_agent_dim]
            # 🔥 关键修复：同步截断 agent_embeddings
            if agent_emb_batch is not None:
                agent_emb_batch = agent_emb_batch[:, :, :target_agent_dim, :]
            max_N = target_agent_dim  # 更新 max_N
        elif max_N < target_agent_dim:
            # 输出有更少 Agent，需要 padding（不应该发生，但为了安全）
            pad_size = target_agent_dim - max_N
            logits_batch = F.pad(logits_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
            alpha_batch = F.pad(alpha_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
            gate_weights_batch = F.pad(gate_weights_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
            output_agent_mask = F.pad(output_agent_mask, (0, pad_size, 0, 0, 0, 0), value=False)
            # 🔥 关键修复：同步 padding agent_embeddings
            if agent_emb_batch is not None:
                agent_emb_batch = F.pad(agent_emb_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
            max_N = target_agent_dim  # 更新 max_N

        # 确保 y_agent 的维度与对齐后的 logits 匹配
        if y_agent.shape[1] != max_N:
            if y_agent.shape[1] > max_N:
                # 截断 y_agent
                y_agent = y_agent[:, :max_N]
                agent_mask = agent_mask[:, :max_N]
            else:
                # 扩展 y_agent（不应该发生）
                pad_size = max_N - y_agent.shape[1]
                y_agent = F.pad(y_agent, (0, pad_size, 0, 0), value=0)
                agent_mask = F.pad(agent_mask, (0, pad_size, 0, 0), value=False)

        # 🔥 关键修复：使用 update 而不是重新赋值，避免覆盖已有字段
        # 先构建基础输出字典
        model_outputs = {
            'logits': logits_batch,
            'alpha': alpha_batch,
            'gate_weights': gate_weights_batch,
            'load': load_batch
        }
        
        # 🔥 关键修复：使用 update 添加其他字段，而不是重新赋值
        # 如果存在 step_logits，添加到输出字典
        if step_logits_batch is not None:
            model_outputs['step_logits'] = step_logits_batch
        
        # 🔥 关键修复：保留 agent_embeddings（用于 ASTRA-CL 对比学习）
        # 注意：agent_emb_batch 已经在上面同步截断/对齐了，这里直接添加
        if has_agent_embeddings and agent_emb_batch is not None:
            # 确保维度匹配（双重检查，虽然上面已经处理过了）
            if agent_emb_batch.shape[2] != max_N:
                if agent_emb_batch.shape[2] > max_N:
                    agent_emb_batch = agent_emb_batch[:, :, :max_N, :]
                elif agent_emb_batch.shape[2] < max_N:
                    pad_size = max_N - agent_emb_batch.shape[2]
                    agent_emb_batch = F.pad(agent_emb_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
            model_outputs['agent_embeddings'] = agent_emb_batch
        
        # 🔥 关键修复：保留 global_feat（用于 SupConLoss 对比学习）
        if has_global_feat and global_feat_batch is not None:
            model_outputs['global_feat'] = global_feat_batch
        
        # 🔥 关键修复：保留 state_value（用于强化学习 Critic）
        if has_state_value and state_value_batch is not None:
            model_outputs['state_value'] = state_value_batch
        
        # 🔥 调试信息：在传递给 loss 函数之前验证 step_logits
        if batch_idx == 0 and epoch == 0:
            print(f"\n[DEBUG] Before Loss Calculation:")
            print(f"  model_outputs keys: {list(model_outputs.keys())}")
            if 'step_logits' in model_outputs:
                print(f"  ✅ step_logits shape: {model_outputs['step_logits'].shape}")
                print(f"  step_logits dtype: {model_outputs['step_logits'].dtype}")
            else:
                print(f"  ❌ step_logits MISSING in model_outputs!")
                print(f"  step_logits_batch is None: {step_logits_batch is None}")

        # 构建掩码字典（损失函数期望的格式）
        masks = {
            'agent_mask': agent_mask,  # [B, max_agents] 用于损失计算
            'seq_mask': output_seq_mask,  # [B, max_T] 输出序列掩码
        }

        # === 1. 计算监督损失 (Supervised Loss) ===
        loss_dict = loss_fn(model_outputs, targets, masks)
        sup_loss = loss_dict['total_loss']
        
        # 🔥 关键修复：验证监督损失是否为 NaN，如果是则跳过该 batch 或使用备用损失
        if torch.isnan(sup_loss) or torch.isinf(sup_loss):
            error_msg = f"[CRITICAL ERROR] Supervised loss is NaN/Inf at batch {batch_idx}, epoch {epoch}!"
            if logger:
                logger.log(error_msg, to_terminal=True)
            else:
                print(error_msg)
            error_msg = f"  Agent loss: {loss_dict['agent_loss'].item():.6f}"
            if logger:
                logger.log(error_msg, to_terminal=True)
            else:
                print(error_msg)
            error_msg = f"  Step loss: {loss_dict['step_loss'].item():.6f}"
            if logger:
                logger.log(error_msg, to_terminal=True)
            else:
                print(error_msg)
            error_msg = f"  Aux loss: {loss_dict['aux_loss'].item():.6f}"
            if logger:
                logger.log(error_msg, to_terminal=True)
            else:
                print(error_msg)
            error_msg = f"  Skipping this batch to prevent NaN propagation."
            if logger:
                logger.log(error_msg, to_terminal=True)
            else:
                print(error_msg)
            # 使用仅 Agent Loss 作为备用（假设 Agent Loss 是稳定的）
            if not (torch.isnan(loss_dict['agent_loss']) or torch.isinf(loss_dict['agent_loss'])):
                sup_loss = loss_fn.w_agent * loss_dict['agent_loss']
                print(f"  Using agent_loss only as fallback: {sup_loss.item():.6f}")
            else:
                print(f"  ❌ All loss components are NaN/Inf, cannot proceed with this batch.")
                continue  # 跳过这个 batch
        
        # === 2. 计算对比损失 (Contrastive Loss) ===
        # 🔥 ASTRA-CL: Counterfactual Node-Level Contrast Loss
        cl_loss = torch.tensor(0.0).to(device)
        
        if has_healed and 'agent_embeddings' in model_outputs:
            # 提取 Mutated 图的 Agent embeddings
            emb_mut = model_outputs['agent_embeddings']  # [B, max_T, max_N, D]
            
            # 处理 Healed 图：需要单独前向传播
            healed_graph_list = batch['healed_graph_list']
            healed_outputs = []
            
            for healed_graphs in healed_graph_list:
                # 将图数据移动到设备
                healed_graphs_device = [graph.to(device) for graph in healed_graphs]
                # 对每个 Healed 图序列进行前向传播
                with torch.set_grad_enabled(True):  # 需要梯度用于对比学习
                    healed_out = model(healed_graphs_device)
                    healed_outputs.append(healed_out)
            
            # 对齐 Healed embeddings（与 Mutated 相同的对齐逻辑）
            if healed_outputs and 'agent_embeddings' in healed_outputs[0]:
                # 对齐到相同的形状 [B, max_T, max_N, D]
                B_healed = len(healed_outputs)
                emb_heal_batch = torch.zeros(B_healed, max_T, max_N, emb_mut.shape[3], 
                                            device=emb_mut.device, dtype=emb_mut.dtype)
                
                for i, out in enumerate(healed_outputs):
                    emb_heal = out['agent_embeddings']  # [T, N, D]
                    T_h, N_h, D_h = emb_heal.shape
                    emb_heal_batch[i, :min(T_h, max_T), :min(N_h, max_N), :] = emb_heal[:min(T_h, max_T), :min(N_h, max_N), :]
                
                # 使用最后一个时间步的 embeddings
                emb_mut_final = emb_mut[:, -1, :, :]  # [B, max_N, D]
                emb_heal_final = emb_heal_batch[:, -1, :, :]  # [B_healed, max_N, D]
                
                # 确保 batch size 匹配
                if B_healed == B:
                    # 计算对比损失
                    # 需要将 y_agent 转换为索引格式
                    mistake_agent_idx = y_agent.argmax(dim=1)  # [B]
                    
                    cl_loss = astra_cl_criterion(
                        emb_mut_final, 
                        emb_heal_final, 
                        mistake_agent_idx,
                        agent_mask
                    )
        
        # 如果没有配对数据，使用原有的 SupConLoss（基于 global_feat）
        elif 'global_feat' in model_outputs:
            global_feat = model_outputs['global_feat']  # [B, d_model] (批处理对齐后)
            
            # 🔥 关键修复：Batch Size 检查（防止小 batch 崩溃）
            if global_feat.shape[0] < 2:
                cl_loss = torch.tensor(0.0).to(device)
            else:
                # 获取标签
                if 'mistake_type' in batch:
                    cl_labels = batch['mistake_type'].to(device)  # [B]
                else:
                    # 🔥 优化：使用 y_agent + y_step 组合作为伪标签
                    # 这样可以区分"同一 Agent 在不同时间步的故障"（更细粒度）
                    # 使用哈希函数将 (agent_id, step) 映射到类别 ID
                    true_agent_idx = y_agent.argmax(dim=1)  # [B]
                    # 创建组合标签：agent_id * max_step + step_id
                    # 假设 max_step 不超过 1000，这样组合是唯一的
                    max_step_for_hash = 1000
                    cl_labels = true_agent_idx * max_step_for_hash + y_step  # [B]
                
                # 确保维度匹配
                if global_feat.shape[0] == B and cl_labels.shape[0] == B:
                    cl_loss = contrastive_criterion(global_feat, cl_labels)
                else:
                    cl_loss = torch.tensor(0.0).to(device)
        
        # === 3. 计算 MAPPO 强化学习损失 (RL Loss) ===
        # A. 获取 Action (即模型的预测)
        # Agent Action: 选哪个 Agent
        # 🔥 修复：适应 num_classes=1，直接使用分数
        logits_last = model_outputs['logits'][:, -1, :, :]  # [B, N, 1]
        scores = logits_last.squeeze(-1)  # [B, N] - 每个 Agent 的故障分数
        
        # 对齐维度
        B_act, N_act = scores.shape
        target_N_act = y_agent.shape[1]
        valid_N_act = min(N_act, target_N_act)
        
        scores = scores[:, :valid_N_act]
        agent_mask_act = agent_mask[:, :valid_N_act]
        
        # 应用掩码：将无效 Agent 的分数设为负无穷
        scores_masked = scores.clone()
        scores_masked[~agent_mask_act.bool()] = -1e9
        
        # 使用 softmax 将分数转换为概率分布
        agent_probs = F.softmax(scores_masked, dim=-1)  # [B, valid_N_act]
        
        # 对每个样本，选择概率最大的 Agent
        dist_agent = torch.distributions.Categorical(probs=agent_probs)
        action_agent = dist_agent.sample()  # [B] 采样动作
        log_prob_agent = dist_agent.log_prob(action_agent)  # [B]
        
        # Step Action: 选哪一步
        if 'step_logits' in model_outputs:
            step_logits_act = model_outputs['step_logits']  # [B, T]
            # 将 -inf 替换为很小的值，避免 softmax 问题
            step_logits_safe = step_logits_act.clone()
            step_logits_safe[step_logits_safe == float('-inf')] = -1e9
            step_probs = F.softmax(step_logits_safe, dim=-1)  # [B, T]
            dist_step = torch.distributions.Categorical(probs=step_probs)
            action_step = dist_step.sample()  # [B]
            log_prob_step = dist_step.log_prob(action_step)  # [B]
        else:
            # Fallback: 如果没有 step_logits，使用均匀分布
            action_step = torch.zeros(B, dtype=torch.long, device=device)
            log_prob_step = torch.zeros(B, device=device)
        
        # B. 计算 Reward (奖励) - 优化版：添加 Shaped Reward
        # 规则：Agent 对给 +0.5, Step 对给 +0.5, 全对额外 +1.0
        # 新增：距离奖励（如果预测步骤在真实步骤的前后 1 步范围内，给 0.2 分）
        # 必须 detach，不需要梯度
        with torch.no_grad():
            true_agent = y_agent.argmax(dim=1)  # [B]
            true_step = y_step  # [B]
            
            # 基础奖励
            r_agent = (action_agent == true_agent).float() * 0.5  # [B]
            r_step = (action_step == true_step).float() * 0.5  # [B]
            
            # 完美奖励
            r_bonus = ((action_agent == true_agent) & (action_step == true_step)).float() * 1.0  # [B]
            
            # 🔥 新增：距离奖励 (Shaped Reward) - 减少稀疏性
            # 如果预测步骤在真实步骤的前后 1 步范围内，给 0.2 分
            step_diff = torch.abs(action_step - true_step)  # [B]
            r_proximity = (step_diff <= 1).float() * 0.2  # [B]
            
            rewards = r_agent + r_step + r_bonus + r_proximity  # [B]
            
            # 计算 Advantage (优势) = Reward - Critic_Value
            # 简单的单步 PPO，Advantage = R - V(s)
            if 'state_value' in model_outputs:
                values = model_outputs['state_value'].squeeze(-1)  # [B] (批处理对齐后)
                if values.shape[0] != B:
                    # 维度不匹配，使用零值
                    values = torch.zeros(B, device=device)
                advantages = rewards - values  # [B]
            else:
                # 如果没有 state_value，使用零值
                values = torch.zeros(B, device=device)
                advantages = rewards  # [B]
        
        # C. 计算 PPO Loss (Actor Loss + Critic Loss)
        # Critic Loss: MSE(Value, Reward)
        if 'state_value' in model_outputs:
            state_value_act = model_outputs['state_value'].squeeze(-1)  # [B] (批处理对齐后)
            if state_value_act.shape[0] != B:
                state_value_act = torch.zeros(B, device=device)
            critic_loss = F.mse_loss(state_value_act, rewards)
        else:
            critic_loss = torch.tensor(0.0).to(device)
        
        # Actor Loss: -log_prob * advantage
        # 这里简化处理，不保留旧策略 (Approximate PPO)
        pg_loss = -(log_prob_agent + log_prob_step) * advantages.detach()  # [B]
        pg_loss = pg_loss.mean()  # 标量
        
        rl_loss = pg_loss + 0.5 * critic_loss
        
        # === 4. 总损失与反向传播 ===
        total_loss = W_SUP * sup_loss + W_CL * cl_loss + W_RL * rl_loss

        # 🔥 增强调试信息：在第一个 batch 打印详细信息（保存到日志）
        if batch_idx == 0:
            debug_msg = f"\n[DEBUG] Loss Calculation - Batch {batch_idx} (Epoch {epoch}):"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Loss Weights: W_SUP={W_SUP}, W_CL={W_CL}, W_RL={W_RL} {'(Warm-up: RL disabled)' if W_RL == 0.0 else '(RL enabled)'}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Max T: {max_T}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Seq Mask Sums: {output_seq_mask.sum(dim=1).tolist()}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  Y_Step: {y_step.tolist()}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            
            # 检查 step_logits 是否存在
            if 'step_logits' in model_outputs:
                step_logits_shape = model_outputs['step_logits'].shape
                debug_msg = f"  ✅ Step Logits Shape: {step_logits_shape}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                debug_msg = f"  Step Logits dtype: {model_outputs['step_logits'].dtype}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                debug_msg = f"  Step Logits device: {model_outputs['step_logits'].device}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                
                # 检查 step_logits 的值范围
                sl = model_outputs['step_logits']
                debug_msg = f"  Step Logits stats: min={sl.min().item():.4f}, max={sl.max().item():.4f}, mean={sl.mean().item():.4f}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                debug_msg = f"  Step Logits -inf count: {(sl == float('-inf')).sum().item()}/{sl.numel()}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                
                # 检查每个样本的 step_logits 在标签位置的值
                for i in range(B):
                    y_step_i = y_step[i].item()
                    if y_step_i >= 0 and y_step_i < step_logits_shape[1]:
                        logit_at_label = model_outputs['step_logits'][i, y_step_i].item()
                        print(f"    Sample {i}: y_step={y_step_i}, logit_at_label={logit_at_label:.4f}")
                    else:
                        print(f"    Sample {i}: y_step={y_step_i} (out of range, max_T={max_T})")
            else:
                print("  ❌ Step Logits NOT in model_outputs!")
                print(f"  Available keys: {list(model_outputs.keys())}")
                raise RuntimeError("CRITICAL: step_logits missing in model_outputs after batch alignment!")
            
            print(f"  Step Loss: {loss_dict['step_loss'].item():.6f}")
            # 检查每个样本的掩码状态
            for i in range(B):
                y_step_i = y_step[i].item()
                mask_sum = output_seq_mask[i].sum().item()
                mask_at_label = output_seq_mask[i, y_step_i].item() if y_step_i >= 0 and y_step_i < max_T else False
                print(f"    Sample {i}: y_step={y_step_i}, mask_sum={mask_sum}, mask_at_label={mask_at_label}")

        # 反向传播（支持梯度累积）
        # 缩放损失（梯度累积时）
        scaled_loss = total_loss / gradient_accumulation_steps
        scaled_loss.backward()
        
        # 🔥 关键修复：验证 Critic 网络的梯度回传
        # 检查 Critic 参数是否有梯度（确保梯度正确回传）
        if hasattr(model, 'critic') and 'state_value' in model_outputs:
            critic_has_grad = False
            critic_grad_norm = 0.0
            for param in model.critic.parameters():
                if param.grad is not None:
                    critic_has_grad = True
                    critic_grad_norm += param.grad.norm().item() ** 2
            critic_grad_norm = critic_grad_norm ** 0.5
            
            # 在第一个 batch 打印梯度信息和 Reward 统计
            if batch_idx == 0 and epoch == 0:
                print(f"\n[DEBUG] Critic Gradient Check:")
                print(f"  Critic has gradient: {critic_has_grad}")
                if critic_has_grad:
                    print(f"  Critic gradient norm: {critic_grad_norm:.6f}")
                else:
                    print(f"  ⚠️  WARNING: Critic has NO gradient! This may indicate a problem with gradient flow.")
                
                # 🔥 新增：打印 Reward 统计信息（监控 Shaped Reward 效果）
                print(f"\n[DEBUG] Reward Statistics:")
                print(f"  Reward components (mean over batch):")
                print(f"    r_agent: {r_agent.mean().item():.4f} (Agent correct reward)")
                print(f"    r_step: {r_step.mean().item():.4f} (Step correct reward)")
                print(f"    r_bonus: {r_bonus.mean().item():.4f} (Perfect match bonus)")
                print(f"    r_proximity: {r_proximity.mean().item():.4f} (Distance reward - NEW)")
                print(f"    Total reward: {rewards.mean().item():.4f}")
                print(f"  Reward range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
                print(f"  Advantage range: [{advantages.min().item():.4f}, {advantages.max().item():.4f}]")
        
        # 🔥 梯度累积：只在累积步骤达到时才更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        elif (batch_idx + 1) == len(dataloader):
            # 最后一个batch，即使没达到累积步数也要更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 累计损失
        total_loss_val = total_loss.item()
        total_loss += total_loss_val
        total_agent_loss += loss_dict['agent_loss'].item()
        total_step_loss += loss_dict['step_loss'].item()
        total_aux_loss += loss_dict['aux_loss'].item()
        total_cl_loss += cl_loss.item()
        total_rl_loss += rl_loss.item()

        # 计算指标
        metrics = compute_metrics(model_outputs, targets, masks)
        all_metrics['agent_accuracy'].append(metrics['agent_accuracy'])
        all_metrics['step_accuracy'].append(metrics['step_accuracy'])

        # 简化进度条输出，只显示总损失
        pbar.set_postfix({'Loss': f"{total_loss_val:.4f}"})

    # 计算平均损失和指标
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_agent_loss = total_agent_loss / num_batches
    avg_step_loss = total_step_loss / num_batches
    avg_aux_loss = total_aux_loss / num_batches
    avg_cl_loss = total_cl_loss / num_batches
    avg_rl_loss = total_rl_loss / num_batches

    avg_agent_acc = np.mean(all_metrics['agent_accuracy'])
    avg_step_acc = np.mean(all_metrics['step_accuracy'])

    return {
        'loss': avg_loss,
        'agent_loss': avg_agent_loss,
        'step_loss': avg_step_loss,
        'aux_loss': avg_aux_loss,
        'cl_loss': avg_cl_loss,
        'rl_loss': avg_rl_loss,
        'agent_accuracy': avg_agent_acc,
        'step_accuracy': avg_step_acc
    }


def validate(model: nn.Module,
            dataloader: DataLoader,
            loss_fn: ASTRALoss,
            device: torch.device,
            logger: Optional[TrainingLogger] = None) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    all_metrics = {'agent_accuracy': [], 'step_accuracy': []}
    
    # 🔥 辅助函数：统一处理 debug 信息的打印和日志保存
    def debug_log(msg: str, to_terminal: bool = True):
        """将 debug 信息保存到日志文件，可选是否同时打印到终端"""
        if logger:
            logger.log(msg, to_terminal=to_terminal)
        else:
            if to_terminal:
                print(msg, flush=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            graph_lists = batch['graph_list']
            y_agent = batch['y_agent'].to(device)
            y_step = batch['y_step'].to(device)
            agent_mask = batch['agent_mask'].to(device)
            seq_mask = batch['seq_mask'].to(device)

            # 前向传播
            batch_outputs = []
            for graph_list in graph_lists:
                # 将图数据移动到设备
                graph_list_device = [graph.to(device) for graph in graph_list]
                output = model(graph_list_device)
                batch_outputs.append(output)

            # 合并输出（使用与训练相同的批处理逻辑）
            B = len(graph_lists)

            # 找到批次中的最大序列长度和 Agent 数量
            max_T = max(out['logits'].shape[0] for out in batch_outputs)
            max_N = max(out['logits'].shape[1] for out in batch_outputs)
            num_classes = batch_outputs[0]['logits'].shape[2]
            num_experts = batch_outputs[0]['gate_weights'].shape[2]

            # 🔥 关键修复：在创建掩码之前，先检查 y_step 是否需要扩展 max_T
            # 这确保 output_seq_mask 能够覆盖所有有效的 y_step 位置
            y_step_cpu = batch['y_step']  # 还在 CPU 上
            max_y_step = y_step_cpu.max().item() if y_step_cpu.numel() > 0 and y_step_cpu.max() >= 0 else -1
            if max_y_step >= 0 and max_y_step >= max_T:
                # 需要扩展 max_T 以包含越界的 y_step
                max_T = max_y_step + 1

            # 初始化批处理张量（使用扩展后的 max_T）
            logits_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['logits'].dtype)
            alpha_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['alpha'].dtype)
            gate_weights_batch = torch.zeros(B, max_T, max_N, num_experts, device=device, dtype=batch_outputs[0]['gate_weights'].dtype)

            # 创建输出掩码（用于损失计算时忽略 padding）
            # 🔥 关键修复：使用扩展后的 max_T 创建掩码
            output_seq_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
            output_agent_mask = torch.zeros(B, max_T, max_N, dtype=torch.bool, device=device)

            # 填充每个样本的输出
            for i, out in enumerate(batch_outputs):
                T_i = out['logits'].shape[0]
                N_i = out['logits'].shape[1]

                # 复制实际数据
                logits_batch[i, :T_i, :N_i, :] = out['logits']
                alpha_batch[i, :T_i, :N_i, :] = out['alpha']
                gate_weights_batch[i, :T_i, :N_i, :] = out['gate_weights']

                # 设置掩码：标记模型输出的有效时间步
                output_seq_mask[i, :T_i] = True
                output_agent_mask[i, :T_i, :N_i] = True
                
                # 🔥 关键修复：确保 y_step 所在的位置也被标记为有效
                # 即使 y_step 超出了模型输出长度（由于 max_T 扩展），也要标记为有效
                y_step_i = y_step[i].item() if y_step.numel() > i else -1
                if y_step_i >= 0 and y_step_i < max_T:
                    output_seq_mask[i, y_step_i] = True  # 强制激活标签位置的掩码

            # load 需要特殊处理（可能是 [num_experts] 或 [T, num_experts]）
            load_list = [out['load'] for out in batch_outputs]
            if load_list[0].dim() == 1:
                # [num_experts] -> [B, num_experts]
                load_batch = torch.stack(load_list, dim=0)
            elif load_list[0].dim() == 2:
                # [T, num_experts] -> [B, max_T, num_experts]
                max_T_load = max(load.shape[0] for load in load_list)
                load_batch = torch.zeros(B, max_T_load, num_experts, device=device, dtype=load_list[0].dtype)
                for i, load in enumerate(load_list):
                    T_load = load.shape[0]
                    load_batch[i, :T_load, :] = load
            else:
                load_batch = torch.stack(load_list, dim=0)

            # step_logits 处理（如果存在）
            # 🔥 关键修复：检查所有输出是否都有 step_logits，并给出明确警告
            has_step_logits = all('step_logits' in out for out in batch_outputs)
            if has_step_logits:
                step_logits_list = [out['step_logits'] for out in batch_outputs]
                # step_logits 形状是 [T]，需要对齐到扩展后的 max_T（已经在上面扩展过了）
                # 使用 -inf 填充越界时间步，表示这些时间步不可预测
                step_logits_batch = torch.full((B, max_T), float('-inf'), device=device, dtype=step_logits_list[0].dtype)
                for i, step_logits in enumerate(step_logits_list):
                    T_i = step_logits.shape[0]
                    # 🔥 关键修复：限制复制长度为 min(T_i, max_T)，防止索引越界
                    # 额外检查：确保 step_logits 不为空且索引有效
                    if T_i > 0:
                        copy_len = min(T_i, max_T, step_logits.shape[0])
                        if copy_len > 0 and i < step_logits_batch.shape[0]:
                            step_logits_batch[i, :copy_len] = step_logits[:copy_len]
                    # 注意：如果 y_step 越界，step_logits 在越界位置保持 -inf
                    # 但 output_seq_mask 已经标记为 True，损失函数会正确处理
            else:
                # 🔥 严重警告：模型没有返回 step_logits
                missing_count = sum(1 for out in batch_outputs if 'step_logits' not in out)
                if missing_count > 0:
                    print(f"[ERROR] ⚠️  {missing_count}/{B} samples missing 'step_logits' in model output!")
                    print(f"  Available keys in first output: {list(batch_outputs[0].keys())}")
                    # 创建 fallback step_logits（全 -inf，表示无预测）
                    step_logits_batch = torch.full((B, max_T), float('-inf'), device=device, dtype=torch.float32)
                else:
                    step_logits_batch = None

            # 🔥 关键修复：在截断之前提取 agent_embeddings、global_feat、state_value
            has_agent_embeddings = all('agent_embeddings' in out for out in batch_outputs)
            agent_emb_batch = None
            if has_agent_embeddings:
                agent_emb_list = [out['agent_embeddings'] for out in batch_outputs]
                # agent_embeddings 形状是 [T, N, D]，需要对齐到 [B, max_T, max_N, D]
                agent_emb_batch = torch.zeros(B, max_T, max_N, agent_emb_list[0].shape[2], 
                                             device=device, dtype=agent_emb_list[0].dtype)
                for i, emb in enumerate(agent_emb_list):
                    T_i, N_i, D_i = emb.shape
                    agent_emb_batch[i, :min(T_i, max_T), :min(N_i, max_N), :] = emb[:min(T_i, max_T), :min(N_i, max_N), :]
            
            # 提取 global_feat 和 state_value（如果存在）
            has_global_feat = all('global_feat' in out for out in batch_outputs)
            global_feat_batch = None
            if has_global_feat:
                global_feat_list = [out['global_feat'] for out in batch_outputs]
                global_feat_batch = torch.stack([feat.squeeze(0) if feat.dim() > 1 and feat.shape[0] == 1 else feat 
                                                for feat in global_feat_list], dim=0)
            
            has_state_value = all('state_value' in out for out in batch_outputs)
            state_value_batch = None
            if has_state_value:
                state_value_list = [out['state_value'] for out in batch_outputs]
                state_value_batch = torch.stack([val.squeeze() if val.dim() > 0 else val.unsqueeze(0) 
                                                for val in state_value_list], dim=0)
                if state_value_batch.dim() == 1:
                    state_value_batch = state_value_batch.unsqueeze(-1)

            # 对齐 Agent 维度
            target_agent_dim = agent_mask.shape[1]  # 目标 Agent 维度（来自标签）

            if max_N > target_agent_dim:
                # 输出有更多 Agent，截断到输入的数量
                logits_batch = logits_batch[:, :, :target_agent_dim, :]
                alpha_batch = alpha_batch[:, :, :target_agent_dim, :]
                gate_weights_batch = gate_weights_batch[:, :, :target_agent_dim, :]
                output_agent_mask = output_agent_mask[:, :, :target_agent_dim]
                # 🔥 关键修复：同步截断 agent_embeddings
                if agent_emb_batch is not None:
                    agent_emb_batch = agent_emb_batch[:, :, :target_agent_dim, :]
                max_N = target_agent_dim  # 更新 max_N
            elif max_N < target_agent_dim:
                # 输出有更少 Agent，需要 padding（不应该发生，但为了安全）
                pad_size = target_agent_dim - max_N
                logits_batch = F.pad(logits_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
                alpha_batch = F.pad(alpha_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
                gate_weights_batch = F.pad(gate_weights_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
                output_agent_mask = F.pad(output_agent_mask, (0, pad_size, 0, 0, 0, 0), value=False)
                # 🔥 关键修复：同步 padding agent_embeddings
                if agent_emb_batch is not None:
                    agent_emb_batch = F.pad(agent_emb_batch, (0, 0, 0, pad_size, 0, 0, 0, 0))
                max_N = target_agent_dim  # 更新 max_N

            # 确保 y_agent 的维度与对齐后的 logits 匹配
            if y_agent.shape[1] != max_N:
                if y_agent.shape[1] > max_N:
                    # 截断 y_agent
                    y_agent = y_agent[:, :max_N]
                    agent_mask = agent_mask[:, :max_N]
                else:
                    # 扩展 y_agent（不应该发生）
                    pad_size = max_N - y_agent.shape[1]
                    y_agent = F.pad(y_agent, (0, pad_size, 0, 0), value=0)
                    agent_mask = F.pad(agent_mask, (0, pad_size, 0, 0), value=False)

            # 🔥 关键修复：构建输出字典，确保所有字段都被添加
            model_outputs = {
                'logits': logits_batch,
                'alpha': alpha_batch,
                'gate_weights': gate_weights_batch,
                'load': load_batch
            }

            # 如果存在 step_logits，添加到输出字典
            if step_logits_batch is not None:
                model_outputs['step_logits'] = step_logits_batch
            
            # 🔥 关键修复：添加 agent_embeddings（用于 ASTRA-CL 对比学习）
            if has_agent_embeddings and agent_emb_batch is not None:
                model_outputs['agent_embeddings'] = agent_emb_batch
            
            # 🔥 关键修复：添加 global_feat（用于 SupConLoss 对比学习）
            if has_global_feat and global_feat_batch is not None:
                model_outputs['global_feat'] = global_feat_batch
            
            # 🔥 关键修复：添加 state_value（用于强化学习 Critic）
            if has_state_value and state_value_batch is not None:
                model_outputs['state_value'] = state_value_batch

            # 构建目标字典
            targets = {'y_agent': y_agent, 'y_step': y_step}
            
            # 构建掩码字典（损失函数期望的格式）
            masks = {'agent_mask': agent_mask, 'seq_mask': output_seq_mask}

            loss_dict = loss_fn(model_outputs, targets, masks)
            
            # 🔥 关键修复：验证损失是否为 NaN，如果是则跳过该样本
            val_loss = loss_dict['total_loss']
            if torch.isnan(val_loss) or torch.isinf(val_loss):
                print(f"[WARNING] Validation loss is NaN/Inf, skipping this batch.")
                continue
            
            total_loss += val_loss.item()

            metrics = compute_metrics(model_outputs, targets, masks)
            all_metrics['agent_accuracy'].append(metrics['agent_accuracy'])
            all_metrics['step_accuracy'].append(metrics['step_accuracy'])
            
            # 🔥 关键修复：打印每个 batch 的准确率，帮助诊断问题（保存到日志）
            batch_idx = len(all_metrics['agent_accuracy']) - 1
            if batch_idx < 3:  # 打印前3个 batch
                debug_msg = f"  [Val Batch {batch_idx}] Agent Acc: {metrics['agent_accuracy']:.6f}, Step Acc: {metrics['step_accuracy']:.6f}"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
            
            # 🔥 调试：在第一个 batch 打印验证集的详细信息（保存到日志）
            if len(all_metrics['agent_accuracy']) == 1:  # 第一个 batch
                debug_msg = f"\n[DEBUG] Validation Batch 0:"
                if logger:
                    logger.log(debug_msg, to_terminal=True)
                else:
                    print(debug_msg)
                debug_log(f"  Batch size: {B}")
                debug_log(f"  y_agent shape: {y_agent.shape}")
                debug_log(f"  y_agent sum per sample: {y_agent.sum(dim=1).tolist()}")
                debug_log(f"  agent_mask shape: {agent_mask.shape}")
                debug_log(f"  agent_mask sum per sample: {agent_mask.sum(dim=1).tolist()}")
                # 检查有多少样本有有效标签
                has_label = y_agent.sum(dim=1) > 0
                debug_log(f"  Samples with valid labels: {has_label.sum().item()}/{B}")
                if has_label.any():
                    # 打印预测和真实标签
                    logits_val = model_outputs['logits']  # [B, T, N, 1]
                    seq_mask_val = masks['seq_mask']  # [B, T]
                    
                    # 🔥 修复：根据每个样本的实际序列长度提取 scores
                    B_val, T_val, N_val, _ = logits_val.shape
                    scores_val = torch.zeros(B_val, N_val, device=logits_val.device, dtype=logits_val.dtype)
                    
                    for i in range(B_val):
                        valid_steps = seq_mask_val[i].nonzero(as_tuple=True)[0]
                        if valid_steps.numel() > 0:
                            last_step = valid_steps[-1].item()
                            scores_val[i] = logits_val[i, last_step, :, 0]
                        else:
                            scores_val[i] = 0.0
                    
                    # 继续原有的维度对齐逻辑
                    valid_N_val = min(N_val, y_agent.shape[1])
                    scores_val = scores_val[:, :valid_N_val]
                    y_agent_val = y_agent[:, :valid_N_val]
                    mask_val = agent_mask[:, :valid_N_val]
                    
                    # 🔥 详细调试：检查 logits 的统计信息（保存到日志）
                    debug_log(f"  Logits shape: {logits_val.shape}")
                    debug_log(f"  Scores shape: {scores_val.shape}")
                    debug_log(f"  Scores (first 3 samples, first 10 agents):")
                    for i in range(min(3, B_val)):
                        debug_log(f"    Sample {i}: {scores_val[i, :10].tolist()}")
                    
                    # 应用掩码
                    scores_masked_val = scores_val.clone()
                    scores_masked_val[~mask_val.bool()] = -1e9
                    debug_log(f"  Scores after masking (first 3 samples, first 10 agents):")
                    for i in range(min(3, B_val)):
                        debug_log(f"    Sample {i}: {scores_masked_val[i, :10].tolist()}")
                    
                    # 检查预测逻辑
                    true_idx_val = y_agent_val.argmax(dim=1)  # 🔥 修复：先定义 true_idx_val
                    pred_idx_val = scores_masked_val.argmax(dim=1)
                    debug_log(f"  Predictions (all): {pred_idx_val.tolist()}")
                    debug_log(f"  True labels (all): {true_idx_val.tolist()}")
                    
                    # 检查是否有NaN或Inf
                    if torch.isnan(scores_val).any():
                        debug_log(f"  [WARNING] Scores contain NaN!")
                    if torch.isinf(scores_val).any():
                        debug_log(f"  [WARNING] Scores contain Inf!")
                    
                    # 检查logits的统计
                    debug_log(f"  Logits stats: min={logits_val.min().item():.4f}, max={logits_val.max().item():.4f}, mean={logits_val.mean().item():.4f}")
                    debug_log(f"  Scores stats: min={scores_val.min().item():.4f}, max={scores_val.max().item():.4f}, mean={scores_val.mean().item():.4f}")
                    debug_log(f"  Scores statistics:")
                    debug_log(f"    Mean: {scores_val.mean().item():.4f}, Std: {scores_val.std().item():.4f}")
                    debug_log(f"    Min: {scores_val.min().item():.4f}, Max: {scores_val.max().item():.4f}")
                    debug_log(f"  Scores per sample (first 5, first 10 agents):")
                    for i in range(min(5, B_val)):
                        debug_log(f"    Sample {i}: {scores_val[i, :10].tolist()}")
                    debug_log(f"  Scores after masking (first 5, first 10 agents):")
                    scores_masked_debug = scores_val.clone()
                    scores_masked_debug[~mask_val.bool()] = -1e9
                    for i in range(min(5, B_val)):
                        debug_log(f"    Sample {i}: {scores_masked_debug[i, :10].tolist()}")
                    debug_log(f"  Agent mask (first 5, first 10 agents):")
                    for i in range(min(5, B_val)):
                        debug_log(f"    Sample {i}: {mask_val[i, :10].tolist()}")
                    debug_log(f"  Scores stats (first sample): min={scores_val[0].min().item():.4f}, max={scores_val[0].max().item():.4f}, mean={scores_val[0].mean().item():.4f}, std={scores_val[0].std().item():.4f}")
                    debug_log(f"  Scores values (first sample, first 10): {scores_val[0, :10].tolist()}")
                    debug_log(f"  Mask (first sample, first 10): {mask_val[0, :10].tolist()}")
                    
                    scores_masked_val = scores_val.clone()
                    scores_masked_val[~mask_val.bool()] = -1e9
                    
                    # 🔥 检查掩码后的分数
                    debug_log(f"  Masked scores (first sample, first 10): {scores_masked_val[0, :10].tolist()}")
                    debug_log(f"  Masked scores max (first sample): {scores_masked_val[0].max().item():.4f} at index {scores_masked_val[0].argmax().item()}")
                    
                    pred_idx_val = scores_masked_val.argmax(dim=1)
                    true_idx_val = y_agent_val.argmax(dim=1)
                    
                    # 🔥 关键修复：详细分析预测分布，诊断"全 0 预测"问题
                    unique_preds, pred_counts = torch.unique(pred_idx_val, return_counts=True)
                    pred_dist = dict(zip(unique_preds.tolist(), pred_counts.tolist()))
                    debug_log(f"  Prediction distribution: {pred_dist}")
                    
                    unique_true, true_counts = torch.unique(true_idx_val, return_counts=True)
                    true_dist = dict(zip(unique_true.tolist(), true_counts.tolist()))
                    debug_log(f"  True label distribution: {true_dist}")
                    
                    # 🔥 关键诊断：检查是否预测坍缩到单一值
                    if len(pred_dist) == 1:
                        collapsed_idx = list(pred_dist.keys())[0]
                        collapsed_count = pred_dist[collapsed_idx]
                        debug_log(f"  ⚠️ [CRITICAL] 预测坍缩：所有 {collapsed_count} 个样本都预测为 Agent {collapsed_idx}")
                        debug_log(f"     这通常意味着模型陷入了'多数类坍缩'或'默认输出模式'")
                    
                    # 🔥 关键诊断：检查预测 0 的比例
                    pred_0_count = (pred_idx_val == 0).sum().item()
                    pred_0_ratio = pred_0_count / len(pred_idx_val)
                    debug_log(f"  预测为 0 的比例: {pred_0_ratio:.2%} ({pred_0_count}/{len(pred_idx_val)})")
                    if pred_0_ratio > 0.8:
                        debug_log(f"  ⚠️ [WARNING] 超过 80% 的样本预测为 Agent 0，模型可能坍缩到默认输出")
                    
                    debug_log(f"  Predictions (first 10): {pred_idx_val[:10].tolist()}")
                    debug_log(f"  True labels (first 10): {true_idx_val[:10].tolist()}")
                    debug_log(f"  Correct (first 10): {(pred_idx_val[:10] == true_idx_val[:10]).tolist()}")
                    
                    # 🔥 关键诊断：检查 logits 的分布，看是否有明显的 bias
                    logits_at_0 = logits_val[:, -1, 0, 0] if logits_val.dim() == 4 else logits_val[:, -1, 0]
                    logits_mean = logits_val[:, -1, :, 0].mean(dim=1) if logits_val.dim() == 4 else logits_val[:, -1, :].mean(dim=1)
                    debug_log(f"  Logits at Agent 0 (mean): {logits_at_0.mean().item():.4f}, std: {logits_at_0.std().item():.4f}")
                    debug_log(f"  Logits mean across all agents: {logits_mean.mean().item():.4f}, std: {logits_mean.std().item():.4f}")
                    if logits_at_0.mean().item() > logits_mean.mean().item() + 0.5:
                        debug_log(f"  ⚠️ [WARNING] Agent 0 的 logits 明显高于平均值，可能存在 bias")
                    
                    # 🔥 检查是否有 logits 全为 0 或 NaN
                    if torch.isnan(scores_val).any():
                        debug_log(f"  [WARNING] Found NaN in scores!")
                    if (scores_val == 0).all(dim=1).any():
                        debug_log(f"  [WARNING] Some samples have all-zero scores!")
                    if (scores_val.abs() < 1e-6).all(dim=1).any():
                        debug_log(f"  [WARNING] Some samples have near-zero scores!")
                debug_log(f"  Agent accuracy (this batch): {metrics['agent_accuracy']:.6f}")

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    
    # 🔥 关键修复：验证集准确率计算方式
    # 问题：如果按 batch 平均，可能掩盖模型的变化
    # 解决方案：确保准确率计算正确，并添加调试信息
    if all_metrics['agent_accuracy']:
        avg_agent_acc = np.mean(all_metrics['agent_accuracy'])
        # 🔥 调试：打印每个 batch 的准确率分布（保存到日志）
        if len(all_metrics['agent_accuracy']) > 1:
            acc_values = all_metrics['agent_accuracy']
            debug_msg = f"  [Val Debug] Batch accuracies: {[f'{a:.4f}' for a in acc_values[:5]]}... (showing first 5)"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
            debug_msg = f"  [Val Debug] Mean acc: {avg_agent_acc:.6f}, Std: {np.std(acc_values):.6f}"
            if logger:
                logger.log(debug_msg, to_terminal=True)
            else:
                print(debug_msg)
    else:
        avg_agent_acc = 0.0
    
    if all_metrics['step_accuracy']:
        avg_step_acc = np.mean(all_metrics['step_accuracy'])
    else:
        avg_step_acc = 0.0

    return {
        'loss': avg_loss,
        'agent_accuracy': avg_agent_acc,
        'step_accuracy': avg_step_acc
    }


def main():
    SAVE_WINDOW_SIZE = 5  # 定义窗口大小
    window_best_acc = 0.0  # 初始化当前窗口的最佳准确率
    """主训练函数"""
    import argparse

    # 固定随机种子，确保实验可复现
    seed_everything(42)

    parser = argparse.ArgumentParser(description='Train ASTRA-MoE model')
    parser.add_argument('--data_dir', type=str, default='outputs', help='Data directory')
    parser.add_argument('--processed_dir', type=str, default='processed_data', help='Processed data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints_large', help='Output directory for checkpoints')
    parser.add_argument('--max_seq_len', type=int, default=160, help='Maximum sequence length (updated to cover test data max length 130 + 30 margin)')
    parser.add_argument('--max_agents', type=int, default=50, help='Maximum number of agents')  # 🔥 修复：从 10 增加到 50
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (reduced to prevent overfitting)')
    parser.add_argument('--num_hgt_layers', type=int, default=2, help='Number of HGT layers (deeper to capture 2nd-order neighbor relations)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--force_cpu', action='store_true', help='Force use CPU (for debugging RTX 5070 compatibility issues)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--force_reprocess', action='store_true', default=True, help='Force reprocess data (ignore cache)')
    parser.add_argument('--no_force_reprocess', dest='force_reprocess', action='store_false', help='Use cached processed data if available')
    parser.add_argument('--debug_overfit', action='store_true', help='Debug mode: Overfit on single batch (200 epochs, balanced loss weights)')
    # 🔥 添加损失权重参数（用于控制对比学习）
    parser.add_argument('--w_sup', type=float, default=1.0, help='Supervised loss weight (default: 1.0)')
    parser.add_argument('--w_cl', type=float, default=2.0, help='Contrastive loss weight (default: 2.0, increased to break "all-0 prediction" collapse. Recommended: >= 1.0)')
    parser.add_argument('--w_rl', type=float, default=0.0, help='Reinforcement learning loss weight (default: 0.0, disabled)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (default: 1, no accumulation). Effective batch size = batch_size * gradient_accumulation_steps')

    args = parser.parse_args()

    # 🔥 创建带时间戳的训练目录
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建训练开始时间戳
    train_start_time = datetime.now()
    timestamp = train_start_time.strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"train_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志目录（在训练目录下）
    log_dir = output_dir / "logs"
    logger = TrainingLogger(log_dir)
    
    # 🔥 关键修复：自动设置 Hand-Crafted 诊断文件路径到训练日志目录
    hc_emb_debug_file = log_dir / "hc_emb_debug.txt"
    hc_match_debug_file = log_dir / "hc_match_debug.txt"
    os.environ['HC_EMB_DEBUG_FILE'] = str(hc_emb_debug_file)
    os.environ['HC_DEBUG_FILE'] = str(hc_match_debug_file)
    
    logger.log(f"训练开始时间: {train_start_time.strftime('%Y-%m-%d %H:%M:%S')}", to_terminal=True)
    logger.log(f"Checkpoint 保存目录: {output_dir}", to_terminal=True)
    logger.log(f"训练日志目录: {log_dir}", to_terminal=True)
    logger.log(f"Hand-Crafted Embedding 诊断文件: {hc_emb_debug_file}", to_terminal=True)
    logger.log(f"Hand-Crafted 匹配诊断文件: {hc_match_debug_file}", to_terminal=True)

    # 设备 - 检查 CUDA 兼容性
    print("=" * 60)
    print("设备检查")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    # 深度诊断 GPU 状态
    # 抑制 sm_120 警告（如果 GPU 计算测试成功，警告可以忽略）
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

    cuda_available = torch.cuda.is_available()
    device_count = 0
    valid_devices = []

    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        try:
            device_count = torch.cuda.device_count()
            print(f"GPU 数量: {device_count}")

            # 特殊处理：即使 device_count 为 0，也尝试直接测试 cuda:0
            # 因为某些情况下 PyTorch 可能报告 device_count=0 但实际可以使用
            if device_count == 0:
                print("⚠️  device_count 为 0，但尝试直接测试 cuda:0...")
                try:
                    # 尝试直接创建张量测试
                    test_tensor = torch.zeros(1).cuda()
                    _ = test_tensor + 1
                    del test_tensor
                    torch.cuda.empty_cache()
                    # 如果能成功，说明 GPU 实际上可用
                    valid_devices.append(0)
                    device_count = 1
                    print("  ✓ 直接测试成功！GPU 实际上可用（可能是 sm_120 兼容模式）")
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        capability = torch.cuda.get_device_capability(0)
                        print(f"  GPU 设备: {gpu_name}")
                        print(f"  计算能力: {capability}")
                        if capability[0] >= 12:
                            print(f"  ⚠️  Blackwell 架构 (sm_{capability[0]}{capability[1]})")
                            print(f"  虽然 PyTorch 显示不兼容，但 GPU 计算测试成功")
                            print(f"  将尝试使用 GPU（兼容模式）")
                    except:
                        pass
                except Exception as e:
                    print(f"  ✗ 直接测试失败: {str(e)}")

            # 尝试访问每个设备以验证是否真正可用
            for i in range(device_count):
                try:
                    # 尝试获取设备属性
                    gpu_name = torch.cuda.get_device_name(i)
                    capability = torch.cuda.get_device_capability(i)
                    props = torch.cuda.get_device_properties(i)

                    print(f"GPU {i}: {gpu_name}")
                    print(f"  计算能力: {capability}")
                    print(f"  显存: {props.total_memory / 1024**3:.2f} GB")

                    # 尝试创建一个测试张量验证设备是否真正可用
                    try:
                        test_tensor = torch.zeros(1, device=f'cuda:{i}')
                        _ = test_tensor + 1
                        del test_tensor
                        torch.cuda.empty_cache()
                        if i not in valid_devices:
                            valid_devices.append(i)
                        print(f"  ✓ 设备 {i} 可用")
                    except RuntimeError as e:
                        print(f"  ✗ 设备 {i} 不可用: {str(e)}")

                    # 检查是否是 RTX 50 系列（sm_120）
                    if capability[0] >= 12:
                        print(f"  ⚠️  检测到计算能力 {capability[0]}.{capability[1]} (Blackwell 架构)")
                        print(f"  当前 PyTorch 版本可能显示不兼容警告")
                        print(f"  但如果 GPU 计算测试成功，将尝试使用 GPU")
                except (AssertionError, RuntimeError, IndexError) as e:
                    print(f"GPU {i}: 无法获取设备信息 ({str(e)})")

            if not valid_devices:
                print("⚠️  未找到可用的 GPU 设备")
                print("可能的原因:")
                print("  1. GPU 被其他进程占用")
                print("  2. GPU 驱动版本不兼容")
                print("  3. CUDA 版本与 PyTorch 不匹配")
                print("  4. GPU 硬件故障")
                print("  5. RTX 50 系列 (sm_120) 架构暂不支持")
        except Exception as e:
            print(f"⚠️  获取 GPU 信息时出错: {str(e)}")
            device_count = 0
    else:
        print("CUDA 不可用")
        print("可能的原因:")
        print("  1. 未安装 CUDA")
        print("  2. PyTorch 未编译 CUDA 支持")
        print("  3. 系统未检测到 NVIDIA GPU")

    print("=" * 60)

    # 🔥 添加 CPU 强制模式支持
    if args.force_cpu:
        print("\n⚠️  强制使用 CPU 模式（--force_cpu 已指定）")
        device = torch.device('cpu')
        print(f"使用设备: {device}")
    elif args.device == 'cuda' and cuda_available and len(valid_devices) > 0:
        try:
            # 使用第一个有效设备
            device_id = valid_devices[0]

            # 特殊处理：如果 device_count 为 0，使用 'cuda' 而不是 'cuda:0'
            # 因为索引访问会失败，但默认设备可能可用
            if device_count == 0:
                # 使用默认 CUDA 设备（不指定索引）
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1  # 执行简单计算
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device('cuda')  # 使用默认设备，不指定索引
                print(f"\n✓ 使用设备: {device} (GPU 兼容模式 - device_count=0 但计算可用)")
            else:
                # 正常情况：使用指定设备索引
                test_tensor = torch.zeros(1, device=f'cuda:{device_id}')
                _ = test_tensor + 1  # 执行简单计算
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device(f'cuda:{device_id}')
                try:
                    gpu_name = torch.cuda.get_device_name(device_id)
                    print(f"\n✓ 使用设备: {device} (GPU: {gpu_name})")
                except:
                    print(f"\n✓ 使用设备: {device}")
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n✗ CUDA 测试失败: {error_msg}")
            print("\n" + "=" * 60)
            print("CUDA 兼容性问题诊断")
            print("=" * 60)

            # 检查是否是 sm_120 兼容性问题
            if "no kernel image is available" in error_msg.lower() or "not compatible" in error_msg.lower():
                # 安全地获取设备能力（避免在设备无效时崩溃）
                capability = None
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        capability = torch.cuda.get_device_capability(0)
                except (AssertionError, RuntimeError):
                    pass  # 设备无效，无法获取能力

                if capability and capability[0] >= 12:
                    print("检测到 RTX 50 系列 GPU (Blackwell 架构, sm_120)")
                    print("当前 PyTorch 版本不支持 sm_120 计算能力")
                    print("\n重要说明:")
                    print("  RTX 5070 等 Blackwell 架构 GPU 需要 PyTorch 从源码编译支持")
                    print("  目前官方发布的版本（包括 nightly）尚未完全支持 sm_120")
                    print("\n临时解决方案:")
                    print("  1. 使用 CPU 模式训练（当前自动回退）")
                    print("  2. 等待 PyTorch 官方发布支持 sm_120 的版本")
                    print("  3. 从源码编译 PyTorch（需要 CUDA Toolkit 和编译环境）")
                    print("\n长期解决方案:")
                    print("  关注 PyTorch GitHub 仓库，等待官方支持:")
                    print("  https://github.com/pytorch/pytorch/issues")
                    print("\n当前将使用 CPU 模式继续训练")
                else:
                    print("可能是 CUDA 版本不兼容或其他问题")
                    print("建议:")
                    print("1. 运行: python check_cuda.py 检查详细错误")
                    print("2. 运行: upgrade_pytorch.bat 升级 PyTorch")
            else:
                print("CUDA 错误详情:", error_msg)
                print("建议运行: python check_cuda.py 检查详细错误")

            print("=" * 60)
            print("\n回退到 CPU 模式")
            device = torch.device('cpu')
            print("⚠️  警告: CPU 训练速度较慢，建议修复 CUDA 问题后使用 GPU")
        except Exception as e:
            print(f"\n✗ 意外的 CUDA 错误: {e}")
            print("回退到 CPU 模式")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        if args.device == 'cuda':
            if not cuda_available:
                print(f"\n警告: 请求使用 CUDA，但 CUDA 不可用")
            elif device_count == 0:
                print(f"\n警告: 请求使用 CUDA，但未检测到 GPU 设备")
            elif len(valid_devices) == 0:
                print(f"\n警告: 请求使用 CUDA，但所有 GPU 设备都不可用")
            print("回退到 CPU 模式")
        print(f"\n使用设备: {device}")
    print()

    # 数据集
    print("加载数据集...")
    if args.force_reprocess:
        print("⚠️  强制重新处理数据（忽略缓存）")
        # 如果强制重新处理，删除旧的分片缓存目录
        # 🔥 注意：train_gpu.sh 已经删除了全局词表并重新构建
        # 这里只删除数据缓存，不删除全局词表（因为 train_gpu.sh 已经重新构建了）
        processed_path = Path(args.processed_dir) if args.processed_dir else Path("processed_data")
        cache_dir = processed_path / "cache"
        if cache_dir.exists():
            print(f"  删除旧分片缓存目录: {cache_dir}")
            import shutil
            shutil.rmtree(cache_dir)
        # 🔥 注意：全局词表 (converter_state.pt) 由 train_gpu.sh 管理
        # train_gpu.sh 会在训练前重新构建全局词表，所以这里不需要删除
    else:
        print("使用缓存数据（如果可用）")

    # ================= 数据加载优化 =================
    print("\n" + "="*60)
    print("数据集初始化 (训练/验证分离)")
    print("="*60)
    
    # 🔥 训练集：开启配对 (enable_pairing=True)
    # 用于计算 Contrastive Loss
    print("正在初始化数据集（配对模式）...")
    print("⚠️  注意：数据加载可能需要较长时间，请耐心等待...", flush=True)
    import sys
    sys.stdout.flush()
    
    try:
        full_dataset = WhoWhenDataset(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            max_agents=args.max_agents,
            processed_dir=args.processed_dir,
            force_reprocess=args.force_reprocess,
            enable_pairing=True  # 🔥 重点：训练时开启配对
        )
        print(f"✅ 数据集初始化完成！数据集大小: {len(full_dataset)}", flush=True)
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n❌ 数据加载被用户中断（Ctrl+C）", flush=True)
        raise
    except Exception as e:
        print(f"\n❌ 数据集初始化失败: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 训练集：直接使用Subset（保持配对模式）
    from torch.utils.data import Subset
    train_dataset = train_subset
    
    # 🔥 验证集：创建包装器，强制返回单样本（即使原dataset是配对模式）
    class SingleSampleWrapper:
        """包装器：将配对数据转换为单样本数据（用于验证/测试）"""
        def __init__(self, subset):
            self.subset = subset
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            sample = self.subset[idx]
            
            # 如果返回的是配对数据，只返回mutated部分
            if isinstance(sample, dict) and 'mutated' in sample:
                return sample['mutated']
            # 否则直接返回（已经是单样本）
            return sample
    
    val_dataset = SingleSampleWrapper(val_subset)
    
    print(f"训练集大小: {len(train_dataset)} (配对模式)")
    print(f"验证集大小: {len(val_dataset)} (单样本模式)")
    print("="*60 + "\n")

    # 添加调试：检查标签统计（仅训练集）
    print("\n检查训练集标签统计（前10个样本）...")
    y_agent_count = 0
    y_step_count = 0
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        # 训练集在配对模式下返回 {'mutated': ..., 'healed': ...}
        if isinstance(sample, dict) and 'mutated' in sample:
            labels = sample['mutated'].get('labels', {})
        else:
            labels = sample.get('labels', {})
        y_agent = labels.get('y_agent', -1)
        y_step = labels.get('y_step', -1)
        if y_agent >= 0:
            y_agent_count += 1
        if y_step >= 0:
            y_step_count += 1
        if i < 3:  # 打印前3个样本的详细信息
            print(f"  样本 {i}: y_agent={y_agent}, y_step={y_step}, "
                  f"mistake_agent={labels.get('mistake_agent_name', 'N/A')}, "
                  f"mistake_step={labels.get('mistake_step_str', 'N/A')}")
    print(f"  标签统计: y_agent有效={y_agent_count}/10, y_step有效={y_step_count}/10")
    if y_agent_count == 0:
        print("  ⚠️  警告: 没有找到有效的 y_agent 标签！")
    if y_step_count == 0:
        print("  ⚠️  警告: 没有找到有效的 y_step 标签！")
    print()

    # 🔥 优化：检查 Batch Size 是否足够大（对比学习需要）
    print("\n" + "="*60)
    print("Batch Size 检查 (对比学习依赖)")
    print("="*60)
    if args.batch_size < 16:
        print(f"⚠️  警告: 当前 batch_size={args.batch_size} < 16")
        print("  对比学习 (SupCon) 需要足够大的 batch size 才能找到正样本对")
        print("  如果 batch_size 太小，一个 batch 里可能根本没有同类的正样本对")
        print("  后果: mask_positive 全为 0，对比学习模块直接失效")
        print("\n  建议:")
        print("    1. 如果显存允许，将 batch_size 增加到 >= 16")
        print("    2. 如果显存不够，考虑使用 Gradient Accumulation (梯度累积)")
        print("      例如: --batch_size 4 --gradient_accumulation_steps 4 (等效 batch_size=16)")
        print("    3. 或者暂时禁用对比学习 (设置 W_CL=0)")
        print("="*60 + "\n")
    else:
        print(f"✓ Batch size={args.batch_size} >= 16，满足对比学习要求")
        print("="*60 + "\n")

    # 数据加载器
    # 🔥 ASTRA-CL: 训练集启用配对模式，验证集禁用配对模式
    print("\n" + "="*60)
    print("数据加载器配置")
    print("="*60)
    # 🔥 修复：检查是否有配对数据，如果没有则禁用配对模式
    has_pairs = (hasattr(train_dataset, 'pairs') and 
                train_dataset.pairs and 
                len(train_dataset.pairs) > 0)
    
    if has_pairs:
        print("训练集: enable_pairing=True (用于对比学习)")
    else:
        print("训练集: enable_pairing=False (没有配对数据，对比学习不可用)")
    print("验证集: enable_pairing=False (模拟真实推理场景)")
    print("="*60 + "\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, max_seq_len=args.max_seq_len, max_agents=args.max_agents, is_paired=has_pairs)  # 🔥 根据是否有配对数据决定
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, max_seq_len=args.max_seq_len, max_agents=args.max_agents, is_paired=False)  # 🔥 验证集：关闭配对
    )

    # 模型
    print("初始化模型...")
    model = ASTRAMoE(
        node_feat_dim=8192,  # 🔥 Qwen3-8B: 4096 (嵌入) + 4096 (元数据)
        edge_feat_dim=32,
        d_model=256,  # 使用命令行参数，默认64
        num_heads=4,
        num_hgt_layers=args.num_hgt_layers,  # 使用命令行参数，默认2
        num_temporal_layers=2,
        num_experts=4,
        num_classes=1,  # 🔥🔥🔥 关键修改：从 args.max_agents 改为 1（每个 Agent 输出一个故障分数）
        dropout=args.dropout,  # 使用命令行参数，默认0.5
        max_seq_len=args.max_seq_len
    ).to(device)

    # 诊断：检查模型是否真的在 GPU 上
    model_device = next(model.parameters()).device
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型实际设备: {model_device}")
    if model_device.type == 'cuda':
        print(f"✓ 模型已成功移动到 GPU: {model_device}")
        # 测试 GPU 计算
        try:
            test_tensor = torch.randn(10, 10, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"✓ GPU 计算测试成功")
        except Exception as e:
            print(f"⚠️  GPU 计算测试失败: {e}")
            print("   将回退到 CPU 模式")
            device = torch.device('cpu')
            model = model.to(device)
    else:
        print(f"⚠️  模型在 CPU 上，训练速度会较慢")

    # 损失函数
    # 🔥 过拟合测试模式：使用平衡的权重
    if args.debug_overfit:
        print("\n" + "="*60)
        print("🔧 过拟合测试模式 (Debug Overfit Mode)")
        print("="*60)
        print("  - Loss 权重: w_agent=1.0, w_step=1.0, w_aux=0.01 (平衡)")
        print("  - 学习率: 5e-4 (固定，无调度器)")
        print("  - 训练: 仅使用第一个 batch，200 epochs")
        print("  - 目标: Loss 应降至 0.001，Accuracy 应达到 100%")
        print("="*60 + "\n")
        loss_fn = ASTRALoss(
            w_agent=5.0,   # 平衡权重
            w_step=1.0,    # 平衡权重（不再主导）
            w_aux=0.01,
            focal_alpha=0.25,
            focal_gamma=2.0,
            mask_agent0=True  # 🔥 启用去偏机制
        )
    else:
        # 正常训练模式：平衡 Loss 权重，修复权重失衡问题
        # 🔥 修复：Step Loss 是 Agent Loss 的 50 倍，导致模型偏科
        # 新策略：提高 Agent 权重，降低 Step 权重，强制平衡
        loss_fn = ASTRALoss(
            w_agent=10.0,  # 🔥 激进的 Agent 权重，迫使模型关注 Agent 预测
            w_step=0.1,    # 🔥 压制 Step Loss，避免主导训练
            w_aux=0.0,
            focal_alpha=0.25,
            focal_gamma=5.0,  # 🔥 极高的 Gamma，只关注难样本（非 0 样本）
            mask_agent0=True  # 🔥 启用去偏机制，打破模型坍缩
        )
        
        # 🔥 关键修复：打印损失函数配置，确保 Focal Loss 参数正确
        if logger:
            logger.log(f"\n[Loss Config] Focal Loss gamma={5.0} (极高值，只关注难样本)", to_terminal=True)
            logger.log(f"[Loss Config] Agent weight={10.0}, Step weight={0.1}, Aux weight={0.0}", to_terminal=True)
            logger.log(f"[Loss Config] Contrastive weight (W_CL)={args.w_cl} (建议 >= 1.0 以打破坍缩，当前: {args.w_cl})", to_terminal=True)
            logger.log(f"[Loss Config] mask_agent0=True (启用去偏机制，抑制 Agent 0 预测)", to_terminal=True)
            if args.w_cl < 1.0:
                logger.log(f"[WARNING] W_CL={args.w_cl} < 1.0，可能导致模型坍缩到全 0 预测！建议增加到 1.0 或更高", to_terminal=True)

    # 优化器
    if args.debug_overfit:
        # 过拟合测试模式：使用更高的学习率，固定学习率（无调度器）
        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = None  # 不使用学习率调度器
        print(f"  优化器: AdamW, lr=5e-4 (固定)")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复检查点
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"从检查点恢复: epoch {start_epoch}, 最佳准确率: {best_val_acc:.4f}")

    # 训练循环
    print("开始训练...")
    
    # 🔥 关键修复：检查数据集是否为空
    if len(train_dataset) == 0:
        raise RuntimeError("❌ 训练数据集为空！无法开始训练。")
    
    print(f"✅ 训练数据集大小: {len(train_dataset)} 个样本", flush=True)
    print(f"✅ 验证数据集大小: {len(val_dataset)} 个样本", flush=True)
    
    # 🔥 关键修复：检查DataLoader
    if len(train_loader) == 0:
        raise RuntimeError("❌ 训练DataLoader为空！无法开始训练。")
    
    print(f"✅ 训练批次数: {len(train_loader)}", flush=True)
    print(f"✅ 验证批次数: {len(val_loader)}", flush=True)
    print("", flush=True)

    # Early Stopping 相关变量
    patience = 25  # 🔥 增加 patience：从 10 增加到 25，给模型更多学习时间
    best_val_loss = float('inf')
    patience_counter = 0

    try:
        # 调试：打印第一个 batch 的 y_step 信息
        print("\n" + "="*80)
        print("调试信息: 检查第一个训练 batch 的 y_step 标签")
        print("="*80)
        first_batch = next(iter(train_loader))
        y_step_batch = first_batch.get('y_step', torch.tensor([]))
        seq_mask_batch = first_batch.get('seq_mask', torch.tensor([]))
        graph_lists = first_batch.get('graph_list', [])

        print(f"第一个 batch 大小: {len(graph_lists)}")
        print(f"y_step 值: {y_step_batch.tolist()}")
        if seq_mask_batch.numel() > 0:
            seq_lens = seq_mask_batch.sum(dim=1).tolist()
            print(f"每个样本的序列长度: {seq_lens}")
            print(f"seq_mask 形状: {seq_mask_batch.shape}")

        # 检查每个样本
        valid_y_step_count = 0
        for i in range(len(graph_lists)):
            y_step_i = y_step_batch[i].item() if y_step_batch.numel() > i else -1
            seq_len_i = len(graph_lists[i]) if i < len(graph_lists) else 0
            valid_timesteps = seq_mask_batch[i].sum().item() if seq_mask_batch.numel() > 0 and i < seq_mask_batch.shape[0] else 0

            print(f"\n  样本 {i}:")
            print(f"    y_step: {y_step_i}")
            print(f"    图序列长度: {seq_len_i}")
            print(f"    有效时间步数 (seq_mask): {valid_timesteps}")

            if y_step_i >= 0:
                valid_y_step_count += 1
                if y_step_i >= seq_len_i:
                    print(f"    ⚠️  警告: y_step ({y_step_i}) >= 图序列长度 ({seq_len_i})")
                if y_step_i >= valid_timesteps:
                    print(f"    ⚠️  警告: y_step ({y_step_i}) >= 有效时间步数 ({valid_timesteps})")

        print(f"\n有效 y_step 标签数量: {valid_y_step_count}/{len(graph_lists)}")
        if valid_y_step_count == 0:
            print("  ❌ 所有 y_step 标签都是 -1！这是 Step Loss 为 0 的原因。")
            print("  建议:")
            print("    1. 运行: python debug_labels.py 检查标签生成过程")
            print("    2. 运行: python check_step_issue.py 检查数据加载")
            print("    3. 检查 data_adapter.py 中的 convert 方法是否正确处理 mistake_step")
        print("="*80 + "\n")

        # 🔥 过拟合测试模式：只使用第一个 batch，训练 200 个 epoch
        if args.debug_overfit:
            print("\n" + "="*60)
            print("🚀 开始过拟合测试：单 Batch 训练")
            print("="*60)
            
            # 获取第一个 batch
            single_batch = next(iter(train_loader))
            print(f"  使用 Batch 大小: {len(single_batch['graph_list'])}")
            print(f"  训练 Epochs: 200")
            print("="*60 + "\n")
            
            # 创建单 batch 的 DataLoader（用于 train_epoch）
            # 我们需要修改 train_epoch 来支持单 batch 模式，或者创建一个包装器
            # 最简单的方法：创建一个只包含一个 batch 的 DataLoader
            # 注意：DataLoader 已在文件顶部导入，不需要再次导入
            
            # 为了兼容 train_epoch，我们需要创建一个特殊的 DataLoader
            # 但 train_epoch 期望的是包含 graph_list 等字段的 batch
            # 所以我们直接在这里实现单 batch 训练循环
            
            overfit_epochs = 200
            for epoch in range(overfit_epochs):
                # 直接在这个 batch 上训练
                model.train()
                
                # 移动到设备
                graph_lists = single_batch['graph_list']
                y_agent = single_batch['y_agent'].to(device)
                y_step = single_batch['y_step'].to(device)
                agent_mask = single_batch['agent_mask'].to(device)
                seq_mask = single_batch['seq_mask'].to(device)
                
                # 前向传播
                batch_outputs = []
                for graph_list in graph_lists:
                    graph_list_device = [graph.to(device) for graph in graph_list]
                    output = model(graph_list_device)
                    batch_outputs.append(output)
                
                # 合并批处理输出（复用 train_epoch 的逻辑）
                B = len(graph_lists)
                max_T = max(out['logits'].shape[0] for out in batch_outputs)
                max_N = max(out['logits'].shape[1] for out in batch_outputs)
                num_classes = batch_outputs[0]['logits'].shape[2]
                num_experts = batch_outputs[0]['gate_weights'].shape[2]
                
                # 检查 y_step 是否需要扩展 max_T
                y_step_cpu = single_batch['y_step']
                max_y_step = y_step_cpu.max().item() if y_step_cpu.numel() > 0 and y_step_cpu.max() >= 0 else -1
                if max_y_step >= 0 and max_y_step >= max_T:
                    max_T = max_y_step + 1
                
                # 初始化批处理张量
                logits_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['logits'].dtype)
                alpha_batch = torch.zeros(B, max_T, max_N, num_classes, device=device, dtype=batch_outputs[0]['alpha'].dtype)
                gate_weights_batch = torch.zeros(B, max_T, max_N, num_experts, device=device, dtype=batch_outputs[0]['gate_weights'].dtype)
                
                output_seq_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
                output_agent_mask = torch.zeros(B, max_T, max_N, dtype=torch.bool, device=device)
                
                # 填充每个样本的输出
                for i, out in enumerate(batch_outputs):
                    T_i = out['logits'].shape[0]
                    N_i = out['logits'].shape[1]
                    logits_batch[i, :T_i, :N_i, :] = out['logits']
                    alpha_batch[i, :T_i, :N_i, :] = out['alpha']
                    gate_weights_batch[i, :T_i, :N_i, :] = out['gate_weights']
                    output_seq_mask[i, :T_i] = True
                    output_agent_mask[i, :T_i, :N_i] = True
                    y_step_i = y_step[i].item() if y_step.numel() > i else -1
                    if y_step_i >= 0 and y_step_i < max_T:
                        output_seq_mask[i, y_step_i] = True
                
                # load 处理
                load_list = [out['load'] for out in batch_outputs]
                if load_list[0].dim() == 1:
                    load_batch = torch.stack(load_list, dim=0)
                elif load_list[0].dim() == 2:
                    max_T_load = max(load.shape[0] for load in load_list)
                    load_batch = torch.zeros(B, max_T_load, num_experts, device=device, dtype=load_list[0].dtype)
                    for i, load in enumerate(load_list):
                        T_load = load.shape[0]
                        load_batch[i, :T_load, :] = load
                else:
                    load_batch = torch.stack(load_list, dim=0)
                
                # step_logits 处理（与 train_epoch 保持一致）
                step_logits_batch = None
                if 'step_logits' in batch_outputs[0]:
                    step_logits_list = [out['step_logits'] for out in batch_outputs]
                    # step_logits 形状是 [T]，需要对齐到 max_T
                    # 使用 -inf 填充越界时间步，表示这些时间步不可预测（与 train_epoch 保持一致）
                    step_logits_batch = torch.full((B, max_T), float('-inf'), device=device, dtype=step_logits_list[0].dtype)
                    for i, step_logits in enumerate(step_logits_list):
                        # 🔥 关键修复：使用实际的序列长度（从 logits 获取），而不是 step_logits 的长度
                        # 因为模型输出的 step_logits 长度可能与序列长度不匹配
                        T_i_actual = batch_outputs[i]['logits'].shape[0]  # 实际的序列长度
                        T_i_step = step_logits.shape[0]  # step_logits 的长度
                        
                        # 如果 step_logits 长度小于序列长度，需要填充
                        if T_i_step < T_i_actual:
                            # 填充到序列长度
                            padding = torch.full((T_i_actual - T_i_step,), float('-inf'), device=device, dtype=step_logits.dtype)
                            step_logits = torch.cat([step_logits, padding], dim=0)
                            T_i_step = T_i_actual
                        elif T_i_step > T_i_actual:
                            # 截断到序列长度
                            step_logits = step_logits[:T_i_actual]
                            T_i_step = T_i_actual
                        
                        # 复制到批处理张量（最多到 max_T）
                        # 🔥 关键修复：确保索引不越界
                        copy_len = min(T_i_step, max_T, step_logits.shape[0])
                        if copy_len > 0:
                            step_logits_batch[i, :copy_len] = step_logits[:copy_len]
                        # 如果 y_step 越界，确保该位置的掩码也被正确设置（已在前面处理）
                
                # Agent 维度对齐
                if y_agent.shape[1] > max_N:
                    y_agent = y_agent[:, :max_N]
                    agent_mask = agent_mask[:, :max_N]
                elif y_agent.shape[1] < max_N:
                    pad_size = max_N - y_agent.shape[1]
                    y_agent = F.pad(y_agent, (0, pad_size, 0, 0), value=0)
                    agent_mask = F.pad(agent_mask, (0, pad_size, 0, 0), value=False)
                
                # 更新模型输出
                model_outputs = {
                    'logits': logits_batch,
                    'alpha': alpha_batch,
                    'gate_weights': gate_weights_batch,
                    'load': load_batch
                }
                if step_logits_batch is not None:
                    model_outputs['step_logits'] = step_logits_batch
                
                masks = {
                    'agent_mask': agent_mask,
                    'seq_mask': output_seq_mask,
                }
                
                targets = {
                    'y_agent': y_agent,
                    'y_step': y_step,
                }
                
                # 计算损失
                loss_dict = loss_fn(model_outputs, targets, masks)
                loss = loss_dict['total_loss']
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 计算准确率（使用与 train_epoch 相同的 compute_metrics 函数）
                with torch.no_grad():
                    # 构建 masks（compute_metrics 期望 agent_mask 是 [B, N]）
                    # 使用原始的 agent_mask（从 batch 中获取，已经是 [B, N]）
                    metrics_masks = {
                        'agent_mask': agent_mask,  # [B, N]
                        'seq_mask': output_seq_mask,  # [B, T]
                    }
                    
                    metrics = compute_metrics(model_outputs, targets, metrics_masks)
                    agent_acc = metrics['agent_accuracy']
                    step_acc = metrics['step_accuracy']
                
                # 打印指标（每 10 个 epoch 打印一次，或前 20 个 epoch 每次都打印）
                if (epoch + 1) % 10 == 0 or epoch < 20:
                    print(f"Epoch {epoch+1:3d}/200 | Loss: {loss.item():.6f} | "
                          f"L_agent: {loss_dict['agent_loss'].item():.6f} | "
                          f"L_step: {loss_dict['step_loss'].item():.6f} | "
                          f"Agent Acc: {agent_acc:.4f} | Step Acc: {step_acc:.4f}")
                
                # 如果达到完美准确率，提前结束
                if agent_acc >= 1.0 and step_acc >= 1.0 and loss.item() < 0.001:
                    print(f"\n✅ 过拟合成功！Epoch {epoch+1}: Loss={loss.item():.6f}, Agent Acc={agent_acc:.4f}, Step Acc={step_acc:.4f}")
                    break
            
            print("\n" + "="*60)
            print("过拟合测试完成")
            print("="*60)
            print(f"最终 Loss: {loss.item():.6f}")
            print(f"最终 Agent Acc: {agent_acc:.4f}")
            print(f"最终 Step Acc: {step_acc:.4f}")
            if agent_acc >= 0.95 and step_acc >= 0.95:
                print("✅ 模型具备学习能力（代码逻辑正确）")
            else:
                print("⚠️  模型未能完全过拟合，可能需要检查代码逻辑")
            print("="*60 + "\n")
            
        else:
            # 正常训练模式
            for epoch in range(start_epoch, args.epochs):
                # 🔥 关键修复：每个epoch开始时打印确认
                print(f"\n{'='*60}", flush=True)
                print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
                print(f"{'='*60}", flush=True)
                import sys
                sys.stdout.flush()
                
                # 训练
                try:
                    train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, logger,
                                               w_sup=args.w_sup, w_cl=args.w_cl, w_rl=args.w_rl,
                                               gradient_accumulation_steps=args.gradient_accumulation_steps)
                except Exception as e:
                    error_msg = f"❌ 训练Epoch {epoch+1} 失败: {type(e).__name__}: {str(e)}"
                    print(error_msg, flush=True)
                    import traceback
                    traceback.print_exc()
                    if logger:
                        logger.log(error_msg, to_terminal=True)
                        logger.log(f"堆栈跟踪:\n{traceback.format_exc()}", to_terminal=False)
                    raise  # 重新抛出异常，停止训练

                # 验证
                try:
                    val_metrics = validate(model, val_loader, loss_fn, device, logger)
                except Exception as e:
                    error_msg = f"❌ 验证Epoch {epoch+1} 失败: {type(e).__name__}: {str(e)}"
                    print(error_msg, flush=True)
                    import traceback
                    traceback.print_exc()
                    if logger:
                        logger.log(error_msg, to_terminal=True)
                        logger.log(f"堆栈跟踪:\n{traceback.format_exc()}", to_terminal=False)
                    raise  # 重新抛出异常，停止训练

                # 更新学习率
                current_lr = None
                if scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']

                # 使用logger记录epoch指标（终端显示简洁版，文件保存详细版）
                logger.log_epoch_metrics(epoch, args.epochs, train_metrics, val_metrics, current_lr)

                # Early Stopping 检查
                current_val_loss = val_metrics['loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.log(f"验证损失未下降 (连续 {patience_counter}/{patience} 个 epoch)", to_terminal=False)

                # 保存检查点字典（每个epoch都保存）
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'best_val_acc': best_val_acc,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': {
                        'data_dir': args.data_dir,
                        'max_seq_len': args.max_seq_len,
                        'max_agents': args.max_agents,
                        'batch_size': args.batch_size,
                        'lr': args.lr,
                        'num_epochs': args.epochs,
                        'train_start_time': timestamp,
                        # 🔥 保存模型结构参数，确保评估时能正确加载
                        'model_config': {
                            'node_feat_dim': 8192,  # 🔥 Qwen3-8B: 4096 (嵌入) + 4096 (元数据)
                            'edge_feat_dim': 32,  # 固定值
                            'd_model': args.d_model,
                            'num_heads': 4,  # 当前硬编码值
                            'num_hgt_layers': args.num_hgt_layers,
                            'num_temporal_layers': 2,  # 当前硬编码值
                            'num_experts': 4,  # 当前硬编码值
                            'num_classes': 1,  # 🔥 修复：改为 1（每个 Agent 输出一个故障分数）
                            'dropout': args.dropout,
                            'max_seq_len': args.max_seq_len
                        }
                    }
                }

                # 1. 始终保存最新的 (Latest) - 方便意外中断恢复
                data_name = Path(args.data_dir).name
                latest_file = output_dir / f'latest_{data_name}_{timestamp}.pt'
                # 🔥 修复：强制创建目录（防止保存失败）
                latest_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, latest_file)

                # 2. 始终保存全局最好的 (Global Best)
                if val_metrics['agent_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['agent_accuracy']
                    checkpoint['best_val_acc'] = best_val_acc
                    best_file = output_dir / f'best_global_acc{best_val_acc:.4f}_{timestamp}.pt'
                    # 🔥 修复：强制创建目录（防止保存失败）
                    best_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, best_file)
                    logger.log(f"  🌟 全局最佳更新: {best_file.name} (Acc: {best_val_acc:.4f})", to_terminal=True)

                # 3. 保存每5个Epoch里的局部最优 (Local Best within Window)
                current_acc = val_metrics['agent_accuracy']
                window_idx = (epoch) // SAVE_WINDOW_SIZE  # 第几个窗口 (0, 1, 2...)

                # 如果当前 Acc 比当前窗口记录的最好值还高，就覆盖保存
                if current_acc > window_best_acc:
                    window_best_acc = current_acc
                    # 覆盖当前窗口的最佳文件（文件名包含时间戳和acc）
                    window_file = output_dir / f'best_epoch{window_idx * SAVE_WINDOW_SIZE + 1}to{(window_idx + 1) * SAVE_WINDOW_SIZE}_acc{window_best_acc:.4f}_{timestamp}.pt'
                    # 🔥 修复：强制创建目录（防止保存失败）
                    window_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, window_file)
                    logger.log(f"  💾 窗口({window_idx * 5 + 1}-{window_idx * 5 + 5})最佳更新: {window_file.name} (Acc: {window_best_acc:.4f})", to_terminal=False)

                # 如果当前是一个窗口的结束 (例如第 5, 10, 15... 个 epoch)
                # 重置窗口最佳 Acc，为下一个窗口做准备
                if (epoch + 1) % SAVE_WINDOW_SIZE == 0:
                    window_best_acc = 0.0  # 重置



                # Early Stopping
                if patience_counter >= patience:
                    logger.log(f"\n早停触发: 验证损失连续 {patience} 个 epoch 未下降", to_terminal=True)
                    logger.log(f"最佳验证损失: {best_val_loss:.4f}", to_terminal=True)
                    logger.log(f"最佳验证准确率: {best_val_acc:.4f}", to_terminal=True)
                    break
            
            # 训练结束，关闭logger
            logger.close()

    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            print("\n" + "="*60)
            print("CUDA 错误: GPU 计算不兼容")
            print("="*60)
            print(f"错误详情: {e}")
            print("\n解决方案:")
            print("1. 使用 CPU 训练（推荐用于测试）:")
            print("   python train.py --device cpu --data_dir outputs --output_dir checkpoints --epochs 50")
            print("\n2. 检查 CUDA 兼容性:")
            print("   - 运行: nvidia-smi 查看 GPU 信息")
            print("   - 检查 PyTorch CUDA 版本: python -c \"import torch; print(torch.version.cuda)\"")
            print("   - 检查系统 CUDA 版本: nvcc --version")
            print("\n3. 重新安装兼容的 PyTorch:")
            print("   访问 https://pytorch.org/get-started/locally/ 获取正确的安装命令")
            print("="*60)
        else:
            raise
    else:
        print("训练完成！")


if __name__ == "__main__":
    main()


"""
ASTRA-MoE 混合损失函数实现

实现论文 Section 3.5 中定义的混合损失函数：
1. Agent 归因损失 (L_focal) - 使用 Focal Loss 处理类别不平衡
2. 步骤预测损失 (L_step) - 时间步分类损失
3. 专家负载均衡损失 (L_aux) - 防止 MoE 坍缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不平衡问题
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    其中：
    - p_t 是预测概率
    - alpha_t 是类别权重
    - gamma 是聚焦参数（gamma > 0 时，难分类样本权重更大）
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: 类别权重平衡因子
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 'mean', 'sum', 或 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: 预测logits [B, T, N, 2] 或 [B, N, 2] 或 [B*T*N, 2]
            targets: 真实标签 [B, T, N] 或 [B, N] 或 [B*T*N]，值为 0 或 1
            mask: 有效位置掩码 [B, T, N] 或 [B, N] 或 [B*T*N]，True 表示有效位置
        
        Returns:
            损失值
        """
        # 展平处理
        # 🔥 修复：使用 reshape 而不是 view，因为 tensor 可能不连续
        if inputs.dim() == 4:
            # [B, T, N, C] -> [B*T*N, C]
            B, T, N, C = inputs.shape
            inputs = inputs.reshape(-1, C)  # [B*T*N, 2]
            targets = targets.reshape(-1)  # [B*T*N]
            if mask is not None:
                mask = mask.reshape(-1)  # [B*T*N]
        elif inputs.dim() == 3:
            # [B, N, C] -> [B*N, C]
            B, N, C = inputs.shape
            inputs = inputs.reshape(-1, C)  # [B*N, 2]
            targets = targets.reshape(-1)  # [B*N]
            if mask is not None:
                mask = mask.reshape(-1)  # [B*N]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')  # [B*T*N] 或 [B*N]
        
        # 计算概率
        p = torch.exp(-ce_loss)  # p_t = exp(-ce_loss)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        # 应用掩码（如果有）
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            if self.reduction == 'mean':
                # 只对有效位置求平均
                return focal_loss.sum() / (mask.float().sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


class ASTRALoss(nn.Module):
    """
    ASTRA-MoE 混合损失函数
    
    总损失 = w1 * L_agent + w2 * L_step + w3 * L_aux
    
    其中：
    - L_agent: Agent 归因损失（Focal Loss）
    - L_step: 步骤预测损失（CrossEntropy）
    - L_aux: 专家负载均衡损失
    """
    
    def __init__(self,
                 w_agent: float = 1.0,
                 w_step: float = 1.0,
                 w_aux: float = 0.01,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 aux_alpha: float = 0.01,
                 mask_agent0: bool = True):
        """
        Args:
            w_agent: Agent 归因损失权重
            w_step: 步骤预测损失权重
            w_aux: 专家负载均衡损失权重
            focal_alpha: Focal Loss 的 alpha 参数
            focal_gamma: Focal Loss 的 gamma 参数
            aux_alpha: 负载均衡损失的权重系数
            mask_agent0: 是否在训练时抑制 Agent 0 的预测（打破模型坍缩）
        """
        super().__init__()
        self.w_agent = w_agent
        self.w_step = w_step
        self.w_aux = w_aux
        self.aux_alpha = aux_alpha
        self.mask_agent0 = mask_agent0  # 🔥 新增：去偏机制
        
        # Agent 归因损失（Focal Loss）
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # 步骤预测损失（CrossEntropy）
        self.step_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def compute_agent_loss(self,
                           logits: torch.Tensor,
                           y_agent: torch.Tensor,
                           agent_mask: torch.Tensor,
                           seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算 Agent 归因损失 (修复版：基于分数的排序损失)
        
        Args:
            logits: [B, T, N, 1] - 每个 Agent 的故障分数 (Logits)
            y_agent: [B, N] - 真实标签 (One-hot), 1 表示该 Agent 是故障源
            agent_mask: [B, N] - 有效 Agent 掩码
            seq_mask: [B, T] - 序列掩码，用于找到每个样本的实际最后一个时间步
        """
        B, T, N, C = logits.shape
        
        # 🔥 关键修复：根据每个样本的实际序列长度提取 scores
        # 问题：之前统一取 logits[:, -1, :, :] 会导致短序列样本取到padding位置（全零）
        # 后果：模型在训练时收到错误的梯度信号，导致无法学习，准确率恒定且低
        if seq_mask is not None:
            scores = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
            for i in range(B):
                valid_steps = seq_mask[i].nonzero(as_tuple=True)[0]
                if valid_steps.numel() > 0:
                    last_step = valid_steps[-1].item()
                    scores[i] = logits[i, last_step, :, 0]
                else:
                    scores[i] = 0.0
        else:
            # Fallback：如果没有提供 seq_mask，使用原来的逻辑（但会有bug）
            scores = logits[:, -1, :, :] # [B, N, 1]
            scores = scores.squeeze(-1)  # [B, N] - 每个 Agent 的故障得分
        
        # 2. 对齐维度 (处理 N 与 y_agent 不一致的情况)
        target_N = y_agent.shape[1]
        valid_N = min(N, target_N)
        
        scores = scores[:, :valid_N]      # [B, valid_N]
        targets = y_agent[:, :valid_N]    # [B, valid_N]
        mask = agent_mask[:, :valid_N]    # [B, valid_N]
        
        # 3. 掩码处理：将无效 Agent 的分数设为负无穷 (防止 softmax 选中)
        # 注意：targets 是 float 类型 (0.0 或 1.0)，需要转换
        
        # 4. 计算 Loss
        # 这是一个多分类问题：在 valid_N 个 Agent 中选出一个
        # 我们可以直接使用 CrossEntropy，但需要将 One-hot target 转为 Index
        
        # 过滤掉没有有效标签的样本 (防止 NaN)
        has_label = targets.sum(dim=1) > 0
        valid_indices = torch.where(has_label)[0]
        
        if len(valid_indices) == 0:
            return scores.sum() * 0.0
            
        scores_valid = scores[valid_indices] # [B_valid, valid_N]
        targets_valid = targets[valid_indices] # [B_valid, valid_N]
        mask_valid = mask[valid_indices]     # [B_valid, valid_N]
        
        # 应用掩码到分数 (无效节点得分 -inf)
        scores_masked = scores_valid.clone()
        scores_masked[~mask_valid.bool()] = -1e9
        
        # 🔥 关键修复：如果开启 mask_agent0，则在训练初期抑制 Agent 0 的 Logits
        # 这是一种强力的"去偏"手段，打破模型坍缩到 Agent 0
        if self.mask_agent0 and self.training:
            # 获取真实标签索引
            target_indices = targets_valid.argmax(dim=1)  # [B_valid]
            is_not_agent0 = (target_indices != 0)
            
            # 仅对真实标签不是 0 的样本，抑制 Agent 0 的预测
            # 将 Agent 0 的分数减去一个大值，使其 Softmax 概率变小
            if is_not_agent0.any():
                scores_masked[is_not_agent0, 0] -= 5.0  # 惩罚 Agent 0
        
        # 获取目标索引（如果上面已经计算过，这里需要重新计算）
        target_indices = targets_valid.argmax(dim=1)  # [B_valid]
        
        # 标准 CrossEntropy
        loss = F.cross_entropy(scores_masked, target_indices)
        
        return loss
    
    def compute_step_loss(self,
                         step_logits: torch.Tensor,
                         y_step: torch.Tensor,
                         seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算步骤预测损失
        
        使用模型输出的 step_logits 计算 CrossEntropy Loss
        
        Args:
            step_logits: 模型输出的步骤预测 logits [B, T]
                       每个时间步的 logit，表示该时间步是故障步的概率
                       无效时间步的 logits 应为 -inf
            y_step: 真实故障时间步 [B]，值为 0 到 T-1 之间的整数，-1 表示无效
            seq_mask: 序列掩码 [B, T]，True 表示有效时间步
        
        Returns:
            损失值（如果计算失败或没有有效标签，返回0.0）
        """
        B, T = step_logits.shape
        
        # 🔥 修正 3: 简化逻辑 - 信任 collate_fn 的输出（无效标签用 -1，无效 logits 用 -inf）
        # 过滤无效标签（y_step == -1）
        valid_mask = (y_step >= 0) & (y_step < T)
        
        if not valid_mask.any():
            # 如果没有有效标签，返回带梯度的零张量
            zero_loss = step_logits.sum() * 0.0
            # 🔥 关键修复：确保返回的零张量不会产生 NaN
            if torch.isnan(zero_loss) or torch.isinf(zero_loss):
                return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
            return zero_loss
        
        # 获取有效的 logits 和标签
        valid_step_logits = step_logits[valid_mask]  # [valid_B, T]
        valid_y_step = y_step[valid_mask].long()  # [valid_B]
        
        # 🔥 关键修复：验证 valid_y_step 中的所有值都在有效范围内
        if (valid_y_step < 0).any() or (valid_y_step >= T).any():
            # 如果发现越界索引，截断到有效范围
            valid_y_step = torch.clamp(valid_y_step, min=0, max=T - 1)
        
        # 如果提供了 seq_mask，只对有效时间步计算损失
        if seq_mask is not None:
            # 🔥 修正 3: 简化逻辑 - 信任 collate_fn 的输出
            # 确保 seq_mask 的形状与 step_logits 一致
            if seq_mask.shape[1] != T:
                if seq_mask.shape[1] > T:
                    seq_mask = seq_mask[:, :T]
                else:
                    pad_size = T - seq_mask.shape[1]
                    seq_mask = F.pad(seq_mask, (0, pad_size), value=False)
            
            valid_seq_mask = seq_mask[valid_mask]  # [valid_B, T]
            # 确保每个样本的标签位置在 seq_mask 中为 True
            batch_indices = torch.arange(valid_y_step.shape[0], device=valid_y_step.device)
            # 🔥 关键修复：确保索引不越界
            safe_y_step = torch.clamp(valid_y_step, min=0, max=T - 1)
            valid_seq_mask[batch_indices, safe_y_step] = True
            
            # 将无效时间步的 logits 设为 -inf
            masked_logits = torch.where(valid_seq_mask, valid_step_logits, 
                                       torch.tensor(float('-inf'), device=valid_step_logits.device, dtype=valid_step_logits.dtype))
            
            # 🔥 关键修复：检查 masked_logits 在标签位置是否全部为 -inf
            # 如果所有有效样本在标签位置的 logit 都是 -inf，会导致 CrossEntropyLoss 产生 NaN
            label_logits = masked_logits[batch_indices, safe_y_step]  # [valid_B]
            if (label_logits == float('-inf')).all():
                # 所有标签位置的 logits 都是 -inf，无法计算损失，返回 0
                print(f"[WARNING] All label logits are -inf in step_loss computation, returning 0.0")
                return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
            
            try:
                step_loss = self.step_loss_fn(masked_logits, safe_y_step)
                # 🔥 关键修复：检查计算结果是否为 NaN 或 Inf
                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    print(f"[WARNING] Step loss is NaN/Inf, returning 0.0. Check model outputs and labels.")
                    return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
            except Exception as e:
                print(f"[ERROR] Step loss computation failed: {e}, returning 0.0")
                return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
        else:
            # 直接计算 CrossEntropy Loss
            try:
                step_loss = self.step_loss_fn(valid_step_logits, valid_y_step)
                # 🔥 关键修复：检查计算结果是否为 NaN 或 Inf
                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    print(f"[WARNING] Step loss is NaN/Inf, returning 0.0. Check model outputs and labels.")
                    return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
            except Exception as e:
                print(f"[ERROR] Step loss computation failed: {e}, returning 0.0")
                return torch.tensor(0.0, device=step_logits.device, requires_grad=True)
        
        return step_loss
    
    def compute_aux_loss(self,
                        gate_weights: torch.Tensor,
                        load: torch.Tensor,
                        agent_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算专家负载均衡损失
        
        公式: L_aux = alpha * N * sum_i(f_i * P_i)
        
        其中：
        - f_i 是专家 i 的负载（load）
        - P_i 是专家 i 的平均门控权重
        - N 是专家数量
        
        Args:
            gate_weights: 门控权重 [B, T, N, num_experts] 或 [B, T, N, num_experts]
            load: 专家负载 [B, num_experts] 或 [num_experts]
            agent_mask: Agent 掩码 [B, N]，True 表示有效 Agent
        
        Returns:
            损失值
        """
        # 处理 load 的维度
        if load.dim() == 1:
            # [num_experts] -> [1, num_experts]
            load = load.unsqueeze(0)
        
        # 计算每个专家的平均门控权重
        # gate_weights: [B, T, N, num_experts]
        if gate_weights.dim() == 4:
            B, T, N, num_experts = gate_weights.shape
            
            # 应用 agent_mask（如果有）
            if agent_mask is not None:
                # 扩展 mask 到 [B, T, N, 1]
                agent_mask_expanded = agent_mask.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1)  # [B, T, N, 1]
                # 只对有效 Agent 求平均
                masked_weights = gate_weights * agent_mask_expanded.float()
                # 计算有效 Agent 数量
                valid_agents = agent_mask_expanded.float().sum(dim=(1, 2), keepdim=True)  # [B, 1, 1, 1]
                # 将 valid_agents 从 [B, 1, 1, 1] 转换为 [B, 1] 以匹配 [B, num_experts] 的维度
                valid_agents_flat = valid_agents.squeeze(-1).squeeze(-1)  # [B, 1]
                P = masked_weights.sum(dim=(1, 2)) / (valid_agents_flat + 1e-8)  # [B, num_experts]
            else:
                P = gate_weights.mean(dim=(1, 2))  # [B, num_experts]
        else:
            # 如果维度不对，直接平均
            P = gate_weights.mean(dim=tuple(range(gate_weights.dim() - 1)))  # [..., num_experts]
        
        # 扩展 load 到匹配 P 的 batch 维度
        if load.shape[0] == 1 and P.shape[0] > 1:
            load = load.expand_as(P)
        
        # 计算负载均衡损失
        # L_aux = alpha * N * sum_i(f_i * P_i)
        num_experts = load.shape[-1]
        aux_loss = self.aux_alpha * num_experts * (load * P).sum(dim=-1).mean()
        
        return aux_loss
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出字典，包含：
                - 'logits': [B, T, N, 2] Agent 故障概率 logits
                - 'alpha': [B, T, N, num_classes] Dirichlet 分布参数（可选）
                - 'gate_weights': [B, T, N, num_experts] 门控权重
                - 'load': [B, num_experts] 或 [num_experts] 专家负载
                - 'step_logits': [B, T] 步骤预测 logits（可选）
            targets: 真实标签字典，包含：
                - 'y_agent': [B, N] Agent 故障标签（0 或 1）
                - 'y_step': [B] 故障时间步（0 到 T-1 的整数，-1 表示无效）
            masks: 掩码字典，包含：
                - 'agent_mask': [B, N] Agent 掩码，True 表示有效 Agent
                - 'seq_mask': [B, T] 序列掩码，True 表示有效时间步
        
        Returns:
            损失字典，包含：
                - 'total_loss': 总损失
                - 'agent_loss': Agent 归因损失
                - 'step_loss': 步骤预测损失
                - 'aux_loss': 专家负载均衡损失
        """
        # 提取输出
        logits = outputs['logits']  # [B, T, N, 2]
        gate_weights = outputs['gate_weights']  # [B, T, N, num_experts]
        load = outputs['load']  # [B, num_experts] 或 [num_experts]
        
        # 提取标签
        y_agent = targets['y_agent']  # [B, N]
        y_step = targets['y_step']  # [B]
        
        # 提取掩码
        agent_mask = masks.get('agent_mask', None) if masks else None
        seq_mask = masks.get('seq_mask', None) if masks else None
        
        # 1. Agent 归因损失
        # 🔥 关键修复：传入 seq_mask 以正确提取每个样本的最后一个有效时间步
        agent_loss = self.compute_agent_loss(logits, y_agent, agent_mask, seq_mask)
        
        # 2. 步骤预测损失
        # 🔥 关键修复：如果缺少 step_logits，直接报错而不是返回 0
        if 'step_logits' not in outputs:
            raise RuntimeError(
                "CRITICAL ERROR: 'step_logits' missing in loss input!\n"
                f"  Available keys in outputs: {list(outputs.keys())}\n"
                f"  Expected keys: ['logits', 'alpha', 'gate_weights', 'load', 'step_logits']\n"
                "  This indicates the model's forward() method is not returning 'step_logits'."
            )
        
        # 如果模型提供了 step_logits，使用 compute_step_loss 计算损失
        step_logits = outputs['step_logits']  # [B, T]
        step_loss = self.compute_step_loss(step_logits, y_step, seq_mask)
        
        # 🔥 关键修复：验证 step_loss 是否为 NaN 或无效值，如果是则禁用该项
        if torch.isnan(step_loss) or torch.isinf(step_loss) or step_loss.item() == 0.0:
            if torch.isnan(step_loss) or torch.isinf(step_loss):
                print(f"[WARNING] Step loss is NaN/Inf, disabling step loss component in total loss.")
            # 将 step_loss 设为 0（不带梯度），防止污染总损失
            step_loss = torch.tensor(0.0, device=agent_loss.device, requires_grad=False)
            # 同时将权重设为 0，确保不影响总损失
            effective_w_step = 0.0
        else:
            effective_w_step = self.w_step
        
        # 3. 专家负载均衡损失
        aux_loss = self.compute_aux_loss(gate_weights, load, agent_mask)
        
        # 🔥 关键修复：验证所有损失组件是否为 NaN
        if torch.isnan(agent_loss) or torch.isinf(agent_loss):
            raise RuntimeError(f"CRITICAL: Agent loss is NaN/Inf! agent_loss={agent_loss}")
        if torch.isnan(aux_loss) or torch.isinf(aux_loss):
            print(f"[WARNING] Aux loss is NaN/Inf, setting to 0.0")
            aux_loss = torch.tensor(0.0, device=agent_loss.device, requires_grad=False)
            effective_w_aux = 0.0
        else:
            effective_w_aux = self.w_aux
        
        # 4. 总损失（使用有效的权重）
        total_loss = (self.w_agent * agent_loss + 
                     effective_w_step * step_loss + 
                     effective_w_aux * aux_loss)
        
        # 🔥 最终检查：确保 total_loss 不是 NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[CRITICAL ERROR] Total loss is NaN/Inf!")
            print(f"  agent_loss: {agent_loss.item():.6f}, w_agent: {self.w_agent}")
            print(f"  step_loss: {step_loss.item():.6f}, w_step: {effective_w_step}")
            print(f"  aux_loss: {aux_loss.item():.6f}, w_aux: {effective_w_aux}")
            raise RuntimeError("Total loss computation resulted in NaN/Inf. Check individual loss components.")
        
        return {
            'total_loss': total_loss,
            'agent_loss': agent_loss,
            'step_loss': step_loss,
            'aux_loss': aux_loss
        }


class ASTRAContrastiveLoss(nn.Module):
    """
    ASTRA-CL: Counterfactual Node-Level Contrast Loss
    
    基于反事实（Counterfactual）的节点级对比学习：
    - Positive Pair (拉近): Mutated 图中的正常节点 vs. Healed 图中的对应节点
    - Negative Pair (推远): Mutated 图中的故障节点 vs. Healed 图中的对应节点
    """
    
    def __init__(self, margin: float = 1.0, alpha: float = 0.7):
        """
        Args:
            margin: 对比损失的边界（用于故障节点的推远）
            alpha: 故障节点损失的权重（正常节点损失权重为 1-alpha）
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
    
    def forward(self, 
                emb_mut: torch.Tensor, 
                emb_heal: torch.Tensor, 
                mistake_agent_idx: torch.Tensor,
                agent_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算反事实对比损失
        
        Args:
            emb_mut: [Batch, Num_Agents, Hidden_Dim] - 故障图的 Agent 嵌入
            emb_heal: [Batch, Num_Agents, Hidden_Dim] - 修复图的 Agent 嵌入
            mistake_agent_idx: [Batch] - 真实的故障 Agent 索引（-1 表示无效）
            agent_mask: [Batch, Num_Agents] - Agent 掩码（可选）
        
        Returns:
            损失值
        """
        B, N, D = emb_mut.shape
        
        # 确保 emb_heal 的形状与 emb_mut 匹配
        if emb_heal.shape != emb_mut.shape:
            # 如果形状不匹配，尝试对齐
            B_h, N_h, D_h = emb_heal.shape
            if B_h != B:
                raise ValueError(f"Batch size mismatch: emb_mut={B}, emb_heal={B_h}")
            if D_h != D:
                raise ValueError(f"Hidden dim mismatch: emb_mut={D}, emb_heal={D_h}")
            
            # 对齐 Agent 数量（取较小值）
            N = min(N, N_h)
            emb_mut = emb_mut[:, :N, :]
            emb_heal = emb_heal[:, :N, :]
            if agent_mask is not None:
                agent_mask = agent_mask[:, :N]
        
        loss = 0.0
        valid_count = 0
        
        for b in range(B):
            idx = mistake_agent_idx[b].item()
            if idx < 0 or idx >= N:
                continue  # 跳过无效数据
            
            # 获取当前样本的掩码（如果有）
            if agent_mask is not None:
                sample_mask = agent_mask[b, :N]  # [N]
            else:
                sample_mask = torch.ones(N, dtype=torch.bool, device=emb_mut.device)
            
            # 1. 故障节点对比 (Negative Pair): 距离越大越好
            # 提取故障 Agent 在两个图中的 Embedding
            h_mut_target = emb_mut[b, idx]  # [D]
            h_heal_target = emb_heal[b, idx]  # [D]
            
            # 计算余弦相似度（我们希望它越小越好，即距离越大越好）
            cos_sim_target = F.cosine_similarity(h_mut_target.unsqueeze(0), h_heal_target.unsqueeze(0), dim=1)  # [1]
            cos_sim_target = cos_sim_target.squeeze(0)  # scalar
            
            # 损失：我们希望 cos_sim_target 接近 -1 或 0（即不相似）
            # 使用 hinge loss: max(0, margin - distance)，但这里我们希望距离大
            # 所以使用: max(0, cos_sim_target - (-margin)) = max(0, cos_sim_target + margin)
            # 或者更简单：直接惩罚相似度（希望相似度为负）
            loss_target = F.relu(cos_sim_target + self.margin)  # 如果相似度 > -margin，则惩罚
            
            # 2. 正常节点对比 (Positive Pair): 距离越小越好（保持稳定性）
            # 算出非故障节点的掩码
            normal_mask = sample_mask.clone()
            normal_mask[idx] = False
            
            if normal_mask.any():
                h_mut_others = emb_mut[b, normal_mask]  # [num_normal, D]
                h_heal_others = emb_heal[b, normal_mask]  # [num_normal, D]
                
                # 正常节点的表示应该保持一致（相似度应该接近 1）
                cos_sim_others = F.cosine_similarity(h_mut_others, h_heal_others, dim=1)  # [num_normal]
                
                # 损失：我们希望 cos_sim_others 接近 1（即相似）
                # 使用: 1 - cos_sim_others（距离）
                loss_others = (1 - cos_sim_others).mean()
            else:
                loss_others = 0.0
            
            # 总 Loss: 故障节点的排斥 + 正常节点的吸引
            loss += self.alpha * loss_target + (1 - self.alpha) * loss_others
            valid_count += 1
        
        if valid_count == 0:
            # 如果没有有效样本，返回零损失（带梯度）
            return emb_mut.sum() * 0.0
        
        return loss / valid_count


class SupConLoss(nn.Module):
    """
    有监督对比学习损失 (Supervised Contrastive Loss)
    让同一类故障的样本在特征空间聚集，不同类的排斥。
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim] (模型的中间层输出)
            labels: [batch_size] (故障类型 label, e.g., Tool Error=0, Logic Error=1...)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 特征归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 掩码：自己不能和自己对比
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        
        # 标签相同的为正样本 (Positive Mask)
        labels = labels.view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        # 移除对角线
        mask_positive = mask_positive * (~mask).float()
        
        # 计算 Logits
        # 这里简化处理：使用 InfoNCE 变体
        exp_logits = torch.exp(similarity_matrix) * (~mask).float()
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 🔥 关键修复：检查是否有正样本对
        # 如果 batch 里没有正样本对（即 mask_positive 全为 0），返回 0 Loss
        num_positive_pairs = mask_positive.sum().item()
        if num_positive_pairs == 0:
            # 没有正样本对，对比学习无效，返回 0 Loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 计算平均损失
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + 1e-8)
        loss = - mean_log_prob_pos
        return loss.mean()


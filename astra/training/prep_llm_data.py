"""
将图数据转换为阶段二（LLM微调）所需的格式

阶段二的目标：
- GNN 已经预测出 Top-K 候选 Agent
- LLM 需要根据这些候选 Agent 的日志，分析出故障原因

数据格式：
- instruction: 包含 Top-K 候选 Agent 的日志内容
- output: 故障原因分析（基于日志内容）
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from tqdm import tqdm
import random


def extract_agent_logs(nodes: Dict[str, Any], agent_ids: List[str], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """
    提取指定 Agent 的日志内容
    
    Args:
        nodes: 节点数据字典
        agent_ids: Agent ID 列表
        history: 历史事件列表（可选，用于提取原始内容）
    
    Returns:
        Dict[agent_id, log_content]
    """
    agent_logs = {}
    
    for agent_id in agent_ids:
        if agent_id in nodes:
            node_data = nodes[agent_id]
            log_content = ""
            
            # 方法1: 从 history 中提取该 Agent 的所有事件内容（保留绝对Step ID）
            if history:
                agent_events = []
                for event_idx, event in enumerate(history):
                    # 检查事件是否属于该 Agent
                    event_agent = event.get('name') or event.get('role') or event.get('sender', '')
                    if event_agent == agent_id or (isinstance(event_agent, str) and agent_id in event_agent):
                        content = event.get('content', '')
                        # 尝试从event中提取Step ID（绝对ID）
                        step_id = event.get('step', event.get('step_id', event.get('timestamp', event_idx)))
                        
                        if content:
                            # 处理不同类型的content
                            if isinstance(content, str):
                                agent_events.append((step_id, content))
                            elif isinstance(content, dict):
                                # 尝试从字典中提取文本
                                text = content.get('text', '') or content.get('content', '') or str(content)
                                if text:
                                    agent_events.append((step_id, text))
                            else:
                                agent_events.append((step_id, str(content)))
                
                if agent_events:
                    # 合并所有事件内容，使用绝对Step ID
                    for step_id, event_content in agent_events:
                        if log_content:
                            log_content += f"\n[Step {step_id}]: {event_content[:500]}"
                        else:
                            log_content = f"[Step {step_id}]: {event_content[:500]}"
            
            # 方法2: 如果 history 中没有，尝试从 features 中提取（使用绝对Step ID）
            if not log_content:
                features = node_data.get('features', {})
                if isinstance(features, dict):
                    # 按时间步排序（t是绝对Step ID）
                    sorted_timesteps = sorted(features.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                    
                    for t in sorted_timesteps:
                        feat = features[t]
                        if isinstance(feat, dict):
                            # 尝试多种可能的键名
                            content_text = (
                                feat.get('content_text', '') or 
                                feat.get('content', '') or
                                feat.get('text', '')
                            )
                            if content_text and content_text.strip():
                                # 🔥 使用绝对Step ID（t），而不是相对索引
                                if log_content:
                                    log_content += f"\n[Step {t}]: {content_text[:500]}"
                                else:
                                    log_content = f"[Step {t}]: {content_text[:500]}"
            
            if log_content:
                agent_logs[agent_id] = log_content
            else:
                # 如果没有日志，尝试从节点类型和其他信息构建描述
                agent_type = node_data.get('type', 'Agent')
                node_info = f"Agent {agent_id} ({agent_type})"
                features = node_data.get('features', {})
                if features:
                    node_info += f"，有 {len(features)} 个时间步的特征"
                agent_logs[agent_id] = f"{node_info}：无详细日志内容"
    
    return agent_logs


def format_instruction_for_stage2(
    ground_truth: Dict[str, Any],
    nodes: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
    include_negative: bool = True
) -> Dict[str, str]:
    """
    格式化阶段二的指令数据
    
    阶段二场景：
    1. GNN 已经预测出 Top-K 候选 Agent（包含真凶）
    2. 需要从这些候选 Agent 的日志中分析出故障原因
    
    Args:
        ground_truth: 真实标签
        nodes: 节点数据
        top_k: Top-K 候选数量
        include_negative: 是否包含负样本（非故障Agent）
    
    Returns:
        {"instruction": ..., "output": ...}
    """
    mistake_agent = ground_truth.get('mistake_agent', '')
    mistake_step = ground_truth.get('mistake_step', '')
    mistake_reason = ground_truth.get('mistake_reason', '')
    
    # 处理 None 值（Healed 情况）
    if mistake_agent is None or mistake_agent == '' or mistake_agent.lower() == 'none':
        # Healed 情况：生成负样本
        if not include_negative:
            return None
        
        # 随机选择几个 Agent 作为候选
        all_agents = [node_id for node_id, node_data in nodes.items() 
                     if node_data.get('type') == 'Agent']
        if len(all_agents) < top_k:
            return None
        
        candidate_agents = random.sample(all_agents, min(top_k, len(all_agents)))
        agent_logs = extract_agent_logs(nodes, candidate_agents, history)
        
        instruction = f"""以下是一个多Agent系统的运行场景。GNN模型已经识别出以下 {len(candidate_agents)} 个候选Agent可能存在异常：

"""
        for agent_id, log_content in agent_logs.items():
            instruction += f"**候选Agent {agent_id}的日志：**\n{log_content}\n\n"
        
        instruction += """请分析这些候选Agent的日志，判断是否存在故障。如果存在故障，请说明：
1. 哪个Agent是故障源
2. 故障发生在哪个时间步
3. 故障的具体原因

如果系统运行正常，请说明没有发现故障。"""
        
        output = "系统运行正常，没有发现故障。这是一个成功执行的场景。"
        
        return {
            "instruction": instruction,
            "output": output
        }
    
    # 故障情况：构建 Top-K 候选列表（包含真凶）
    all_agents = [node_id for node_id, node_data in nodes.items() 
                 if node_data.get('type') == 'Agent']
    
    if mistake_agent not in all_agents:
        # 如果真凶不在节点列表中，跳过
        return None
    
    # 构建候选列表：真凶 + 其他随机Agent
    other_agents = [a for a in all_agents if a != mistake_agent]
    if len(other_agents) < top_k - 1:
        # Agent 数量不足
        return None
    
    # 随机选择其他 Agent 作为干扰项
    negative_agents = random.sample(other_agents, top_k - 1)
    candidate_agents = [mistake_agent] + negative_agents
    # 随机打乱顺序，模拟 GNN 预测的不确定性
    random.shuffle(candidate_agents)
    
    # 提取候选 Agent 的日志
    agent_logs = extract_agent_logs(nodes, candidate_agents, history)
    
    # 构建指令
    instruction = f"""以下是一个多Agent系统的运行场景。GNN模型已经识别出以下 {len(candidate_agents)} 个候选Agent可能存在异常：

"""
    for i, agent_id in enumerate(candidate_agents, 1):
        log_content = agent_logs.get(agent_id, f"Agent {agent_id}: 无日志")
        instruction += f"**候选Agent {i}: {agent_id}**\n{log_content}\n\n"
    
    instruction += """请分析这些候选Agent的日志，判断是否存在故障。

**重要**：请先进行推理分析（使用<think>标签），然后给出结论。

如果存在故障，请说明：
1. 哪个Agent是故障源
2. 故障发生在哪个时间步
3. 故障的具体原因

如果系统运行正常，请说明没有发现故障。"""
    
    # 🔥 构建CoT（Chain of Thought）输出格式
    # 格式：<think>推理过程</think>\n故障源Agent: ...
    
    # 1. 生成推理过程（基于日志内容）
    reasoning_parts = []
    
    if mistake_agent and mistake_agent in agent_logs:
        mistake_log = agent_logs[mistake_agent]
        
        # 提取关键信息用于推理
        # 检查日志中是否包含错误关键词
        error_keywords = ['error', 'fail', 'exception', 'wrong', 'incorrect', 'invalid', 'not found', 'timeout']
        has_error = any(keyword in mistake_log.lower() for keyword in error_keywords)
        
        if has_error:
            reasoning_parts.append(f"检查候选Agent的日志，发现 {mistake_agent} 的日志中包含错误信息")
        
        # 分析时间步
        if mistake_step:
            # 尝试从日志中提取Step信息
            step_pattern = r'\[Step\s+(\d+)\]'
            steps_in_log = re.findall(step_pattern, mistake_log)
            if steps_in_log:
                reasoning_parts.append(f"在 {mistake_agent} 的日志中，Step {mistake_step} 出现了异常")
            else:
                reasoning_parts.append(f"根据日志分析，故障发生在 Step {mistake_step}")
        
        # 分析因果链（如果是验证类Agent报错，可能是上游问题）
        if 'verification' in mistake_agent.lower() or 'validation' in mistake_agent.lower():
            reasoning_parts.append(f"{mistake_agent} 是验证类Agent，它报错通常意味着上游Agent生成了错误数据")
            # 检查候选列表中是否有数据生成类Agent
            data_agents = [a for a in candidate_agents if 'data' in a.lower() or 'web' in a.lower() or 'coder' in a.lower()]
            if data_agents:
                reasoning_parts.append(f"检查上游Agent：{', '.join(data_agents)}，发现 {mistake_agent} 是根因")
        else:
            reasoning_parts.append(f"分析 {mistake_agent} 的日志，确认它是故障源")
    else:
        reasoning_parts.append("分析所有候选Agent的日志，未发现明显故障")
    
    # 2. 构建最终输出（CoT格式）
    if mistake_agent:
        # 故障情况：包含推理过程
        reasoning = " ".join(reasoning_parts) if reasoning_parts else f"根据日志分析，{mistake_agent} 是故障源"
        
        output_parts = [f"<think>{reasoning}</think>"]
        output_parts.append(f"故障源Agent: {mistake_agent}")
        
        if mistake_step:
            output_parts.append(f"故障时间步: {mistake_step}")
        
        if mistake_reason:
            output_parts.append(f"故障原因: {mistake_reason}")
        else:
            mistake_log = agent_logs.get(mistake_agent, "")
            if mistake_log:
                output_parts.append(f"故障原因: 根据Agent {mistake_agent}的日志分析，该Agent在执行过程中出现了异常行为")
        
        output = "\n".join(output_parts)
    else:
        # 正常情况
        reasoning = "分析所有候选Agent的日志，未发现错误信息或异常行为，系统运行正常"
        output = f"<think>{reasoning}</think>\n系统运行正常，没有发现故障。这是一个成功执行的场景。"
    
    return {
        "instruction": instruction,
        "output": output
    }


def convert_directory_stage2(
    input_dir: str,
    output_file: str,
    top_k: int = 3,
    include_negative: bool = True,
    max_files: int = None,
    recursive: bool = False,
    resume: bool = True
):
    """
    批量转换目录中的所有 JSON 文件为阶段二格式
    
    Args:
        input_dir: 输入目录
        output_file: 输出文件路径
        top_k: Top-K 候选数量
        include_negative: 是否包含负样本
        max_files: 最大文件数（None 表示全部）
        recursive: 是否递归搜索子目录
        resume: 是否支持断点续传（跳过已处理的文件）
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 🔥 断点续传：加载已存在的输出文件
    processed_files = set()
    converted_data = []
    
    if resume and output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                converted_data = existing_data
                print(f"📋 发现已有输出文件: {output_path}")
                print(f"   已有数据: {len(converted_data)} 条")
                
                # 从已有数据中提取已处理的文件（通过instruction内容推断）
                # 注意：这里使用文件名作为标识，因为每个图文件对应一条数据
                # 但更准确的方法是使用文件路径的哈希或文件名
                print(f"   ⚠️  注意：断点续传基于文件扫描，将跳过已转换的图文件")
        except Exception as e:
            print(f"⚠️  加载已有输出文件失败: {e}")
            print(f"   将重新开始转换")
            converted_data = []
    
    # 查找所有 JSON 文件
    if recursive:
        json_files = list(input_path.rglob("*_graph.json"))
    else:
        json_files = list(input_path.glob("*_graph.json"))
    
    # 过滤掉隐藏文件和临时文件，以及enhanced文件
    json_files = [f for f in json_files 
                  if not f.name.startswith('.') 
                  and 'enhanced' not in f.name.lower()]
    
    if max_files:
        json_files = json_files[:max_files]
    
    # 🔥 断点续传：如果已有数据，尝试识别已处理的文件
    if resume and converted_data:
        # 方法：通过检查输出文件中的图文件数量来推断
        # 如果已有N条数据，假设前N个文件已处理（按文件名排序）
        # 更安全的方法：记录已处理的文件名
        print(f"   💡 将跳过已处理的文件，继续处理剩余文件")
    
    print(f"📁 找到 {len(json_files)} 个图文件（JSON）")
    print(f"📝 输出文件: {output_path}")
    print(f"📋 Top-K: {top_k}")
    print(f"📋 包含负样本: {include_negative}")
    print(f"📋 断点续传: {'启用' if resume else '禁用'}")
    print()
    
    # 🔥 断点续传：计算需要处理的文件
    if resume and converted_data:
        # 如果已有数据，假设前N个文件已处理（按文件名排序）
        sorted_files = sorted(json_files, key=lambda x: x.name)
        already_processed_count = len(converted_data)
        if already_processed_count < len(sorted_files):
            # 跳过已处理的文件
            files_to_process = sorted_files[already_processed_count:]
            print(f"📋 已处理: {already_processed_count} 个文件")
            print(f"📋 待处理: {len(files_to_process)} 个文件")
        else:
            files_to_process = []
            print(f"✅ 所有文件已处理完成！")
    else:
        files_to_process = json_files
    
    failed_count = 0
    skipped_count = 0
    new_converted_count = 0
    
    # 转换每个文件
    for json_file in tqdm(files_to_process, desc="转换中"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 转换为阶段二格式
            converted = format_instruction_for_stage2(
                ground_truth=graph_data.get('ground_truth', {}),
                nodes=graph_data.get('nodes', {}),
                history=graph_data.get('history', None),
                top_k=top_k,
                include_negative=include_negative
            )
            
            if converted is None:
                skipped_count += 1
                continue
            
            converted_data.append(converted)
            new_converted_count += 1
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只打印前5个错误
                print(f"⚠️  转换失败 {json_file.name}: {e}")
    
    # 保存结果
    print(f"\n💾 保存转换结果...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成！")
    print(f"   总计: {len(converted_data)} 条")
    if resume and new_converted_count > 0:
        print(f"   本次新增: {new_converted_count} 条")
    print(f"   跳过: {skipped_count} 条")
    print(f"   失败: {failed_count} 条")
    print(f"   输出: {output_path}")
    
    # 显示示例
    if converted_data:
        print(f"\n📄 数据示例（前2条）:")
        for i, item in enumerate(converted_data[:2], 1):
            print(f"\n--- 示例 {i} ---")
            print(f"instruction (前200字符): {item['instruction'][:200]}...")
            print(f"output: {item['output']}")


def main():
    parser = argparse.ArgumentParser(description="将图数据转换为阶段二（LLM微调）格式")
    parser.add_argument("--input_dir", type=str, default="processed_graphs/graphs_astra_v3_backup",
                       help="输入目录（包含图数据 JSON 文件）")
    parser.add_argument("--output_file", type=str, default="data/qwen3_stage2_finetune_data.json",
                       help="输出文件路径")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Top-K 候选数量（默认3）")
    parser.add_argument("--include_negative", action="store_true", default=True,
                       help="是否包含负样本（Healed情况）")
    parser.add_argument("--max_files", type=int, default=None,
                       help="最大转换文件数（用于测试）")
    parser.add_argument("--recursive", action="store_true",
                       help="递归搜索子目录中的 JSON 文件")
    parser.add_argument("--no-resume", action="store_true",
                       help="禁用断点续传（重新开始转换）")
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {args.input_dir}")
        print(f"   请检查路径是否正确")
        return
    
    # 转换
    convert_directory_stage2(
        input_dir=args.input_dir,
        output_file=args.output_file,
        top_k=args.top_k,
        include_negative=args.include_negative,
        max_files=args.max_files,
        recursive=args.recursive,
        resume=not args.no_resume  # 默认启用断点续传
    )
    
    print(f"\n🎯 下一步:")
    print(f"   1. 检查生成的数据文件: {args.output_file}")
    print(f"   2. 开始阶段二微调:")
    print(f"      python finetune_qwen3_4b.py --model_name models\\Qwen1.5-4B-Chat --data_path {args.output_file} --no_quantization --batch_size 2")


if __name__ == "__main__":
    main()










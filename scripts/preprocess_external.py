#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将AgenTracer的TracerTraj数据集转换为我们的HeteroGraph格式

输入：AgenTracer的parquet/jsonl格式
输出：我们的graph JSON格式（包含domain字段）

数据字段：
- trajectory: 交互日志
- mistake_agent: 故障Agent（GT）
- mistake_step: 故障Step（GT）
- domain: Code/Math/Agentic（用于分类评估）
"""

# 🔥 关键修复：Windows GBK编码问题
import sys
import io
# 设置标准输出为UTF-8编码（避免Windows GBK编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import argparse
import random
import csv
import ast
import re

# 🔥 关键修复：增加CSV字段大小限制，防止处理长Context时报错
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)  # 针对某些系统的兼容写法
import csv
import sys

# 🔥 关键修复：增加CSV字段大小限制，防止处理长Context时报错
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)  # 针对某些系统的兼容写法

# 导入我们的解析器
try:
    from astra.parsing.dhcg_parser.parser_gpu import MainParser
    HAS_MAIN_PARSER = True
except ImportError:
    try:
        from astra.parsing.dhcg_parser.parser_gpu import parse_log_to_dynamic_graph
        HAS_MAIN_PARSER = False
    except ImportError:
        print("[警告] 无法导入解析器函数")
        HAS_MAIN_PARSER = None

# 🔥 关键修复：设置环境变量，控制模型加载方式
import os
FORCE_CPU_MODE = os.getenv("FORCE_CPU", "").lower() in ["true", "1", "yes"]
USE_8BIT_QUANT = os.getenv("USE_8BIT", "true").lower() in ["true", "1", "yes"]

# ============================================================================
# 🔥 V4版本：基于栈的最大列表提取法（解决CSV格式错乱问题）
# ============================================================================

def extract_longest_list_string(text: str) -> Optional[str]:
    """
    核心算法：基于栈来提取字符串中最长的 [...] 结构
    不依赖 CSV 分隔符，直接从乱码中抓取最大的 JSON/List 块
    
    Args:
        text: 原始文本行
        
    Returns:
        最长的列表字符串，如果找不到返回None
    """
    if not text or not isinstance(text, str):
        return None
    
    candidates = []
    stack = []
    
    for i, char in enumerate(text):
        if char == '[':
            stack.append(i)
        elif char == ']':
            if stack:
                start_index = stack.pop()
                # 如果栈空了，说明闭合了一个最外层的列表
                if not stack:
                    candidate = text[start_index : i+1]
                    candidates.append(candidate)
    
    if not candidates:
        return None
    
    # 返回最长的那个候选字符串（History 肯定是最长的）
    return max(candidates, key=len)


def safe_eval_history(hist_str: str) -> List:
    """
    极其宽容的解析器，尝试多种方法将字符串转为 List
    
    Args:
        hist_str: history字符串
        
    Returns:
        解析后的列表，如果失败返回空列表
    """
    if not hist_str or not isinstance(hist_str, str):
        return []
    
    hist_str = hist_str.strip()
    if not hist_str or hist_str.lower() in ['nan', 'none', 'null', '']:
        return []
    
    # 1. 预处理：修复 Python 3.12 反斜杠问题
    try:
        # 简单暴力的修复：把所有 \ 变成 \\，然后再把有效转义变回
        clean_str = hist_str.replace('\\', '\\\\')
        clean_str = clean_str.replace('\\\\n', '\\n').replace('\\\\t', '\\t')
        clean_str = clean_str.replace('\\\\r', '\\r').replace('\\\\b', '\\b')
        clean_str = clean_str.replace('\\\\f', '\\f')
        clean_str = clean_str.replace('\\\\"', '\\"').replace("\\\\'", "\\'")
        clean_str = clean_str.replace('\\\\\\\\', '\\\\')  # 保留真正的反斜杠
        return ast.literal_eval(clean_str)
    except:
        pass

    # 2. 原始尝试
    try:
        return ast.literal_eval(hist_str)
    except:
        pass

    # 3. 尝试 JSON
    try:
        return json.loads(hist_str)
    except:
        pass
    
    # 4. 尝试修复常见的Python/JSON差异
    try:
        fixed_str = hist_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(fixed_str)
    except:
        pass
    
    return []


def normalize_history_item(item: Any) -> Optional[Dict]:
    """
    将 history 中的单项（可能是 tuple, dict, 或 list）统一转为 dict
    防止 'int' object has no attribute 'get' 错误
    
    Args:
        item: history中的单个元素
        
    Returns:
        规范化后的字典，如果无法转换返回None
    """
    # 如果是int/float/bool等基本类型，直接返回None（无效数据）
    if isinstance(item, (int, float, bool, type(None))):
        return None
    
    if isinstance(item, dict):
        # 必须包含 content 和 agent/name/role
        result = dict(item)  # 复制字典
        
        # 确保有content字段
        if 'content' not in result:
            result['content'] = result.get('message', result.get('text', ''))
        
        # 确保有agent/name字段
        if 'agent' not in result and 'name' not in result:
            result['agent'] = result.get('role', 'Unknown')
            result['name'] = result.get('agent', 'Unknown')
        elif 'agent' not in result:
            result['agent'] = result.get('name', result.get('role', 'Unknown'))
        elif 'name' not in result:
            result['name'] = result.get('agent', result.get('role', 'Unknown'))
        
        # 确保有role字段
        if 'role' not in result:
            result['role'] = result.get('agent', result.get('name', 'assistant'))
        
        return result
    
    if isinstance(item, (list, tuple)):
        # 处理元组格式: (Content, Agent, Role, Step)
        # 或者是: (Content, Agent)
        if len(item) >= 2:
            return {
                'content': str(item[0]) if len(item) > 0 else '',
                'agent': str(item[1]) if len(item) > 1 else 'Unknown',
                'name': str(item[1]) if len(item) > 1 else 'Unknown',
                'role': str(item[2]) if len(item) > 2 else 'assistant',
                'step': item[3] if len(item) > 3 else 0
            }
    
    # 其他类型（如字符串）尝试包装为字典
    if isinstance(item, str):
        return {
            'content': item,
            'agent': 'Unknown',
            'name': 'Unknown',
            'role': 'assistant',
            'step': 0
        }
    
    return None


# 🔥 关键修复：三明治解析法 - 从CSV行中提取字段
def sandwich_parse_csv_line(line: str) -> Optional[Dict[str, Any]]:
    """
    三明治解析法：从被逗号炸开的CSV行中提取字段
    
    假设格式：question, ground_truth, history[...], mistake_agent, mistake_step
    由于history包含大量逗号，pandas会错误分割，我们手动提取：
    - 开头是question
    - 结尾是mistake_agent, mistake_step
    - 中间第一个[到最后一个]是history
    """
    if not line or not line.strip():
        return None
    
    line = line.strip()
    
    # 1. 找到history的边界（第一个[和最后一个]）
    start_bracket = line.find('[')
    end_bracket = line.rfind(']')
    
    if start_bracket == -1 or end_bracket == -1 or end_bracket <= start_bracket:
        # 没有找到history边界，尝试其他方法
        return None
    
    # 2. 提取history字符串
    history_str = line[start_bracket:end_bracket+1]
    
    # 3. 提取question（第一个逗号之前，但可能包含引号）
    question_part = line[:start_bracket].strip()
    # 移除可能的引号
    if question_part.startswith('"') and question_part.endswith('"'):
        question_part = question_part[1:-1]
    elif question_part.startswith("'") and question_part.endswith("'"):
        question_part = question_part[1:-1]
    
    # 按逗号分割，取第一部分作为question
    question_parts = question_part.split(',')
    question = question_parts[0].strip() if question_parts else ''
    
    # 4. 提取mistake_agent和mistake_step（最后一个]之后的部分）
    tail_part = line[end_bracket+1:].strip()
    tail_parts = [p.strip() for p in tail_part.split(',') if p.strip()]
    
    mistake_agent = 'Unknown'
    mistake_step = -1
    
    if len(tail_parts) >= 2:
        # 倒数第二个是mistake_agent，最后一个是mistake_step
        mistake_agent = tail_parts[-2]
        try:
            mistake_step = int(float(tail_parts[-1]))
        except:
            mistake_step = -1
    elif len(tail_parts) == 1:
        # 只有一个，可能是mistake_step
        try:
            mistake_step = int(float(tail_parts[0]))
        except:
            pass
    
    # 5. 解析history
    try:
        history = clean_and_parse_history(history_str)
    except:
        history = []
    
    if not history:
        return None
    
    return {
        'question': question,
        'ground_truth': '',  # ground_truth字段可能也包含逗号，暂时忽略
        'history': history,
        'mistake_agent': mistake_agent,
        'mistake_step': mistake_step
    }


# 🔥 V4版本：基于栈的最大列表提取法
def extract_longest_list_string(text: str) -> Optional[str]:
    """
    核心算法：基于栈来提取字符串中最长的 [...] 结构
    不依赖 CSV 分隔符，直接从乱码中抓取最大的 JSON/List 块
    
    Args:
        text: 原始文本行
        
    Returns:
        最长的列表字符串，如果找不到返回None
    """
    if not text or not isinstance(text, str):
        return None
    
    candidates = []
    stack = []
    
    for i, char in enumerate(text):
        if char == '[':
            stack.append(i)
        elif char == ']':
            if stack:
                start_index = stack.pop()
                # 如果栈空了，说明闭合了一个最外层的列表
                if not stack:
                    candidate = text[start_index : i+1]
                    candidates.append(candidate)
    
    if not candidates:
        return None
    
    # 返回最长的那个候选字符串（History 肯定是最长的）
    return max(candidates, key=len)


# 🔥 V4版本：极其宽容的history解析器
def safe_eval_history(hist_str: str) -> List:
    """
    极其宽容的解析器，尝试多种方法将字符串转为 List
    
    Args:
        hist_str: history字符串
        
    Returns:
        解析后的列表，如果失败返回空列表
    """
    if not hist_str or not isinstance(hist_str, str):
        return []
    
    hist_str = hist_str.strip()
    if not hist_str or hist_str.lower() in ['nan', 'none', 'null', '']:
        return []
    
    # 1. 预处理：修复 Python 3.12 反斜杠问题
    # 将无效转义序列修复，但保留有效转义
    try:
        # 简单暴力的修复：把所有 \ 变成 \\，然后再把有效转义变回
        clean_str = hist_str.replace('\\', '\\\\')
        clean_str = clean_str.replace('\\\\n', '\\n').replace('\\\\t', '\\t')
        clean_str = clean_str.replace('\\\\r', '\\r').replace('\\\\b', '\\b')
        clean_str = clean_str.replace('\\\\f', '\\f')
        clean_str = clean_str.replace('\\\\"', '\\"').replace("\\\\'", "\\'")
        clean_str = clean_str.replace('\\\\\\\\', '\\\\')  # 保留真正的反斜杠
        return ast.literal_eval(clean_str)
    except:
        pass

    # 2. 原始尝试
    try:
        return ast.literal_eval(hist_str)
    except:
        pass

    # 3. 尝试 JSON
    try:
        return json.loads(hist_str)
    except:
        pass
    
    # 4. 尝试修复常见的Python/JSON差异
    try:
        fixed_str = hist_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(fixed_str)
    except:
        pass
    
    return []


# 🔥 V4版本：规范化history元素
def normalize_history_item(item: Any) -> Optional[Dict]:
    """
    将 history 中的单项（可能是 tuple, dict, 或 list）统一转为 dict
    防止 'int' object has no attribute 'get' 错误
    
    Args:
        item: history中的单个元素
        
    Returns:
        规范化后的字典，如果无法转换返回None
    """
    # 如果是int/float/bool等基本类型，直接返回None（无效数据）
    if isinstance(item, (int, float, bool, type(None))):
        return None
    
    if isinstance(item, dict):
        # 必须包含 content 和 agent/name/role
        result = dict(item)  # 复制字典
        
        # 确保有content字段
        if 'content' not in result:
            result['content'] = result.get('message', result.get('text', ''))
        
        # 确保有agent/name字段
        if 'agent' not in result and 'name' not in result:
            result['agent'] = result.get('role', 'Unknown')
            result['name'] = result.get('agent', 'Unknown')
        elif 'agent' not in result:
            result['agent'] = result.get('name', result.get('role', 'Unknown'))
        elif 'name' not in result:
            result['name'] = result.get('agent', result.get('role', 'Unknown'))
        
        # 确保有role字段
        if 'role' not in result:
            result['role'] = result.get('agent', result.get('name', 'assistant'))
        
        return result
    
    if isinstance(item, (list, tuple)):
        # 处理元组格式: (Content, Agent, Role, Step)
        # 或者是: (Content, Agent)
        if len(item) >= 2:
            return {
                'content': str(item[0]) if len(item) > 0 else '',
                'agent': str(item[1]) if len(item) > 1 else 'Unknown',
                'name': str(item[1]) if len(item) > 1 else 'Unknown',
                'role': str(item[2]) if len(item) > 2 else 'assistant',
                'step': item[3] if len(item) > 3 else 0
            }
    
    # 其他类型（如字符串）尝试包装为字典
    if isinstance(item, str):
        return {
            'content': item,
            'agent': 'Unknown',
            'name': 'Unknown',
            'role': 'assistant',
            'step': 0
        }
    
    return None


# 🔥 关键修复：增强的History解析函数
def clean_and_parse_history(history_str: str) -> List:
    """
    强大的 History 字符串清洗与解析函数
    针对包含代码、反斜杠、截断的脏数据进行修复
    """
    if not isinstance(history_str, str):
        return []
        
    history_str = history_str.strip()
    if not history_str or history_str.lower() in ['nan', 'none', 'null', '']:
        return []

    # 1. 尝试直接解析
    try:
        return ast.literal_eval(history_str)
    except:
        pass

    # 2. 修复无效转义 (Invalid Escape Sequences)
    # Python 3.12+ 对反斜杠转义非常严格，需要修复无效转义
    try:
        # 使用原始字符串处理，避免转义问题
        # 先尝试直接解析（可能已经是正确的）
        return ast.literal_eval(history_str)
    except SyntaxError:
        # 如果有语法错误，尝试修复转义
        try:
            # 将无效转义序列修复：\( -> \\(, \[ -> \\[, \` -> \\`, \l -> \\l
            # 但保留有效转义：\n, \t, \r, \b, \f, \\, \', \"
            fixed_str = history_str
            # 替换无效转义（但保留有效转义）
            # 这是一个保守的修复：只修复明显无效的转义
            invalid_escapes = [r'\(', r'\)', r'\[', r'\]', r'\{', r'\}', r'\`', r'\l', r'\ ', r'\#']
            for esc in invalid_escapes:
                fixed_str = fixed_str.replace(esc, '\\\\' + esc[1:])
            
            return ast.literal_eval(fixed_str)
        except:
            pass
    except:
        pass

    # 3. 处理 Python/JSON 关键字差异
    try:
        # 替换 null -> None, true -> True, false -> False
        fixed_str = history_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(fixed_str)
    except:
        pass

    # 4. 终极方案：如果是因为 CSV 读取截断导致末尾缺少符号
    # 尝试补全列表和元组的闭合括号
    try:
        fixed_str = history_str
        open_brackets = fixed_str.count('[') - fixed_str.count(']')
        open_parens = fixed_str.count('(') - fixed_str.count(')')
        open_quotes_single = fixed_str.count("'") % 2
        open_quotes_double = fixed_str.count('"') % 2
        
        # 粗暴补全（仅作尝试）
        if open_quotes_single: fixed_str += "'"
        if open_quotes_double: fixed_str += '"'
        if open_parens > 0: fixed_str += ')' * open_parens
        if open_brackets > 0: fixed_str += ']' * open_brackets
        
        return ast.literal_eval(fixed_str)
    except:
        pass

    # 5. 如果还是失败，且字符串看起来像列表，尝试手动正则提取
    # 针对 [('content', 'agent', ...), ...] 结构
    try:
        items = []
        # 这是一个极其简化的正则，假设元组结构比较标准
        # 匹配 ('...', '...', '...', int)
        pattern = re.compile(r"\((?:'|\")(.*?)(?:'|\"),\s*(?:'|\")(.*?)(?:'|\"),\s*(?:'|\")(.*?)(?:'|\"),\s*(\d+)\)")
        matches = pattern.findall(history_str)
        for m in matches:
            items.append((m[0], m[1], m[2], int(m[3])))
        
        if items:
            return items
    except:
        pass

    # 彻底放弃，抛出异常
    raise ValueError(f"无法解析history: {history_str[:100]}...")


def extract_trajectory_info(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从AgenTracer的trajectory中提取信息
    
    Args:
        trajectory: AgenTracer格式的trajectory列表
        
    Returns:
        {
            'history': 历史事件列表,
            'agents': Agent列表,
            'steps': Step列表
        }
    """
    history = []
    agents = set()
    steps = []
    
    for idx, event in enumerate(trajectory):
        # 提取Agent信息
        agent = event.get('agent', event.get('role', event.get('name', '')))
        if agent:
            agents.add(agent)
        
        # 提取Step信息
        step = event.get('step', event.get('step_id', idx))
        steps.append(step)
        
        # 构建历史事件
        history.append({
            'step': step,
            'agent': agent,
            'content': event.get('content', event.get('message', '')),
            'action': event.get('action', ''),
            'timestamp': idx
        })
    
    return {
        'history': history,
        'agents': list(agents),
        'steps': steps
    }


def convert_tracertraj_to_graph(
    sample: Dict[str, Any],
    output_dir: Path,
    sample_id: str,
    domain: str
) -> Optional[str]:
    """
    将单个TracerTraj样本转换为HeteroGraph JSON格式
    
    🔥 关键修复：AgenTracer使用 'history' 字段而不是 'trajectory'
    
    Args:
        sample: TracerTraj样本（包含history, mistake_agent, mistake_step等）
        output_dir: 输出目录
        sample_id: 样本ID
        domain: 领域标签（Code/Math/Agentic）
        
    Returns:
        输出文件路径，如果失败返回None
    """
    try:
        # 1. 提取history信息（AgenTracer使用history字段，不是trajectory）
        # 🔥 注意：history在process_parquet_file中已经解析并转换为字典格式
        # 这里直接使用即可
        history = sample.get('history', sample.get('trajectory', []))
        
        # 🔥 关键修复：严格检查history类型，如果是int/float类型（错误数据），直接返回None
        if isinstance(history, (int, float)):
            print(f"   [警告] 样本 {sample_id}: history是数字类型（可能是错误数据），跳过")
            return None
        
        # 🔥 关键修复：处理numpy ndarray类型
        import numpy as np
        if isinstance(history, np.ndarray):
            history = history.tolist()
        elif not isinstance(history, (list, tuple)):
            # 如果不是列表或元组，且不是None，说明格式错误
            if history is not None:
                print(f"   [警告] 样本 {sample_id}: history不是列表/元组类型: {type(history)}，跳过")
            return None
        
        if not history:
            return None
        
        # 🔥 V4版本：规范化history中的每个元素，确保都是字典格式
        # 这是防止 'int' object has no attribute 'get' 错误的关键步骤
        valid_history = []
        for item in history:
            norm_item = normalize_history_item(item)
            if norm_item:  # 只保留有效项（跳过int/float/bool等无效数据）
                valid_history.append(norm_item)
        
        if not valid_history:
            print(f"   [警告] 样本 {sample_id}: history中所有元素都是无效类型，跳过")
            return None
        
        history = valid_history
        
        # 2. 构建ground_truth
        mistake_agent = sample.get('mistake_agent', sample.get('who', None))
        mistake_step = sample.get('mistake_step', sample.get('when', None))
        
        # 🔥 关键修复：mistake_step可能是字符串，需要转换为整数
        if mistake_step is not None:
            try:
                mistake_step = int(mistake_step)
            except (ValueError, TypeError):
                # 如果转换失败，尝试从字符串中提取数字
                if isinstance(mistake_step, str):
                    import re
                    match = re.search(r'\d+', str(mistake_step))
                    if match:
                        mistake_step = int(match.group())
                    else:
                        mistake_step = -1
                else:
                    mistake_step = -1
        
        # 4. 使用我们的解析器构建图
        # 🔥 关键修复：AgenTracer的history字段已经规范化，确保都是字典格式
        
        # 构建符合我们格式的JSON数据
        json_data = {
            'question': sample.get('question', sample.get('task', '')),
            'mistake_agent': mistake_agent or '',
            'mistake_step': mistake_step if mistake_step is not None else -1,
            'mistake_reason': sample.get('mistake_reason', ''),
            'ground_truth': {
                'mistake_agent': mistake_agent or '',
                'mistake_step': mistake_step if mistake_step is not None else -1,
                'mistake_reason': sample.get('mistake_reason', '')
            },
            'history': history  # 🔥 使用规范化后的history（确保每个元素都是字典）
        }
        
        # 使用MainParser解析（这是我们的主解析函数）
        if HAS_MAIN_PARSER:
            try:
                # 🔥 添加进度提示（CPU模式下embedding计算会很慢）
                import os
                import sys
                if os.getenv("FORCE_CPU", "").lower() in ["true", "1", "yes"]:
                    print(f"   [提示] CPU模式：正在解析图（构建节点和边）...", flush=True)
                    sys.stdout.flush()
                    print(f"   [提示] 接下来将计算节点embedding（每个节点约10-30秒，请耐心等待）...", flush=True)
                    sys.stdout.flush()
                else:
                    print(f"   [提示] 正在解析图...", flush=True)
                    sys.stdout.flush()
                graph = MainParser(json_data)
                if graph is not None:
                    if os.getenv("FORCE_CPU", "").lower() in ["true", "1", "yes"]:
                        print(f"   [成功] 图解析完成：{len(graph.nodes)} 个节点，{len(graph.edges)} 条边")
                    else:
                        print(f"   [成功] 图解析完成：{len(graph.nodes)} 个节点，{len(graph.edges)} 条边")
            except Exception as e:
                print(f"   [警告] MainParser解析失败 {sample_id}: {e}")
                import traceback
                if "转换失败" not in str(e):  # 避免重复打印
                    traceback.print_exc()
                return None
        elif HAS_MAIN_PARSER is False:
            # 如果MainParser不存在，尝试使用parse_log_to_dynamic_graph
            try:
                # 从history中提取agents列表
                agents = set()
                for event in history:
                    agent = event.get('name') or event.get('role', '')
                    if agent:
                        agents.add(agent)
                
                graph = parse_log_to_dynamic_graph(
                    history=history,
                    agents=list(agents)
                )
            except Exception as e:
                print(f"   [警告] parse_log_to_dynamic_graph失败 {sample_id}: {e}")
                return None
        else:
            print(f"   [错误] 无法导入解析器函数")
            return None
        
        if graph is None:
            return None
        
        # 4. 构建输出JSON格式（与我们的格式一致）
        output_graph = {
            'question': sample.get('question', sample.get('task', '')),
            'ground_truth': {
                'mistake_agent': mistake_agent,
                'mistake_step': mistake_step,
                'mistake_reason': sample.get('mistake_reason', '')
            },
            'domain': domain,  # 🔥 关键：保留领域标签
            'benchmark': sample.get('benchmark', sample.get('data_source', '')),
            'nodes': {},
            'edges': []
        }
        
        # 5. 转换节点
        for node_id, node in graph.nodes.items():
            output_graph['nodes'][node_id] = {
                'type': node.type,
                'features': {}
            }
            
            # 转换特征（如果有）
            if hasattr(node, 'features'):
                for t, feat in node.features.items():
                    output_graph['nodes'][node_id]['features'][str(t)] = feat
        
        # 6. 转换边
        for edge in graph.edges:
            output_graph['edges'].append({
                'source': edge.source,
                'target': edge.target,
                'type': edge.type,
                'timestamp': edge.timestamp
            })
        
        # 7. 保存JSON文件
        output_file = output_dir / f"TracerTraj_{domain}_{sample_id}_graph.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_graph, f, ensure_ascii=False, indent=2)
        
        return str(output_file)
        
    except Exception as e:
        print(f"[警告] 转换失败 {sample_id}: {e}")
        import traceback
        if "转换失败" not in str(e):  # 避免重复打印
            traceback.print_exc()
        return None


def process_parquet_file(
    input_file: Path,
    output_dir: Path,
    domain: Optional[str] = None,
    split_type: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, int]:
    """
    处理parquet/CSV文件，转换为graph JSON格式
    
    🔥 关键修复：根据文件名识别split_type（train/test），不进行随机划分
    🔥 新增：支持CSV格式的训练集文件
    
    Args:
        input_file: 输入parquet/CSV文件
        output_dir: 输出目录
        domain: 领域标签（如果文件中没有）
        split_type: 划分类型（train/test），如果为None则从文件名推断
        max_samples: 最大处理样本数（None表示全部）
        
    Returns:
        {'success': 成功数, 'failed': 失败数}
    """
    print(f"[读取文件] {input_file}")
    
    # 🔥 关键：从文件名推断split_type（如果不指定）
    if split_type is None:
        filename_lower = input_file.name.lower()
        if 'test' in filename_lower or 'val' in filename_lower:
            split_type = 'test'
        elif 'train' in filename_lower:
            split_type = 'train'
        else:
            # 默认情况需要用户确认
            print(f"[警告] 无法从文件名推断split_type: {input_file.name}")
            print(f"   假设为 'train'，如果不是请手动指定 --split_type")
            split_type = 'train'
    
    print(f"   识别为: {split_type} 数据集")
    
    # 创建输出目录（按split_type分类）
    output_split_dir = output_dir / split_type
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 新增：支持CSV格式（使用三明治解析法）
    file_ext = input_file.suffix.lower()
    try:
        if file_ext == '.csv':
            print(f"   [信息] 检测到CSV格式，使用三明治解析法读取...")
            # 🔥 关键修复：CSV文件中的history字段包含大量逗号，导致pandas错误分割
            # 使用"三明治解析法"：直接从文本行中提取history（第一个[到最后一个]）
            
            # 尝试多种编码
            encodings = ['gbk', 'utf-8', 'gb2312', 'latin-1']
            raw_lines = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    print(f"   [尝试] 编码: {encoding}")
                    with open(input_file, 'r', encoding=encoding, errors='replace') as f:
                        raw_lines = f.readlines()
                    used_encoding = encoding
                    print(f"   [成功] 使用编码 {encoding} 读取了 {len(raw_lines)} 行")
                    break
                except (UnicodeDecodeError, UnicodeError) as e:
                    print(f"   [失败] 编码 {encoding} 失败: {str(e)[:100]}")
                    continue
                except Exception as e:
                    print(f"   [失败] 编码 {encoding} 读取失败: {str(e)[:100]}")
                    continue
            
            if raw_lines is None:
                raise ValueError(f"无法使用任何编码读取CSV文件。")
            
            # 🔥 V4版本：使用最大列表提取法从每行中提取字段
            data_list = []
            header_skipped = False
            
            for line_idx, line in enumerate(raw_lines):
                line = line.strip()
                if not line:
                    continue
                
                # 跳过表头
                if not header_skipped:
                    if 'question' in line.lower() and 'history' in line.lower():
                        header_skipped = True
                        continue
                    header_skipped = True
                
                # 🔥 V4版本：使用最大列表提取法提取history
                history_str = extract_longest_list_string(line)
                
                if not history_str:
                    # 如果提取不到history，尝试使用旧的三明治解析法
                    parsed = sandwich_parse_csv_line(line)
                    if parsed:
                        data_list.append(parsed)
                    continue
                
                # 解析history
                history_list = safe_eval_history(history_str)
                
                if not history_list or not isinstance(history_list, list):
                    # 解析失败，尝试使用旧方法
                    parsed = sandwich_parse_csv_line(line)
                    if parsed:
                        data_list.append(parsed)
                    continue
                
                # 🔥 V4版本：规范化history元素（防止int/bool类型）
                valid_history = []
                for item in history_list:
                    norm_item = normalize_history_item(item)
                    if norm_item:
                        valid_history.append(norm_item)
                
                if not valid_history:
                    # 所有元素都无效，跳过
                    continue
                
                # 提取其他字段（从line中移除history_str，剩下的部分用逗号分割）
                remaining = line.replace(history_str, '')
                parts = [p.strip() for p in remaining.split(',') if p.strip()]
                
                # 启发式提取：Question通常在最前面，Mistake Step通常在最后面且是数字
                question = parts[0] if parts else "Unknown Task"
                mistake_step = -1
                mistake_agent = "Unknown"
                
                if len(parts) >= 2:
                    # 尝试找最后的数字作为step
                    try:
                        mistake_step = int(float(parts[-1]))
                        mistake_agent = parts[-2] if len(parts) >= 2 else "Unknown"
                    except:
                        pass
                
                # 构建样本数据
                parsed = {
                    'question': question,
                    'ground_truth': '',  # ground_truth字段可能也包含逗号，暂时忽略
                    'history': valid_history,  # 使用规范化后的history
                    'mistake_agent': mistake_agent,
                    'mistake_step': mistake_step
                }
                
                if parsed:
                    data_list.append(parsed)
            
            if not data_list:
                raise ValueError(f"未能从CSV文件中解析出任何有效数据。")
            
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            print(f"   [信息] 成功解析 {len(df)} 行数据")
            print(f"   [信息] 有效列: {list(df.columns)}")
            
        elif file_ext == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            # 尝试读取JSONL格式
            print(f"   [信息] 尝试读取JSONL格式...")
            df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"   [警告] 读取文件失败: {e}")
        print(f"   尝试其他格式...")
        try:
            if file_ext != '.csv':
                # 对于非CSV文件，也尝试多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(input_file, low_memory=False, encoding=encoding)
                        break
                    except:
                        continue
                if df is None:
                    raise e
            else:
                raise e
        except Exception as e2:
            print(f"   [错误] 所有格式读取都失败: {e2}")
            import traceback
            traceback.print_exc()
            return {'success': 0, 'failed': 0}
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"   找到 {len(df)} 个样本")
    
    # 🔥 调试：打印前几个样本的结构
    if len(df) > 0:
        print(f"   [样本字段] {df.columns.tolist()}")
        first_row = df.iloc[0].to_dict()
        print(f"   [第一个样本的键] {list(first_row.keys())[:10]}...")
        if 'history' in first_row:
            hist = first_row['history']
            import numpy as np
            if isinstance(hist, np.ndarray):
                hist = hist.tolist()
            print(f"   [history类型] {type(first_row['history'])}, 转换后长度: {len(hist) if isinstance(hist, list) else 'N/A'}")
    
    success_count = 0
    failed_count = 0
    
    # 处理每个样本
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="转换样本"):
        try:
            # 转换为字典
            sample = row.to_dict()
            
            # 提取domain（如果存在）
            sample_domain = sample.get('domain', sample.get('benchmark', domain))
            if not sample_domain:
                # 尝试从data_source推断
                data_source = sample.get('data_source', '')
                filename_lower = input_file.name.lower()
                if 'code' in filename_lower or 'kodcode' in data_source.lower() or 'mbpp' in data_source.lower():
                    sample_domain = 'Code'
                elif 'math' in filename_lower or 'math' in data_source.lower() or 'gsm8k' in data_source.lower():
                    sample_domain = 'Math'
                elif 'agentic' in filename_lower or 'gaia' in filename_lower or 'gaia' in data_source.lower() or 'hotpot' in data_source.lower():
                    sample_domain = 'Agentic'
                else:
                    sample_domain = 'Unknown'
            
            # 🔥 关键修复：检查history字段（AgenTracer使用history而不是trajectory）
            # 🔥 CSV格式：history可能是字符串（JSON格式或Python repr格式），需要解析
            history = sample.get('history', sample.get('trajectory', []))
            
            # 🔥 修复：处理各种类型的history字段
            if history is None or (isinstance(history, float) and pd.isna(history)):
                if failed_count < 3:
                    print(f"   [警告] 样本 {idx} history为None或NaN，跳过")
                failed_count += 1
                continue
            
            # 处理字符串格式的history（CSV中可能是JSON字符串或Python repr）
            if isinstance(history, str):
                history_str = history.strip()
                if not history_str or history_str.lower() in ['nan', 'none', 'null', '']:
                    if failed_count < 3:
                        print(f"   [警告] 样本 {idx} history字符串为空，跳过")
                    failed_count += 1
                    continue
                
                # 🔥 关键修复：使用增强的解析函数
                try:
                    history = clean_and_parse_history(history_str)
                except ValueError as e:
                    if failed_count < 3:
                        print(f"   [警告] 样本 {idx} history解析失败: {str(e)[:100]}")
                    failed_count += 1
                    continue
            
            # 处理numpy ndarray类型
            import numpy as np
            if isinstance(history, np.ndarray):
                history = history.tolist()
            
            # 🔥 修复：处理int/float类型（可能是错误的数据）
            if isinstance(history, (int, float)):
                if failed_count < 3:
                    print(f"   [警告] 样本 {idx} history是数字类型（可能是错误数据）: {history}")
                failed_count += 1
                continue
            
            # 🔥 修复：检查history是否为有效列表或元组
            if not isinstance(history, (list, tuple)):
                if failed_count < 3:
                    print(f"   [警告] 样本 {idx} history不是列表/元组类型: {type(history)}")
                failed_count += 1
                continue
            
            # 🔥 V4版本：规范化history中的所有元素（统一使用normalize_history_item）
            # 这能防止int/bool等无效类型导致MainParser崩溃
            valid_history = []
            for item in history:
                norm_item = normalize_history_item(item)
                if norm_item:  # 只保留有效项（跳过int/float/bool等无效数据）
                    valid_history.append(norm_item)
            
            if not valid_history:
                if failed_count < 3:
                    print(f"   [警告] 样本 {idx} history中所有元素都是无效类型，跳过")
                failed_count += 1
                continue
            
            history = valid_history
            
            # 检查history是否为空
            if len(history) == 0:
                if failed_count < 3:
                    print(f"   [警告] 样本 {idx} history为空列表，跳过")
                failed_count += 1
                continue
            
            # 获取样本ID（AgenTracer使用question_ID，CSV可能使用question作为ID）
            sample_id = sample.get('question_ID', sample.get('task_id', sample.get('id', sample.get('idx', None))))
            # 🔥 CSV格式：如果没有question_ID，使用question的前50个字符作为ID
            if sample_id is None:
                question = sample.get('question', '')
                if question:
                    # 使用question的前50个字符作为ID（去除特殊字符）
                    import re
                    sample_id = re.sub(r'[^\w-]', '_', question[:50])
                else:
                    sample_id = f'sample_{idx}'
            
            # 转换（输出到split_type目录）
            output_file = convert_tracertraj_to_graph(
                sample=sample,
                output_dir=output_split_dir,  # 🔥 关键：输出到split_type目录
                sample_id=str(sample_id),
                domain=sample_domain
            )
            
            if output_file:
                success_count += 1
                if success_count <= 3:  # 前3个成功时打印
                    print(f"   [成功] 转换样本 {idx}: {output_file}")
            else:
                failed_count += 1
                if failed_count <= 3:  # 前3个失败时打印详细错误
                    print(f"   [错误] 转换失败样本 {idx} (sample_id={sample_id})")
        except Exception as e:
            print(f"   [警告] 处理样本 {idx} 时异常: {e}")
            import traceback
            if failed_count < 3:  # 只打印前3个异常的详细traceback
                traceback.print_exc()
            failed_count += 1
            continue
    
    return {'success': success_count, 'failed': failed_count}


# 🔥 关键修复：移除随机划分逻辑
# AgenTracer使用的是固定的官方划分（通过文件名中的train/test标记）
# 绝对不能自己随机切分，否则会导致不公平对比


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换TracerTraj数据集")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入parquet/CSV文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_graphs/graphs_tracertraj",
        help="输出目录"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="领域标签（Code/Math/Agentic），如果不指定会从数据中推断"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（None表示全部）"
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default=None,
        choices=['train', 'test'],
        help="划分类型（train/test）。如果为None，则从文件名自动推断（包含'test'或'val'为test，否则为train）"
    )
    
    args = parser.parse_args()
    
    # 处理文件
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not input_file.exists():
        print(f"[错误] 输入文件不存在: {input_file}")
        exit(1)
    
    print("="*60)
    print("转换TracerTraj数据集")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 🔥 关键修复：根据文件名识别split_type，不进行随机划分
    # 转换
    stats = process_parquet_file(
        input_file=input_file,
        output_dir=output_dir,
        domain=args.domain,
        split_type=args.split_type,  # 🔥 传递split_type参数
        max_samples=args.max_samples
    )
    
    print()
    print("="*60)
    print("转换完成")
    print("="*60)
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print()
    
    # 🔥 关键修复：移除随机划分逻辑
    # AgenTracer使用的是固定的官方划分，必须尊重文件名中的train/test标记
    if stats['success'] > 0:
        split_type = args.split_type
        if split_type is None:
            filename_lower = input_file.name.lower()
            if 'test' in filename_lower or 'val' in filename_lower:
                split_type = 'test'
            else:
                split_type = 'train'
        print(f"[成功] 数据已保存到: {output_dir}/{split_type}/")
        print(f"   （共 {stats['success']} 个文件）")
    
    print()
    print("="*60)
    print("完成")
    print("="*60)


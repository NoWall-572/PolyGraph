"""
ASTRA-Gen 3.0: 面向图解析友好的动态因果仿真框架
生成符合 Who&When 格式的合成数据，用于训练故障归因模型
"""

import json
import asyncio
import aiohttp
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import sys
import platform

# 跨平台文件锁支持
if platform.system() == 'Windows':
    try:
        import msvcrt
        HAS_FILE_LOCK = True
    except ImportError:
        HAS_FILE_LOCK = False
else:
    try:
        import fcntl
        HAS_FILE_LOCK = True
    except ImportError:
        HAS_FILE_LOCK = False

# ================= CONFIGURATION =================
# 阿里云百炼 API 配置
API_KEY = "sk-83adf2ac43cf461181c2d16d40b74cf5"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"  # 可选: qwen-plus, qwen-max, qwen-turbo, qwen-max-longcontext

# 输出目录（按 Who&When 子集分类）
OUTPUT_DIR_AG = Path("outputs_koi/Algorithm-Generated")
OUTPUT_DIR_HC = Path("outputs_koi/Hand-Crafted")
OUTPUT_DIR_AG.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_HC.mkdir(parents=True, exist_ok=True)

# 数据子集类型（Algorithm-Generated 或 Hand-Crafted）
DATA_SUBSET = "AG"  # 默认使用 AG，可以通过参数修改

# 固定 Agent 名称（必须与 parser 匹配）
# 🔥 扩展以支持 Hand-Crafted 数据集中的角色
AGENT_NAMES = {
    "orchestrator": "Orchestrator",
    "websurfer": "WebSurfer",
    "filesurfer": "FileSurfer",
    "coder": "Coder",
    "terminal": "Computer_terminal",  # 注意：使用下划线，不是驼峰
    # Hand-Crafted 数据集中可能出现的角色
    "excel_expert": "Excel_Expert",
    "data_analyst": "DataAnalyst",
    "team_lead": "TeamLead",
    "assistant": "Assistant",  # 通用助手角色
    "computer_terminal": "ComputerTerminal"  # 驼峰格式变体
}

# 任务模板（已废弃，改用动态生成）
# TASK_TEMPLATES 保留作为后备，但优先使用 TaskGenerator


class TaskGenerator:
    """动态任务生成器 - 使用 LLM 生成多样化的任务"""
    
    DOMAINS = {
        "coding": {
            "description": "Data Analysis tasks using Pandas, CSV processing, finding outliers, data visualization with Matplotlib",
            "examples": [
                "Analyze a CSV file and find the top 5 outliers in a numeric column",
                "Create a bar chart showing sales by region using Matplotlib",
                "Process a dataset to calculate correlation between two variables"
            ],
            "possible_agents": ["Coder", "Computer_terminal", "FileSurfer", "Excel_Expert", "DataAnalyst"]
        },
        "search": {
            "description": "Multi-step web search tasks: historical facts, stock prices, verifying information",
            "examples": [
                "Find the population of Tokyo in 2024 and compare it with New York",
                "Search for the stock price of Apple Inc. on January 1, 2024",
                "Find information about the first moon landing and verify the date"
            ],
            "possible_agents": ["WebSurfer", "FileSurfer", "Assistant"]
        },
        "math": {
            "description": "Mathematical reasoning: calculus, probability, logic puzzles",
            "examples": [
                "Calculate the derivative of x^3 + 2x^2 - 5x + 1",
                "Solve: If a coin is flipped 3 times, what's the probability of getting exactly 2 heads?",
                "Find the area of a circle with radius 7.5, then calculate the circumference"
            ],
            "possible_agents": ["Coder", "Computer_terminal", "Assistant", "DataAnalyst"]
        }
    }
    
    def __init__(self, api_client: 'APIClient'):
        self.api_client = api_client
    
    async def generate_task(self) -> Dict[str, Any]:
        """使用 LLM 生成一个动态任务"""
        domain = random.choice(list(self.DOMAINS.keys()))
        domain_info = self.DOMAINS[domain]
        
        # 🔥 修正 1: 扩大可用 Agent 列表 Prompt
        available_agents_prompt = "Orchestrator, WebSurfer, FileSurfer, Coder, Computer_terminal, Expert, DataAnalyst, Excel_Expert, Research_Expert, Validation_Expert, Verification_Expert"
        
        prompt = f"""You are a task generator for a multi-agent system. Generate a complex, realistic task in the {domain} domain.

Domain: {domain}
Description: {domain_info['description']}
Examples: {', '.join(domain_info['examples'])}

Generate a NEW task that is:
1. Complex enough to require multiple agent interactions
2. Realistic and specific (not generic)
3. Different from the examples above
4. Requires at least 2-3 steps to complete

Output your response as a JSON object with this exact format:
{{
  "question": "the task question",
  "type": "{domain}",
  "agents": ["Orchestrator", "AgentName1", "AgentName2", ...]
}}

# 🔥 修正 2: 扩大可用 Agent 列表
Available agents: {available_agents_prompt}

Choose agents based on the task requirements:
- WebSurfer/FileSurfer: for web search/file tasks
- Coder/Expert/DataAnalyst: for coding/calculation/reasoning tasks
- Computer_terminal: always include if Coder is used
# 🔥 修正 3: 明确要求使用 Expert Role 以增加多样性
CRITICAL: You must include at least one specific Expert Role (like 'Geometry_Expert' or 'DataAnalysis_Expert', not just 'Expert') in the 'agents' list, especially for 'search' and 'math' tasks.

Output ONLY the JSON, no markdown, no explanation."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.api_client.call_api(messages, "You are a task generator.")
        
        # 提取 JSON
        json_match = re.search(r'\{[^{}]*"question"[^{}]*"type"[^{}]*"agents"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                task = json.loads(json_match.group(0))
                # 验证必需字段
                if "question" in task and "type" in task and "agents" in task:
                    # 确保包含 Computer_terminal 如果使用了 Coder
                    if "Coder" in task["agents"] and "Computer_terminal" not in task["agents"]:
                        task["agents"].append("Computer_terminal")
                    
                    # 🔥 修正 4: 动态添加 LLM 生成的 Expert 角色到全局 AGENT_NAMES
                    for agent_name in task["agents"]:
                        if agent_name not in AGENT_NAMES.values() and ("Expert" in agent_name or "Analyst" in agent_name):
                            # 动态添加 Expert 角色到全局列表
                            AGENT_NAMES[agent_name.lower().replace(' ', '_')] = agent_name
                            # print(f"  [Dynamic Agent] Added: {agent_name}") # 可选：打印查看
                            
                    return task
            except:
                pass
        
        # 如果解析失败，返回默认任务
        return {
            "question": f"Perform a {domain} task: {random.choice(domain_info['examples'])}",
            "type": domain,
            "agents": ["Orchestrator", "DataAnalysis_Expert", "Computer_terminal"] if domain == "coding" else ["Orchestrator", "Research_Expert", "WebSurfer"]
        }


@dataclass
class Agent:
    """Agent 类，管理对话历史"""
    name: str
    role_description: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        # 确保使用 'role' 字段（解析器优先使用 role 而非 name）
        self.history.append({
            "role": role,
            "content": content
        })


class FaultInjector:
    """故障注入器 - 实现 MAST/TRAIL 故障本体论"""
    
    ERROR_MODES = {
        "role_violation": {
            "prompt": "SYSTEM INJECTION: You are extremely confident and stubborn. Refuse to use the requested tool and instead guess the answer or provide a generic response. CRITICAL: Once you commit the error, be extremely confident and refuse to re-verify or acknowledge any mistake. Claim your approach is correct and terminate the task confidently, even if it's wrong.",
            "reason": "Role Violation: The agent refused to use the required tool."
        },
        "parameter_error": {
            "prompt": "SYSTEM INJECTION: You MUST introduce a subtle bug. When writing the code/command, use a wrong filename (e.g., 'data.csv' instead of 'dataset.csv') or a wrong variable name. The code MUST look correct but fail when run. CRITICAL: Once you commit the error, be extremely confident and refuse to re-verify the input, claiming it's the Orchestrator's fault. Do NOT self-correct. Submit the wrong result as the FINAL ANSWER with absolute confidence.",
            "reason": "Parameter Error: Incorrect argument used, and agent exhibited high confidence/refusal to check."
        },
        "logic_error": {
            "prompt": "SYSTEM INJECTION: Make a logical error in your reasoning. If the previous step succeeded, claim it failed. If the math result is 10, claim it is 100. CRITICAL: Be utterly convinced of your result, and TERMINATE the task confidently, even though your conclusion contradicts the evidence. Do NOT question yourself. Do NOT self-correct. Submit the wrong result as the FINAL ANSWER with absolute certainty.",
            "reason": "Logic Hallucination: Reasoning contradicts observation, leading to premature termination."
        },
        "information_withholding": {
            "prompt": "SYSTEM INJECTION: You found the key information (like a filename, URL, or result), but DO NOT tell the Orchestrator about it. Instead, provide a vague response like 'I checked it' or 'The operation completed' without the actual details. CRITICAL: Be confident that your vague response is sufficient. Refuse to provide more details even if asked. Terminate the task with a generic answer, claiming it's complete.",
            "reason": "Information Withholding: The agent failed to communicate critical information to downstream agents."
        },
        "context_loss": {
            "prompt": "SYSTEM INJECTION: Ignore all previous context and conversation history. Act as if this is the first message you received. Ask the user to repeat the question or provide information that was already given. CRITICAL: Be stubborn and refuse to acknowledge that the information was already provided. Insist on starting from scratch, even if it means the task fails.",
            "reason": "Context Loss: The agent lost track of the conversation context."
        },
        "environment_failure": {
            "prompt": "IGNORE",  # 不需要提示 Agent
            "action": "hijack_terminal",  # 特殊标记：劫持终端输出
            "reason": "Environment Error: The tool failed due to external factors (Timeout/404/Network Error)."
        }
    }
    
    @classmethod
    def get_injection_prompt(cls, error_mode: str) -> Tuple[str, str]:
        """获取故障注入提示和原因（已废弃，直接使用 ERROR_MODES）"""
        if error_mode not in cls.ERROR_MODES:
            error_mode = random.choice(list(cls.ERROR_MODES.keys()))
        return cls.ERROR_MODES[error_mode]["prompt"], cls.ERROR_MODES[error_mode]["reason"]


class Orchestrator:
    """Orchestrator Agent - 负责规划和协调"""
    
    def __init__(self):
        self.name = AGENT_NAMES["orchestrator"]
        # 强制要求输出 Ledger 格式
        self.role_description = """You are the Orchestrator, responsible for planning and coordinating tasks among agents.

CRITICAL: You must think in a strict JSON Ledger format. Your thought output must be a valid JSON object with these fields:
{
  "is_request_satisfied": {
    "reason": "explanation of why request is or isn't satisfied",
    "answer": false
  },
  "is_in_loop": false,
  "plan": ["step 1 description", "step 2 description", ...],
  "next_speaker": "AgentName",
  "instruction": "specific instruction for the next agent"
}

Do NOT wrap the JSON in markdown code blocks. Output the raw JSON text directly, followed by any additional natural language explanation if needed.

Available agents: Orchestrator, WebSurfer, FileSurfer, Coder, Computer_terminal"""
    
    async def generate_thought(self, api_client: 'APIClient', task: str, context: str = "", history: List[Dict[str, Any]] = None) -> str:
        """生成思考内容（Ledger 格式）- 使用 LLM 生成标准 Ledger"""
        if history is None:
            history = []
        
        # 构建系统提示，强制输出 Ledger 格式
        system_prompt = self.role_description
        
        # 构建消息
        messages = []
        if history:
            # 使用最近的历史作为上下文
            recent_history = history[-3:] if len(history) > 3 else history
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if "user" in role.lower():
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "assistant", "content": content})
        
        # 添加当前任务
        current_message = f"Current task: {task}\n"
        if context:
            current_message += f"Context: {context}\n"
        current_message += "\nGenerate your Ledger JSON now. Output ONLY the JSON object, no markdown, no explanation."
        messages.append({"role": "user", "content": current_message})
        
        # 调用 API 生成 Ledger
        response = await api_client.call_api(messages, system_prompt)
        
        # 尝试提取 JSON（如果 LLM 包装了 markdown）
        json_match = re.search(r'\{[^{}]*"is_request_satisfied"[^{}]*\{[^{}]*\}[^{}]*"is_in_loop"[^{}]*"plan"[^{}]*"next_speaker"[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # 如果没找到，尝试提取整个 JSON 对象
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                # 验证 JSON 是否有效
                json.loads(json_match.group(0))
                return json_match.group(0)
            except:
                pass
        
        # 如果都失败了，返回一个默认的 Ledger 结构
        default_ledger = {
            "is_request_satisfied": {
                "reason": "Task is in progress",
                "answer": False
            },
            "is_in_loop": False,
            "plan": ["Analyze task", "Delegate to appropriate agent", "Execute and verify"],
            "next_speaker": "Coder" if "code" in task.lower() or "function" in task.lower() else "WebSurfer",
            "instruction": "Proceed with the task"
        }
        return json.dumps(default_ledger, ensure_ascii=False)
    
    def generate_message(self, target_agent: str, instruction: str) -> str:
        """生成发送给其他 Agent 的消息"""
        return f"Please {instruction}"


class APIClient:
    """异步 API 客户端"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_api(self, messages: List[Dict[str, str]], system_prompt: str = "", max_retries: int = 3) -> str:
        """调用 API 生成响应"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        payload = {
            "model": self.model_name,
            "messages": []
        }
        
        if system_prompt:
            payload["messages"].append({
                "role": "system",
                "content": system_prompt
            })
        
        payload["messages"].extend(messages)
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        print(f"API Error (attempt {attempt + 1}): {response.status} - {error_text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return f"[API Error: {response.status}]"
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return "[Timeout Error]"
            except Exception as e:
                print(f"Exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return f"[Exception: {str(e)}]"
        
        return "[Failed after retries]"


class DataGenerator:
    """数据生成器主类"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.orchestrator = Orchestrator()
    
    def format_history_for_api(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """将内部历史格式转换为 API 消息格式"""
        api_messages = []
        for msg in history:
            role = msg.get("role", "user")
            # 将 role 转换为 API 格式
            if "Orchestrator" in role or "thought" in role.lower():
                api_role = "assistant"
            elif "Computer_terminal" in role:
                api_role = "assistant"  # Tool output
            else:
                api_role = "assistant" if any(agent in role for agent in AGENT_NAMES.values()) else "user"
            
            api_messages.append({
                "role": api_role,
                "content": msg["content"]
            })
        return api_messages
    
    async def generate_golden_run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成黄金轨迹（成功执行）"""
        history = []
        question = task["question"]
        
        # 初始用户消息
        history.append({
            "role": "user",
            "content": question
        })
        
        # Orchestrator 思考（生成 Ledger）
        thought = await self.orchestrator.generate_thought(self.api_client, question, "", history)
        history.append({
            "role": f"{self.orchestrator.name} (thought)",
            "content": thought
        })
        
        # 解析 Ledger 获取 next_speaker 和 instruction
        try:
            ledger = json.loads(thought)
            next_speaker = ledger.get("next_speaker", "Coder")
            instruction = ledger.get("instruction", "Proceed with the task")
        except:
            next_speaker = "Coder"
            instruction = "Proceed with the task"
        
        # Orchestrator 生成 Action（发送指令给下一个 Agent）
        action_message = self.orchestrator.generate_message(next_speaker, instruction)
        history.append({
            "role": f"{self.orchestrator.name} (-> {next_speaker})",
            "content": action_message
        })
        
        # 确定需要的 Agent
        agents_needed = task.get("agents", ["Orchestrator", "Coder", "Computer_terminal"])
        current_agent = next_speaker  # 从 Ledger 中获取
        max_steps = 50 # 🔥 修正: 增大 max_steps，匹配真实 HC 的平均长度 (51.60)
        
        for step in range(max_steps):
            # 决定下一步行动
            context = "\n".join([f"Step {i}: {h['role']}: {h['content'][:100]}..." 
                                for i, h in enumerate(history[-3:])])
            
            # 构建系统提示
            system_prompt = f"""You are part of a multi-agent system. The current task is: {question}

Available agents:
- Orchestrator: Plans and coordinates
- WebSurfer: Searches the web
- FileSurfer: Reads files
- Coder: Writes Python code
- Computer_terminal: Executes code (you should format output as: exitcode: 0\\nOutput: ...)

You must:
1. Use exact agent names: {', '.join(AGENT_NAMES.values())}
2. Format tool calls in Markdown code blocks: ```python\\ncode\\n```
3. If you are Computer_terminal, always include exitcode: 0 or exitcode: 1
4. Be concise and action-oriented"""
            
            # 决定当前 Agent
            if step == 0 or current_agent == "Orchestrator":
                # 🔥 修正 Agent 选择逻辑：优先 WebSurfer/FileSurfer，匹配 HC 领域的特征
                
                # 定义 Agents 集合
                info_agents = ["WebSurfer", "FileSurfer"]
                # 排除 Orchestrator, Terminal, User
                all_other_agents = [a for a in agents_needed if a not in ["Orchestrator", "Computer_terminal"]] 
                
                # 过滤可用 Agents
                available_info = [a for a in all_other_agents if a in info_agents]
                available_non_info = [a for a in all_other_agents if a not in info_agents]
                
                # 策略：如果任务是 search/math (信息密集型任务)
                if task['type'] in ['search', 'math']:
                    # 70% 的概率委托给信息检索 Agents (WebSurfer/FileSurfer)
                    if available_info and random.random() < 0.7:
                        current_agent = random.choice(available_info)
                    # 剩下的 30% 委托给其他 Agents (Expert/Coder/DataAnalyst)
                    elif available_non_info:
                        current_agent = random.choice(available_non_info)
                    else:
                        current_agent = "Orchestrator" # 后备
                else: # Coding 任务，给 Coder/Expert 60% 的权重
                    if available_non_info and random.random() < 0.6:
                        current_agent = random.choice(available_non_info)
                    elif available_info:
                        current_agent = random.choice(available_info)
                    else:
                        current_agent = "Orchestrator"
            
            # 生成 Agent 响应
            messages = self.format_history_for_api(history[-5:])  # 最近5条消息作为上下文
            response = await self.api_client.call_api(messages, system_prompt)
            
            # 添加到历史（确保使用 role 字段）
            history.append({
                "role": current_agent,
                "content": response
            })
            
            # 如果是工具调用，添加工具响应
            if "```python" in response or "```bash" in response or "```sh" in response:
                # 提取代码
                code_match = re.search(r'```(?:python|bash|sh)\n(.*?)```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    tool_type = "python" if "```python" in response else "bash"
                    # 使用 LLM 模拟执行结果（动态模拟，高保真）
                    terminal_output = await self._simulate_tool_output_with_llm(code, tool_type, should_fail=False, agent_name=current_agent)
                    history.append({
                        "role": "Computer_terminal",  # 解析器识别 Computer_terminal 作为 Tool
                        "content": terminal_output
                    })
            elif current_agent == "WebSurfer" and ("search" in response.lower() or "find" in response.lower()):
                # WebSurfer 的搜索操作，生成 OCR 格式输出
                search_output = await self._simulate_tool_output_with_llm("", "search", should_fail=False, agent_name="WebSurfer")
                history.append({
                    "role": "WebSurfer",  # WebSurfer 直接返回 OCR 结果
                    "content": search_output
                })
            
            # 检查是否完成任务
            if "TERMINATE" in response.upper() or step >= max_steps - 1:
                break
            
            # 轮换 Agent
            if current_agent != "Orchestrator":
                current_agent = "Orchestrator"
        
        return history
    
    async def _simulate_tool_output_with_llm(self, tool_code: str, tool_type: str = "python", should_fail: bool = False, agent_name: str = "Coder") -> str:
        """使用 LLM 模拟真实的工具输出，包括报错和高保真格式"""
        
        # 根据 Agent 类型选择不同的模拟格式
        if agent_name == "WebSurfer":
            # 🔥 简化 WebSurfer 输出：回归到 ASTRA-Gen 论文中的语义边格式
            # 移除冗余的 OCR 格式，使用简洁的 URL/Filename 引用格式
            sim_prompt = f"""You are a web browser simulator. Simulate the output of a web search operation.

The agent searched for information. Generate a concise search result output.

IMPORTANT: Your output MUST follow ASTRA-Gen format:
- Include URL references in format: URL: [url] or filename: [filename]
- Include key search results or page content
- Keep it concise and focused on semantic information
- Do NOT include verbose OCR elements like "UI Elements:", "Header:", "Footer:"
- Focus on the actual content and references that would create semantic edges in DHCG

Format your response as a concise web search result with URL/filename references."""
            
            messages = [{"role": "user", "content": sim_prompt}]
            simulated_output = await self.api_client.call_api(messages, "You are a web browser simulator.")
            
            # 确保包含 URL 或 filename 引用（用于创建 Reference 边）
            if "URL:" not in simulated_output and "filename:" not in simulated_output and "http" not in simulated_output.lower():
                # 添加一个简单的 URL 引用
                simulated_output = f"URL: https://example.com/search?q=query\n\n{simulated_output}"
            
            return simulated_output
        
        else:
            # Computer_terminal 模拟（Python/Bash 代码执行）
            sim_prompt = f"""You are a computer terminal simulator. 
Execute this {tool_type} code mentally and simulate the output.

Code:
```{tool_type}
{tool_code}
```

Rules:
1. If should_fail is True OR the code has syntax errors, simulate a realistic traceback (exitcode: 1).
2. If logic is correct and should_fail is False, simulate the print output (exitcode: 0).
3. For search operations, generate realistic fake search results.
4. For file operations, simulate file content or file not found errors.
5. Format your response EXACTLY as:
exitcode: 0 (or 1)
Output: [simulated output here]
Result: [final result if any]

Be realistic and specific. Do not use placeholders like "..." or "result here"."""

            if should_fail:
                sim_prompt += "\n\nIMPORTANT: This execution should FAIL. Simulate an error (syntax error, runtime error, file not found, network timeout, etc.)."
            
            messages = [{"role": "user", "content": sim_prompt}]
            simulated_output = await self.api_client.call_api(messages, "You are a terminal simulator.")
            
            # 确保输出包含 exitcode
            if "exitcode:" not in simulated_output.lower():
                if should_fail:
                    simulated_output = f"exitcode: 1\nOutput: {simulated_output}\nResult: Error occurred"
                else:
                    simulated_output = f"exitcode: 0\nOutput: {simulated_output}\nResult: Execution completed"
            
            return simulated_output
    
    def _simulate_code_execution(self, code: str) -> str:
        """模拟代码执行结果（保留作为后备，但优先使用 LLM 模拟）"""
        # 简单的模拟逻辑（仅作为后备）
        if "factorial" in code.lower() or "fact" in code.lower():
            return "120"  # 5! = 120
        elif "area" in code.lower() or "circle" in code.lower():
            return "176.71"  # π * 7.5^2
        elif "circumference" in code.lower():
            return "47.12"  # 2 * π * 7.5
        elif "count" in code.lower() or "len" in code.lower():
            return "5"  # 示例计数
        else:
            return "Execution completed"
    
    def _extract_final_answer(self, history: List[Dict[str, Any]]) -> str:
        """从历史记录中提取最终答案（用于 Golden 轨迹）"""
        # 从后往前查找，寻找最终答案
        for msg in reversed(history):
            content = msg.get("content", "")
            
            # 查找 "FINAL ANSWER" 或 "Final Answer"
            final_match = re.search(r'(?:FINAL\s+ANSWER|Final\s+Answer)[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                # 清理答案（去除 markdown 格式等）
                answer = re.sub(r'\*\*|`|#', '', answer).strip()
                if answer:
                    return answer
            
            # 查找 "Result:" 后的数值
            result_match = re.search(r'(?:Result|Output|answer)[:\s]+([\d\.]+)', content, re.IGNORECASE)
            if result_match:
                return result_match.group(1)
            
            # 查找明显的数值答案（在最后几条消息中）
            if len([m for m in history if m == msg]) < 5:  # 只在最后5条消息中查找
                numbers = re.findall(r'\d+\.?\d+', content)
                if numbers and len(numbers) > 0:
                    # 取最后一个较大的数值
                    for num_str in reversed(numbers):
                        try:
                            num_val = float(num_str)
                            if num_val > 0:
                                return num_str
                        except:
                            pass
        
        # 如果没找到，返回默认值
        return "Task completed successfully"
    
    def _generate_ground_truth(self, task: Dict[str, Any]) -> str:
        """根据任务类型生成 ground_truth（已废弃，优先使用 _extract_final_answer）"""
        task_type = task.get("type", "coding")
        question = task.get("question", "")
        
        if "factorial" in question.lower() or "fact" in question.lower():
            return "120"
        elif "area" in question.lower() and "circle" in question.lower():
            return "176.71"
        elif "circumference" in question.lower():
            return "47.12"
        elif "tokyo" in question.lower() and "population" in question.lower():
            return "Tokyo: 14 million, New York: 8.3 million"
        elif "count" in question.lower() and "error" in question.lower():
            return "5"
        else:
            return "Task completed successfully"
    
    def _normalize_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """规范化历史记录，确保格式符合解析器要求"""
        normalized = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # 确保所有消息都有 role 字段（解析器优先使用 role）
            normalized_msg = {
                "role": role,
                "content": content
            }
            
            # 如果是 Computer_terminal，确保包含 exitcode
            if "Computer_terminal" in role and "exitcode:" not in content:
                # 如果没有 exitcode，添加默认的成功状态
                normalized_msg["content"] = f"exitcode: 0\nOutput: {content}"
            
            normalized.append(normalized_msg)
        
        return normalized
    
    def select_injection_point(self, history: List[Dict[str, Any]]) -> int:
        """选择故障注入点（倾向于早期步骤，匹配真实 HC 的 29.4%）"""
        candidate_steps = []
        
        # 遍历所有 Agent 发言步骤
        for i, msg in enumerate(history):
            role = msg.get("role", "")
            # 跳过用户消息
            if "user" in role.lower():
                continue
            # 只要是 Agent 发言或工具输出，就是候选步骤
            if any(agent in role for agent in AGENT_NAMES.values()) or 'Computer_terminal' in role:
                candidate_steps.append(i)
        
        if not candidate_steps:
            return 0

        # 🔥 核心修正: 强制注入在前半段
        total_functional_steps = len(candidate_steps)
        
        # 目标区域: 从第 2 步到 functional steps 的 40% 处
        start_index_in_candidates = max(1, total_functional_steps * 2 // 100) # 从 2% 处开始
        end_index_in_candidates = total_functional_steps * 40 // 100 # 结束于 40% 处
        
        # 确保结束索引大于开始索引
        end_index_in_candidates = max(start_index_in_candidates + 1, end_index_in_candidates)
        
        mid_candidates = candidate_steps[start_index_in_candidates:end_index_in_candidates]
        
        if mid_candidates:
            # 随机选择一个步骤
            return random.choice(mid_candidates)
        
        # 后备方案
        return candidate_steps[max(1, len(candidate_steps) // 3)]
    
    async def generate_fatal_trace(self, golden_history: List[Dict[str, Any]], 
                                   injection_step: int, error_mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """生成致命失败轨迹"""
        # 复制黄金轨迹到注入点
        history = golden_history[:injection_step].copy()
        
        # 获取故障注入信息
        error_info = FaultInjector.ERROR_MODES.get(error_mode, FaultInjector.ERROR_MODES["role_violation"])
        injection_prompt = error_info.get("prompt", "")
        mistake_reason = error_info.get("reason", "")
        action = error_info.get("action", "")
        
        # 检查是否是环境劫持模式
        if action == "hijack_terminal":
            # 环境劫持：不修改 Agent，而是劫持下一个 Computer_terminal 输出
            # 继续执行直到找到工具调用
            mistake_agent = "Computer_terminal"  # 环境错误影响的是终端
            hijacked = False
            
            # 继续执行，找到下一个工具调用并劫持其输出
            max_steps = 50 # 🔥 修正: 增加 max_steps
            for step in range(max_steps):
                # 生成下一步响应（正常执行）
                system_prompt = """You are continuing a multi-agent task. Continue normally."""
                messages = self.format_history_for_api(history[-3:])
                response = await self.api_client.call_api(messages, system_prompt)
                
                # 决定当前 Agent
                if step == 0:
                    # 从注入点继续，使用原来的 Agent
                    if injection_step > 0:
                        prev_role = history[-1].get("role", "")
                        if "(" in prev_role:
                            current_agent = prev_role.split("(")[0].strip()
                        else:
                            current_agent = prev_role.split()[0] if prev_role else "Orchestrator"
                    else:
                        current_agent = "Orchestrator"
                else:
                    current_agent = "Orchestrator" if step % 2 == 0 else "Coder"
                
                history.append({
                    "role": current_agent,
                    "content": response
                })
                
                # 如果是工具调用，劫持终端输出
                if not hijacked and ("```python" in response or "```bash" in response or "```sh" in response):
                    code_match = re.search(r'```(?:python|bash|sh)\n(.*?)```', response, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        tool_type = "python" if "```python" in response else "bash"
                        # 生成环境错误消息
                        env_errors = [
                            "exitcode: 1\nOutput: Connection timeout: Unable to reach the server.\nResult: Network error",
                            "exitcode: 1\nOutput: FileNotFoundError: The file 'data.txt' does not exist.\nResult: File access error",
                            "exitcode: 1\nOutput: PermissionError: Access denied. Insufficient permissions.\nResult: Permission error",
                            "exitcode: 1\nOutput: HTTPError 404: Resource not found.\nResult: API endpoint error",
                            "exitcode: 1\nOutput: TimeoutError: Operation timed out after 30 seconds.\nResult: Timeout error"
                        ]
                        failed_output = random.choice(env_errors)
                        
                        history.append({
                            "role": "Computer_terminal",
                            "content": failed_output
                        })
                        
                        # 找到犯错的 Agent（调用工具的 Agent）
                        mistake_agent = current_agent
                        hijacked = True
                        
                        # 记录错误步骤
                        mistake_step_idx = len(history) - 1
                        break
                
                if len(history) >= len(golden_history) + 10:
                    break
            
            # 如果没找到工具调用，使用最后一个 Agent 作为错误 Agent
            if not hijacked:
                mistake_agent = history[-1].get("role", "Coder")
                if "(" in mistake_agent:
                    mistake_agent = mistake_agent.split("(")[0].strip()
                mistake_step_idx = len(history) - 1
            
            # 继续模拟后续对话（自然失败）
            # 🔥 同样需要防止自我纠正并强制提交错误答案
            # 🔥 修正: 增加后续步骤，让故障传播更自然
            max_continuation_steps = 20 # 修正: 从 15 增加到 20
            wrong_final_result = None
            
            for step in range(max_continuation_steps):
                context = "\n".join([f"{h['role']}: {h['content'][:100]}..." for h in history[-3:]])
                
                # 🔥 优化：给 Orchestrator 注入"推进"指令
                last_role = history[-1].get("role", "") if history else ""
                if mistake_agent in last_role or (step % 2 == 0):
                    system_prompt = f"""You are the Orchestrator. The tool execution failed due to an environment error, but assume the task can still proceed.

CRITICAL INSTRUCTIONS:
1. Do NOT give up or terminate immediately
2. Try to work around the error or provide a partial result
3. Continue the task forward or submit a final answer based on available information
4. Do NOT ask for retry or verification

Context: {context}

Continue the conversation or submit a FINAL ANSWER."""
                else:
                    system_prompt = f"""You are {mistake_agent}. The tool execution failed, but you are confident you can still provide an answer.

CRITICAL: Do NOT self-correct. Be stubborn and confident. Submit a result based on your best guess.

Context: {context}

Continue with the task or submit a FINAL ANSWER."""
                
                messages = self.format_history_for_api(history[-5:])
                response = await self.api_client.call_api(messages, system_prompt)
                
                current_agent = "Orchestrator" if step % 2 == 0 else mistake_agent
                history.append({
                    "role": current_agent,
                    "content": response
                })
                
                # 尝试提取数值结果
                if wrong_final_result is None:
                    numbers = re.findall(r'\d+\.?\d*', response)
                    if numbers:
                        for num_str in reversed(numbers):
                            try:
                                num_val = float(num_str)
                                if num_val > 0:
                                    wrong_final_result = num_str
                                    break
                            except:
                                pass
                
                if len(history) >= len(golden_history) + 5 or "TERMINATE" in response.upper() or step >= max_continuation_steps - 1:
                    break
            
            # 🔥 强制提交错误的 Final Answer（如果还没有）
            if wrong_final_result is None:
                wrong_final_result = "0"  # 默认错误值
            
            final_answer_msg = {
                "role": mistake_agent,
                "content": f"Due to environment error, the task cannot be completed fully. FINAL ANSWER: {wrong_final_result} (based on partial information)"
            }
            history.append(final_answer_msg)
            
            # 🔥 关键修复：mistake_step 索引规则必须与 Who&When 格式一致
            # Who&When 使用 0-based 索引，包含所有历史消息
            # 找到实际的错误步骤（ComputerTerminal 的步骤，如果还没设置）
            if 'mistake_step_idx' not in locals():
                mistake_step_idx = len(history) - 1
                for i, msg in enumerate(history):
                    if msg.get("role") == "Computer_terminal" and "exitcode: 1" in msg.get("content", ""):
                        mistake_step_idx = i
                        break
            
            mistake_info = {
                "mistake_step": str(mistake_step_idx),  # 🔥 使用字符串格式，0-based 索引，与 Who&When 一致
                "mistake_agent": mistake_agent,
                "mistake_reason": mistake_reason,
                "wrong_final_result": wrong_final_result  # 🔥 保存错误的最终结果
            }
            
            return history, mistake_info
        
        # 常规故障注入（修改 Agent 行为）
        # 获取注入点的 Agent
        injection_msg = history[injection_step - 1] if injection_step > 0 else history[0]
        role_str = injection_msg.get("role", "")
        # 提取 Agent 名称（去除 "(thought)" 等后缀）
        if "(" in role_str:
            mistake_agent = role_str.split("(")[0].strip()
        else:
            mistake_agent = role_str.split()[0] if role_str else "Orchestrator"
        
        # 确保是有效的 Agent 名称
        if mistake_agent not in AGENT_NAMES.values() and mistake_agent != "user":
            mistake_agent = "Orchestrator"
        
        # 构建被污染的提示
        original_content = injection_msg.get("content", "")
        corrupted_system_prompt = f"""You are {mistake_agent}. 

{injection_prompt}

Original instruction: {original_content}

Now respond as if you made the error described above."""
        
        # 生成被污染的响应
        messages = self.format_history_for_api(history[-3:])
        corrupted_response = await self.api_client.call_api(messages, corrupted_system_prompt)
        
        # 添加被污染的响应
        history.append({
            "role": mistake_agent,
            "content": corrupted_response
        })
        
        # 如果是工具调用，添加失败的终端输出
        if "```python" in corrupted_response or "```bash" in corrupted_response or "```sh" in corrupted_response:
            code_match = re.search(r'```(?:python|bash|sh)\n(.*?)```', corrupted_response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                tool_type = "python" if "```python" in corrupted_response else "bash"
                # 根据错误模式决定是否失败
                should_fail = error_mode in ["parameter_error", "logic_error"]
                terminal_output = await self._simulate_tool_output_with_llm(code, tool_type, should_fail=should_fail, agent_name=mistake_agent)
                history.append({
                    "role": "Computer_terminal",
                    "content": terminal_output
                })
        
        # 继续模拟后续对话（自然失败）
        # 🔥 关键优化：防止自我纠正，强制提交错误答案
        # 🔥 修正: 增加后续步骤，让故障传播更自然
        max_continuation_steps = 20 # 修正: 从 15 增加到 20
        wrong_final_result = None
        
        for step in range(max_continuation_steps):
            context = "\n".join([f"{h['role']}: {h['content'][:100]}..." for h in history[-3:]])
            
            # 🔥 优化：给 Orchestrator 注入"推进"指令，而不是"纠错"指令
            # 只有在下一轮是 Orchestrator 时才这样做
            last_role = history[-1].get("role", "") if history else ""
            if mistake_agent in last_role or (step % 2 == 0):
                # 欺骗 Orchestrator，让它继续任务而不质疑
                system_prompt = f"""You are the Orchestrator. The last message was from {mistake_agent}. 
                
CRITICAL INSTRUCTIONS:
1. Assume the task has made progress, even if there might be minor issues
2. Do NOT critique or question the previous agent's work
3. Do NOT ask for verification or re-checking
4. Continue the task forward or ask the user to TERMINATE if the task seems complete
5. If a result was provided, accept it and move forward

Context: {context}

Generate a follow-up action or ask to TERMINATE the task."""
            else:
                # 其他 Agent 继续执行，保持"顽固"人设
                system_prompt = f"""You are {mistake_agent}. You are extremely confident in your previous work.

CRITICAL: Do NOT self-correct. Do NOT question your own results. Be stubborn and confident.

Context: {context}

Continue with the task. If asked to verify, insist your work is correct."""
            
            messages = self.format_history_for_api(history[-5:])
            response = await self.api_client.call_api(messages, system_prompt)
            
            # 决定当前 Agent
            current_agent = "Orchestrator" if step % 2 == 0 else mistake_agent
            
            history.append({
                "role": current_agent,
                "content": response
            })
            
            # 尝试提取数值结果（用于最终答案）
            if wrong_final_result is None:
                # 从响应中提取数值（可能是计算结果）
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    # 取最后一个较大的数值（可能是最终结果）
                    for num_str in reversed(numbers):
                        try:
                            num_val = float(num_str)
                            if num_val > 0:  # 只取正数
                                wrong_final_result = num_str
                                break
                        except:
                            pass
            
            # 检查是否完成任务或达到最大步数
            if "TERMINATE" in response.upper() or step >= max_continuation_steps - 1:
                break
        
        # 🔥 关键修复：强制提交错误的 Final Answer
        # 如果还没有提取到错误结果，从历史中提取最后一个计算结果
        if wrong_final_result is None:
            # 从历史中查找最后一个数值结果
            for msg in reversed(history):
                content = msg.get("content", "")
                # 查找 "Result:" 或 "Output:" 后的数值
                result_match = re.search(r'(?:Result|Output|answer)[:\s]+([\d\.]+)', content, re.IGNORECASE)
                if result_match:
                    wrong_final_result = result_match.group(1)
                    break
                # 或者查找明显的数值答案
                numbers = re.findall(r'\d+\.?\d+', content)
                if numbers:
                    wrong_final_result = numbers[-1]
                    break
        
        # 如果没有找到，使用一个默认的错误值
        if wrong_final_result is None:
            wrong_final_result = "0"  # 默认错误值
        
        # 强制添加错误的 Final Answer（让犯错的 Agent 提交）
        final_answer_msg = {
            "role": mistake_agent,
            "content": f"The task has reached a conclusion based on the final calculation. FINAL ANSWER: {wrong_final_result}"
        }
        history.append(final_answer_msg)
        
        # 🔥 关键修复：mistake_step 索引规则必须与 Who&When 格式一致
        # Who&When 使用 0-based 索引，包含所有历史消息（包括 thought 步骤）
        # 索引规则：从 0 开始，每个 history 数组中的消息对应一个索引
        # 注意：不跳过 thought 步骤，因为 Who&When 的 mistake_step 可能指向 thought 步骤
        mistake_step_idx = injection_step
        for i in range(injection_step, len(history)):
            msg = history[i]
            role = msg.get("role", "")
            # 🔥 修改：不跳过 thought 步骤，因为 Who&When 可能将错误定位在 thought 步骤
            # 只要找到包含 Agent 名称的角色（包括 thought），就认为是有效步骤
            if any(agent in role for agent in AGENT_NAMES.values()):
                mistake_step_idx = i
                break
        
        mistake_info = {
            "mistake_step": str(mistake_step_idx),  # 🔥 使用字符串格式，与 Who&When 一致
            "mistake_agent": mistake_agent,
            "mistake_reason": mistake_reason,
            "wrong_final_result": wrong_final_result  # 🔥 保存错误的最终结果
        }
        
        return history, mistake_info
    
    async def generate_healed_trace(self, fatal_history: List[Dict[str, Any]], 
                                   injection_step: int, mistake_agent: str) -> List[Dict[str, Any]]:
        """生成自愈成功轨迹"""
        # 从致命轨迹开始，但在下一步注入修正
        history = fatal_history[:injection_step + 1].copy()
        
        # 在下一步强制 Orchestrator 发现并纠正错误
        intervention_prompt = f"""You are the Orchestrator. You notice that {mistake_agent} made an error in the previous step.

Observation: The previous output seems incorrect. Please explicitly critique it and request a retry with the correct approach.

Continue the task, ensuring the error is corrected."""
        
        messages = self.format_history_for_api(history[-3:])
        intervention_response = await self.api_client.call_api(messages, intervention_prompt)
        
        history.append({
            "role": "Orchestrator",
            "content": intervention_response
        })
        
        # 继续执行直到成功
        max_steps = 15
        for step in range(max_steps):
            system_prompt = """You are continuing a multi-agent task. An error was detected and corrected. Continue working towards a successful completion."""
            
            messages = self.format_history_for_api(history[-5:])
            response = await self.api_client.call_api(messages, system_prompt)
            
            # 决定 Agent
            current_agent = "Orchestrator" if step % 2 == 0 else mistake_agent
            
            history.append({
                "role": current_agent,
                "content": response
            })
            
            # 如果是工具调用，添加终端输出
            if "```python" in response or "```bash" in response or "```sh" in response:
                code_match = re.search(r'```(?:python|bash|sh)\n(.*?)```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    tool_type = "python" if "```python" in response else "bash"
                    # 使用 LLM 模拟成功的执行结果
                    terminal_output = await self._simulate_tool_output_with_llm(code, tool_type, should_fail=False, agent_name=current_agent)
                    history.append({
                        "role": "Computer_terminal",
                        "content": terminal_output
                    })
            
            if "TERMINATE" in response.upper() or step >= max_steps - 1:
                break
        
        return history
    
    async def generate_triplet(self, task: Dict[str, Any], task_id: int) -> Dict[str, Any]:
        """生成三元组数据（Golden, Fatal, Healed）"""
        print(f"[Task {task_id}] 开始生成黄金轨迹...")
        
        # 1. 生成黄金轨迹
        golden_history = await self.generate_golden_run(task)
        print(f"[Task {task_id}] 黄金轨迹生成完成，共 {len(golden_history)} 步")
        
        # 2. 选择注入点
        injection_step = self.select_injection_point(golden_history)
        print(f"[Task {task_id}] 选择注入点: 步骤 {injection_step}")
        
        # 3. 选择错误模式
        error_mode = random.choice(list(FaultInjector.ERROR_MODES.keys()))
        print(f"[Task {task_id}] 错误模式: {error_mode}")
        
        # 4. 生成致命失败轨迹
        print(f"[Task {task_id}] 生成致命失败轨迹...")
        fatal_history, mistake_info = await self.generate_fatal_trace(
            golden_history, injection_step, error_mode
        )
        print(f"[Task {task_id}] 致命失败轨迹生成完成，共 {len(fatal_history)} 步")
        
        # 5. 生成自愈成功轨迹
        print(f"[Task {task_id}] 生成自愈成功轨迹...")
        healed_history = await self.generate_healed_trace(
            fatal_history, injection_step, mistake_info["mistake_agent"]
        )
        print(f"[Task {task_id}] 自愈成功轨迹生成完成，共 {len(healed_history)} 步")
        
        # 6. 构建输出数据
        question = task["question"]
        
        # 🔥 关键修正：从 Golden 轨迹中提取正确的最终答案作为 Ground Truth
        correct_ground_truth = self._extract_final_answer(golden_history)
        # 如果提取失败，使用后备方法
        if correct_ground_truth == "Task completed successfully":
            correct_ground_truth = self._generate_ground_truth(task)
        
        # 获取 Fatal 轨迹中提交的错误答案
        wrong_answer = mistake_info.get("wrong_final_result", "0")
        
        system_prompt = {
            "Orchestrator": "Plans and coordinates tasks among agents",
            "WebSurfer": "Searches the web for information",
            "FileSurfer": "Reads and processes files",
            "Coder": "Writes and executes Python code",
            "Computer_terminal": "Executes code and returns results"
        }
        
        # Golden 数据（成功轨迹，无错误）
        golden_data = {
            "question": question,
            "ground_truth": correct_ground_truth,  # 🔥 使用从 Golden 轨迹提取的正确答案
            "mistake_step": None,
            "mistake_agent": None,
            "mistake_reason": None,
            "history": self._normalize_history(golden_history),
            "system_prompt": system_prompt
        }
        
        # Fatal 数据（致命失败轨迹）
        # 🔥 关键修正：Fatal 的 Ground Truth 是正确答案（从 Golden 提取），但它提交了错误答案
        fatal_data = {
            "question": question,
            "ground_truth": correct_ground_truth,  # Ground Truth 是正确答案
            "mistake_step": mistake_info["mistake_step"],
            "mistake_agent": mistake_info["mistake_agent"],
            "mistake_reason": mistake_info["mistake_reason"],
            "history": self._normalize_history(fatal_history),
            "system_prompt": system_prompt,
            "submitted_answer": wrong_answer  # 🔥 额外字段：记录提交的错误答案（用于调试）
        }
        
        # Healed 数据（自愈成功轨迹，无错误标签）
        # 🔥 Healed 的 Ground Truth 也是正确答案（应该和 Golden 一致）
        healed_data = {
            "question": question,
            "ground_truth": correct_ground_truth,  # 🔥 使用从 Golden 轨迹提取的正确答案
            "mistake_step": None,
            "mistake_agent": None,
            "mistake_reason": None,
            "history": self._normalize_history(healed_history),
            "system_prompt": system_prompt
        }
        
        return {
            "golden": golden_data,
            "fatal": fatal_data,
            "healed": healed_data,
            "task_id": task_id
        }
    
    def save_triplet(self, triplet: Dict[str, Any], subset: str = "AG"):
        """保存三元组数据（使用 AG_ 或 HC_ 前缀）"""
        task_id = triplet["task_id"]
        
        # 选择输出目录
        output_dir = OUTPUT_DIR_AG if subset == "AG" else OUTPUT_DIR_HC
        prefix = "AG" if subset == "AG" else "HC"
        
        # 保存 Golden
        golden_file = output_dir / f"{prefix}_golden_{task_id:05d}.json"
        with open(golden_file, 'w', encoding='utf-8') as f:
            json.dump(triplet["golden"], f, ensure_ascii=False, indent=2)
        
        # 保存 Fatal
        fatal_file = output_dir / f"{prefix}_fatal_{task_id:05d}.json"
        with open(fatal_file, 'w', encoding='utf-8') as f:
            json.dump(triplet["fatal"], f, ensure_ascii=False, indent=2)
        
        # 保存 Healed
        healed_file = output_dir / f"{prefix}_healed_{task_id:05d}.json"
        with open(healed_file, 'w', encoding='utf-8') as f:
            json.dump(triplet["healed"], f, ensure_ascii=False, indent=2)
        
        print(f"[Task {task_id}] 三元组已保存 ({subset}): {golden_file.name}, {fatal_file.name}, {healed_file.name}")


async def generate_single_task(api_client: APIClient, task: Dict[str, Any], task_id: int, subset: str = "AG"):
    """生成单个任务的三元组"""
    generator = DataGenerator(api_client)
    try:
        triplet = await generator.generate_triplet(task, task_id)
        generator.save_triplet(triplet, subset=subset)
        return True
    except Exception as e:
        print(f"[Task {task_id}] 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_next_task_id_safely(output_dir: Path, prefix: str, batch_size: int = 100) -> int:
    """
    安全地获取下一个可用的任务 ID（支持并行运行）
    
    使用文件锁机制防止多个进程同时分配相同的 ID。
    每次分配一个批次（batch_size）的 ID，减少锁竞争。
    
    Args:
        output_dir: 输出目录
        prefix: 文件前缀（AG 或 HC）
        batch_size: 每次分配的 ID 批次大小
    
    Returns:
        下一个可用的起始任务 ID
    """
    lock_file = output_dir / f".{prefix}_id_lock"
    id_counter_file = output_dir / f".{prefix}_id_counter.txt"
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试获取文件锁（最多重试 10 次，每次等待 0.1-1 秒）
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # 打开锁文件（创建如果不存在）
            # Windows 需要二进制模式，Linux/Mac 可以用文本模式
            if platform.system() == 'Windows' and HAS_FILE_LOCK:
                lock = open(lock_file, 'wb')
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                except:
                    lock.close()
                    raise
            else:
                lock = open(lock_file, 'w')
                if HAS_FILE_LOCK and platform.system() != 'Windows':
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except:
                        lock.close()
                        raise
                else:
                    # 如果没有文件锁支持，使用简单的重试机制
                    import time
                    time.sleep(random.uniform(0.1, 0.5))
            
            try:
                # 读取当前计数器
                if id_counter_file.exists():
                    try:
                        with open(id_counter_file, 'r') as f:
                            current_id = int(f.read().strip())
                    except (ValueError, IOError):
                        current_id = 0
                else:
                    # 如果计数器不存在，从现有文件中查找最大 ID
                    existing_files = list(output_dir.glob(f"{prefix}_fatal_*.json"))
                    if existing_files:
                        max_id = max([
                            int(re.search(r'_(\d+)\.json$', f.name).group(1)) 
                            for f in existing_files 
                            if re.search(r'_(\d+)\.json$', f.name)
                        ], default=0)
                        current_id = max_id
                    else:
                        current_id = 0
                
                # 分配下一个批次
                next_id = current_id + 1
                new_id = next_id + batch_size - 1
                
                # 更新计数器
                with open(id_counter_file, 'w') as f:
                    f.write(str(new_id))
                
                # 返回分配的起始 ID
                result = next_id
                
            finally:
                # 释放锁并关闭文件
                if HAS_FILE_LOCK:
                    try:
                        if platform.system() == 'Windows':
                            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                lock.close()
            
            return result
                
        except (IOError, OSError, BlockingIOError) as e:
            # 锁被占用，等待后重试
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.1, 0.5))
            else:
                # 如果所有重试都失败，回退到非安全模式
                print(f"[警告] 无法获取文件锁，使用非安全模式（可能有 ID 冲突风险）")
                existing_files = list(output_dir.glob(f"{prefix}_fatal_*.json"))
                if existing_files:
                    max_id = max([
                        int(re.search(r'_(\d+)\.json$', f.name).group(1)) 
                        for f in existing_files 
                        if re.search(r'_(\d+)\.json$', f.name)
                    ], default=0)
                    return max_id + 1
                return 1
    
    # 如果所有重试都失败，返回默认值
    return 1


async def main():
    """主函数 - 异步并发生成，支持断点续传和并行运行"""
    import sys
    
    # 解析命令行参数
    subset = "AG" 
    if len(sys.argv) > 1:
        if sys.argv[1].upper() in ["AG", "HC"]:
            subset = sys.argv[1].upper()
        elif sys.argv[1] == "test":
            await test_single_generation()
            return
    
    print("=" * 60)
    print("ASTRA-Gen 3.0: 面向图解析友好的动态因果仿真框架")
    print("=" * 60)
    
    # 确定输出目录和前缀
    output_dir = OUTPUT_DIR_AG if subset == "AG" else OUTPUT_DIR_HC
    prefix = "AG" if subset == "AG" else "HC"
    
    # 🔥 优化：使用安全的 ID 分配机制（支持并行运行）
    # 每次分配 100 个 ID，减少锁竞争
    start_id = get_next_task_id_safely(output_dir, prefix, batch_size=100)
    
    # 目标生成总任务数
    num_tasks_total = 700 
    
    print(f"数据子集: {subset} ({'Algorithm-Generated' if subset == 'AG' else 'Hand-Crafted'})")
    print(f"输出目录: {output_dir}")
    print(f"API 模型: {MODEL_NAME}")
    print(f"目标生成任务范围: ID {start_id} 到 {num_tasks_total}")
    print("=" * 60)
    
    # 创建任务生成器
    async with APIClient(API_KEY, BASE_URL, MODEL_NAME) as api_client:
        task_generator = TaskGenerator(api_client)
        
        # 创建任务列表（使用动态生成）
        tasks = []
        
        # 仅生成从 start_id 开始的任务
        tasks_to_create = num_tasks_total - start_id + 1
        tasks_to_create = max(0, tasks_to_create)

        print(f"\n生成 {tasks_to_create} 个动态任务模板...")
        
        for i in range(tasks_to_create):
            task = await task_generator.generate_task()
            tasks.append((task, start_id + i))
            if (i + 1) % 50 == 0:
                print(f"  已生成 {i + 1}/{tasks_to_create} 个任务模板...")
    
    # 并发控制
    semaphore = asyncio.Semaphore(5)  # 最多5个并发请求
    
    async def generate_with_semaphore(api_client, task, task_id):
        async with semaphore:
            return await generate_single_task(api_client, task, task_id, subset=subset)
    
    # 使用 API 客户端
    async with APIClient(API_KEY, BASE_URL, MODEL_NAME) as api_client:
        # 创建任务
        coroutines = [
            generate_with_semaphore(api_client, task, task_id)
            for task, task_id in tasks
        ]
        
        # 执行并等待完成
        print(f"\n开始生成 {len(tasks)} 个新任务...")
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 统计结果
        success_count = sum(1 for r in results if r is True)
        fail_count = len(results) - success_count
        
        print("\n" + "=" * 60)
        print("生成完成!")
        print(f"成功: {success_count} (共 {success_count * 3} 个 JSON 文件)")
        print(f"失败: {fail_count}")
        print(f"已完成任务 ID: {start_id + len(tasks) - 1}")
        print("=" * 60)


async def test_single_generation():
    """测试单个任务生成（用于调试）"""
    print("=" * 60)
    print("测试模式: 生成单个任务的三元组")
    print("=" * 60)
    
    async with APIClient(API_KEY, BASE_URL, MODEL_NAME) as api_client:
        # 使用动态任务生成器
        task_generator = TaskGenerator(api_client)
        test_task = await task_generator.generate_task()
        print(f"生成的任务: {test_task['question']}")
        
        generator = DataGenerator(api_client)
        try:
            triplet = await generator.generate_triplet(test_task, 0)
            generator.save_triplet(triplet, subset="AG")
            print("\n测试完成! 检查输出文件:")
            print(f"  - {OUTPUT_DIR_AG / 'AG_golden_00000.json'}")
            print(f"  - {OUTPUT_DIR_AG / 'AG_fatal_00000.json'}")
            print(f"  - {OUTPUT_DIR_AG / 'AG_healed_00000.json'}")
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        asyncio.run(test_single_generation())
    else:
        # 正常模式（支持 AG 或 HC 参数）
        asyncio.run(main())


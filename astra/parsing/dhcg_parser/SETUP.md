# 环境设置指南

## 问题诊断

如果遇到 `ModuleNotFoundError: No module named 'sentence_transformers'` 错误，请按照以下步骤操作：

## 解决方案

### 方法1: 使用项目根目录的虚拟环境（推荐）

1. **激活虚拟环境**：
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD
   venv\Scripts\activate.bat
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **安装依赖**：
   ```bash
   # 从项目根目录安装所有依赖
   pip install -r requirements.txt
   
   # 或者只安装解析器需要的依赖
   pip install sentence-transformers
   ```

3. **验证安装**：
   ```bash
   python check_env.py
   ```

### 方法2: 使用Python安装脚本

直接运行安装脚本：
```bash
python install_requirements.py
```

### 方法3: 手动安装

```bash
pip install sentence-transformers
```

## 验证环境

运行环境检查脚本：
```bash
python check_env.py
```

如果所有检查通过，你应该看到：
```
✓ Python 版本符合要求
✓ 正在使用虚拟环境
✓ sentence-transformers 已安装
✓ 所有依赖已安装！
```

## 运行解析器

环境配置完成后，可以运行解析器：

```bash
# 处理单个文件
python parser.py "Who&When/Algorithm-Generated/1.json"

# 批量处理目录
python parser.py "Who&When/Algorithm-Generated"
```

## 常见问题

### Q: 如何确认是否在虚拟环境中？
A: 运行 `python -c "import sys; print(sys.prefix)"`，如果路径包含 `venv`，说明在虚拟环境中。

### Q: 安装后仍然报错？
A: 确保：
1. 虚拟环境已激活
2. 使用的 Python 解释器是虚拟环境中的（检查 `sys.executable`）
3. 重新安装：`pip install --force-reinstall sentence-transformers`

### Q: Conda 环境问题？
A: 如果使用 Conda，可能需要：
```bash
conda activate your_env_name
pip install sentence-transformers
```


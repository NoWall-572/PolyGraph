"""
解析 ASTRA-Gen 3.0 生成的数据集并构建图
将解析结果按子集分类存储到两个子目录中
"""

import sys
from pathlib import Path
import argparse

# 添加 dhcg_parser 到路径
sys.path.insert(0, str(Path(__file__).parent))

from astra.parsing.dhcg_parser.parser import process_directory


def main():
    """主函数：解析两个子集并分类存储"""
    parser = argparse.ArgumentParser(
        description='解析 ASTRA-Gen 3.0 数据集并构建图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认路径解析（需要 --save 参数才会保存结果）
  python parse_astra_v3_dataset.py --save
  
  # 指定输入和输出目录，解析并保存
  python parse_astra_v3_dataset.py --input outputs_koi --output processed_graphs/graphs_astra_v3 --subset both --save
  
  # 只解析其中一个子集
  python parse_astra_v3_dataset.py --subset AG --save
  python parse_astra_v3_dataset.py --subset HC --save
  
  # 静默模式（只显示总结信息）
  python parse_astra_v3_dataset.py --input outputs_koi --output processed_graphs/graphs_astra_v3 --subset both --quiet --save
  
断点续传功能:
  - 使用 --save 参数时，脚本会自动跳过已经解析过的 JSON 文件
  - 已解析的文件通过检查输出目录中是否存在对应的 _graph.json 文件来判断
  - 可以随时停止脚本，再次运行时会自动从上次停止的地方继续
  - 输出文件命名格式: {子目录名}_{原文件名}_graph.json
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='outputs_koi',
        help='输入数据根目录（包含 Algorithm-Generated 和 Hand-Crafted 子目录，默认: outputs_koi）'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='outputs/graphs_astra_v3',
        help='输出根目录（将创建两个子目录存放解析结果，默认: outputs/graphs_astra_v3）'
    )
    parser.add_argument(
        '--subset',
        type=str,
        choices=['AG', 'HC', 'both'],
        default='both',
        help='要解析的子集：AG (Algorithm-Generated), HC (Hand-Crafted), both (两者，默认)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，只显示总结信息'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='是否保存解析后的图数据到输出目录（默认: False）'
    )
    
    args = parser.parse_args()
    
    # 转换为 Path 对象
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    # 检查输入目录是否存在
    if not input_root.exists():
        print(f"错误: 输入目录 '{input_root}' 不存在")
        sys.exit(1)
    
    # 定义子集映射
    subset_config = {
        'AG': {
            'input_dir': input_root / 'Algorithm-Generated',
            'output_dir': output_root / 'Algorithm-Generated',
            'name': 'Algorithm-Generated'
        },
        'HC': {
            'input_dir': input_root / 'Hand-Crafted',
            'output_dir': output_root / 'Hand-Crafted',
            'name': 'Hand-Crafted'
        }
    }
    
    # 确定要处理的子集
    subsets_to_process = []
    if args.subset == 'both':
        subsets_to_process = ['AG', 'HC']
    else:
        subsets_to_process = [args.subset]
    
    print("=" * 80)
    print("ASTRA-Gen 3.0 数据集解析器")
    print("=" * 80)
    print(f"输入根目录: {input_root.absolute()}")
    print(f"输出根目录: {output_root.absolute()}")
    print(f"处理子集: {', '.join([subset_config[s]['name'] for s in subsets_to_process])}")
    print("=" * 80)
    
    total_results = {
        'AG': {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0},
        'HC': {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    }
    
    # 处理每个子集
    for subset_key in subsets_to_process:
        config = subset_config[subset_key]
        input_dir = config['input_dir']
        output_dir = config['output_dir']
        
        print(f"\n{'=' * 80}")
        print(f"处理子集: {config['name']}")
        print(f"输入目录: {input_dir.absolute()}")
        print(f"输出目录: {output_dir.absolute()}")
        print(f"{'=' * 80}\n")
        
        # 检查输入目录是否存在
        if not input_dir.exists():
            print(f"⚠️  警告: 输入目录 '{input_dir}' 不存在，跳过")
            continue
        
        # 检查是否有 JSON 文件
        json_files = list(input_dir.glob("*.json"))
        # 🔥 过滤掉enhanced文件（enhanced文件是嵌套结构，不适用于图格式转换）
        # 图格式转换只需要golden/fatal/healed文件（扁平结构）
        json_files = [f for f in json_files if "enhanced" not in f.name.lower()]
        if not json_files:
            if not args.quiet:
                print(f"⚠️  警告: 在 '{input_dir}' 中未找到 JSON 文件，跳过")
            continue
        
        # 创建输出目录（确保存在）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 断点续传：检查已解析的文件数量（仅在 save 模式下有效）
        if args.save:
            # 预先检查已解析的文件数量
            source_dir_name = input_dir.name
            safe_dir_name = source_dir_name.replace('/', '_').replace('\\', '_').replace('&', '_')
            already_parsed = 0
            for json_file in json_files:
                output_file_name = f"{safe_dir_name}_{json_file.stem}_graph.json"
                output_file_path = output_dir / output_file_name
                if output_file_path.exists():
                    already_parsed += 1
            
            if already_parsed > 0:
                print(f"📋 发现 {already_parsed}/{len(json_files)} 个文件已解析，将自动跳过")
        
        # 解析该子集（process_directory 内部会处理跳过逻辑）
        results = process_directory(
            directory=input_dir,
            verbose=not args.quiet,
            save_result=args.save,
            output_dir=output_dir
        )
        
        # 记录结果
        total_results[subset_key] = {
            'total': results['total'],
            'success': results['success'],
            'failed': results['failed'],
            'skipped': results.get('skipped', 0)  # 支持跳过统计
        }
    
    # 打印总结
    print("\n" + "=" * 80)
    print("解析完成总结")
    print("=" * 80)
    
    for subset_key in subsets_to_process:
        config = subset_config[subset_key]
        stats = total_results[subset_key]
        print(f"\n{config['name']}:")
        print(f"  总计: {stats['total']} 个文件")
        print(f"  成功: {stats['success']} 个文件")
        print(f"  失败: {stats['failed']} 个文件")
        if stats.get('skipped', 0) > 0:
            print(f"  跳过: {stats['skipped']} 个文件（已解析）")
        if stats['total'] > 0:
            success_rate = (stats['success'] / stats['total']) * 100
            print(f"  成功率: {success_rate:.1f}%")
    
    # 计算总计
    total_files = sum(total_results[s]['total'] for s in subsets_to_process)
    total_success = sum(total_results[s]['success'] for s in subsets_to_process)
    total_failed = sum(total_results[s]['failed'] for s in subsets_to_process)
    total_skipped = sum(total_results[s].get('skipped', 0) for s in subsets_to_process)
    
    print(f"\n总计:")
    print(f"  总计: {total_files} 个文件")
    print(f"  成功: {total_success} 个文件")
    print(f"  失败: {total_failed} 个文件")
    if total_skipped > 0:
        print(f"  跳过: {total_skipped} 个文件（已解析）")
    if total_files > 0:
        overall_success_rate = (total_success / total_files) * 100
        print(f"  总成功率: {overall_success_rate:.1f}%")
    
    print(f"\n输出目录: {output_root.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()


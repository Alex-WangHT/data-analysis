"""
快速使用示例脚本
展示如何使用标准化工具
"""

from main import DataStandardizer

def example_1_full_pipeline():
    """
    示例 1: 完整标准化流程
    分布分析 -> Box-Cox 变换 -> Z-score 标准化
    """
    print("=" * 60)
    print("示例 1: 完整标准化流程")
    print("=" * 60)
    
    custom_config = {
        'missing': {
            'strategy': 'interpolate',
        },
        'distribution': {
            'skewness_threshold': 0.5,
            'test_normal': True,
        },
        'output': {
            'output_file': '标准化数据_完整流程.xlsx',
            'transform_params_file': '变换参数_完整流程.json',
        }
    }
    
    standardizer = DataStandardizer(custom_config)
    
    result = standardizer.run_standardization_pipeline()
    
    standardizer.save_results()
    
    print("\n结果预览 (前5行):")
    print(result.head())
    
    return result


def example_2_simple_zscore():
    """
    示例 2: 简单 Z-score 标准化
    跳过 Box-Cox，直接标准化
    """
    print("\n" + "=" * 60)
    print("示例 2: 简单 Z-score 标准化")
    print("=" * 60)
    
    standardizer = DataStandardizer()
    
    result = standardizer.run_standardization_pipeline(skip_boxcox=True)
    
    standardizer.save_results(
        output_file='标准化数据_简单Zscore.xlsx',
        save_params=True
    )
    
    return result


def example_3_analyze_only():
    """
    示例 3: 仅进行分布分析
    """
    print("\n" + "=" * 60)
    print("示例 3: 仅进行分布分析")
    print("=" * 60)
    
    from data_loader import DataLoader
    from boxcox_transformer import DistributionAnalyzer
    
    loader = DataLoader()
    loader.load_and_preprocess()
    numeric_data = loader.get_numeric_data()
    
    analyzer = DistributionAnalyzer()
    analyzer.calculate_distribution_stats(numeric_data)
    skewed, normal = analyzer.classify_columns()
    
    recommendations = analyzer.get_transformation_recommendations()
    print("\n变换建议:")
    for col, rec in recommendations.items():
        print(f"  {col}: {rec}")


def example_4_load_params():
    """
    示例 4: 使用已保存的参数变换新数据
    """
    print("\n" + "=" * 60)
    print("示例 4: 使用已保存的参数")
    print("=" * 60)
    
    standardizer = DataStandardizer()
    
    try:
        standardized_new = standardizer.load_params_and_transform(
            params_file='变换参数_完整流程.json'
        )
        print("\n使用已保存参数变换成功!")
        print(standardized_new.head())
    except FileNotFoundError:
        print("未找到参数文件，请先运行示例 1 生成参数文件")


if __name__ == '__main__':
    print("选择要运行的示例:")
    print("  1. 完整标准化流程（推荐）")
    print("  2. 简单 Z-score 标准化")
    print("  3. 仅分布分析")
    print("  4. 运行所有示例")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == '1':
        example_1_full_pipeline()
    elif choice == '2':
        example_2_simple_zscore()
    elif choice == '3':
        example_3_analyze_only()
    elif choice == '4':
        example_1_full_pipeline()
        example_2_simple_zscore()
        example_3_analyze_only()
        example_4_load_params()
    else:
        print("默认运行示例 1")
        example_1_full_pipeline()

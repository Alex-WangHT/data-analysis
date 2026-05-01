"""
验证所有模块能否正确导入
"""

import sys

print("=" * 60)
print("验证模块导入")
print("=" * 60)

# 测试 1: 导入 config
print("\n[1/6] 测试导入 config...")
try:
    from config import (
        DATA_CONFIG, 
        COLUMN_CONFIG, 
        MISSING_CONFIG,
        DISTRIBUTION_CONFIG,
        BOXCOX_CONFIG,
        ZSCORE_CONFIG,
        OUTPUT_CONFIG
    )
    print("    ✓ config.py 导入成功")
except Exception as e:
    print(f"    ✗ config.py 导入失败: {e}")
    sys.exit(1)

# 测试 2: 导入 data_loader
print("\n[2/6] 测试导入 data_loader...")
try:
    from data_loader import DataLoader
    print("    ✓ data_loader.py 导入成功")
    print(f"    DataLoader 类已定义: {DataLoader}")
except Exception as e:
    print(f"    ✗ data_loader.py 导入失败: {e}")
    sys.exit(1)

# 测试 3: 导入 boxcox_transformer
print("\n[3/6] 测试导入 boxcox_transformer...")
try:
    from boxcox_transformer import DistributionAnalyzer, BoxCoxTransformer
    print("    ✓ boxcox_transformer.py 导入成功")
    print(f"    DistributionAnalyzer 类已定义: {DistributionAnalyzer}")
    print(f"    BoxCoxTransformer 类已定义: {BoxCoxTransformer}")
except Exception as e:
    print(f"    ✗ boxcox_transformer.py 导入失败: {e}")
    sys.exit(1)

# 测试 4: 导入 zscore_scaler
print("\n[4/6] 测试导入 zscore_scaler...")
try:
    from zscore_scaler import ZScoreScaler, StandardizationPipeline
    print("    ✓ zscore_scaler.py 导入成功")
    print(f"    ZScoreScaler 类已定义: {ZScoreScaler}")
    print(f"    StandardizationPipeline 类已定义: {StandardizationPipeline}")
except Exception as e:
    print(f"    ✗ zscore_scaler.py 导入失败: {e}")
    sys.exit(1)

# 测试 5: 导入 main
print("\n[5/6] 测试导入 main...")
try:
    from main import DataStandardizer
    print("    ✓ main.py 导入成功")
    print(f"    DataStandardizer 类已定义: {DataStandardizer}")
except Exception as e:
    print(f"    ✗ main.py 导入失败: {e}")
    sys.exit(1)

# 测试 6: 检查类的方法
print("\n[6/6] 检查关键类的方法...")

# 检查 DataStandardizer 的关键方法
try:
    standardizer_methods = [
        'run_standardization_pipeline',
        'save_results',
        'load_params_and_transform'
    ]
    for method in standardizer_methods:
        if hasattr(DataStandardizer, method):
            print(f"    ✓ DataStandardizer.{method}() 存在")
        else:
            print(f"    ⚠ DataStandardizer.{method}() 不存在")
except Exception as e:
    print(f"    检查方法时出错: {e}")

print("\n" + "=" * 60)
print("所有模块导入验证通过!")
print("=" * 60)

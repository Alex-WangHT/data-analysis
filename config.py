"""
配置文件
包含数据路径、列名配置、处理参数等
"""

# 数据文件配置
DATA_CONFIG = {
    'excel_file': '统计局数据.xlsx',
    'sheet_name': None,  # None 表示读取所有工作表，或指定工作表名
}

# 列名配置
COLUMN_CONFIG = {
    'id_columns': ['地区', '年份', '时间'],  # 标识列，不进行标准化处理
    'numeric_columns': None,  # None 表示自动识别所有数值列
}

# 缺失值处理配置
MISSING_CONFIG = {
    'strategy': 'interpolate',  # 可选: 'drop', 'mean', 'median', 'interpolate', 'ffill', 'bfill'
    'interpolate_method': 'linear',  # 插值方法: 'linear', 'time', 'index', 'values'
}

# 分布分析配置
DISTRIBUTION_CONFIG = {
    'skewness_threshold': 0.5,  # 偏度阈值，超过此值视为偏态分布
    'test_normal': True,  # 是否进行正态性检验
    'alpha': 0.05,  # 正态性检验显著性水平
}

# Box-Cox 变换配置
BOXCOX_CONFIG = {
    'lmbda': None,  # None 表示自动选择最优 lambda
    'add_constant': 1e-6,  # 为确保数据为正，添加的常数
}

# Z-score 标准化配置
ZSCORE_CONFIG = {
    'use_robust': False,  # 是否使用稳健标准化（中位数和绝对中位数偏差）
}

# 输出配置
OUTPUT_CONFIG = {
    'save_to_excel': True,
    'output_file': '标准化数据.xlsx',
    'save_transform_params': True,
    'transform_params_file': '变换参数.json',
}

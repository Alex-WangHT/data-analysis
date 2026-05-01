"""
透视表数据配置
针对 Excel 透视表格式：
- 省份列使用合并单元格，NaN需要向前填充
- 列结构：省份 | 指标 | 单位 | 年份列(2020年, 2021年...)
"""

# 数据文件配置
PIVOT_CONFIG = {
    'excel_file': '统计局数据.xlsx',
    
    'sheet_filter': {
        'keyword': '透视',           # 只处理名称包含此关键词的工作表
        'exclude_keywords': [],      # 排除包含这些关键词的表
    },
    
    'column_structure': {
        'province_col_idx': 0,       # 省份列的索引（第1列）
        'indicator_col_idx': 1,      # 指标列的索引（第2列）
        'unit_col_idx': 2,           # 单位列的索引（第3列），如果没有设为 None
        'year_start_col_idx': 3,     # 年份数据开始的列索引（第4列）
    },
    
    'merge_cell_handling': {
        'enabled': True,             # 是否启用合并单元格处理
        'forward_fill_columns': [0], # 需要向前填充的列索引（省份列）
        'forward_fill_method': 'ffill',  # 填充方法 'ffill' = 向前填充
    },
    
    'column_names': {
        'province': '省份',
        'indicator': '指标',
        'unit': '单位',
        'year': '年份',
        'value': '数值',
    },
    
    'year_column_pattern': {
        'has_suffix': True,          # 年份列是否有后缀，如 "2020年"
        'suffix': '年',              # 后缀内容
        'extract_year': True,        # 是否从列名中提取年份数字
    }
}

# 归一化配置
PIVOT_NORMALIZATION = {
    'within_province': {
        'enabled': True,
        'method': 'zscore',          # 'zscore' | 'minmax' | 'robust'
        'axis': 'by_indicator',       # 'by_indicator' = 每个指标在年份上归一化; 'by_year' = 每年对指标归一化
    },
    
    'across_province': {
        'enabled': True,
        'method': 'zscore',
        'group_by': 'year',          # 'year' = 每年对各省归一化; 'overall' = 整体归一化
    }
}

# 相关性分析配置
PIVOT_CORRELATION = {
    'within_province': {
        'method': 'pearson',         # 'pearson' | 'spearman' | 'kendall'
        'min_obs': 5,                # 最小观测数要求
        'alpha': 0.05,               # 显著性水平
    },
    
    'across_province': {
        'method': 'pearson',
        'group_by': 'indicator',     # 'indicator' = 按指标分析; 'year' = 按年份分析
        'min_obs': 5,
        'alpha': 0.05,
    }
}

# 输出配置
PIVOT_OUTPUT = {
    'save_to_excel': True,
    'output_file': '透视表分析结果.xlsx',
    
    'save_within_province': True,
    'within_province_sheet': '省份内分析',
    
    'save_across_province': True,
    'across_province_sheet': '省份间分析',
    
    'save_params': True,
    'params_file': '分析参数.json',
}

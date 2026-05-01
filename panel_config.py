"""
面板数据配置文件
适用于省-年-指标结构的数据
"""

# 数据文件配置
PANEL_DATA_CONFIG = {
    'excel_file': '统计局数据.xlsx',
    
    'sheet_type': 'pivot',  # 'pivot' = 透视表, 'long' = 长格式
    
    'pivot_structure': {
        'row_index': 'province',  # 行是什么: 'province'=省份, 'year'=年份
        'col_index': 'year',      # 列是什么: 'year'=年份, 'province'=省份, 'indicator'=指标
        'value_type': 'indicator' # 值是什么: 'indicator'=每个单元格是一个指标值
    },
    
    'id_columns': {
        'province': ['地区', '省份', '省', 'region', 'province'],
        'year': ['年份', '年', 'year', '时间'],
        'indicator': ['指标', '指标名称', 'variable', 'indicator']
    },
}

# 归一化配置
NORMALIZATION_CONFIG = {
    'method': 'by_indicator',  # 按指标归一化
    
    'by_indicator': {
        'group_by': 'year',  # 按年份分组，每年内对各省归一化
        'method': 'zscore',   # 'zscore' | 'minmax' | 'robust'
    },
    
    'by_province': {
        'group_by': 'indicator',  # 按指标分组，对该省的所有年份归一化
        'method': 'zscore',
    },
    
    'overall': {
        'method': 'zscore',
    }
}

# 相关性分析配置
CORRELATION_CONFIG = {
    'method': 'pearson',  # 'pearson' | 'spearman' | 'kendall'
    
    'analysis_type': 'cross_province',  # 跨省市分析
    
    'cross_province': {
        'target_indicator': None,  # 目标指标，None 表示所有指标两两比较
        'group_by': 'year',        # 按年分组计算，或 'overall' 整体
    },
    
    'time_series': {
        'group_by': 'province',    # 按省分组，计算该省内指标的时序相关性
    }
}

# 输出配置
PANEL_OUTPUT_CONFIG = {
    'save_normalized': True,
    'normalized_file': '标准化面板数据.xlsx',
    
    'save_correlation': True,
    'correlation_file': '相关性分析结果.xlsx',
    
    'save_params': True,
    'params_file': '变换参数.json',
}

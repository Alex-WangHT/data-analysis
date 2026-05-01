"""
省份内分析模块
在每个省份内部进行：
1. 指标数据的归一化
2. 指标之间的相关性分析
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from pivot_config import PIVOT_NORMALIZATION, PIVOT_CORRELATION


class WithinProvinceAnalyzer:
    """
    省份内分析器
    对每个省份内部的指标进行归一化和相关性分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化省份内分析器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        self.norm_config = {**PIVOT_NORMALIZATION['within_province'], 
                            **self.config.get('within_norm', {})}
        self.corr_config = {**PIVOT_CORRELATION['within_province'],
                            **self.config.get('within_corr', {})}
        
        self.normalized_data: Dict[str, pd.DataFrame] = {}
        self.transform_params: Dict[str, Dict[str, Any]] = {}
        self.correlation_results: Dict[str, Dict[str, Any]] = {}
        
        self.provinces: List[str] = []
        self.indicators: List[str] = []
        self.years: List[int] = []
    
    def set_data_info(self, provinces: List[str], indicators: List[str], years: List[int]):
        """
        设置数据信息
        
        参数:
            provinces: 省份列表
            indicators: 指标列表
            years: 年份列表
        """
        self.provinces = provinces
        self.indicators = indicators
        self.years = years
    
    def _zscore(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Z-score 标准化"""
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'mean': 0, 'std': 1}
        
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        
        if std_val == 0:
            std_val = 1e-6
        
        normalized = data.copy()
        normalized[valid_mask] = (valid_data - mean_val) / std_val
        
        return normalized, {'mean': mean_val, 'std': std_val}
    
    def _minmax(self, data: pd.Series, feature_range: Tuple[float, float] = (0, 1)
                ) -> Tuple[pd.Series, Dict[str, float]]:
        """Min-Max 归一化"""
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'min': 0, 'max': 1, 'range_min': feature_range[0], 'range_max': feature_range[1]}
        
        min_val = valid_data.min()
        max_val = valid_data.max()
        range_min, range_max = feature_range
        
        normalized = data.copy()
        if max_val == min_val:
            normalized[valid_mask] = (range_min + range_max) / 2
        else:
            normalized[valid_mask] = (valid_data - min_val) / (max_val - min_val) * (range_max - range_min) + range_min
        
        return normalized, {'min': min_val, 'max': max_val, 'range_min': range_min, 'range_max': range_max}
    
    def _robust_zscore(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """稳健 Z-score（中位数 + MAD）"""
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'median': 0, 'mad': 1}
        
        median_val = valid_data.median()
        mad = np.median(np.abs(valid_data - median_val))
        
        if mad == 0:
            mad = 1e-6
        
        normalized = data.copy()
        normalized[valid_mask] = (valid_data - median_val) / (mad * 1.4826)
        
        return normalized, {'median': median_val, 'mad': mad}
    
    def normalize_province(self, province_data: pd.DataFrame, province_name: str) -> pd.DataFrame:
        """
        对单个省份的数据进行归一化
        
        参数:
            province_data: 该省份的数据（行=年份，列=指标）
            province_name: 省份名
            
        返回:
            归一化后的数据
        """
        print(f"\n  归一化省份: {province_name}")
        
        method = self.norm_config['method']
        axis = self.norm_config['axis']
        
        normalized = province_data.copy()
        
        if province_name not in self.transform_params:
            self.transform_params[province_name] = {}
        
        year_col = province_data.columns[0] if len(province_data.columns) > 0 else None
        indicator_cols = [c for c in province_data.columns if c != year_col]
        
        if axis == 'by_indicator':
            for indicator in indicator_cols:
                if indicator not in province_data.columns:
                    continue
                
                data = province_data[indicator]
                
                if method == 'zscore':
                    norm_data, params = self._zscore(data)
                elif method == 'minmax':
                    norm_data, params = self._minmax(data)
                elif method == 'robust':
                    norm_data, params = self._robust_zscore(data)
                else:
                    norm_data, params = self._zscore(data)
                
                normalized[indicator] = norm_data
                self.transform_params[province_name][indicator] = {
                    'method': method,
                    'params': params
                }
        
        elif axis == 'by_year':
            for idx, row in province_data.iterrows():
                year = row[year_col] if year_col else idx
                
                row_data = row[indicator_cols]
                
                if method == 'zscore':
                    mean_val = row_data.mean()
                    std_val = row_data.std()
                    if std_val == 0:
                        std_val = 1e-6
                    normalized_row = (row_data - mean_val) / std_val
                    params = {'mean': mean_val, 'std': std_val}
                elif method == 'minmax':
                    min_val = row_data.min()
                    max_val = row_data.max()
                    if max_val == min_val:
                        normalized_row = row_data * 0 + 0.5
                    else:
                        normalized_row = (row_data - min_val) / (max_val - min_val)
                    params = {'min': min_val, 'max': max_val}
                else:
                    normalized_row = row_data
                    params = {}
                
                for indicator in indicator_cols:
                    if indicator in normalized_row:
                        normalized.loc[idx, indicator] = normalized_row[indicator]
                
                if str(year) not in self.transform_params[province_name]:
                    self.transform_params[province_name][str(year)] = {
                        'method': method,
                        'params': params
                    }
        
        print(f"    归一化方法: {method}, 方式: {axis}")
        
        return normalized
    
    def normalize_all_provinces(self, data_by_province: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对所有省份进行归一化
        
        参数:
            data_by_province: 按省份组织的数据
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 70)
        print("省份内归一化")
        print("=" * 70)
        
        for province, data in data_by_province.items():
            normalized = self.normalize_province(data, province)
            self.normalized_data[province] = normalized
        
        return self.normalized_data
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """
        计算两个变量的相关系数
        
        参数:
            x: 变量 X
            y: 变量 Y
            
        返回:
            相关系数结果
        """
        method = self.corr_config['method']
        min_obs = self.corr_config['min_obs']
        alpha = self.corr_config['alpha']
        
        valid_mask = ~x.isnull() & ~y.isnull()
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < min_obs:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': method,
                'note': f'观测数不足 (需要 {min_obs})'
            }
        
        try:
            if method == 'pearson':
                corr, p_value = pearsonr(x_valid, y_valid)
            elif method == 'spearman':
                corr, p_value = spearmanr(x_valid, y_valid)
            elif method == 'kendall':
                corr, p_value = kendalltau(x_valid, y_valid)
            else:
                corr, p_value = pearsonr(x_valid, y_valid)
            
            significant = p_value < alpha
            
            return {
                'correlation': corr,
                'p_value': p_value,
                'n_obs': len(x_valid),
                'significant': significant,
                'method': method,
                'abs_correlation': abs(corr)
            }
            
        except Exception as e:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': method,
                'error': str(e)
            }
    
    def analyze_province_correlation(self, province_data: pd.DataFrame, province_name: str,
                                       use_normalized: bool = True) -> Dict[str, Any]:
        """
        分析单个省份内的指标相关性
        
        参数:
            province_data: 该省份的数据
            province_name: 省份名
            use_normalized: 是否使用归一化后的数据
            
        返回:
            相关性分析结果
        """
        print(f"\n  分析省份: {province_name}")
        
        if use_normalized and province_name in self.normalized_data:
            data = self.normalized_data[province_name]
            print(f"    使用归一化数据")
        else:
            data = province_data
        
        year_col = data.columns[0] if len(data.columns) > 0 else None
        indicator_cols = [c for c in data.columns if c != year_col]
        
        n_indicators = len(indicator_cols)
        
        result = {
            'province': province_name,
            'indicators': indicator_cols,
            'n_years': len(data),
            'correlation_matrix': pd.DataFrame(),
            'p_value_matrix': pd.DataFrame(),
            'significant_pairs': [],
            'pairwise': {}
        }
        
        if n_indicators < 2:
            print(f"    警告: 指标数不足 ({n_indicators} 个)，无法计算相关性")
            return result
        
        corr_matrix = pd.DataFrame(np.eye(n_indicators), 
                                    index=indicator_cols, 
                                    columns=indicator_cols)
        p_matrix = pd.DataFrame(np.ones((n_indicators, n_indicators)),
                                index=indicator_cols,
                                columns=indicator_cols)
        
        for i, ind1 in enumerate(indicator_cols):
            for j, ind2 in enumerate(indicator_cols):
                if i == j:
                    continue
                
                corr_result = self._calculate_correlation(
                    data[ind1], data[ind2]
                )
                
                corr_matrix.loc[ind1, ind2] = corr_result['correlation']
                p_matrix.loc[ind1, ind2] = corr_result['p_value']
                
                if i < j:
                    pair_key = f"{ind1} ↔ {ind2}"
                    result['pairwise'][pair_key] = corr_result
                    
                    if corr_result.get('significant', False):
                        result['significant_pairs'].append({
                            'indicator_1': ind1,
                            'indicator_2': ind2,
                            'correlation': corr_result['correlation'],
                            'p_value': corr_result['p_value'],
                            'n_obs': corr_result['n_obs'],
                            'abs_correlation': abs(corr_result['correlation'])
                        })
        
        result['correlation_matrix'] = corr_matrix
        result['p_value_matrix'] = p_matrix
        
        result['significant_pairs'].sort(
            key=lambda x: x['abs_correlation'], reverse=True
        )
        
        print(f"    指标数: {n_indicators}, 年份数: {result['n_years']}")
        print(f"    显著相关对: {len(result['significant_pairs'])} 个")
        
        for pair in result['significant_pairs'][:5]:
            sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
            print(f"      {pair['indicator_1']} ↔ {pair['indicator_2']}: "
                  f"r={pair['correlation']:.4f}, p={pair['p_value']:.4f} {sig}")
        
        return result
    
    def analyze_all_provinces(self, data_by_province: Dict[str, pd.DataFrame],
                               use_normalized: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        分析所有省份的指标相关性
        
        参数:
            data_by_province: 按省份组织的数据
            use_normalized: 是否使用归一化后的数据
            
        返回:
            所有省份的相关性分析结果
        """
        print("\n" + "=" * 70)
        print("省份内指标相关性分析")
        print("=" * 70)
        
        for province, data in data_by_province.items():
            result = self.analyze_province_correlation(data, province, use_normalized)
            self.correlation_results[province] = result
        
        self._print_summary()
        
        return self.correlation_results
    
    def _print_summary(self):
        """打印分析汇总"""
        print("\n" + "=" * 70)
        print("省份内相关性分析汇总")
        print("=" * 70)
        
        all_significant = defaultdict(list)
        
        for province, result in self.correlation_results.items():
            for pair in result['significant_pairs']:
                pair_key = f"{pair['indicator_1']} ↔ {pair['indicator_2']}"
                all_significant[pair_key].append({
                    'province': province,
                    'correlation': pair['correlation'],
                    'p_value': pair['p_value']
                })
        
        print("\n跨省份一致的显著相关对:")
        for pair_key, results in all_significant.items():
            n_provinces = len(results)
            if n_provinces >= 2:
                avg_corr = np.mean([r['correlation'] for r in results])
                
                pos_count = sum(1 for r in results if r['correlation'] > 0)
                neg_count = sum(1 for r in results if r['correlation'] < 0)
                
                direction = "正相关" if pos_count > neg_count else "负相关"
                
                print(f"\n  {pair_key}:")
                print(f"    显著省份数: {n_provinces}")
                print(f"    平均相关系数: {avg_corr:.4f}")
                print(f"    方向: {direction} (正:{pos_count}, 负:{neg_count})")
                
                for r in results[:5]:
                    sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else '*')
                    print(f"    {r['province']}: r={r['correlation']:.4f} {sig}")
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取所有结果
        
        返回:
            结果字典
        """
        return {
            'normalized_data': self.normalized_data,
            'transform_params': self.transform_params,
            'correlation_results': self.correlation_results,
            'config': {
                'normalization': self.norm_config,
                'correlation': self.corr_config
            }
        }

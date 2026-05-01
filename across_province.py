"""
省份间分析模块
分析不同省份之间的关系：
1. 同一指标在不同省份间的相关性
2. 省份之间的相似性分析
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from pivot_config import PIVOT_NORMALIZATION, PIVOT_CORRELATION


class AcrossProvinceAnalyzer:
    """
    省份间分析器
    分析不同省份之间的指标相关性和相似性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化省份间分析器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        self.norm_config = {**PIVOT_NORMALIZATION['across_province'],
                            **self.config.get('across_norm', {})}
        self.corr_config = {**PIVOT_CORRELATION['across_province'],
                            **self.config.get('across_corr', {})}
        
        self.normalized_data: Dict[str, pd.DataFrame] = {}
        self.transform_params: Dict[str, Dict[str, Any]] = {}
        self.correlation_results: Dict[str, Any] = {}
        
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
    
    def _minmax(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Min-Max 归一化"""
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'min': 0, 'max': 1}
        
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        normalized = data.copy()
        if max_val == min_val:
            normalized[valid_mask] = 0.5
        else:
            normalized[valid_mask] = (valid_data - min_val) / (max_val - min_val)
        
        return normalized, {'min': min_val, 'max': max_val}
    
    def normalize_indicator(self, indicator_data: pd.DataFrame, indicator_name: str,
                            group_by: str = 'year') -> pd.DataFrame:
        """
        对单个指标的数据进行归一化（省份间比较）
        
        参数:
            indicator_data: 该指标的数据（行=省份，列=年份 或 行=年份，列=省份）
            indicator_name: 指标名
            group_by: 分组方式 'year' 或 'overall'
            
        返回:
            归一化后的数据
        """
        print(f"\n  归一化指标: {indicator_name}")
        
        method = self.norm_config['method']
        
        normalized = indicator_data.copy()
        
        if indicator_name not in self.transform_params:
            self.transform_params[indicator_name] = {}
        
        prov_col = indicator_data.columns[0] if len(indicator_data.columns) > 0 else None
        year_cols = [c for c in indicator_data.columns if c != prov_col]
        
        if group_by == 'year':
            for year in year_cols:
                if year not in indicator_data.columns:
                    continue
                
                data = indicator_data[year]
                
                if method == 'zscore':
                    norm_data, params = self._zscore(data)
                elif method == 'minmax':
                    norm_data, params = self._minmax(data)
                else:
                    norm_data, params = self._zscore(data)
                
                normalized[year] = norm_data
                self.transform_params[indicator_name][str(year)] = {
                    'method': method,
                    'params': params
                }
        
        else:
            all_values = []
            for year in year_cols:
                if year in indicator_data.columns:
                    all_values.extend(indicator_data[year].dropna().tolist())
            
            if len(all_values) > 0:
                all_series = pd.Series(all_values)
                
                if method == 'zscore':
                    mean_val = all_series.mean()
                    std_val = all_series.std()
                    if std_val == 0:
                        std_val = 1e-6
                    
                    for year in year_cols:
                        if year in normalized.columns:
                            normalized[year] = (indicator_data[year] - mean_val) / std_val
                    
                    self.transform_params[indicator_name]['overall'] = {
                        'method': method,
                        'mean': mean_val,
                        'std': std_val
                    }
        
        print(f"    归一化方法: {method}, 分组: {group_by}")
        
        return normalized
    
    def normalize_all_indicators(self, data_by_indicator: Dict[str, pd.DataFrame],
                                   group_by: str = 'year') -> Dict[str, pd.DataFrame]:
        """
        对所有指标进行省份间归一化
        
        参数:
            data_by_indicator: 按指标组织的数据
            group_by: 分组方式
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 70)
        print("省份间归一化（按指标）")
        print("=" * 70)
        
        for indicator, data in data_by_indicator.items():
            normalized = self.normalize_indicator(data, indicator, group_by)
            self.normalized_data[indicator] = normalized
        
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
    
    def analyze_by_indicator(self, data_by_indicator: Dict[str, pd.DataFrame],
                              use_normalized: bool = True) -> Dict[str, Any]:
        """
        按指标分析省份间的相关性
        
        分析逻辑：
        - 对每个指标，获取各省份在各年份的值
        - 计算省份之间的相关系数（时间维度上的相似性）
        - 或计算各年份省份间的相关系数（横截面相似性）
        
        参数:
            data_by_indicator: 按指标组织的数据
            use_normalized: 是否使用归一化后的数据
            
        返回:
            相关性分析结果
        """
        print("\n" + "=" * 70)
        print("省份间相关性分析（按指标）")
        print("=" * 70)
        
        group_by = self.corr_config.get('group_by', 'indicator')
        
        result = {
            'analysis_type': 'across_province_by_indicator',
            'group_by': group_by,
            'indicators': self.indicators,
            'provinces': self.provinces,
            'by_indicator': {},
            'overall_similarity': None
        }
        
        for indicator in self.indicators:
            if indicator not in data_by_indicator:
                continue
            
            print(f"\n【{indicator}】")
            
            if use_normalized and indicator in self.normalized_data:
                data = self.normalized_data[indicator]
            else:
                data = data_by_indicator[indicator]
            
            prov_col = data.columns[0] if len(data.columns) > 0 else None
            year_cols = [c for c in data.columns if c != prov_col]
            
            if prov_col is None or len(year_cols) < 2:
                print(f"    警告: 数据不足，跳过")
                continue
            
            provinces = data[prov_col].tolist()
            n_provinces = len(provinces)
            
            if n_provinces < 2:
                print(f"    警告: 省份数不足 ({n_provinces} 个)，跳过")
                continue
            
            ind_result = {
                'indicator': indicator,
                'provinces': provinces,
                'year_cols': year_cols,
                'correlation_matrix': pd.DataFrame(),
                'p_value_matrix': pd.DataFrame(),
                'similar_provinces': [],
                'pairwise': {}
            }
            
            prov_data = data.set_index(prov_col)[year_cols]
            
            corr_matrix = pd.DataFrame(np.eye(n_provinces),
                                        index=provinces,
                                        columns=provinces)
            p_matrix = pd.DataFrame(np.ones((n_provinces, n_provinces)),
                                    index=provinces,
                                    columns=provinces)
            
            for i, prov1 in enumerate(provinces):
                for j, prov2 in enumerate(provinces):
                    if i == j:
                        continue
                    
                    corr_result = self._calculate_correlation(
                        prov_data.loc[prov1], prov_data.loc[prov2]
                    )
                    
                    corr_matrix.loc[prov1, prov2] = corr_result['correlation']
                    p_matrix.loc[prov1, prov2] = corr_result['p_value']
                    
                    if i < j:
                        pair_key = f"{prov1} ↔ {prov2}"
                        ind_result['pairwise'][pair_key] = corr_result
                        
                        if corr_result.get('significant', False):
                            ind_result['similar_provinces'].append({
                                'province_1': prov1,
                                'province_2': prov2,
                                'correlation': corr_result['correlation'],
                                'p_value': corr_result['p_value'],
                                'n_obs': corr_result['n_obs'],
                                'abs_correlation': abs(corr_result['correlation'])
                            })
            
            ind_result['correlation_matrix'] = corr_matrix
            ind_result['p_value_matrix'] = p_matrix
            
            ind_result['similar_provinces'].sort(
                key=lambda x: x['abs_correlation'], reverse=True
            )
            
            result['by_indicator'][indicator] = ind_result
            
            print(f"    省份数: {n_provinces}, 年份数: {len(year_cols)}")
            print(f"    显著相似的省份对: {len(ind_result['similar_provinces'])} 个")
            
            for pair in ind_result['similar_provinces'][:5]:
                sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
                print(f"      {pair['province_1']} ↔ {pair['province_2']}: "
                      f"r={pair['correlation']:.4f}, p={pair['p_value']:.4f} {sig}")
        
        self.correlation_results = result
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: Dict[str, Any]):
        """打印分析汇总"""
        print("\n" + "=" * 70)
        print("省份间相关性分析汇总")
        print("=" * 70)
        
        if 'by_indicator' not in result:
            return
        
        all_similar = defaultdict(list)
        
        for indicator, ind_result in result['by_indicator'].items():
            for pair in ind_result.get('similar_provinces', []):
                pair_key = f"{pair['province_1']} ↔ {pair['province_2']}"
                all_similar[pair_key].append({
                    'indicator': indicator,
                    'correlation': pair['correlation'],
                    'p_value': pair['p_value']
                })
        
        print("\n多指标一致的相似省份对:")
        for pair_key, results in all_similar.items():
            n_indicators = len(results)
            if n_indicators >= 2:
                avg_corr = np.mean([r['correlation'] for r in results])
                
                print(f"\n  {pair_key}:")
                print(f"    显著指标数: {n_indicators}")
                print(f"    平均相关系数: {avg_corr:.4f}")
                
                for r in results[:5]:
                    sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else '*')
                    print(f"    {r['indicator']}: r={r['correlation']:.4f} {sig}")
    
    def analyze_province_similarity(self, data_by_indicator: Dict[str, pd.DataFrame],
                                      use_normalized: bool = True) -> Dict[str, Any]:
        """
        分析省份之间的综合相似性（基于所有指标）
        
        参数:
            data_by_indicator: 按指标组织的数据
            use_normalized: 是否使用归一化后的数据
            
        返回:
            相似性分析结果
        """
        print("\n" + "=" * 70)
        print("省份综合相似性分析")
        print("=" * 70)
        
        all_indicator_data = {}
        
        for indicator in self.indicators:
            if indicator not in data_by_indicator:
                continue
            
            if use_normalized and indicator in self.normalized_data:
                data = self.normalized_data[indicator]
            else:
                data = data_by_indicator[indicator]
            
            prov_col = data.columns[0] if len(data.columns) > 0 else None
            if prov_col is None:
                continue
            
            melted = data.melt(id_vars=[prov_col], var_name='年份', value_name=indicator)
            melted['年份'] = melted['年份'].astype(str)
            melted['_key'] = melted[prov_col] + '_' + melted['年份']
            
            all_indicator_data[indicator] = melted.set_index('_key')[indicator]
        
        if not all_indicator_data:
            print("警告: 没有有效数据")
            return {}
        
        combined = pd.concat(all_indicator_data.values(), axis=1, keys=all_indicator_data.keys())
        
        combined['省份'] = combined.index.str.split('_').str[0]
        combined['年份'] = combined.index.str.split('_').str[1]
        
        provinces = combined['省份'].unique()
        
        print(f"省份数: {len(provinces)}")
        print(f"指标数: {len([c for c in combined.columns if c not in ['省份', '年份']])}")
        
        province_vectors = {}
        for province in provinces:
            prov_data = combined[combined['省份'] == province].copy()
            prov_data = prov_data.sort_values('年份')
            
            indicator_cols = [c for c in prov_data.columns if c not in ['省份', '年份']]
            
            flattened = []
            for year in sorted(prov_data['年份'].unique()):
                year_data = prov_data[prov_data['年份'] == year]
                for ind in indicator_cols:
                    if ind in year_data.columns:
                        flattened.append(year_data[ind].iloc[0] if len(year_data) > 0 else np.nan)
            
            province_vectors[province] = pd.Series(flattened)
        
        n_provinces = len(province_vectors)
        if n_provinces < 2:
            print("警告: 省份数不足")
            return {}
        
        provinces_list = list(province_vectors.keys())
        
        similarity_matrix = pd.DataFrame(np.eye(n_provinces),
                                          index=provinces_list,
                                          columns=provinces_list)
        
        similar_pairs = []
        
        for i, prov1 in enumerate(provinces_list):
            for j, prov2 in enumerate(provinces_list):
                if i >= j:
                    continue
                
                corr_result = self._calculate_correlation(
                    province_vectors[prov1], province_vectors[prov2]
                )
                
                similarity_matrix.loc[prov1, prov2] = corr_result['correlation']
                similarity_matrix.loc[prov2, prov1] = corr_result['correlation']
                
                if corr_result.get('significant', False):
                    similar_pairs.append({
                        'province_1': prov1,
                        'province_2': prov2,
                        'correlation': corr_result['correlation'],
                        'p_value': corr_result['p_value'],
                        'abs_correlation': abs(corr_result['correlation'])
                    })
        
        similar_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        result = {
            'analysis_type': 'province_similarity',
            'provinces': provinces_list,
            'similarity_matrix': similarity_matrix,
            'similar_pairs': similar_pairs
        }
        
        print(f"\n综合相似的省份对: {len(similar_pairs)} 个")
        for pair in similar_pairs[:10]:
            sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
            print(f"  {pair['province_1']} ↔ {pair['province_2']}: "
                  f"r={pair['correlation']:.4f}, p={pair['p_value']:.4f} {sig}")
        
        return result
    
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

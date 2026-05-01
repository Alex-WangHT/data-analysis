"""
指标相关性分析模块
专门用于计算指标与指标之间的相关性

支持两种分析方式：
1. 省份内分析：每个省份内部，指标A与指标B在时间上的相关性
2. 整体分析：所有省所有年放在一起，指标A与指标B的相关性
"""

import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class IndicatorCorrelationAnalyzer:
    """
    指标相关性分析器
    专门计算指标与指标之间的相关性
    """
    
    def __init__(self, method: str = 'pearson', alpha: float = 0.05, min_obs: int = 5):
        """
        初始化分析器
        
        参数:
            method: 相关系数方法 'pearson' | 'spearman' | 'kendall'
            alpha: 显著性水平
            min_obs: 最小观测数要求
        """
        self.method = method
        self.alpha = alpha
        self.min_obs = min_obs
        
        self.within_province_results: Dict[str, Dict[str, Any]] = {}
        self.overall_results: Dict[str, Any] = {}
        
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
    
    def _is_near_constant(self, data: pd.Series, threshold: float = 1e-6) -> bool:
        """
        检查数据是否几乎是常数
        
        参数:
            data: 数据序列
            threshold: 标准差阈值
            
        返回:
            是否几乎是常数
        """
        valid_data = data.dropna()
        
        if len(valid_data) < 2:
            return True
        
        std_val = valid_data.std()
        range_val = valid_data.max() - valid_data.min()
        
        if std_val < threshold or range_val < threshold:
            return True
        
        return False
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series,
                                x_name: str = 'X', y_name: str = 'Y') -> Dict[str, Any]:
        """
        计算两个变量的相关系数
        
        参数:
            x: 变量 X
            y: 变量 Y
            x_name: X 名称（用于警告信息）
            y_name: Y 名称（用于警告信息）
            
        返回:
            相关系数结果
        """
        valid_mask = ~x.isnull() & ~y.isnull()
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < self.min_obs:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': self.method,
                'note': f'观测数不足 (需要 {self.min_obs}, 实际 {len(x_valid)})'
            }
        
        if self._is_near_constant(x_valid):
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': self.method,
                'note': f'指标 "{x_name}" 几乎是常数，无法计算有效相关性'
            }
        
        if self._is_near_constant(y_valid):
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': self.method,
                'note': f'指标 "{y_name}" 几乎是常数，无法计算有效相关性'
            }
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                
                if self.method == 'pearson':
                    corr, p_value = pearsonr(x_valid, y_valid)
                elif self.method == 'spearman':
                    corr, p_value = spearmanr(x_valid, y_valid)
                elif self.method == 'kendall':
                    corr, p_value = kendalltau(x_valid, y_valid)
                else:
                    corr, p_value = pearsonr(x_valid, y_valid)
                
                significant = p_value < self.alpha
                
                return {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_obs': len(x_valid),
                    'significant': significant,
                    'method': self.method,
                    'abs_correlation': abs(corr)
                }
                
        except Warning as w:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': self.method,
                'note': f'计算警告: {str(w)}'
            }
            
        except Exception as e:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'significant': False,
                'method': self.method,
                'note': f'计算错误: {str(e)}'
            }
    
    def analyze_within_province(self, data_by_province: Dict[str, pd.DataFrame],
                                  year_col: str = '年份') -> Dict[str, Dict[str, Any]]:
        """
        省份内指标相关性分析
        
        分析逻辑：
        - 对每个省份
        - 获取该省各指标在各年份的值
        - 计算指标A与指标B的相关系数（时间维度上的相关性）
        
        参数:
            data_by_province: 按省份组织的数据（行=年份，列=指标）
            year_col: 年份列名
            
        返回:
            各省份的相关性分析结果
        """
        print("\n" + "=" * 70)
        print("省份内指标相关性分析")
        print("=" * 70)
        print(f"\n分析方法: 对每个省份，计算该省内各指标在时间维度上的相关性")
        print(f"相关系数方法: {self.method}")
        print(f"显著性水平: α = {self.alpha}")
        
        for province, data in data_by_province.items():
            print(f"\n【{province}】")
            
            indicator_cols = [c for c in data.columns if c != year_col]
            n_indicators = len(indicator_cols)
            
            if n_indicators < 2:
                print(f"    警告: 指标数不足 ({n_indicators} 个)，无法计算相关性")
                continue
            
            result = {
                'province': province,
                'n_years': len(data),
                'indicators': indicator_cols,
                'correlation_matrix': pd.DataFrame(),
                'p_value_matrix': pd.DataFrame(),
                'significant_pairs': [],
                'pairwise': {},
                'notes': []
            }
            
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
                        data[ind1], data[ind2], ind1, ind2
                    )
                    
                    corr_matrix.loc[ind1, ind2] = corr_result['correlation']
                    p_matrix.loc[ind1, ind2] = corr_result['p_value']
                    
                    if i < j:
                        pair_key = f"{ind1} ↔ {ind2}"
                        result['pairwise'][pair_key] = corr_result
                        
                        if 'note' in corr_result and corr_result['note']:
                            result['notes'].append(f"{pair_key}: {corr_result['note']}")
                        
                        if corr_result.get('significant', False):
                            result['significant_pairs'].append({
                                'indicator_1': ind1,
                                'indicator_2': ind2,
                                'correlation': corr_result['correlation'],
                                'p_value': corr_result['p_value'],
                                'n_obs': corr_result['n_obs'],
                                'abs_correlation': corr_result.get('abs_correlation', abs(corr_result['correlation']))
                            })
            
            result['correlation_matrix'] = corr_matrix
            result['p_value_matrix'] = p_matrix
            
            result['significant_pairs'].sort(
                key=lambda x: x['abs_correlation'], reverse=True
            )
            
            self.within_province_results[province] = result
            
            print(f"    年份数: {result['n_years']}")
            print(f"    指标数: {n_indicators}")
            print(f"    显著相关对: {len(result['significant_pairs'])} 个")
            
            if result['notes']:
                print(f"    警告/说明:")
                for note in result['notes'][:5]:
                    print(f"      - {note}")
            
            for pair in result['significant_pairs'][:5]:
                sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
                direction = "正相关" if pair['correlation'] > 0 else "负相关"
                print(f"      {pair['indicator_1']} ↔ {pair['indicator_2']}: "
                      f"r={pair['correlation']:.4f} ({direction}), p={pair['p_value']:.4f} {sig}")
        
        self._print_within_province_summary()
        
        return self.within_province_results
    
    def _print_within_province_summary(self):
        """打印省份内分析汇总"""
        print("\n" + "=" * 70)
        print("省份内相关性分析汇总")
        print("=" * 70)
        
        all_significant = defaultdict(list)
        
        for province, result in self.within_province_results.items():
            for pair in result.get('significant_pairs', []):
                pair_key = f"{pair['indicator_1']} ↔ {pair['indicator_2']}"
                all_significant[pair_key].append({
                    'province': province,
                    'correlation': pair['correlation'],
                    'p_value': pair['p_value']
                })
        
        if all_significant:
            print("\n跨省份一致的显著相关对:")
            for pair_key, results in all_significant.items():
                n_provinces = len(results)
                if n_provinces >= 2:
                    avg_corr = np.mean([r['correlation'] for r in results])
                    
                    pos_count = sum(1 for r in results if r['correlation'] > 0)
                    neg_count = sum(1 for r in results if r['correlation'] < 0)
                    
                    direction = "正相关为主" if pos_count > neg_count else "负相关为主"
                    
                    print(f"\n  {pair_key}:")
                    print(f"    显著省份数: {n_provinces}")
                    print(f"    平均相关系数: {avg_corr:.4f}")
                    print(f"    方向: {direction} (正:{pos_count}, 负:{neg_count})")
                    
                    for r in results[:5]:
                        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else '*')
                        print(f"    {r['province']}: r={r['correlation']:.4f} {sig}")
        else:
            print("\n没有在多个省份中都显著的相关对")
    
    def analyze_overall(self, all_data: pd.DataFrame,
                        province_col: str = '省份',
                        indicator_col: str = '指标',
                        year_col: str = '年份',
                        value_col: str = '数值') -> Dict[str, Any]:
        """
        整体指标相关性分析
        
        分析逻辑：
        - 把所有省份所有年份的数据放在一起
        - 每一行 = 某个省某年的所有指标值
        - 计算指标A与指标B的相关系数
        
        参数:
            all_data: 长格式数据
            province_col: 省份列名
            indicator_col: 指标列名
            year_col: 年份列名
            value_col: 数值列名
            
        返回:
            整体相关性分析结果
        """
        print("\n" + "=" * 70)
        print("整体指标相关性分析（所有省所有年）")
        print("=" * 70)
        print(f"\n分析方法: 把所有省所有年的数据放在一起，计算指标间的相关性")
        print(f"相关系数方法: {self.method}")
        print(f"显著性水平: α = {self.alpha}")
        
        pivoted = all_data.pivot_table(
            index=[province_col, year_col],
            columns=indicator_col,
            values=value_col,
            aggfunc='first'
        ).reset_index()
        
        indicator_cols = [c for c in pivoted.columns if c not in [province_col, year_col]]
        n_indicators = len(indicator_cols)
        
        print(f"\n数据形状: {len(pivoted)} 行 × {len(pivoted.columns)} 列")
        print(f"指标数: {n_indicators}")
        
        if n_indicators < 2:
            print("警告: 指标数不足，无法计算相关性")
            return {}
        
        result = {
            'analysis_type': 'overall',
            'n_obs': len(pivoted),
            'indicators': indicator_cols,
            'correlation_matrix': pd.DataFrame(),
            'p_value_matrix': pd.DataFrame(),
            'significant_pairs': [],
            'pairwise': {},
            'notes': []
        }
        
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
                    pivoted[ind1], pivoted[ind2], ind1, ind2
                )
                
                corr_matrix.loc[ind1, ind2] = corr_result['correlation']
                p_matrix.loc[ind1, ind2] = corr_result['p_value']
                
                if i < j:
                    pair_key = f"{ind1} ↔ {ind2}"
                    result['pairwise'][pair_key] = corr_result
                    
                    if 'note' in corr_result and corr_result['note']:
                        result['notes'].append(f"{pair_key}: {corr_result['note']}")
                    
                    if corr_result.get('significant', False):
                        result['significant_pairs'].append({
                            'indicator_1': ind1,
                            'indicator_2': ind2,
                            'correlation': corr_result['correlation'],
                            'p_value': corr_result['p_value'],
                            'n_obs': corr_result['n_obs'],
                            'abs_correlation': corr_result.get('abs_correlation', abs(corr_result['correlation']))
                        })
        
        result['correlation_matrix'] = corr_matrix
        result['p_value_matrix'] = p_matrix
        
        result['significant_pairs'].sort(
            key=lambda x: x['abs_correlation'], reverse=True
        )
        
        self.overall_results = result
        
        print(f"\n显著相关对: {len(result['significant_pairs'])} 个")
        
        if result['notes']:
            print(f"\n警告/说明:")
            for note in result['notes']:
                print(f"  - {note}")
        
        print(f"\n所有指标对的相关性（按显著性排序）:")
        for pair in result['significant_pairs']:
            sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else ('*' if pair['p_value'] < 0.05 else ''))
            direction = "正相关" if pair['correlation'] > 0 else "负相关"
            strength = "强相关" if abs(pair['correlation']) > 0.7 else (
                "中等相关" if abs(pair['correlation']) > 0.5 else "弱相关"
            )
            print(f"  {pair['indicator_1']} ↔ {pair['indicator_2']}: "
                  f"r={pair['correlation']:.4f} ({direction}, {strength}), "
                  f"p={pair['p_value']:.4f} {sig}")
        
        return result
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取所有结果
        
        返回:
            结果字典
        """
        return {
            'within_province': self.within_province_results,
            'overall': self.overall_results,
            'config': {
                'method': self.method,
                'alpha': self.alpha,
                'min_obs': self.min_obs
            }
        }

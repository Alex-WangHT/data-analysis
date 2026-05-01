"""
面板数据相关性分析模块
支持多种相关性分析方法，针对面板数据结构优化
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from panel_config import CORRELATION_CONFIG


class CorrelationAnalyzer:
    """
    相关性分析器
    支持跨省市、时序等多种相关性分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化相关性分析器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.corr_config = {**CORRELATION_CONFIG, **self.config.get('correlation', {})}
        
        self.method = self.corr_config.get('method', 'pearson')
        self.results: Dict[str, Any] = {}
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series,
                                method: Optional[str] = None) -> Dict[str, Any]:
        """
        计算两个变量之间的相关系数
        
        参数:
            x: 变量 X
            y: 变量 Y
            method: 相关系数方法 'pearson' | 'spearman' | 'kendall'
            
        返回:
            包含相关系数和p值的字典
        """
        method = method or self.method
        
        valid_mask = ~x.isnull() & ~y.isnull()
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'method': method,
                'significant': False
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
            
            significant = p_value < 0.05
            
            return {
                'correlation': corr,
                'p_value': p_value,
                'n_obs': len(x_valid),
                'method': method,
                'significant': significant,
                'abs_correlation': abs(corr)
            }
            
        except Exception as e:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': len(x_valid),
                'method': method,
                'significant': False,
                'error': str(e)
            }
    
    def cross_province_correlation(self, wide_data: pd.DataFrame,
                                     group_by: str = 'year',
                                     method: Optional[str] = None) -> Dict[str, Any]:
        """
        跨省市指标相关性分析
        
        分析逻辑：
        - 按年份分组（group_by='year'）
        - 对每一年，计算所有省份的指标之间的相关性
        - 或整体计算（group_by='overall'）
        
        参数:
            wide_data: 宽格式数据，列=['地区', '年份', '指标1', '指标2', ...]
            group_by: 分组方式 'year' 或 'overall'
            method: 相关系数方法
            
        返回:
            相关性分析结果
        """
        print("\n" + "=" * 60)
        print("跨省市指标相关性分析")
        print("=" * 60)
        print(f"分组方式: {group_by}")
        
        method = method or self.method
        print(f"相关系数方法: {method}")
        
        indicator_cols = [c for c in wide_data.columns if c not in ['地区', '年份']]
        print(f"\n指标列表: {indicator_cols}")
        
        results = {
            'analysis_type': 'cross_province',
            'group_by': group_by,
            'method': method,
            'indicators': indicator_cols,
            'by_group': {},
            'overall': None
        }
        
        if group_by == 'year':
            years = sorted(wide_data['年份'].unique())
            print(f"\n年份数量: {len(years)}")
            
            for year in years:
                year_data = wide_data[wide_data['年份'] == year].copy()
                
                if len(year_data) < 3:
                    print(f"\n  年份 {year}: 样本量不足 ({len(year_data)} 个省份)，跳过")
                    continue
                
                print(f"\n  年份 {year}: {len(year_data)} 个省份")
                
                year_corr = {
                    'correlation_matrix': pd.DataFrame(),
                    'p_values': pd.DataFrame(),
                    'significant_pairs': [],
                    'pairwise': {}
                }
                
                corr_matrix = np.zeros((len(indicator_cols), len(indicator_cols)))
                p_matrix = np.ones((len(indicator_cols), len(indicator_cols)))
                
                for i, ind1 in enumerate(indicator_cols):
                    for j, ind2 in enumerate(indicator_cols):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                            p_matrix[i, j] = 1.0
                            continue
                        
                        corr_result = self._calculate_correlation(
                            year_data[ind1], year_data[ind2], method
                        )
                        
                        corr_matrix[i, j] = corr_result['correlation']
                        p_matrix[i, j] = corr_result['p_value']
                        
                        if i < j:
                            pair_key = f"{ind1} ↔ {ind2}"
                            year_corr['pairwise'][pair_key] = corr_result
                            
                            if corr_result['significant']:
                                year_corr['significant_pairs'].append({
                                    'indicator_1': ind1,
                                    'indicator_2': ind2,
                                    'correlation': corr_result['correlation'],
                                    'p_value': corr_result['p_value'],
                                    'abs_correlation': abs(corr_result['correlation'])
                                })
                
                year_corr['correlation_matrix'] = pd.DataFrame(
                    corr_matrix, index=indicator_cols, columns=indicator_cols
                )
                year_corr['p_values'] = pd.DataFrame(
                    p_matrix, index=indicator_cols, columns=indicator_cols
                )
                
                year_corr['significant_pairs'].sort(
                    key=lambda x: x['abs_correlation'], reverse=True
                )
                
                results['by_group'][year] = year_corr
                
                print(f"    显著相关的指标对: {len(year_corr['significant_pairs'])} 个")
                for pair in year_corr['significant_pairs'][:5]:
                    sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
                    print(f"      {pair['indicator_1']} ↔ {pair['indicator_2']}: "
                          f"r={pair['correlation']:.4f}, p={pair['p_value']:.4f} {sig}")
        
        else:
            print(f"\n整体分析: {len(wide_data)} 个观测")
            
            overall_corr = {
                'correlation_matrix': pd.DataFrame(),
                'p_values': pd.DataFrame(),
                'significant_pairs': [],
                'pairwise': {}
            }
            
            corr_matrix = np.zeros((len(indicator_cols), len(indicator_cols)))
            p_matrix = np.ones((len(indicator_cols), len(indicator_cols)))
            
            for i, ind1 in enumerate(indicator_cols):
                for j, ind2 in enumerate(indicator_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        p_matrix[i, j] = 1.0
                        continue
                    
                    corr_result = self._calculate_correlation(
                        wide_data[ind1], wide_data[ind2], method
                    )
                    
                    corr_matrix[i, j] = corr_result['correlation']
                    p_matrix[i, j] = corr_result['p_value']
                    
                    if i < j:
                        pair_key = f"{ind1} ↔ {ind2}"
                        overall_corr['pairwise'][pair_key] = corr_result
                        
                        if corr_result['significant']:
                            overall_corr['significant_pairs'].append({
                                'indicator_1': ind1,
                                'indicator_2': ind2,
                                'correlation': corr_result['correlation'],
                                'p_value': corr_result['p_value'],
                                'abs_correlation': abs(corr_result['correlation'])
                            })
            
            overall_corr['correlation_matrix'] = pd.DataFrame(
                corr_matrix, index=indicator_cols, columns=indicator_cols
            )
            overall_corr['p_values'] = pd.DataFrame(
                p_matrix, index=indicator_cols, columns=indicator_cols
            )
            
            overall_corr['significant_pairs'].sort(
                key=lambda x: x['abs_correlation'], reverse=True
            )
            
            results['overall'] = overall_corr
            
            print(f"\n  显著相关的指标对: {len(overall_corr['significant_pairs'])} 个")
            for pair in overall_corr['significant_pairs'][:10]:
                sig = '***' if pair['p_value'] < 0.001 else ('**' if pair['p_value'] < 0.01 else '*')
                print(f"    {pair['indicator_1']} ↔ {pair['indicator_2']}: "
                      f"r={pair['correlation']:.4f}, p={pair['p_value']:.4f} {sig}")
        
        self.results = results
        return results
    
    def time_series_correlation(self, wide_data: pd.DataFrame,
                                  group_by: str = 'province',
                                  method: Optional[str] = None) -> Dict[str, Any]:
        """
        时序相关性分析（按省份分组）
        
        分析逻辑：
        - 按省份分组
        - 对每个省份，计算该省内不同指标在时间上的相关性
        
        参数:
            wide_data: 宽格式数据
            group_by: 分组方式 'province'
            method: 相关系数方法
            
        返回:
            相关性分析结果
        """
        print("\n" + "=" * 60)
        print("时序相关性分析（按省份）")
        print("=" * 60)
        
        method = method or self.method
        
        indicator_cols = [c for c in wide_data.columns if c not in ['地区', '年份']]
        provinces = sorted(wide_data['地区'].unique())
        
        print(f"省份数量: {len(provinces)}")
        print(f"指标列表: {indicator_cols}")
        
        results = {
            'analysis_type': 'time_series',
            'group_by': group_by,
            'method': method,
            'indicators': indicator_cols,
            'by_province': {},
            'summary': defaultdict(list)
        }
        
        for province in provinces:
            prov_data = wide_data[wide_data['地区'] == province].copy()
            prov_data = prov_data.sort_values('年份')
            
            if len(prov_data) < 5:
                print(f"\n  省份 {province}: 时间点不足 ({len(prov_data)} 年)，跳过")
                continue
            
            print(f"\n  省份 {province}: {len(prov_data)} 年数据")
            
            prov_corr = {
                'correlation_matrix': pd.DataFrame(),
                'p_values': pd.DataFrame(),
                'significant_pairs': [],
                'pairwise': {}
            }
            
            corr_matrix = np.zeros((len(indicator_cols), len(indicator_cols)))
            p_matrix = np.ones((len(indicator_cols), len(indicator_cols)))
            
            for i, ind1 in enumerate(indicator_cols):
                for j, ind2 in enumerate(indicator_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        p_matrix[i, j] = 1.0
                        continue
                    
                    corr_result = self._calculate_correlation(
                        prov_data[ind1], prov_data[ind2], method
                    )
                    
                    corr_matrix[i, j] = corr_result['correlation']
                    p_matrix[i, j] = corr_result['p_value']
                    
                    if i < j:
                        pair_key = f"{ind1} ↔ {ind2}"
                        prov_corr['pairwise'][pair_key] = corr_result
                        
                        if corr_result['significant']:
                            prov_corr['significant_pairs'].append({
                                'indicator_1': ind1,
                                'indicator_2': ind2,
                                'correlation': corr_result['correlation'],
                                'p_value': corr_result['p_value'],
                                'abs_correlation': abs(corr_result['correlation'])
                            })
                            
                            results['summary'][pair_key].append({
                                'province': province,
                                'correlation': corr_result['correlation'],
                                'p_value': corr_result['p_value']
                            })
            
            prov_corr['correlation_matrix'] = pd.DataFrame(
                corr_matrix, index=indicator_cols, columns=indicator_cols
            )
            prov_corr['p_values'] = pd.DataFrame(
                p_matrix, index=indicator_cols, columns=indicator_cols
            )
            
            results['by_province'][province] = prov_corr
            
            print(f"    显著相关的指标对: {len(prov_corr['significant_pairs'])} 个")
        
        print("\n" + "=" * 60)
        print("时序相关性汇总")
        print("=" * 60)
        
        print("\n跨省份一致的显著相关:")
        for pair_key, prov_results in results['summary'].items():
            n_provinces = len(prov_results)
            if n_provinces >= 3:
                avg_corr = np.mean([r['correlation'] for r in prov_results])
                print(f"\n  {pair_key}:")
                print(f"    显著省份数: {n_provinces}")
                print(f"    平均相关系数: {avg_corr:.4f}")
                for r in prov_results[:5]:
                    sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else '*')
                    print(f"    {r['province']}: r={r['correlation']:.4f} {sig}")
        
        return results
    
    def export_results_to_excel(self, results: Dict[str, Any],
                                  output_file: str = '相关性分析结果.xlsx'):
        """
        导出相关性分析结果到 Excel
        
        参数:
            results: 分析结果
            output_file: 输出文件名
        """
        print("\n" + "=" * 60)
        print(f"导出结果到: {output_file}")
        print("=" * 60)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            if results['analysis_type'] == 'cross_province':
                if results['overall'] is not None:
                    overall = results['overall']
                    
                    overall['correlation_matrix'].to_excel(writer, sheet_name='相关系数矩阵')
                    overall['p_values'].to_excel(writer, sheet_name='P值矩阵')
                    
                    if overall['significant_pairs']:
                        sig_df = pd.DataFrame(overall['significant_pairs'])
                        sig_df['显著性'] = sig_df['p_value'].apply(
                            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                        )
                        sig_df.to_excel(writer, sheet_name='显著相关对', index=False)
                
                if results['by_group']:
                    all_pairs = []
                    for year, year_result in results['by_group'].items():
                        for pair in year_result['significant_pairs']:
                            pair['年份'] = year
                            all_pairs.append(pair)
                    
                    if all_pairs:
                        all_pairs_df = pd.DataFrame(all_pairs)
                        all_pairs_df['显著性'] = all_pairs_df['p_value'].apply(
                            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                        )
                        all_pairs_df.to_excel(writer, sheet_name='分年度显著相关', index=False)
            
            elif results['analysis_type'] == 'time_series':
                all_sig = []
                for province, prov_result in results['by_province'].items():
                    for pair in prov_result['significant_pairs']:
                        pair['省份'] = province
                        all_sig.append(pair)
                
                if all_sig:
                    all_sig_df = pd.DataFrame(all_sig)
                    all_sig_df['显著性'] = all_sig_df['p_value'].apply(
                        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                    )
                    all_sig_df.to_excel(writer, sheet_name='分省份显著相关', index=False)
            
            info_df = pd.DataFrame({
                '项目': [
                    '分析类型',
                    '分组方式',
                    '相关系数方法',
                    '指标数量',
                    '生成时间'
                ],
                '值': [
                    results['analysis_type'],
                    results.get('group_by', 'N/A'),
                    results['method'],
                    len(results['indicators']),
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            })
            info_df.to_excel(writer, sheet_name='分析信息', index=False)
            
            indicators_df = pd.DataFrame({
                '指标名称': results['indicators'],
                '序号': range(1, len(results['indicators']) + 1)
            })
            indicators_df.to_excel(writer, sheet_name='指标列表', index=False)
        
        print(f"结果已导出到: {output_file}")
    
    def print_correlation_heatmap_data(self, results: Dict[str, Any],
                                         year: Optional[int] = None):
        """
        打印相关系数矩阵（用于可视化）
        
        参数:
            results: 分析结果
            year: 指定年份（仅对按年分组有效）
        """
        if results['analysis_type'] == 'cross_province':
            if year is not None and year in results['by_group']:
                corr_matrix = results['by_group'][year]['correlation_matrix']
                print(f"\n相关系数矩阵 (年份 {year}):")
            elif results['overall'] is not None:
                corr_matrix = results['overall']['correlation_matrix']
                print(f"\n整体相关系数矩阵:")
            else:
                print("没有可用的相关系数矩阵")
                return
        else:
            print("此函数仅适用于跨省市分析结果")
            return
        
        print("\n" + "=" * 80)
        print("相关系数矩阵 (上三角显示):")
        print("=" * 80)
        
        cols = corr_matrix.columns.tolist()
        
        header = " " * 12
        for col in cols:
            header += f"{col[:10]:>12}"
        print(header)
        
        for i, row in enumerate(cols):
            row_str = f"{row[:10]:<12}"
            for j, col in enumerate(cols):
                if i >= j:
                    row_str += " " * 12
                else:
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.7:
                        row_str += f"{'***' if val > 0 else ''}{val:>9.3f}   "
                    elif abs(val) > 0.5:
                        row_str += f"{'**' if val > 0 else ''}{val:>10.3f}  "
                    elif abs(val) > 0.3:
                        row_str += f"{'*' if val > 0 else ''}{val:>11.3f} "
                    else:
                        row_str += f"{val:>12.3f}"
            print(row_str)
        
        print("\n注: * |r| > 0.3, ** |r| > 0.5, *** |r| > 0.7 (仅为参考，非统计显著性)")

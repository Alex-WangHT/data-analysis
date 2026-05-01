"""
面板数据归一化模块
支持按指标、按省份、整体等多种归一化方式
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, boxcox
from typing import Dict, List, Optional, Tuple, Any

from panel_config import NORMALIZATION_CONFIG


class PanelNormalizer:
    """
    面板数据归一化器
    支持多种归一化策略，针对面板数据结构优化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化归一化器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.norm_config = {**NORMALIZATION_CONFIG, **self.config.get('normalization', {})}
        
        self.transform_params: Dict[str, Dict[str, Any]] = {}
        self.method = self.norm_config.get('method', 'by_indicator')
        
        self.original_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None
    
    def _zscore(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Z-score 标准化
        
        参数:
            data: 输入数据
            
        返回:
            (标准化后的数据, 参数字典)
        """
        data = data.copy()
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'mean': 0, 'std': 1}
        
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        
        if std_val == 0:
            std_val = 1e-6
        
        data[valid_mask] = (valid_data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val}
        
        return data, params
    
    def _robust_zscore(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """
        稳健 Z-score 标准化（使用中位数和MAD）
        
        参数:
            data: 输入数据
            
        返回:
            (标准化后的数据, 参数字典)
        """
        data = data.copy()
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'median': 0, 'mad': 1}
        
        median_val = valid_data.median()
        
        mad = np.median(np.abs(valid_data - median_val))
        if mad == 0:
            mad = 1e-6
        
        data[valid_mask] = (valid_data - median_val) / (mad * 1.4826)
        params = {'median': median_val, 'mad': mad}
        
        return data, params
    
    def _minmax(self, data: pd.Series, feature_range: Tuple[float, float] = (0, 1)
                 ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Min-Max 归一化
        
        参数:
            data: 输入数据
            feature_range: 目标范围
            
        返回:
            (归一化后的数据, 参数字典)
        """
        data = data.copy()
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return data, {'min': 0, 'max': 1, 'range_min': feature_range[0], 'range_max': feature_range[1]}
        
        min_val = valid_data.min()
        max_val = valid_data.max()
        range_min, range_max = feature_range
        
        if max_val == min_val:
            data[valid_mask] = (range_min + range_max) / 2
        else:
            data[valid_mask] = (valid_data - min_val) / (max_val - min_val) * (range_max - range_min) + range_min
        
        params = {'min': min_val, 'max': max_val, 'range_min': range_min, 'range_max': range_max}
        
        return data, params
    
    def _boxcox_transform(self, data: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Box-Cox 变换（用于偏态分布）
        
        参数:
            data: 输入数据
            
        返回:
            (变换后的数据, 参数字典)
        """
        data = data.copy()
        valid_mask = ~data.isnull()
        valid_data = data[valid_mask]
        
        if len(valid_data) < 2:
            return data, {'lambda': None, 'constant': 0}
        
        min_val = valid_data.min()
        constant = 0.0
        if min_val <= 0:
            constant = abs(min_val) + 1e-6
            valid_data = valid_data + constant
        
        try:
            transformed, lmbda = boxcox(valid_data)
            data[valid_mask] = transformed
            params = {'lambda': lmbda, 'constant': constant, 'method': 'boxcox'}
        except:
            lmbda = 0
            transformed = np.log(valid_data)
            data[valid_mask] = transformed
            params = {'lambda': lmbda, 'constant': constant, 'method': 'log'}
        
        return data, params
    
    def normalize_by_indicator(self, panel_data: pd.DataFrame,
                                group_by: str = 'year',
                                method: str = 'zscore',
                                handle_skewness: bool = False) -> pd.DataFrame:
        """
        按指标归一化（省间横向比较）
        
        对每个指标：
        - 如果 group_by='year': 每年内，对所有省份的该指标进行归一化
        - 如果 group_by='overall': 所有年份所有省份整体归一化
        
        参数:
            panel_data: 面板数据（长格式）
            group_by: 分组方式 'year' 或 'overall'
            method: 归一化方法 'zscore' | 'robust' | 'minmax'
            handle_skewness: 是否先处理偏态分布
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 60)
        print("按指标归一化（省间横向比较）")
        print("=" * 60)
        print(f"分组方式: {group_by}")
        print(f"归一化方法: {method}")
        
        result = panel_data.copy()
        result['标准化数值'] = np.nan
        
        indicators = sorted(result['指标'].unique())
        print(f"\n指标数量: {len(indicators)}")
        
        self.transform_params = {
            'method': f'by_indicator_{group_by}_{method}',
            'indicators': {}
        }
        
        for indicator in indicators:
            print(f"\n处理指标: {indicator}")
            
            indicator_mask = result['指标'] == indicator
            indicator_data = result[indicator_mask].copy()
            
            if handle_skewness:
                valid_vals = indicator_data['数值'].dropna()
                if len(valid_vals) > 10:
                    skewness = skew(valid_vals)
                    if abs(skewness) > 0.5:
                        print(f"  检测到偏态分布 (偏度={skewness:.4f})，执行 Box-Cox 变换")
                        transformed, bc_params = self._boxcox_transform(indicator_data['数值'])
                        indicator_data['数值'] = transformed
                        
                        if indicator not in self.transform_params['indicators']:
                            self.transform_params['indicators'][indicator] = {}
                        self.transform_params['indicators'][indicator]['boxcox'] = bc_params
            
            if group_by == 'year':
                years = sorted(indicator_data['年份'].unique())
                print(f"  年份数: {len(years)}")
                
                if indicator not in self.transform_params['indicators']:
                    self.transform_params['indicators'][indicator] = {}
                self.transform_params['indicators'][indicator]['years'] = {}
                
                for year in years:
                    year_mask = indicator_data['年份'] == year
                    year_data = indicator_data[year_mask]
                    
                    if method == 'zscore':
                        normalized, params = self._zscore(year_data['数值'])
                    elif method == 'robust':
                        normalized, params = self._robust_zscore(year_data['数值'])
                    elif method == 'minmax':
                        normalized, params = self._minmax(year_data['数值'])
                    else:
                        normalized, params = self._zscore(year_data['数值'])
                    
                    result.loc[indicator_mask & year_mask, '标准化数值'] = normalized.values
                    self.transform_params['indicators'][indicator]['years'][year] = params
                    
            else:
                if method == 'zscore':
                    normalized, params = self._zscore(indicator_data['数值'])
                elif method == 'robust':
                    normalized, params = self._robust_zscore(indicator_data['数值'])
                elif method == 'minmax':
                    normalized, params = self._minmax(indicator_data['数值'])
                else:
                    normalized, params = self._zscore(indicator_data['数值'])
                
                result.loc[indicator_mask, '标准化数值'] = normalized.values
                self.transform_params['indicators'][indicator] = {'overall': params}
        
        print("\n" + "-" * 60)
        print("归一化统计:")
        for indicator in indicators:
            mask = result['指标'] == indicator
            orig_stats = result.loc[mask, '数值'].describe()
            norm_stats = result.loc[mask, '标准化数值'].describe()
            
            print(f"\n  {indicator}:")
            print(f"    原始 - 均值={orig_stats['mean']:.4f}, 标准差={orig_stats['std']:.4f}")
            print(f"    标准化 - 均值={norm_stats['mean']:.4f}, 标准差={norm_stats['std']:.4f}")
        
        self.original_data = panel_data.copy()
        self.normalized_data = result
        
        return result
    
    def normalize_by_province(self, panel_data: pd.DataFrame,
                               group_by: str = 'indicator',
                               method: str = 'zscore') -> pd.DataFrame:
        """
        按省份归一化（省内纵向比较）
        
        对每个省份的每个指标，在所有年份间进行归一化
        
        参数:
            panel_data: 面板数据
            group_by: 分组方式
            method: 归一化方法
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 60)
        print("按省份归一化（省内纵向比较）")
        print("=" * 60)
        
        result = panel_data.copy()
        result['标准化数值'] = np.nan
        
        provinces = sorted(result['地区'].unique())
        indicators = sorted(result['指标'].unique())
        
        self.transform_params = {
            'method': f'by_province_{method}',
            'provinces': {}
        }
        
        for province in provinces:
            print(f"\n处理省份: {province}")
            
            province_mask = result['地区'] == province
            province_data = result[province_mask]
            
            self.transform_params['provinces'][province] = {}
            
            for indicator in indicators:
                indicator_mask = province_data['指标'] == indicator
                indicator_data = province_data[indicator_mask]
                
                if len(indicator_data) == 0:
                    continue
                
                if method == 'zscore':
                    normalized, params = self._zscore(indicator_data['数值'])
                elif method == 'robust':
                    normalized, params = self._robust_zscore(indicator_data['数值'])
                elif method == 'minmax':
                    normalized, params = self._minmax(indicator_data['数值'])
                else:
                    normalized, params = self._zscore(indicator_data['数值'])
                
                result.loc[province_mask & (result['指标'] == indicator), '标准化数值'] = normalized.values
                self.transform_params['provinces'][province][indicator] = params
        
        self.original_data = panel_data.copy()
        self.normalized_data = result
        
        return result
    
    def normalize_overall(self, panel_data: pd.DataFrame,
                          method: str = 'zscore') -> pd.DataFrame:
        """
        整体归一化（不分组）
        
        参数:
            panel_data: 面板数据
            method: 归一化方法
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 60)
        print("整体归一化")
        print("=" * 60)
        
        result = panel_data.copy()
        result['标准化数值'] = np.nan
        
        indicators = sorted(result['指标'].unique())
        
        self.transform_params = {
            'method': f'overall_{method}',
            'indicators': {}
        }
        
        for indicator in indicators:
            print(f"\n处理指标: {indicator}")
            
            indicator_mask = result['指标'] == indicator
            indicator_data = result[indicator_mask]
            
            if method == 'zscore':
                normalized, params = self._zscore(indicator_data['数值'])
            elif method == 'robust':
                normalized, params = self._robust_zscore(indicator_data['数值'])
            elif method == 'minmax':
                normalized, params = self._minmax(indicator_data['数值'])
            else:
                normalized, params = self._zscore(indicator_data['数值'])
            
            result.loc[indicator_mask, '标准化数值'] = normalized.values
            self.transform_params['indicators'][indicator] = params
        
        self.original_data = panel_data.copy()
        self.normalized_data = result
        
        return result
    
    def pivot_normalized_to_wide(self, normalized_data: Optional[pd.DataFrame] = None,
                                   value_col: str = '标准化数值') -> pd.DataFrame:
        """
        将归一化后的长格式数据转换为宽格式
        
        宽格式:
            列: ['地区', '年份', '指标1', '指标2', ...]
            值: 标准化后的数值
        
        参数:
            normalized_data: 归一化后的数据
            value_col: 值列名
            
        返回:
            宽格式数据
        """
        if normalized_data is None:
            normalized_data = self.normalized_data
        
        if normalized_data is None:
            raise ValueError("请先提供或执行归一化")
        
        print("\n" + "=" * 60)
        print("转换为宽格式（用于相关性分析）")
        print("=" * 60)
        
        wide = normalized_data.pivot_table(
            index=['地区', '年份'],
            columns='指标',
            values=value_col,
            aggfunc='first'
        ).reset_index()
        
        print(f"宽格式形状: {wide.shape}")
        print(f"列名: {list(wide.columns)}")
        
        return wide
    
    def get_transform_params(self) -> Dict[str, Any]:
        """
        获取变换参数
        
        返回:
            变换参数字典
        """
        return self.transform_params.copy()

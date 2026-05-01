"""
Z-score 标准化模块
负责执行 Z-score 标准化和逆变换，支持稳健标准化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from config import ZSCORE_CONFIG


class ZScoreScaler:
    """
    Z-score 标准化器类
    支持标准 Z-score 和稳健 Z-score 标准化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Z-score 标准化器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.zscore_config = {**ZSCORE_CONFIG, **self.config.get('zscore', {})}
        
        self.use_robust = self.zscore_config['use_robust']
        
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.medians: Dict[str, float] = {}
        self.mads: Dict[str, float] = {}
    
    @staticmethod
    def _calculate_mad(data: np.ndarray) -> float:
        """
        计算绝对中位数偏差 (MAD)
        
        参数:
            data: 输入数组
            
        返回:
            MAD 值
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return mad
    
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None):
        """
        计算标准化参数（均值、标准差或中位数、MAD）
        
        参数:
            data: 输入 DataFrame
            columns: 需要标准化的列，None 表示所有列
        """
        if columns is None:
            columns = list(data.columns)
        
        print("\n" + "=" * 60)
        print("计算标准化参数")
        print("=" * 60)
        
        if self.use_robust:
            print("使用稳健标准化 (中位数 + MAD)")
        else:
            print("使用标准 Z-score 标准化 (均值 + 标准差)")
        
        print("-" * 60)
        
        for col in columns:
            if col not in data.columns:
                continue
            
            col_data = data[col].dropna().values
            
            if len(col_data) == 0:
                print(f"  警告: 列 '{col}' 无有效数据")
                continue
            
            if self.use_robust:
                median = np.median(col_data)
                mad = self._calculate_mad(col_data)
                
                if mad == 0:
                    mad = 1e-6
                    print(f"  警告: 列 '{col}' MAD 为 0，使用极小值代替")
                
                self.medians[col] = median
                self.mads[col] = mad
                
                print(f"  {col}: 中位数 = {median:.4f}, MAD = {mad:.4f}")
            else:
                mean_val = np.mean(col_data)
                std_val = np.std(col_data, ddof=1)
                
                if std_val == 0:
                    std_val = 1e-6
                    print(f"  警告: 列 '{col}' 标准差为 0，使用极小值代替")
                
                self.means[col] = mean_val
                self.stds[col] = std_val
                
                print(f"  {col}: 均值 = {mean_val:.4f}, 标准差 = {std_val:.4f}")
    
    def transform(self, data: pd.DataFrame, 
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        执行标准化变换
        
        参数:
            data: 输入 DataFrame
            columns: 需要标准化的列，None 表示所有已拟合的列
            
        返回:
            标准化后的 DataFrame
        """
        if columns is None:
            if self.use_robust:
                columns = list(self.medians.keys())
            else:
                columns = list(self.means.keys())
        
        result = data.copy()
        
        print("\n" + "=" * 60)
        print("执行 Z-score 标准化")
        print("=" * 60)
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if self.use_robust:
                if col not in self.medians:
                    print(f"  警告: 列 '{col}' 未拟合，跳过")
                    continue
                
                median = self.medians[col]
                mad = self.mads[col]
                
                result[col] = (data[col] - median) / (mad * 1.4826)
            else:
                if col not in self.means:
                    print(f"  警告: 列 '{col}' 未拟合，跳过")
                    continue
                
                mean_val = self.means[col]
                std_val = self.stds[col]
                
                result[col] = (data[col] - mean_val) / std_val
            
            print(f"  {col}: 已标准化")
        
        print("\n标准化后数据统计:")
        for col in columns:
            if col not in result.columns:
                continue
            
            col_data = result[col].dropna()
            print(f"  {col}: 均值 = {col_data.mean():.4f}, "
                  f"标准差 = {col_data.std():.4f}")
        
        return result
    
    def fit_transform(self, data: pd.DataFrame, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        拟合并变换
        
        参数:
            data: 输入 DataFrame
            columns: 需要标准化的列
            
        返回:
            标准化后的 DataFrame
        """
        self.fit(data, columns)
        return self.transform(data, columns)
    
    def inverse_transform(self, data: pd.DataFrame, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        执行逆变换（恢复原始尺度）
        
        参数:
            data: 标准化后的 DataFrame
            columns: 需要逆变换的列
            
        返回:
            原始尺度的 DataFrame
        """
        if columns is None:
            if self.use_robust:
                columns = list(self.medians.keys())
            else:
                columns = list(self.means.keys())
        
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if self.use_robust:
                if col not in self.medians:
                    continue
                
                median = self.medians[col]
                mad = self.mads[col]
                
                result[col] = data[col] * (mad * 1.4826) + median
            else:
                if col not in self.means:
                    continue
                
                mean_val = self.means[col]
                std_val = self.stds[col]
                
                result[col] = data[col] * std_val + mean_val
        
        return result
    
    def get_scaler_params(self) -> Dict[str, Any]:
        """
        获取标准化参数（用于保存和加载）
        
        返回:
            标准化参数字典
        """
        return {
            'use_robust': self.use_robust,
            'means': self.means.copy(),
            'stds': self.stds.copy(),
            'medians': self.medians.copy(),
            'mads': self.mads.copy()
        }
    
    def load_scaler_params(self, params: Dict[str, Any]):
        """
        加载标准化参数
        
        参数:
            params: 标准化参数字典
        """
        self.use_robust = params.get('use_robust', self.use_robust)
        self.means = params.get('means', {})
        self.stds = params.get('stds', {})
        self.medians = params.get('medians', {})
        self.mads = params.get('mads', {})


class StandardizationPipeline:
    """
    标准化流水线类
    整合 Box-Cox 变换和 Z-score 标准化的完整流程
    """
    
    def __init__(self, boxcox_transformer=None, zscore_scaler=None):
        """
        初始化标准化流水线
        
        参数:
            boxcox_transformer: Box-Cox 变换器实例
            zscore_scaler: Z-score 标准化器实例
        """
        self.boxcox_transformer = boxcox_transformer
        self.zscore_scaler = zscore_scaler
        
        self.skewed_columns: List[str] = []
        self.normal_columns: List[str] = []
        self.all_numeric_columns: List[str] = []
    
    def set_columns(self, skewed_columns: List[str], 
                    normal_columns: List[str]):
        """
        设置列分类
        
        参数:
            skewed_columns: 偏态分布列
            normal_columns: 近似正态分布列
        """
        self.skewed_columns = skewed_columns
        self.normal_columns = normal_columns
        self.all_numeric_columns = skewed_columns + normal_columns
    
    def transform(self, data: pd.DataFrame, 
                  distribution_stats: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        执行完整的标准化流程
        
        参数:
            data: 输入 DataFrame
            distribution_stats: 分布统计信息
            
        返回:
            标准化后的 DataFrame
        """
        result = data.copy()
        
        if self.skewed_columns and self.boxcox_transformer:
            result = self.boxcox_transformer.transform(
                result, self.skewed_columns, distribution_stats
            )
        
        if self.zscore_scaler:
            result = self.zscore_scaler.fit_transform(
                result, self.all_numeric_columns
            )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的逆变换流程
        
        参数:
            data: 标准化后的 DataFrame
            
        返回:
            原始尺度的 DataFrame
        """
        result = data.copy()
        
        if self.zscore_scaler:
            result = self.zscore_scaler.inverse_transform(
                result, self.all_numeric_columns
            )
        
        if self.skewed_columns and self.boxcox_transformer:
            for col in self.skewed_columns:
                if col in result.columns:
                    result[col] = self.boxcox_transformer.inverse_transform_column(
                        result[col], col
                    )
        
        return result
    
    def get_all_params(self) -> Dict[str, Any]:
        """
        获取所有变换参数
        
        返回:
            所有变换参数字典
        """
        params = {
            'skewed_columns': self.skewed_columns,
            'normal_columns': self.normal_columns,
            'all_numeric_columns': self.all_numeric_columns,
        }
        
        if self.boxcox_transformer:
            params['boxcox'] = self.boxcox_transformer.get_transform_params()
        
        if self.zscore_scaler:
            params['zscore'] = self.zscore_scaler.get_scaler_params()
        
        return params
    
    def load_all_params(self, params: Dict[str, Any]):
        """
        加载所有变换参数
        
        参数:
            params: 所有变换参数字典
        """
        self.skewed_columns = params.get('skewed_columns', [])
        self.normal_columns = params.get('normal_columns', [])
        self.all_numeric_columns = params.get('all_numeric_columns', [])
        
        if self.boxcox_transformer and 'boxcox' in params:
            self.boxcox_transformer.load_transform_params(params['boxcox'])
        
        if self.zscore_scaler and 'zscore' in params:
            self.zscore_scaler.load_scaler_params(params['zscore'])

"""
分布分析和 Box-Cox 变换模块
负责分析数据分布特征、检测偏态分布、执行 Box-Cox 变换
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, normaltest, boxcox
from typing import Dict, List, Optional, Tuple, Any

from config import DISTRIBUTION_CONFIG, BOXCOX_CONFIG


class DistributionAnalyzer:
    """
    分布分析器类
    负责分析数据分布特征，识别偏态分布
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分布分析器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.dist_config = {**DISTRIBUTION_CONFIG, **self.config.get('distribution', {})}
        
        self.skewness_threshold = self.dist_config['skewness_threshold']
        self.test_normal = self.dist_config['test_normal']
        self.alpha = self.dist_config['alpha']
        
        self.distribution_stats: Dict[str, Dict[str, Any]] = {}
        self.skewed_columns: List[str] = []
        self.normal_columns: List[str] = []
    
    def calculate_distribution_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        计算每列的分布统计量
        
        参数:
            data: 数值型 DataFrame
            
        返回:
            每列的分布统计量字典
        """
        stats_dict = {}
        
        print("\n" + "=" * 60)
        print("分布统计分析")
        print("=" * 60)
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            skewness = skew(col_data)
            kurt = kurtosis(col_data)
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            
            is_skewed = abs(skewness) > self.skewness_threshold
            
            normal_p_value = None
            is_normal = None
            if self.test_normal and len(col_data) >= 8:
                try:
                    stat, normal_p_value = normaltest(col_data)
                    is_normal = normal_p_value > self.alpha
                except:
                    normal_p_value = None
                    is_normal = None
            
            stats_dict[col] = {
                'skewness': skewness,
                'kurtosis': kurt,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'is_skewed': is_skewed,
                'normal_p_value': normal_p_value,
                'is_normal': is_normal,
                'sample_count': len(col_data)
            }
            
            skew_dir = "右偏(正偏)" if skewness > 0 else "左偏(负偏)"
            print(f"\n【{col}】")
            print(f"  偏度: {skewness:.4f} ({skew_dir})")
            print(f"  峰度: {kurt:.4f}")
            print(f"  均值: {mean_val:.4f}, 中位数: {median_val:.4f}")
            print(f"  标准差: {std_val:.4f}")
            print(f"  范围: [{min_val:.4f}, {max_val:.4f}]")
            
            if is_skewed:
                print(f"  ⚠️  检测到偏态分布 (|偏度| > {self.skewness_threshold})")
            
            if is_normal is not None:
                norm_status = "服从正态分布" if is_normal else "不服从正态分布"
                print(f"  正态性检验: p={normal_p_value:.4f}, {norm_status} (α={self.alpha})")
        
        self.distribution_stats = stats_dict
        return stats_dict
    
    def classify_columns(self) -> Tuple[List[str], List[str]]:
        """
        根据分布特征分类列
        
        返回:
            (偏态分布列列表, 近似正态分布列列表)
        """
        if not self.distribution_stats:
            raise ValueError("请先调用 calculate_distribution_stats()")
        
        skewed = []
        normal = []
        
        for col, stats in self.distribution_stats.items():
            if stats['is_skewed']:
                skewed.append(col)
            else:
                normal.append(col)
        
        self.skewed_columns = skewed
        self.normal_columns = normal
        
        print("\n" + "=" * 60)
        print("列分类结果")
        print("=" * 60)
        print(f"偏态分布列 ({len(skewed)} 个): {skewed}")
        print(f"近似正态分布列 ({len(normal)} 个): {normal}")
        
        return skewed, normal
    
    def get_transformation_recommendations(self) -> Dict[str, str]:
        """
        获取变换建议
        
        返回:
            每列的变换建议字典
        """
        recommendations = {}
        
        for col, stats in self.distribution_stats.items():
            if stats['is_skewed']:
                skewness = stats['skewness']
                if skewness > 0:
                    recommendations[col] = "建议: Box-Cox 变换 (右偏数据)"
                else:
                    recommendations[col] = "建议: 先反射再 Box-Cox 变换 (左偏数据)"
            else:
                recommendations[col] = "建议: 直接 Z-score 标准化"
        
        return recommendations


class BoxCoxTransformer:
    """
    Box-Cox 变换器类
    负责执行 Box-Cox 变换和逆变换
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Box-Cox 变换器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.boxcox_config = {**BOXCOX_CONFIG, **self.config.get('boxcox', {})}
        
        self.add_constant = self.boxcox_config['add_constant']
        self.lambdas: Dict[str, float] = {}
        self.constants: Dict[str, float] = {}
        self.reflected_columns: List[str] = []
    
    def _ensure_positive(self, data: pd.Series, col_name: str) -> Tuple[pd.Series, float]:
        """
        确保数据为正数（Box-Cox 变换要求）
        
        参数:
            data: 输入数据 Series
            col_name: 列名
            
        返回:
            (正数化后的数据, 添加的常数)
        """
        min_val = data.min()
        
        if min_val <= 0:
            constant = abs(min_val) + self.add_constant
            data_positive = data + constant
            print(f"    列 '{col_name}': 最小值 {min_val:.4f} <= 0，添加常数 {constant:.6f}")
        else:
            constant = 0.0
            data_positive = data.copy()
        
        return data_positive, constant
    
    def _reflect_data(self, data: pd.Series) -> pd.Series:
        """
        反射数据（用于处理左偏分布）
        
        参数:
            data: 输入数据 Series
            
        返回:
            反射后的数据
        """
        max_val = data.max()
        reflected = max_val - data + self.add_constant
        return reflected
    
    def transform_column(self, data: pd.Series, col_name: str, 
                          is_left_skewed: bool = False) -> Tuple[pd.Series, float]:
        """
        对单列进行 Box-Cox 变换
        
        参数:
            data: 输入数据 Series
            col_name: 列名
            is_left_skewed: 是否为左偏分布
            
        返回:
            (变换后的数据, lambda 值)
        """
        data_to_transform = data.copy()
        
        if is_left_skewed:
            print(f"  列 '{col_name}': 左偏分布，先进行反射处理")
            data_to_transform = self._reflect_data(data_to_transform)
            if col_name not in self.reflected_columns:
                self.reflected_columns.append(col_name)
        
        data_positive, constant = self._ensure_positive(data_to_transform, col_name)
        self.constants[col_name] = constant
        
        if self.boxcox_config['lmbda'] is not None:
            lmbda = self.boxcox_config['lmbda']
            print(f"  列 '{col_name}': 使用指定 lambda = {lmbda}")
        else:
            try:
                _, lmbda = boxcox(data_positive)
                print(f"  列 '{col_name}': 最优 lambda = {lmbda:.4f}")
            except Exception as e:
                print(f"  列 '{col_name}': Box-Cox 优化失败，使用默认 lambda=0.5")
                print(f"    错误: {e}")
                lmbda = 0.5
        
        self.lambdas[col_name] = lmbda
        
        if lmbda == 0:
            transformed = np.log(data_positive)
        else:
            transformed = (np.power(data_positive, lmbda) - 1) / lmbda
        
        return transformed, lmbda
    
    def transform(self, data: pd.DataFrame, skewed_columns: List[str],
                  distribution_stats: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        对指定列进行 Box-Cox 变换
        
        参数:
            data: 输入 DataFrame
            skewed_columns: 需要变换的偏态分布列
            distribution_stats: 分布统计信息（用于判断偏度方向）
            
        返回:
            变换后的 DataFrame
        """
        print("\n" + "=" * 60)
        print("执行 Box-Cox 变换")
        print("=" * 60)
        
        result = data.copy()
        
        for col in skewed_columns:
            if col not in data.columns:
                continue
            
            print(f"\n处理列: {col}")
            
            col_data = data[col].copy()
            
            stats = distribution_stats.get(col, {})
            skewness = stats.get('skewness', 0)
            is_left_skewed = skewness < 0
            
            transformed_data, lmbda = self.transform_column(
                col_data, col, is_left_skewed
            )
            
            result[col] = transformed_data
            
            new_skewness = skew(transformed_data.dropna())
            print(f"    变换后偏度: {new_skewness:.4f} (原偏度: {skewness:.4f})")
            
            if abs(new_skewness) > abs(skewness):
                print(f"    ⚠️  警告: 变换后偏度反而增大")
        
        print("\n" + "-" * 60)
        print("Box-Cox 变换参数汇总:")
        for col in self.lambdas:
            reflected = " (已反射)" if col in self.reflected_columns else ""
            print(f"  {col}{reflected}: lambda = {self.lambdas[col]:.4f}, "
                  f"常数 = {self.constants.get(col, 0):.6f}")
        
        return result
    
    def inverse_transform_column(self, transformed_data: pd.Series, 
                                  col_name: str) -> pd.Series:
        """
        对单列进行 Box-Cox 逆变换
        
        参数:
            transformed_data: 变换后的数据
            col_name: 列名
            
        返回:
            原始尺度的数据
        """
        if col_name not in self.lambdas:
            raise ValueError(f"列 '{col_name}' 未进行过 Box-Cox 变换")
        
        lmbda = self.lambdas[col_name]
        constant = self.constants.get(col_name, 0.0)
        
        if lmbda == 0:
            data_positive = np.exp(transformed_data)
        else:
            data_positive = np.power(lmbda * transformed_data + 1, 1 / lmbda)
        
        original_data = data_positive - constant
        
        if col_name in self.reflected_columns:
            if hasattr(self, '_original_max'):
                original_data = self._original_max - original_data + self.add_constant
        
        return original_data
    
    def get_transform_params(self) -> Dict[str, Any]:
        """
        获取变换参数（用于保存和加载）
        
        返回:
            变换参数字典
        """
        return {
            'lambdas': self.lambdas.copy(),
            'constants': self.constants.copy(),
            'reflected_columns': self.reflected_columns.copy(),
            'add_constant': self.add_constant
        }
    
    def load_transform_params(self, params: Dict[str, Any]):
        """
        加载变换参数
        
        参数:
            params: 变换参数字典
        """
        self.lambdas = params.get('lambdas', {})
        self.constants = params.get('constants', {})
        self.reflected_columns = params.get('reflected_columns', [])
        self.add_constant = params.get('add_constant', self.add_constant)

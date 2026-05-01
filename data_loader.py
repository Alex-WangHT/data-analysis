"""
数据加载和预处理模块
负责读取 Excel 数据、处理缺失值、识别数值列
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from config import DATA_CONFIG, COLUMN_CONFIG, MISSING_CONFIG


class DataLoader:
    """
    数据加载器类
    负责读取 Excel 数据并进行基础预处理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器
        
        参数:
            config: 配置字典，如未提供则使用默认配置
        """
        self.config = config or {}
        self.data_config = {**DATA_CONFIG, **self.config.get('data', {})}
        self.column_config = {**COLUMN_CONFIG, **self.config.get('column', {})}
        self.missing_config = {**MISSING_CONFIG, **self.config.get('missing', {})}
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.numeric_columns: List[str] = []
        self.id_columns: List[str] = []
    
    def load_excel(self) -> pd.DataFrame:
        """
        从 Excel 文件加载数据
        
        返回:
            加载的 DataFrame
        """
        excel_file = self.data_config['excel_file']
        sheet_name = self.data_config['sheet_name']
        
        if sheet_name is None:
            xls = pd.ExcelFile(excel_file)
            if len(xls.sheet_names) == 1:
                sheet_name = xls.sheet_names[0]
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            else:
                dfs = []
                for sheet in xls.sheet_names:
                    df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
                    df_sheet['_sheet_name'] = sheet
                    dfs.append(df_sheet)
                df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        self.raw_data = df.copy()
        print(f"成功加载数据，形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        return df
    
    def identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        识别标识列和数值列
        
        参数:
            df: 输入 DataFrame
            
        返回:
            (标识列列表, 数值列列表)
        """
        configured_id_columns = self.column_config['id_columns']
        configured_numeric_columns = self.column_config['numeric_columns']
        
        actual_id_columns = []
        for col in configured_id_columns:
            if col in df.columns:
                actual_id_columns.append(col)
        
        if configured_numeric_columns is not None:
            actual_numeric_columns = [
                col for col in configured_numeric_columns 
                if col in df.columns and col not in actual_id_columns
            ]
        else:
            actual_numeric_columns = [
                col for col in df.columns 
                if col not in actual_id_columns and 
                pd.api.types.is_numeric_dtype(df[col].dtype)
            ]
        
        self.id_columns = actual_id_columns
        self.numeric_columns = actual_numeric_columns
        
        print(f"识别到标识列: {actual_id_columns}")
        print(f"识别到数值列: {actual_numeric_columns}")
        print(f"数值列数量: {len(actual_numeric_columns)}")
        
        return actual_id_columns, actual_numeric_columns
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.Series:
        """
        检查缺失值
        
        参数:
            df: 输入 DataFrame
            
        返回:
            每列缺失值数量的 Series
        """
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df) * 100).round(2)
        
        print("\n缺失值统计:")
        for col in df.columns:
            if missing_stats[col] > 0:
                print(f"  {col}: {missing_stats[col]} 个缺失值 ({missing_percent[col]}%)")
        
        if missing_stats.sum() == 0:
            print("  无缺失值")
        
        return missing_stats
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        参数:
            df: 输入 DataFrame
            
        返回:
            处理缺失值后的 DataFrame
        """
        strategy = self.missing_config['strategy']
        numeric_cols = self.numeric_columns
        
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=numeric_cols)
            print(f"删除包含缺失值的行后，形状: {df_processed.shape}")
            
        elif strategy == 'mean':
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            print("使用均值填充缺失值")
            
        elif strategy == 'median':
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            print("使用中位数填充缺失值")
            
        elif strategy == 'interpolate':
            method = self.missing_config['interpolate_method']
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].interpolate(method=method)
            print(f"使用 {method} 插值填充缺失值")
            
        elif strategy == 'ffill':
            df_processed = df_processed.fillna(method='ffill')
            print("使用前向填充缺失值")
            
        elif strategy == 'bfill':
            df_processed = df_processed.fillna(method='bfill')
            print("使用后向填充缺失值")
        
        remaining_missing = df_processed[numeric_cols].isnull().sum().sum()
        if remaining_missing > 0:
            print(f"警告: 仍有 {remaining_missing} 个缺失值未处理")
            df_processed = df_processed.dropna(subset=numeric_cols)
            print(f"删除剩余缺失值行后，形状: {df_processed.shape}")
        
        return df_processed
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """
        执行完整的加载和预处理流程
        
        返回:
            预处理后的 DataFrame
        """
        print("=" * 60)
        print("开始数据加载和预处理")
        print("=" * 60)
        
        df = self.load_excel()
        
        self.identify_columns(df)
        
        self.check_missing_values(df)
        
        df_processed = self.handle_missing_values(df)
        
        self.processed_data = df_processed
        
        print("\n" + "=" * 60)
        print("数据预处理完成")
        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"处理后数据形状: {self.processed_data.shape}")
        print("=" * 60)
        
        return self.processed_data
    
    def get_numeric_data(self) -> pd.DataFrame:
        """
        获取仅包含数值列的数据
        
        返回:
            仅包含数值列的 DataFrame
        """
        if self.processed_data is None:
            raise ValueError("数据尚未加载，请先调用 load_and_preprocess()")
        
        return self.processed_data[self.numeric_columns].copy()
    
    def get_combined_data(self, processed_numeric: pd.DataFrame) -> pd.DataFrame:
        """
        将处理后的数值列与标识列合并
        
        参数:
            processed_numeric: 处理后的数值列 DataFrame
            
        返回:
            合并后的 DataFrame
        """
        if self.processed_data is None:
            raise ValueError("数据尚未加载")
        
        id_data = self.processed_data[self.id_columns].copy()
        combined = pd.concat([id_data.reset_index(drop=True), 
                              processed_numeric.reset_index(drop=True)], 
                             axis=1)
        
        return combined

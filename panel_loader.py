"""
面板数据读取和整合模块
处理透视表结构，转换为标准的面板数据格式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from panel_config import PANEL_DATA_CONFIG


class PanelDataLoader:
    """
    面板数据加载器
    负责读取 Excel 中的多个工作表，整合为标准面板数据格式
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化面板数据加载器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.data_config = {**PANEL_DATA_CONFIG, **self.config.get('panel_data', {})}
        
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.panel_data: Optional[pd.DataFrame] = None
        self.provinces: List[str] = []
        self.years: List[int] = []
        self.indicators: List[str] = []
        
        self.data_structure: Dict[str, Any] = {}
    
    def load_all_sheets(self, excel_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载 Excel 中的所有工作表
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            工作表名到 DataFrame 的字典
        """
        excel_file = excel_file or self.data_config['excel_file']
        
        print("=" * 60)
        print("加载 Excel 工作表")
        print("=" * 60)
        
        xls = pd.ExcelFile(excel_file)
        print(f"文件: {excel_file}")
        print(f"工作表数量: {len(xls.sheet_names)}")
        print(f"工作表名称: {xls.sheet_names}")
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            self.raw_data[sheet_name] = df
            print(f"\n【{sheet_name}】")
            print(f"  形状: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(f"  列名: {list(df.columns)}")
            print(f"  前5行预览:")
            print(df.head().to_string())
        
        return self.raw_data
    
    def _detect_sheet_structure(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        检测单个工作表的结构
        
        参数:
            df: 工作表数据
            sheet_name: 工作表名
            
        返回:
            结构信息字典
        """
        structure = {
            'sheet_name': sheet_name,
            'row_type': None,      # 'province' | 'year' | 'indicator' | 'unknown'
            'col_type': None,      # 'year' | 'province' | 'indicator' | 'unknown'
            'value_type': None,    # 'indicator' | 'multiple' | 'unknown'
            'id_cols': [],
            'value_cols': [],
        }
        
        id_cols_config = self.data_config['id_columns']
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            for col_type, keywords in id_cols_config.items():
                for kw in keywords:
                    kw_lower = str(kw).lower()
                    if kw_lower in col_lower or col_lower in kw_lower:
                        if col_type not in structure['id_cols']:
                            structure['id_cols'].append(col_type)
                        break
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 10:
                    structure['value_cols'].append(col)
        
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
        
        year_candidates = []
        for col in numeric_cols:
            if isinstance(col, (int, float)) and 1949 <= col <= 2030:
                year_candidates.append(col)
            elif isinstance(col, str):
                if col.isdigit() and 1949 <= int(col) <= 2030:
                    year_candidates.append(col)
        
        if len(year_candidates) >= 3:
            structure['col_type'] = 'year'
            structure['value_type'] = 'indicator'
        
        first_col = df.columns[0]
        first_col_vals = df[first_col].dropna().unique()
        
        province_keywords = ['省', '市', '区', '自治', '北京', '上海', '广东', '江苏', '浙江', '山东', '河南']
        is_province_col = False
        for val in first_col_vals[:20]:
            val_str = str(val)
            for kw in province_keywords:
                if kw in val_str:
                    is_province_col = True
                    break
            if is_province_col:
                break
        
        if is_province_col:
            structure['row_type'] = 'province'
        
        return structure
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        分析所有工作表的结构，理解数据组织方式
        
        返回:
            整体结构信息
        """
        print("\n" + "=" * 60)
        print("分析数据结构")
        print("=" * 60)
        
        structures = {}
        for sheet_name, df in self.raw_data.items():
            structure = self._detect_sheet_structure(df, sheet_name)
            structures[sheet_name] = structure
            print(f"\n【{sheet_name}】结构分析:")
            print(f"  行类型: {structure['row_type']}")
            print(f"  列类型: {structure['col_type']}")
            print(f"  值类型: {structure['value_type']}")
            print(f"  标识列类型: {structure['id_cols']}")
        
        row_types = set(s['row_type'] for s in structures.values())
        col_types = set(s['col_type'] for s in structures.values())
        
        self.data_structure = {
            'sheet_structures': structures,
            'common_row_type': list(row_types)[0] if len(row_types) == 1 else 'mixed',
            'common_col_type': list(col_types)[0] if len(col_types) == 1 else 'mixed',
        }
        
        print(f"\n整体结构判断:")
        print(f"  行类型: {self.data_structure['common_row_type']}")
        print(f"  列类型: {self.data_structure['common_col_type']}")
        
        return self.data_structure
    
    def _pivot_to_long(self, df: pd.DataFrame, sheet_name: str, 
                        structure: Dict[str, Any]) -> pd.DataFrame:
        """
        将透视表转换为长格式
        
        参数:
            df: 透视表数据
            sheet_name: 工作表名（可能是指标名）
            structure: 结构信息
            
        返回:
            长格式数据
        """
        row_type = structure.get('row_type', 'province')
        col_type = structure.get('col_type', 'year')
        value_type = structure.get('value_type', 'indicator')
        
        id_col = df.columns[0]
        
        if col_type == 'year':
            value_cols = []
            for col in df.columns[1:]:
                if isinstance(col, (int, float)) and 1949 <= col <= 2030:
                    value_cols.append(col)
                elif isinstance(col, str):
                    if col.isdigit() and 1949 <= int(col) <= 2030:
                        value_cols.append(col)
            
            if not value_cols:
                value_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c].dtype)]
            
            melted = pd.melt(
                df,
                id_vars=[id_col],
                value_vars=value_cols,
                var_name='年份',
                value_name='数值'
            )
            
            melted = melted.rename(columns={id_col: '地区'})
            
            if value_type == 'indicator':
                indicator_name = sheet_name
                for kw in ['透视表', '_pivot', '透视', '表']:
                    if kw in indicator_name:
                        indicator_name = indicator_name.replace(kw, '')
                indicator_name = indicator_name.strip()
                
                melted['指标'] = indicator_name
            
            return melted
        
        return pd.DataFrame()
    
    def convert_to_panel(self) -> pd.DataFrame:
        """
        将所有工作表转换为标准面板数据格式
        
        标准格式:
            列: ['地区', '年份', '指标', '数值']
            每一行: 一个地区在某一年的某一个指标值
        
        返回:
            标准面板数据
        """
        print("\n" + "=" * 60)
        print("转换为标准面板数据格式")
        print("=" * 60)
        
        all_data = []
        
        for sheet_name, df in self.raw_data.items():
            print(f"\n处理工作表: {sheet_name}")
            
            structure = self._detect_sheet_structure(df, sheet_name)
            print(f"  检测结构: 行={structure['row_type']}, 列={structure['col_type']}")
            
            try:
                long_df = self._pivot_to_long(df, sheet_name, structure)
                
                if not long_df.empty:
                    print(f"  转换后: {len(long_df)} 行")
                    print(f"  列: {list(long_df.columns)}")
                    
                    required_cols = ['地区', '年份', '数值']
                    if '指标' not in long_df.columns:
                        print(f"  ⚠️  未检测到指标名，使用工作表名: {sheet_name}")
                        long_df['指标'] = sheet_name
                    
                    all_data.append(long_df)
                    
            except Exception as e:
                print(f"  ✗ 转换失败: {e}")
        
        if not all_data:
            raise ValueError("没有成功转换任何工作表")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n合并后数据形状: {combined.shape}")
        print(f"列名: {list(combined.columns)}")
        
        print(f"\n数据预览 (前10行):")
        print(combined.head(10).to_string())
        
        self.provinces = sorted(combined['地区'].dropna().unique().tolist())
        self.years = sorted(combined['年份'].dropna().unique().tolist())
        self.indicators = sorted(combined['指标'].dropna().unique().tolist())
        
        print(f"\n识别到:")
        print(f"  地区数: {len(self.provinces)}")
        print(f"  年份数: {len(self.years)}")
        print(f"  指标数: {len(self.indicators)}")
        
        print(f"\n地区列表 (前10个): {self.provinces[:10]}")
        print(f"年份列表: {self.years}")
        print(f"指标列表: {self.indicators}")
        
        self.panel_data = combined
        return combined
    
    def pivot_to_wide(self, panel_data: Optional[pd.DataFrame] = None,
                       index: List[str] = ['地区', '年份'],
                       columns: str = '指标',
                       values: str = '数值') -> pd.DataFrame:
        """
        将长格式面板数据转换为宽格式
        
        宽格式:
            列: ['地区', '年份', '指标1', '指标2', ...]
            每一行: 一个地区在某一年的所有指标值
        
        参数:
            panel_data: 长格式数据
            index: 索引列
            columns: 要展开的列
            values: 值列
            
        返回:
            宽格式数据
        """
        if panel_data is None:
            panel_data = self.panel_data
        
        if panel_data is None:
            raise ValueError("请先提供或加载面板数据")
        
        print("\n" + "=" * 60)
        print("转换为宽格式")
        print("=" * 60)
        
        wide = panel_data.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc='first'
        ).reset_index()
        
        print(f"宽格式形状: {wide.shape}")
        print(f"列名: {list(wide.columns)}")
        
        return wide
    
    def load_and_process(self, excel_file: Optional[str] = None) -> pd.DataFrame:
        """
        执行完整的加载和处理流程
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            标准面板数据
        """
        self.load_all_sheets(excel_file)
        self.analyze_structure()
        self.convert_to_panel()
        
        return self.panel_data
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        获取数据的基本统计信息
        
        返回:
            统计信息字典
        """
        if self.panel_data is None:
            return {}
        
        stats = {
            'n_provinces': len(self.provinces),
            'n_years': len(self.years),
            'n_indicators': len(self.indicators),
            'provinces': self.provinces,
            'years': self.years,
            'indicators': self.indicators,
            'total_obs': len(self.panel_data),
            'missing_by_indicator': {},
            'missing_by_province': {},
            'missing_by_year': {},
        }
        
        for indicator in self.indicators:
            subset = self.panel_data[self.panel_data['指标'] == indicator]
            missing = subset['数值'].isnull().sum()
            total = len(subset)
            stats['missing_by_indicator'][indicator] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        for province in self.provinces:
            subset = self.panel_data[self.panel_data['地区'] == province]
            missing = subset['数值'].isnull().sum()
            total = len(subset)
            stats['missing_by_province'][province] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        for year in self.years:
            subset = self.panel_data[self.panel_data['年份'] == year]
            missing = subset['数值'].isnull().sum()
            total = len(subset)
            stats['missing_by_year'][year] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        return stats

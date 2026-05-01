"""
透视表数据加载器
处理特定结构：省份 | 指标 | 年份列(2020-2025...)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from pivot_config import PIVOT_CONFIG


class PivotTableLoader:
    """
    透视表加载器
    专门处理：省份 | 指标 | 年份列 结构的透视表
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化透视表加载器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.pivot_config = {**PIVOT_CONFIG, **self.config.get('pivot', {})}
        
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.pivot_sheets: List[str] = []
        
        self.all_data: Optional[pd.DataFrame] = None
        
        self.provinces: List[str] = []
        self.indicators: List[str] = []
        self.years: List[int] = []
        
        self.data_by_province: Dict[str, pd.DataFrame] = {}
        self.data_by_indicator: Dict[str, pd.DataFrame] = {}
    
    def _is_pivot_sheet(self, sheet_name: str) -> bool:
        """
        检查是否为透视表工作表
        
        参数:
            sheet_name: 工作表名
            
        返回:
            是否为透视表
        """
        filter_config = self.pivot_config['sheet_filter']
        keyword = filter_config['keyword']
        exclude_keywords = filter_config['exclude_keywords']
        
        if keyword and keyword not in sheet_name:
            return False
        
        for exc_kw in exclude_keywords:
            if exc_kw in sheet_name:
                return False
        
        return True
    
    def _parse_year_column(self, col_name) -> Optional[int]:
        """
        解析年份列名
        
        参数:
            col_name: 列名
            
        返回:
            年份整数，或 None
        """
        col_str = str(col_name).strip()
        
        if col_str.isdigit():
            year = int(col_str)
            if 1949 <= year <= 2030:
                return year
        
        for prefix in ['年份', '年', 'Y', 'y', 'Year', 'year']:
            if col_str.startswith(prefix):
                num_part = col_str[len(prefix):].strip()
                if num_part.isdigit():
                    year = int(num_part)
                    if 1949 <= year <= 2030:
                        return year
        
        return None
    
    def _detect_structure(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        检测 DataFrame 的结构
        
        参数:
            df: 数据
            sheet_name: 工作表名
            
        返回:
            结构信息
        """
        structure = {
            'sheet_name': sheet_name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'province_col': None,
            'indicator_col': None,
            'year_cols': [],
            'year_values': [],
        }
        
        col_struct = self.pivot_config['column_structure']
        prov_idx = col_struct['province_col_idx']
        ind_idx = col_struct['indicator_col_idx']
        year_start_idx = col_struct['year_start_col_idx']
        
        if len(df.columns) > prov_idx:
            structure['province_col'] = df.columns[prov_idx]
        
        if len(df.columns) > ind_idx:
            structure['indicator_col'] = df.columns[ind_idx]
        
        for col_idx in range(year_start_idx, len(df.columns)):
            col_name = df.columns[col_idx]
            year = self._parse_year_column(col_name)
            if year is not None:
                structure['year_cols'].append(col_name)
                structure['year_values'].append(year)
        
        return structure
    
    def load_excel(self, excel_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载 Excel 文件，识别透视表
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            工作表数据字典
        """
        excel_file = excel_file or self.pivot_config['excel_file']
        
        print("=" * 70)
        print("加载透视表数据")
        print("=" * 70)
        
        print(f"\n文件: {excel_file}")
        
        xls = pd.ExcelFile(excel_file)
        all_sheets = xls.sheet_names
        print(f"总工作表数: {len(all_sheets)}")
        print(f"所有工作表: {all_sheets}")
        
        self.pivot_sheets = [s for s in all_sheets if self._is_pivot_sheet(s)]
        print(f"\n透视表工作表 ({len(self.pivot_sheets)} 个): {self.pivot_sheets}")
        
        if not self.pivot_sheets:
            print("\n警告: 未找到包含 '透视' 关键词的工作表!")
            print("将检查所有工作表的结构...")
            self.pivot_sheets = all_sheets
        
        for sheet_name in self.pivot_sheets:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            self.raw_data[sheet_name] = df
            
            structure = self._detect_structure(df, sheet_name)
            
            print(f"\n【{sheet_name}】")
            print(f"  形状: {structure['n_rows']} 行 × {structure['n_cols']} 列")
            print(f"  列名: {structure['columns']}")
            print(f"  省份列: {structure['province_col']}")
            print(f"  指标列: {structure['indicator_col']}")
            print(f"  年份列: {structure['year_cols']}")
            print(f"  年份值: {structure['year_values']}")
            
            print(f"\n  前5行数据:")
            print(df.head().to_string())
        
        return self.raw_data
    
    def _melt_to_long(self, df: pd.DataFrame, structure: Dict[str, Any]) -> pd.DataFrame:
        """
        将透视表转换为长格式
        
        参数:
            df: 原始透视表
            structure: 结构信息
            
        返回:
            长格式数据
        """
        col_names = self.pivot_config['column_names']
        prov_col = structure['province_col']
        ind_col = structure['indicator_col']
        year_cols = structure['year_cols']
        year_values = structure['year_values']
        
        if not year_cols:
            print(f"    警告: 未检测到年份列，跳过此表")
            return pd.DataFrame()
        
        id_vars = []
        if prov_col is not None and prov_col in df.columns:
            id_vars.append(prov_col)
        if ind_col is not None and ind_col in df.columns:
            id_vars.append(ind_col)
        
        if not id_vars:
            print(f"    警告: 未找到省份或指标列，跳过此表")
            return pd.DataFrame()
        
        actual_year_cols = [c for c in year_cols if c in df.columns]
        
        melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=actual_year_cols,
            var_name='_year_col',
            value_name=col_names['value']
        )
        
        melted[col_names['year']] = melted['_year_col'].apply(self._parse_year_column)
        
        result = melted.drop(columns=['_year_col'])
        
        rename_map = {}
        if prov_col is not None and prov_col in result.columns:
            rename_map[prov_col] = col_names['province']
        if ind_col is not None and ind_col in result.columns:
            rename_map[ind_col] = col_names['indicator']
        
        if rename_map:
            result = result.rename(columns=rename_map)
        
        return result
    
    def consolidate_all_sheets(self) -> pd.DataFrame:
        """
        整合所有透视表
        
        返回:
            整合后的长格式数据
        """
        print("\n" + "=" * 70)
        print("整合所有透视表")
        print("=" * 70)
        
        all_dfs = []
        
        for sheet_name, df in self.raw_data.items():
            print(f"\n处理工作表: {sheet_name}")
            
            structure = self._detect_structure(df, sheet_name)
            melted = self._melt_to_long(df, structure)
            
            if not melted.empty:
                melted['_sheet_name'] = sheet_name
                all_dfs.append(melted)
                print(f"  转换后: {len(melted)} 行")
        
        if not all_dfs:
            raise ValueError("没有成功转换任何工作表")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        col_names = self.pivot_config['column_names']
        prov_col = col_names['province']
        ind_col = col_names['indicator']
        year_col = col_names['year']
        value_col = col_names['value']
        
        print(f"\n整合后数据形状: {combined.shape}")
        print(f"列名: {list(combined.columns)}")
        
        for col in [prov_col, ind_col, year_col]:
            if col in combined.columns:
                combined[col] = combined[col].astype(str).str.strip()
        
        if year_col in combined.columns:
            def safe_int(x):
                try:
                    return int(float(x))
                except:
                    return None
            combined[year_col] = combined[year_col].apply(safe_int)
            combined = combined.dropna(subset=[year_col])
            combined[year_col] = combined[year_col].astype(int)
        
        if prov_col in combined.columns:
            self.provinces = sorted(combined[prov_col].dropna().unique().tolist())
        
        if ind_col in combined.columns:
            self.indicators = sorted(combined[ind_col].dropna().unique().tolist())
        
        if year_col in combined.columns:
            self.years = sorted(combined[year_col].dropna().unique().tolist())
        
        print(f"\n识别到:")
        print(f"  省份数: {len(self.provinces)}")
        print(f"  指标数: {len(self.indicators)}")
        print(f"  年份数: {len(self.years)}")
        
        print(f"\n省份列表: {self.provinces}")
        print(f"指标列表: {self.indicators}")
        print(f"年份列表: {self.years}")
        
        self.all_data = combined
        return combined
    
    def organize_by_province(self) -> Dict[str, pd.DataFrame]:
        """
        按省份组织数据
        
        每个省份的数据格式：
            行 = 年份
            列 = 指标
            
        返回:
            省份名到 DataFrame 的字典
        """
        if self.all_data is None:
            raise ValueError("请先调用 consolidate_all_sheets()")
        
        col_names = self.pivot_config['column_names']
        prov_col = col_names['province']
        ind_col = col_names['indicator']
        year_col = col_names['year']
        value_col = col_names['value']
        
        print("\n" + "=" * 70)
        print("按省份组织数据")
        print("=" * 70)
        
        for province in self.provinces:
            prov_data = self.all_data[self.all_data[prov_col] == province].copy()
            
            if prov_data.empty:
                continue
            
            pivoted = prov_data.pivot_table(
                index=year_col,
                columns=ind_col,
                values=value_col,
                aggfunc='first'
            ).reset_index()
            
            pivoted.index.name = None
            
            self.data_by_province[province] = pivoted
            
            print(f"\n【{province}】")
            print(f"  形状: {pivoted.shape}")
            print(f"  列名: {list(pivoted.columns)}")
            print(f"  前3行:")
            print(pivoted.head(3).to_string())
        
        return self.data_by_province
    
    def organize_by_indicator(self, group_by: str = 'year') -> Dict[str, pd.DataFrame]:
        """
        按指标组织数据
        
        参数:
            group_by: 'year' = 行=省份, 列=年份
                      'province' = 行=年份, 列=省份
            
        返回:
            指标名到 DataFrame 的字典
        """
        if self.all_data is None:
            raise ValueError("请先调用 consolidate_all_sheets()")
        
        col_names = self.pivot_config['column_names']
        prov_col = col_names['province']
        ind_col = col_names['indicator']
        year_col = col_names['year']
        value_col = col_names['value']
        
        print(f"\n{'='*70}")
        print(f"按指标组织数据 (group_by={group_by})")
        print("=" * 70)
        
        for indicator in self.indicators:
            ind_data = self.all_data[self.all_data[ind_col] == indicator].copy()
            
            if ind_data.empty:
                continue
            
            if group_by == 'year':
                pivoted = ind_data.pivot_table(
                    index=prov_col,
                    columns=year_col,
                    values=value_col,
                    aggfunc='first'
                ).reset_index()
            else:
                pivoted = ind_data.pivot_table(
                    index=year_col,
                    columns=prov_col,
                    values=value_col,
                    aggfunc='first'
                ).reset_index()
            
            pivoted.index.name = None
            
            self.data_by_indicator[indicator] = pivoted
            
            print(f"\n【{indicator}】")
            print(f"  形状: {pivoted.shape}")
            print(f"  列名: {list(pivoted.columns)}")
        
        return self.data_by_indicator
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        获取基本统计信息
        
        返回:
            统计信息字典
        """
        if self.all_data is None:
            return {}
        
        col_names = self.pivot_config['column_names']
        prov_col = col_names['province']
        ind_col = col_names['indicator']
        year_col = col_names['year']
        value_col = col_names['value']
        
        stats = {
            'n_provinces': len(self.provinces),
            'n_indicators': len(self.indicators),
            'n_years': len(self.years),
            'provinces': self.provinces,
            'indicators': self.indicators,
            'years': self.years,
            'total_obs': len(self.all_data),
            'missing_total': self.all_data[value_col].isnull().sum(),
            'missing_by_province': {},
            'missing_by_indicator': {},
            'missing_by_year': {},
        }
        
        for province in self.provinces:
            subset = self.all_data[self.all_data[prov_col] == province]
            missing = subset[value_col].isnull().sum()
            total = len(subset)
            stats['missing_by_province'][province] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        for indicator in self.indicators:
            subset = self.all_data[self.all_data[ind_col] == indicator]
            missing = subset[value_col].isnull().sum()
            total = len(subset)
            stats['missing_by_indicator'][indicator] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        for year in self.years:
            subset = self.all_data[self.all_data[year_col] == year]
            missing = subset[value_col].isnull().sum()
            total = len(subset)
            stats['missing_by_year'][year] = {
                'missing': missing,
                'total': total,
                'pct': missing / total * 100 if total > 0 else 0
            }
        
        return stats
    
    def load_and_organize(self, excel_file: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        完整流程：加载、整合、组织数据
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            (按省份组织的数据, 按指标组织的数据)
        """
        self.load_excel(excel_file)
        self.consolidate_all_sheets()
        self.organize_by_province()
        self.organize_by_indicator(group_by='year')
        
        return self.data_by_province, self.data_by_indicator

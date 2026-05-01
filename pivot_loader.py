"""
透视表数据加载器（处理合并单元格格式）

针对 Excel 透视表的典型格式：
┌────────┬─────────────────────────────────┬──────┬───────┬───────┐
│  省份  │              指标               │ 单位 │2020年 │2021年 │
├────────┼─────────────────────────────────┼──────┼───────┼───────┤
│ 上海市  │ 内地居民结婚登记 (万对)          │ 万对 │ 9.15  │ 8.90  │
│  NaN   │ 涉外、华侨及港澳台居民结婚登记 (万对)│ 万对 │ 0.07  │ 0.08  │
│  NaN   │ 离婚率 (‰)                      │  ‰   │ 2.70  │ 1.46  │
└────────┴─────────────────────────────────┴──────┴───────┴───────┘

处理逻辑：
1. 省份列的 NaN 向前填充（用上一行的值）
2. 从年份列名中提取年份（如 "2020年" → 2020）
3. 转换为标准长格式
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from pivot_config import PIVOT_CONFIG


class PivotTableLoader:
    """
    透视表加载器（处理合并单元格格式）
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
        self.units: Dict[str, str] = {}  # 指标到单位的映射
        
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
        
        if keyword:
            if keyword not in str(sheet_name):
                return False
        
        for exc_kw in exclude_keywords:
            if exc_kw in str(sheet_name):
                return False
        
        return True
    
    def _extract_year_from_column(self, col_name) -> Optional[int]:
        """
        从列名中提取年份
        
        支持格式：
        - "2020年" → 2020
        - "2020" → 2020
        - "年份2020" → 2020
        
        参数:
            col_name: 列名
            
        返回:
            年份整数，或 None
        """
        col_str = str(col_name).strip()
        
        year_pattern = self.pivot_config.get('year_column_pattern', {})
        has_suffix = year_pattern.get('has_suffix', True)
        suffix = year_pattern.get('suffix', '年')
        
        if has_suffix and col_str.endswith(suffix):
            year_part = col_str[:-len(suffix)]
            if year_part.isdigit():
                year = int(year_part)
                if 1949 <= year <= 2030:
                    return year
        
        digits = re.findall(r'\d{4}', col_str)
        for d in digits:
            year = int(d)
            if 1949 <= year <= 2030:
                return year
        
        if col_str.isdigit():
            year = int(col_str)
            if 1949 <= year <= 2030:
                return year
        
        return None
    
    def _handle_merge_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理合并单元格（向前填充 NaN）
        
        参数:
            df: 原始 DataFrame
            
        返回:
            处理后的 DataFrame
        """
        merge_config = self.pivot_config.get('merge_cell_handling', {})
        
        if not merge_config.get('enabled', True):
            return df
        
        fill_columns = merge_config.get('forward_fill_columns', [0])
        method = merge_config.get('forward_fill_method', 'ffill')
        
        df_processed = df.copy()
        
        for col_idx in fill_columns:
            if col_idx < len(df_processed.columns):
                col_name = df_processed.columns[col_idx]
                
                if method == 'ffill':
                    df_processed[col_name] = df_processed[col_name].ffill()
                elif method == 'bfill':
                    df_processed[col_name] = df_processed[col_name].bfill()
        
        return df_processed
    
    def _detect_sheet_structure(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        检测工作表结构
        
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
            'unit_col': None,
            'year_cols': [],
            'year_values': [],
            'has_merge_cells': False,
        }
        
        col_struct = self.pivot_config['column_structure']
        prov_idx = col_struct['province_col_idx']
        ind_idx = col_struct['indicator_col_idx']
        unit_idx = col_struct.get('unit_col_idx')
        year_start_idx = col_struct['year_start_col_idx']
        
        if len(df.columns) > prov_idx:
            structure['province_col'] = df.columns[prov_idx]
            
            prov_col = df[df.columns[prov_idx]]
            if prov_col.isnull().any():
                structure['has_merge_cells'] = True
                structure['merge_cells_in_province'] = prov_col.isnull().sum()
        
        if len(df.columns) > ind_idx:
            structure['indicator_col'] = df.columns[ind_idx]
        
        if unit_idx is not None and len(df.columns) > unit_idx:
            structure['unit_col'] = df.columns[unit_idx]
        
        for col_idx in range(year_start_idx, len(df.columns)):
            col_name = df.columns[col_idx]
            year = self._extract_year_from_column(col_name)
            
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
            
            structure = self._detect_sheet_structure(df, sheet_name)
            
            print(f"\n【{sheet_name}】")
            print(f"  形状: {structure['n_rows']} 行 × {structure['n_cols']} 列")
            print(f"  列名: {structure['columns']}")
            print(f"  省份列: {structure['province_col']}")
            print(f"  指标列: {structure['indicator_col']}")
            print(f"  单位列: {structure['unit_col']}")
            print(f"  年份列: {structure['year_cols']}")
            print(f"  年份值: {structure['year_values']}")
            
            if structure['has_merge_cells']:
                print(f"  ⚠️  检测到合并单元格: 省份列有 {structure['merge_cells_in_province']} 个 NaN")
                print(f"      将使用向前填充处理")
            
            print(f"\n  原始数据前5行:")
            print(df.head().to_string())
        
        return self.raw_data
    
    def _melt_to_long(self, df: pd.DataFrame, structure: Dict[str, Any],
                      sheet_name: str) -> pd.DataFrame:
        """
        将透视表转换为长格式
        
        参数:
            df: 原始透视表（已处理合并单元格）
            structure: 结构信息
            sheet_name: 工作表名
            
        返回:
            长格式数据
        """
        col_names = self.pivot_config['column_names']
        
        prov_col = structure['province_col']
        ind_col = structure['indicator_col']
        unit_col = structure['unit_col']
        year_cols = structure['year_cols']
        
        if not year_cols:
            print(f"    警告: 未检测到年份列，跳过此表")
            return pd.DataFrame()
        
        id_vars = []
        if prov_col is not None and prov_col in df.columns:
            id_vars.append(prov_col)
        if ind_col is not None and ind_col in df.columns:
            id_vars.append(ind_col)
        if unit_col is not None and unit_col in df.columns:
            id_vars.append(unit_col)
        
        if not id_vars:
            print(f"    警告: 未找到标识列，跳过此表")
            return pd.DataFrame()
        
        actual_year_cols = [c for c in year_cols if c in df.columns]
        
        melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=actual_year_cols,
            var_name='_year_col',
            value_name=col_names['value']
        )
        
        melted[col_names['year']] = melted['_year_col'].apply(self._extract_year_from_column)
        
        result = melted.drop(columns=['_year_col'])
        
        rename_map = {}
        if prov_col is not None and prov_col in result.columns:
            rename_map[prov_col] = col_names['province']
        if ind_col is not None and ind_col in result.columns:
            rename_map[ind_col] = col_names['indicator']
        if unit_col is not None and unit_col in result.columns:
            rename_map[unit_col] = col_names['unit']
        
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
        print("整合所有透视表（处理合并单元格）")
        print("=" * 70)
        
        all_dfs = []
        
        for sheet_name, df in self.raw_data.items():
            print(f"\n处理工作表: {sheet_name}")
            
            structure = self._detect_sheet_structure(df, sheet_name)
            
            if structure['has_merge_cells']:
                print(f"  处理合并单元格（向前填充 NaN）...")
                df_processed = self._handle_merge_cells(df)
                
                prov_col = structure['province_col']
                if prov_col:
                    print(f"    填充后省份列的唯一值: {df_processed[prov_col].dropna().unique().tolist()}")
            else:
                df_processed = df.copy()
            
            print(f"  转换为长格式...")
            melted = self._melt_to_long(df_processed, structure, sheet_name)
            
            if not melted.empty:
                melted['_sheet_name'] = sheet_name
                all_dfs.append(melted)
                print(f"    转换后: {len(melted)} 行")
                
                print(f"\n    转换后数据预览（前5行）:")
                print(melted.head().to_string())
        
        if not all_dfs:
            raise ValueError("没有成功转换任何工作表")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        col_names = self.pivot_config['column_names']
        prov_col = col_names['province']
        ind_col = col_names['indicator']
        unit_col = col_names['unit']
        year_col = col_names['year']
        value_col = col_names['value']
        
        print(f"\n整合后数据形状: {combined.shape}")
        print(f"列名: {list(combined.columns)}")
        
        for col in [prov_col, ind_col, unit_col]:
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
            
            if unit_col in combined.columns:
                unit_mapping = combined[[ind_col, unit_col]].dropna().drop_duplicates()
                for _, row in unit_mapping.iterrows():
                    self.units[row[ind_col]] = row[unit_col]
        
        if year_col in combined.columns:
            self.years = sorted(combined[year_col].dropna().unique().tolist())
        
        print(f"\n识别到:")
        print(f"  省份数: {len(self.provinces)}")
        print(f"  指标数: {len(self.indicators)}")
        print(f"  年份数: {len(self.years)}")
        print(f"  单位数: {len(self.units)}")
        
        print(f"\n省份列表: {self.provinces}")
        print(f"指标列表: {self.indicators}")
        print(f"年份列表: {self.years}")
        
        if self.units:
            print(f"\n指标-单位映射:")
            for ind, unit in self.units.items():
                print(f"  {ind}: {unit}")
        
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
            'units': self.units,
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

"""
透视表数据分析主流程
整合数据加载、省份内分析、省份间分析
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any

from pivot_config import (
    PIVOT_CONFIG,
    PIVOT_NORMALIZATION,
    PIVOT_CORRELATION,
    PIVOT_OUTPUT
)
from pivot_loader import PivotTableLoader
from within_province import WithinProvinceAnalyzer
from across_province import AcrossProvinceAnalyzer


class PivotDataAnalysis:
    """
    透视表数据分析主类
    整合完整的分析流程
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分析器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        self.output_config = {**PIVOT_OUTPUT, **self.config.get('output', {})}
        
        self.loader = PivotTableLoader(self.config)
        self.within_analyzer = WithinProvinceAnalyzer(self.config)
        self.across_analyzer = AcrossProvinceAnalyzer(self.config)
        
        self.data_by_province: Dict[str, pd.DataFrame] = {}
        self.data_by_indicator: Dict[str, pd.DataFrame] = {}
        
        self.within_results: Dict[str, Any] = {}
        self.across_results: Dict[str, Any] = {}
    
    def load_data(self, excel_file: Optional[str] = None) -> Dict[str, Any]:
        """
        加载并组织数据
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            数据信息
        """
        print("\n" + "#" * 70)
        print("第一步: 加载和组织数据")
        print("#" * 70)
        
        self.loader.load_and_organize(excel_file)
        
        self.data_by_province = self.loader.data_by_province
        self.data_by_indicator = self.loader.data_by_indicator
        
        stats = self.loader.get_basic_stats()
        
        self.within_analyzer.set_data_info(
            self.loader.provinces,
            self.loader.indicators,
            self.loader.years
        )
        self.across_analyzer.set_data_info(
            self.loader.provinces,
            self.loader.indicators,
            self.loader.years
        )
        
        print("\n" + "-" * 70)
        print("数据基本信息:")
        print("-" * 70)
        print(f"  省份数: {stats['n_provinces']}")
        print(f"  指标数: {stats['n_indicators']}")
        print(f"  年份数: {stats['n_years']}")
        print(f"  总观测数: {stats['total_obs']}")
        print(f"  总缺失值: {stats['missing_total']}")
        
        print(f"\n  省份列表: {stats['provinces']}")
        print(f"  指标列表: {stats['indicators']}")
        print(f"  年份列表: {stats['years']}")
        
        if stats['missing_total'] > 0:
            print(f"\n  缺失值按指标:")
            for ind, miss in stats['missing_by_indicator'].items():
                if miss['missing'] > 0:
                    print(f"    {ind}: {miss['missing']}/{miss['total']} ({miss['pct']:.1f}%)")
        
        return stats
    
    def analyze_within_province(self,
                                  normalize: bool = True,
                                  use_normalized: bool = True) -> Dict[str, Any]:
        """
        执行省份内分析
        
        参数:
            normalize: 是否进行归一化
            use_normalized: 相关性分析是否使用归一化数据
            
        返回:
            分析结果
        """
        print("\n" + "#" * 70)
        print("第二步: 省份内分析")
        print("#" * 70)
        
        if not self.data_by_province:
            raise ValueError("请先调用 load_data() 加载数据")
        
        if normalize:
            self.within_analyzer.normalize_all_provinces(self.data_by_province)
        
        self.within_analyzer.analyze_all_provinces(
            self.data_by_province,
            use_normalized=use_normalized
        )
        
        self.within_results = self.within_analyzer.get_results()
        
        return self.within_results
    
    def analyze_across_province(self,
                                 normalize: bool = True,
                                 use_normalized: bool = True,
                                 group_by: str = 'year') -> Dict[str, Any]:
        """
        执行省份间分析
        
        参数:
            normalize: 是否进行归一化
            use_normalized: 相关性分析是否使用归一化数据
            group_by: 分组方式 'year' 或 'overall'
            
        返回:
            分析结果
        """
        print("\n" + "#" * 70)
        print("第三步: 省份间分析")
        print("#" * 70)
        
        if not self.data_by_indicator:
            raise ValueError("请先调用 load_data() 加载数据")
        
        if normalize:
            self.across_analyzer.normalize_all_indicators(
                self.data_by_indicator,
                group_by=group_by
            )
        
        self.across_analyzer.analyze_by_indicator(
            self.data_by_indicator,
            use_normalized=use_normalized
        )
        
        self.across_results = self.across_analyzer.get_results()
        
        return self.across_results
    
    def analyze_province_similarity(self,
                                      use_normalized: bool = True) -> Dict[str, Any]:
        """
        执行省份综合相似性分析
        
        参数:
            use_normalized: 是否使用归一化数据
            
        返回:
            相似性分析结果
        """
        print("\n" + "#" * 70)
        print("第四步: 省份综合相似性分析")
        print("#" * 70)
        
        if not self.data_by_indicator:
            raise ValueError("请先调用 load_data() 加载数据")
        
        similarity_result = self.across_analyzer.analyze_province_similarity(
            self.data_by_indicator,
            use_normalized=use_normalized
        )
        
        if 'correlation_results' not in self.across_results:
            self.across_results['correlation_results'] = {}
        self.across_results['correlation_results']['province_similarity'] = similarity_result
        
        return similarity_result
    
    def _convert_to_serializable(self, obj):
        """转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')
        return obj
    
    def save_results(self,
                      save_to_excel: Optional[bool] = None,
                      save_params: Optional[bool] = None):
        """
        保存所有结果
        
        参数:
            save_to_excel: 是否保存到 Excel
            save_params: 是否保存参数
        """
        print("\n" + "#" * 70)
        print("第五步: 保存结果")
        print("#" * 70)
        
        save_to_excel = save_to_excel if save_to_excel is not None else self.output_config['save_to_excel']
        save_params = save_params if save_params is not None else self.output_config['save_params']
        
        output_file = self.output_config['output_file']
        params_file = self.output_config['params_file']
        
        if save_to_excel:
            self._save_to_excel(output_file)
        
        if save_params:
            self._save_params(params_file)
        
        print("\n保存完成!")
    
    def _save_to_excel(self, output_file: str):
        """保存结果到 Excel"""
        print(f"\n保存到 Excel: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            if self.loader.all_data is not None:
                self.loader.all_data.to_excel(writer, sheet_name='原始数据_长格式', index=False)
            
            if self.within_results:
                norm_data = self.within_results.get('normalized_data', {})
                if norm_data:
                    all_norm = []
                    for province, df in norm_data.items():
                        df_copy = df.copy()
                        df_copy['省份'] = province
                        all_norm.append(df_copy)
                    
                    if all_norm:
                        combined = pd.concat(all_norm, ignore_index=True)
                        combined.to_excel(writer, sheet_name='省份内_标准化数据', index=False)
                
                corr_results = self.within_results.get('correlation_results', {})
                if corr_results:
                    all_sig = []
                    for province, result in corr_results.items():
                        if 'significant_pairs' in result:
                            for pair in result['significant_pairs']:
                                pair['省份'] = province
                                all_sig.append(pair)
                    
                    if all_sig:
                        sig_df = pd.DataFrame(all_sig)
                        sig_df['显著性'] = sig_df['p_value'].apply(
                            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                        )
                        sig_df.to_excel(writer, sheet_name='省份内_显著相关', index=False)
            
            if self.across_results:
                corr_results = self.across_results.get('correlation_results', {})
                
                if 'by_indicator' in corr_results:
                    all_sig = []
                    for indicator, result in corr_results['by_indicator'].items():
                        if 'similar_provinces' in result:
                            for pair in result['similar_provinces']:
                                pair['指标'] = indicator
                                all_sig.append(pair)
                    
                    if all_sig:
                        sig_df = pd.DataFrame(all_sig)
                        sig_df['显著性'] = sig_df['p_value'].apply(
                            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                        )
                        sig_df.to_excel(writer, sheet_name='省份间_显著相似', index=False)
            
            info_data = {
                '项目': [
                    '分析时间',
                    '省份数',
                    '指标数',
                    '年份数',
                    '省份内归一化方法',
                    '省份内相关方法',
                    '省份间归一化方法',
                    '省份间相关方法'
                ],
                '值': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(self.loader.provinces),
                    len(self.loader.indicators),
                    len(self.loader.years),
                    PIVOT_NORMALIZATION['within_province'].get('method', 'N/A'),
                    PIVOT_CORRELATION['within_province'].get('method', 'N/A'),
                    PIVOT_NORMALIZATION['across_province'].get('method', 'N/A'),
                    PIVOT_CORRELATION['across_province'].get('method', 'N/A')
                ]
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_excel(writer, sheet_name='分析信息', index=False)
            
            provinces_df = pd.DataFrame({
                '省份': self.loader.provinces,
                '序号': range(1, len(self.loader.provinces) + 1)
            })
            provinces_df.to_excel(writer, sheet_name='省份列表', index=False)
            
            indicators_df = pd.DataFrame({
                '指标': self.loader.indicators,
                '序号': range(1, len(self.loader.indicators) + 1)
            })
            indicators_df.to_excel(writer, sheet_name='指标列表', index=False)
        
        print(f"  ✓ Excel 文件已保存")
    
    def _save_params(self, params_file: str):
        """保存参数到 JSON"""
        print(f"\n保存参数: {params_file}")
        
        all_params = {
            'data_info': {
                'provinces': self.loader.provinces,
                'indicators': self.loader.indicators,
                'years': self.loader.years
            },
            'within_province': self._convert_to_serializable(
                self.within_results.get('transform_params', {})
            ),
            'across_province': self._convert_to_serializable(
                self.across_results.get('transform_params', {})
            ),
            'config': {
                'pivot': PIVOT_CONFIG,
                'normalization': PIVOT_NORMALIZATION,
                'correlation': PIVOT_CORRELATION
            }
        }
        
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(all_params, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 参数文件已保存")
    
    def run_full_analysis(self,
                          excel_file: Optional[str] = None,
                          do_within: bool = True,
                          do_across: bool = True,
                          do_similarity: bool = True,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        执行完整的分析流程
        
        参数:
            excel_file: Excel 文件路径
            do_within: 是否执行省份内分析
            do_across: 是否执行省份间分析
            do_similarity: 是否执行综合相似性分析
            save_results: 是否保存结果
            
        返回:
            所有分析结果
        """
        print("\n" + "=" * 70)
        print("透视表数据完整分析流程")
        print("=" * 70)
        
        self.load_data(excel_file)
        
        if do_within:
            self.analyze_within_province(normalize=True, use_normalized=True)
        
        if do_across:
            self.analyze_across_province(normalize=True, use_normalized=True)
        
        if do_similarity and do_across:
            self.analyze_province_similarity(use_normalized=True)
        
        if save_results:
            self.save_results()
        
        print("\n" + "=" * 70)
        print("分析完成!")
        print("=" * 70)
        
        return {
            'within_province': self.within_results,
            'across_province': self.across_results,
            'data_info': self.loader.get_basic_stats()
        }


def main():
    """
    命令行主函数
    """
    print("=" * 70)
    print("透视表数据分析工具")
    print("=" * 70)
    
    print("""
分析选项:
  1. 完整分析流程（推荐）
     - 加载数据 → 省份内分析 → 省份间分析 → 综合相似性 → 保存结果
     
  2. 仅省份内分析
     - 每个省份内部的指标相关性
     
  3. 仅省份间分析
     - 不同省份之间的相似性
     
  4. 仅加载和查看数据
     - 不进行分析，仅了解数据结构
    """)
    
    choice = input("请选择 (1-4): ").strip()
    
    analysis = PivotDataAnalysis()
    
    if choice == '1':
        print("\n执行完整分析流程...")
        analysis.run_full_analysis(
            do_within=True,
            do_across=True,
            do_similarity=True,
            save_results=True
        )
        
    elif choice == '2':
        print("\n执行省份内分析...")
        analysis.load_data()
        analysis.analyze_within_province(normalize=True, use_normalized=True)
        analysis.save_results()
        
    elif choice == '3':
        print("\n执行省份间分析...")
        analysis.load_data()
        analysis.analyze_across_province(normalize=True, use_normalized=True)
        analysis.analyze_province_similarity(use_normalized=True)
        analysis.save_results()
        
    elif choice == '4':
        print("\n加载数据...")
        analysis.load_data()
        
    else:
        print("\n默认执行选项 1...")
        analysis.run_full_analysis()


if __name__ == '__main__':
    main()

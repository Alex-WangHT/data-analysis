"""
指标相关性分析主流程
专注于：指标与指标之间的相关性分析

两种分析方式：
1. 省份内分析：每个省份内部，指标A与指标B在时间上的相关性
2. 整体分析：所有省所有年放在一起，指标A与指标B的相关性
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any

from pivot_config import (
    PIVOT_CONFIG,
    PIVOT_OUTPUT
)
from pivot_loader import PivotTableLoader
from indicator_correlation import IndicatorCorrelationAnalyzer


class IndicatorCorrelationAnalysis:
    """
    指标相关性分析主类
    专注于指标与指标之间的相关性分析
    """
    
    def __init__(self, 
                 method: str = 'pearson', 
                 alpha: float = 0.05, 
                 min_obs: int = 5):
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
        
        self.loader = PivotTableLoader()
        self.analyzer = IndicatorCorrelationAnalyzer(
            method=method,
            alpha=alpha,
            min_obs=min_obs
        )
        
        self.data_by_province: Dict[str, pd.DataFrame] = {}
        self.all_data: Optional[pd.DataFrame] = None
        
        self.within_province_results: Dict[str, Dict[str, Any]] = {}
        self.overall_results: Dict[str, Any] = {}
    
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
        self.all_data = self.loader.all_data
        
        stats = self.loader.get_basic_stats()
        
        self.analyzer.set_data_info(
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
        
        print(f"\n  省份列表: {stats['provinces']}")
        print(f"  指标列表: {stats['indicators']}")
        print(f"  年份列表: {stats['years']}")
        
        if stats.get('units'):
            print(f"\n  指标-单位映射:")
            for ind, unit in stats['units'].items():
                print(f"    {ind}: {unit}")
        
        return stats
    
    def analyze_within_province(self) -> Dict[str, Dict[str, Any]]:
        """
        执行省份内指标相关性分析
        
        分析逻辑：
        - 对每个省份
        - 获取该省各指标在各年份的值
        - 计算指标A与指标B的相关系数（时间维度上的相关性）
        
        例如：上海市的"结婚登记"与"离婚率"在2020-2025年是否相关？
        
        返回:
            各省份的相关性分析结果
        """
        print("\n" + "#" * 70)
        print("第二步: 省份内指标相关性分析")
        print("#" * 70)
        
        if not self.data_by_province:
            raise ValueError("请先调用 load_data() 加载数据")
        
        self.within_province_results = self.analyzer.analyze_within_province(
            self.data_by_province,
            year_col='年份'
        )
        
        return self.within_province_results
    
    def analyze_overall(self) -> Dict[str, Any]:
        """
        执行整体指标相关性分析
        
        分析逻辑：
        - 把所有省份所有年份的数据放在一起
        - 每一行 = 某个省某年的所有指标值
        - 计算指标A与指标B的相关系数
        
        例如："GDP"与"失业率"在所有省所有年中是否相关？
        
        返回:
            整体相关性分析结果
        """
        print("\n" + "#" * 70)
        print("第三步: 整体指标相关性分析（所有省所有年）")
        print("#" * 70)
        
        if self.all_data is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        col_names = PIVOT_CONFIG['column_names']
        
        self.overall_results = self.analyzer.analyze_overall(
            self.all_data,
            province_col=col_names['province'],
            indicator_col=col_names['indicator'],
            year_col=col_names['year'],
            value_col=col_names['value']
        )
        
        return self.overall_results
    
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
    
    def save_results(self, output_file: Optional[str] = None,
                     params_file: Optional[str] = None):
        """
        保存结果
        
        参数:
            output_file: 输出 Excel 文件名
            params_file: 参数文件名
        """
        print("\n" + "#" * 70)
        print("第四步: 保存结果")
        print("#" * 70)
        
        output_file = output_file or PIVOT_OUTPUT['output_file']
        params_file = params_file or PIVOT_OUTPUT['params_file']
        
        self._save_to_excel(output_file)
        self._save_params(params_file)
        
        print("\n保存完成!")
    
    def _save_to_excel(self, output_file: str):
        """保存到 Excel"""
        print(f"\n保存到 Excel: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            if self.all_data is not None:
                self.all_data.to_excel(writer, sheet_name='原始数据_长格式', index=False)
            
            if self.within_province_results:
                all_sig = []
                for province, result in self.within_province_results.items():
                    for pair in result.get('significant_pairs', []):
                        pair['省份'] = province
                        all_sig.append(pair)
                
                if all_sig:
                    sig_df = pd.DataFrame(all_sig)
                    sig_df['显著性'] = sig_df['p_value'].apply(
                        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                    )
                    sig_df['相关方向'] = sig_df['correlation'].apply(
                        lambda r: '正相关' if r > 0 else '负相关'
                    )
                    sig_df.to_excel(writer, sheet_name='省份内_显著相关', index=False)
                
                all_notes = []
                for province, result in self.within_province_results.items():
                    for note in result.get('notes', []):
                        all_notes.append({'省份': province, '说明': note})
                
                if all_notes:
                    notes_df = pd.DataFrame(all_notes)
                    notes_df.to_excel(writer, sheet_name='省份内_警告说明', index=False)
            
            if self.overall_results and self.overall_results.get('significant_pairs'):
                sig_df = pd.DataFrame(self.overall_results['significant_pairs'])
                sig_df['显著性'] = sig_df['p_value'].apply(
                    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                )
                sig_df['相关方向'] = sig_df['correlation'].apply(
                    lambda r: '正相关' if r > 0 else '负相关'
                )
                sig_df['相关强度'] = sig_df['correlation'].apply(
                    lambda r: '强相关' if abs(r) > 0.7 else ('中等相关' if abs(r) > 0.5 else '弱相关')
                )
                sig_df.to_excel(writer, sheet_name='整体_显著相关', index=False)
                
                if self.overall_results.get('notes'):
                    notes_df = pd.DataFrame({
                        '说明': self.overall_results['notes']
                    })
                    notes_df.to_excel(writer, sheet_name='整体_警告说明', index=False)
            
            info_data = {
                '项目': [
                    '分析时间',
                    '相关系数方法',
                    '显著性水平 α',
                    '最小观测数要求',
                    '省份数',
                    '指标数',
                    '年份数'
                ],
                '值': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.method,
                    self.alpha,
                    self.min_obs,
                    len(self.loader.provinces),
                    len(self.loader.indicators),
                    len(self.loader.years)
                ]
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_excel(writer, sheet_name='分析信息', index=False)
            
            if self.loader.provinces:
                provinces_df = pd.DataFrame({
                    '省份': self.loader.provinces,
                    '序号': range(1, len(self.loader.provinces) + 1)
                })
                provinces_df.to_excel(writer, sheet_name='省份列表', index=False)
            
            if self.loader.indicators:
                indicators_df = pd.DataFrame({
                    '指标': self.loader.indicators,
                    '序号': range(1, len(self.loader.indicators) + 1),
                    '单位': [self.loader.units.get(ind, '') for ind in self.loader.indicators]
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
                'years': self.loader.years,
                'units': self.loader.units
            },
            'analysis_config': {
                'method': self.method,
                'alpha': self.alpha,
                'min_obs': self.min_obs
            },
            'results_summary': {
                'within_province_n_significant': sum(
                    len(r.get('significant_pairs', [])) 
                    for r in self.within_province_results.values()
                ),
                'overall_n_significant': len(self.overall_results.get('significant_pairs', []))
            }
        }
        
        serializable_params = self._convert_to_serializable(all_params)
        
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_params, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 参数文件已保存")
    
    def run_full_analysis(self,
                          excel_file: Optional[str] = None,
                          do_within_province: bool = True,
                          do_overall: bool = True,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        执行完整的指标相关性分析流程
        
        参数:
            excel_file: Excel 文件路径
            do_within_province: 是否执行省份内分析
            do_overall: 是否执行整体分析
            save_results: 是否保存结果
            
        返回:
            所有分析结果
        """
        print("\n" + "=" * 70)
        print("指标相关性分析完整流程")
        print("=" * 70)
        print(f"\n相关系数方法: {self.method}")
        print(f"显著性水平: α = {self.alpha}")
        
        self.load_data(excel_file)
        
        if do_within_province:
            self.analyze_within_province()
        
        if do_overall:
            self.analyze_overall()
        
        if save_results:
            self.save_results()
        
        print("\n" + "=" * 70)
        print("分析完成!")
        print("=" * 70)
        
        return {
            'within_province': self.within_province_results,
            'overall': self.overall_results,
            'data_info': self.loader.get_basic_stats()
        }


def main():
    """
    命令行主函数
    """
    print("=" * 70)
    print("指标相关性分析工具")
    print("=" * 70)
    
    print("""
分析选项:
  1. 完整分析（推荐）
     - 加载数据 → 省份内分析 → 整体分析 → 保存结果
     
  2. 仅省份内分析
     - 每个省份内部，指标间的时间序列相关性
     
  3. 仅整体分析
     - 所有省所有年放在一起，指标间的相关性
     
  4. 仅加载和查看数据
     - 不进行分析，仅了解数据结构
     
  5. 自定义参数
     - 手动选择相关系数方法和显著性水平
    """)
    
    choice = input("请选择 (1-5): ").strip()
    
    if choice == '1':
        print("\n执行完整分析流程...")
        analysis = IndicatorCorrelationAnalysis(
            method='pearson',
            alpha=0.05,
            min_obs=5
        )
        analysis.run_full_analysis(
            do_within_province=True,
            do_overall=True,
            save_results=True
        )
        
    elif choice == '2':
        print("\n执行省份内分析...")
        analysis = IndicatorCorrelationAnalysis()
        analysis.load_data()
        analysis.analyze_within_province()
        analysis.save_results()
        
    elif choice == '3':
        print("\n执行整体分析...")
        analysis = IndicatorCorrelationAnalysis()
        analysis.load_data()
        analysis.analyze_overall()
        analysis.save_results()
        
    elif choice == '4':
        print("\n加载数据...")
        analysis = IndicatorCorrelationAnalysis()
        analysis.load_data()
        
    elif choice == '5':
        print("\n自定义参数设置:")
        
        print("\n相关系数方法:")
        print("  1. Pearson（皮尔逊，默认，适用于线性关系）")
        print("  2. Spearman（斯皮尔曼，适用于单调关系）")
        print("  3. Kendall（肯德尔，适用于序数数据）")
        method_choice = input("请选择 (1-3，默认1): ").strip()
        
        method_map = {'1': 'pearson', '2': 'spearman', '3': 'kendall'}
        method = method_map.get(method_choice, 'pearson')
        
        alpha_input = input("\n显著性水平 α（默认0.05）: ").strip()
        alpha = float(alpha_input) if alpha_input else 0.05
        
        min_obs_input = input("最小观测数要求（默认5）: ").strip()
        min_obs = int(min_obs_input) if min_obs_input else 5
        
        print(f"\n使用参数: 方法={method}, α={alpha}, 最小观测数={min_obs}")
        
        analysis = IndicatorCorrelationAnalysis(
            method=method,
            alpha=alpha,
            min_obs=min_obs
        )
        analysis.run_full_analysis(
            do_within_province=True,
            do_overall=True,
            save_results=True
        )
        
    else:
        print("\n默认执行选项 1...")
        analysis = IndicatorCorrelationAnalysis()
        analysis.run_full_analysis()


if __name__ == '__main__':
    main()

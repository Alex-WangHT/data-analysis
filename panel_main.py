"""
面板数据分析主流程
整合数据加载、归一化、相关性分析
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any

from panel_config import (
    PANEL_DATA_CONFIG, 
    NORMALIZATION_CONFIG, 
    CORRELATION_CONFIG,
    PANEL_OUTPUT_CONFIG
)
from panel_loader import PanelDataLoader
from panel_normalizer import PanelNormalizer
from panel_correlation import CorrelationAnalyzer


class PanelDataAnalysis:
    """
    面板数据分析主类
    整合完整的数据分析流程
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化面板数据分析
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        self.output_config = {**PANEL_OUTPUT_CONFIG, **self.config.get('output', {})}
        
        self.loader = PanelDataLoader(self.config)
        self.normalizer = PanelNormalizer(self.config)
        self.analyzer = CorrelationAnalyzer(self.config)
        
        self.panel_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None
        self.wide_data: Optional[pd.DataFrame] = None
        self.correlation_results: Dict[str, Any] = {}
    
    def load_data(self, excel_file: Optional[str] = None) -> pd.DataFrame:
        """
        加载并整合数据
        
        参数:
            excel_file: Excel 文件路径
            
        返回:
            面板数据
        """
        print("\n" + "=" * 70)
        print("第一步: 加载和整合数据")
        print("=" * 70)
        
        self.panel_data = self.loader.load_and_process(excel_file)
        
        stats = self.loader.get_basic_stats()
        
        print("\n" + "-" * 70)
        print("数据基本统计:")
        print("-" * 70)
        print(f"  地区数: {stats['n_provinces']}")
        print(f"  年份数: {stats['n_years']}")
        print(f"  指标数: {stats['n_indicators']}")
        print(f"  总观测数: {stats['total_obs']}")
        
        print(f"\n  缺失值按指标:")
        for ind, miss in stats['missing_by_indicator'].items():
            if miss['missing'] > 0:
                print(f"    {ind}: {miss['missing']}/{miss['total']} ({miss['pct']:.1f}%)")
        
        return self.panel_data
    
    def normalize_data(self, 
                       method: str = 'by_indicator',
                       group_by: str = 'year',
                       norm_method: str = 'zscore',
                       handle_skewness: bool = False) -> pd.DataFrame:
        """
        执行归一化
        
        参数:
            method: 归一化策略 'by_indicator' | 'by_province' | 'overall'
            group_by: 分组方式
            norm_method: 归一化方法 'zscore' | 'robust' | 'minmax'
            handle_skewness: 是否处理偏态分布
            
        返回:
            归一化后的数据
        """
        print("\n" + "=" * 70)
        print("第二步: 数据归一化")
        print("=" * 70)
        
        if self.panel_data is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        if method == 'by_indicator':
            self.normalized_data = self.normalizer.normalize_by_indicator(
                self.panel_data,
                group_by=group_by,
                method=norm_method,
                handle_skewness=handle_skewness
            )
        elif method == 'by_province':
            self.normalized_data = self.normalizer.normalize_by_province(
                self.panel_data,
                method=norm_method
            )
        else:
            self.normalized_data = self.normalizer.normalize_overall(
                self.panel_data,
                method=norm_method
            )
        
        self.wide_data = self.normalizer.pivot_normalized_to_wide(self.normalized_data)
        
        return self.normalized_data
    
    def analyze_correlation(self,
                            analysis_type: str = 'cross_province',
                            group_by: str = 'year',
                            corr_method: str = 'pearson') -> Dict[str, Any]:
        """
        执行相关性分析
        
        参数:
            analysis_type: 分析类型 'cross_province' | 'time_series'
            group_by: 分组方式
            corr_method: 相关系数方法 'pearson' | 'spearman' | 'kendall'
            
        返回:
            相关性分析结果
        """
        print("\n" + "=" * 70)
        print("第三步: 相关性分析")
        print("=" * 70)
        
        if self.wide_data is None:
            raise ValueError("请先调用 normalize_data() 进行归一化")
        
        if analysis_type == 'cross_province':
            self.correlation_results = self.analyzer.cross_province_correlation(
                self.wide_data,
                group_by=group_by,
                method=corr_method
            )
        else:
            self.correlation_results = self.analyzer.time_series_correlation(
                self.wide_data,
                method=corr_method
            )
        
        return self.correlation_results
    
    def save_results(self,
                     save_normalized: Optional[bool] = None,
                     save_correlation: Optional[bool] = None,
                     save_params: Optional[bool] = None):
        """
        保存所有结果
        
        参数:
            save_normalized: 是否保存归一化数据
            save_correlation: 是否保存相关性结果
            save_params: 是否保存变换参数
        """
        print("\n" + "=" * 70)
        print("第四步: 保存结果")
        print("=" * 70)
        
        save_normalized = save_normalized if save_normalized is not None else self.output_config['save_normalized']
        save_correlation = save_correlation if save_correlation is not None else self.output_config['save_correlation']
        save_params = save_params if save_params is not None else self.output_config['save_params']
        
        if save_normalized and self.normalized_data is not None:
            norm_file = self.output_config['normalized_file']
            
            with pd.ExcelWriter(norm_file, engine='openpyxl') as writer:
                self.normalized_data.to_excel(writer, sheet_name='长格式_标准化', index=False)
                
                if self.wide_data is not None:
                    self.wide_data.to_excel(writer, sheet_name='宽格式_标准化', index=False)
                
                original_wide = self.loader.pivot_to_wide(self.panel_data)
                original_wide.to_excel(writer, sheet_name='宽格式_原始', index=False)
                
                info_df = pd.DataFrame({
                    '项目': [
                        '数据来源',
                        '归一化方式',
                        '归一化方法',
                        '地区数',
                        '年份数',
                        '指标数',
                        '生成时间'
                    ],
                    '值': [
                        self.loader.data_config.get('excel_file', 'N/A'),
                        self.normalizer.transform_params.get('method', 'N/A'),
                        self.normalizer.norm_config.get('by_indicator', {}).get('method', 'N/A'),
                        len(self.loader.provinces),
                        len(self.loader.years),
                        len(self.loader.indicators),
                        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                })
                info_df.to_excel(writer, sheet_name='数据信息', index=False)
            
            print(f"  ✓ 归一化数据已保存到: {norm_file}")
        
        if save_correlation and self.correlation_results:
            corr_file = self.output_config['correlation_file']
            self.analyzer.export_results_to_excel(self.correlation_results, corr_file)
            print(f"  ✓ 相关性结果已保存到: {corr_file}")
        
        if save_params:
            params_file = self.output_config['params_file']
            all_params = {
                'normalization': self.normalizer.get_transform_params(),
                'data_info': {
                    'provinces': self.loader.provinces,
                    'years': self.loader.years,
                    'indicators': self.loader.indicators,
                }
            }
            
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            serializable_params = convert_to_serializable(all_params)
            
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_params, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 变换参数已保存到: {params_file}")
        
        print("\n所有结果保存完成!")
    
    def run_full_analysis(self,
                          excel_file: Optional[str] = None,
                          norm_method: str = 'by_indicator',
                          group_by: str = 'year',
                          corr_type: str = 'cross_province',
                          corr_method: str = 'pearson'):
        """
        执行完整的分析流程
        
        参数:
            excel_file: Excel 文件路径
            norm_method: 归一化方式
            group_by: 分组方式
            corr_type: 相关性分析类型
            corr_method: 相关系数方法
        """
        print("\n" + "#" * 70)
        print("面板数据完整分析流程")
        print("#" * 70)
        
        self.load_data(excel_file)
        
        self.normalize_data(
            method=norm_method,
            group_by=group_by,
            norm_method='zscore'
        )
        
        self.analyze_correlation(
            analysis_type=corr_type,
            group_by=group_by,
            corr_method=corr_method
        )
        
        self.save_results()
        
        print("\n" + "#" * 70)
        print("分析完成!")
        print("#" * 70)
        
        return self.correlation_results


def main():
    """
    命令行主函数
    """
    print("=" * 70)
    print("面板数据分析工具")
    print("=" * 70)
    
    print("""
分析选项:
  1. 完整分析流程（推荐）
     - 加载数据 → 按指标归一化 → 跨省市相关性分析
     
  2. 仅加载和查看数据
     - 不进行分析，仅了解数据结构
     
  3. 时序相关性分析
     - 按省份分组，分析各指标的时间序列相关性
     
  4. 自定义分析
     - 手动选择参数
    """)
    
    choice = input("请选择 (1-4): ").strip()
    
    analysis = PanelDataAnalysis()
    
    if choice == '1':
        print("\n执行完整分析流程...")
        analysis.run_full_analysis(
            norm_method='by_indicator',
            group_by='year',
            corr_type='cross_province',
            corr_method='pearson'
        )
        
    elif choice == '2':
        print("\n加载数据...")
        analysis.load_data()
        
    elif choice == '3':
        print("\n执行时序相关性分析...")
        analysis.load_data()
        analysis.normalize_data(method='by_province', norm_method='zscore')
        analysis.analyze_correlation(analysis_type='time_series', corr_method='pearson')
        analysis.save_results()
        
    elif choice == '4':
        print("\n自定义分析设置:")
        
        print("\n归一化方式:")
        print("  1. 按指标归一化（省间横向比较，推荐）")
        print("  2. 按省份归一化（省内纵向比较）")
        print("  3. 整体归一化")
        norm_choice = input("请选择 (1-3): ").strip()
        
        norm_method_map = {'1': 'by_indicator', '2': 'by_province', '3': 'overall'}
        norm_method = norm_method_map.get(norm_choice, 'by_indicator')
        
        print("\n相关性分析类型:")
        print("  1. 跨省市相关性（指标间关系，推荐）")
        print("  2. 时序相关性（按省份）")
        corr_choice = input("请选择 (1-2): ").strip()
        
        corr_type_map = {'1': 'cross_province', '2': 'time_series'}
        corr_type = corr_type_map.get(corr_choice, 'cross_province')
        
        analysis.run_full_analysis(
            norm_method=norm_method,
            corr_type=corr_type
        )
        
    else:
        print("\n默认执行选项 1...")
        analysis.run_full_analysis()


if __name__ == '__main__':
    main()

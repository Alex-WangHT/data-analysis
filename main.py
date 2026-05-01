"""
主流程控制模块
整合所有模块，提供完整的标准化流程接口
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any

from config import OUTPUT_CONFIG
from data_loader import DataLoader
from boxcox_transformer import DistributionAnalyzer, BoxCoxTransformer
from zscore_scaler import ZScoreScaler, StandardizationPipeline


class DataStandardizer:
    """
    数据标准化主类
    整合数据加载、分布分析、Box-Cox变换、Z-score标准化的完整流程
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据标准化器
        
        参数:
            config: 完整配置字典，包含各模块的配置
        """
        self.config = config or {}
        self.output_config = {**OUTPUT_CONFIG, **self.config.get('output', {})}
        
        self.data_loader = DataLoader(self.config)
        self.distribution_analyzer = DistributionAnalyzer(self.config)
        self.boxcox_transformer = BoxCoxTransformer(self.config)
        self.zscore_scaler = ZScoreScaler(self.config)
        self.pipeline = StandardizationPipeline(
            self.boxcox_transformer, self.zscore_scaler
        )
        
        self.original_data: Optional[pd.DataFrame] = None
        self.numeric_data: Optional[pd.DataFrame] = None
        self.standardized_data: Optional[pd.DataFrame] = None
        self.final_output: Optional[pd.DataFrame] = None
    
    def run_standardization_pipeline(self, 
                                      skip_boxcox: bool = False,
                                      skip_distribution_analysis: bool = False
                                      ) -> pd.DataFrame:
        """
        执行完整的标准化流水线
        
        参数:
            skip_boxcox: 是否跳过 Box-Cox 变换
            skip_distribution_analysis: 是否跳过分布分析
            
        返回:
            标准化后的完整 DataFrame（包含标识列）
        """
        print("\n" + "=" * 60)
        print("开始数据标准化流程")
        print("=" * 60)
        
        processed_data = self.data_loader.load_and_preprocess()
        self.original_data = processed_data.copy()
        
        self.numeric_data = self.data_loader.get_numeric_data()
        
        if not skip_distribution_analysis:
            self.distribution_analyzer.calculate_distribution_stats(self.numeric_data)
            skewed_columns, normal_columns = self.distribution_analyzer.classify_columns()
            
            recommendations = self.distribution_analyzer.get_transformation_recommendations()
            print("\n" + "=" * 60)
            print("变换建议:")
            print("=" * 60)
            for col, rec in recommendations.items():
                print(f"  {col}: {rec}")
        else:
            print("\n跳过分布分析")
            skewed_columns = list(self.numeric_data.columns) if not skip_boxcox else []
            normal_columns = list(self.numeric_data.columns) if skip_boxcox else []
        
        self.pipeline.set_columns(skewed_columns, normal_columns)
        
        if skip_boxcox:
            print("\n跳过 Box-Cox 变换，直接进行 Z-score 标准化")
            self.standardized_data = self.zscore_scaler.fit_transform(
                self.numeric_data, list(self.numeric_data.columns)
            )
        else:
            distribution_stats = self.distribution_analyzer.distribution_stats
            if not distribution_stats:
                for col in self.numeric_data.columns:
                    col_data = self.numeric_data[col]
                    distribution_stats[col] = {
                        'skewness': skew(col_data.dropna()),
                        'is_skewed': True
                    }
            
            self.standardized_data = self.pipeline.transform(
                self.numeric_data, distribution_stats
            )
        
        self.final_output = self.data_loader.get_combined_data(self.standardized_data)
        
        print("\n" + "=" * 60)
        print("标准化流程完成")
        print("=" * 60)
        
        print("\n标准化前后对比:")
        for col in self.numeric_data.columns:
            orig = self.numeric_data[col].dropna()
            std = self.standardized_data[col].dropna()
            
            print(f"\n  {col}:")
            print(f"    原始 - 均值: {orig.mean():.4f}, 标准差: {orig.std():.4f}")
            print(f"    标准化 - 均值: {std.mean():.4f}, 标准差: {std.std():.4f}")
        
        return self.final_output
    
    def save_results(self, 
                     output_file: Optional[str] = None,
                     save_params: Optional[bool] = None):
        """
        保存标准化结果和变换参数
        
        参数:
            output_file: 输出 Excel 文件名
            save_params: 是否保存变换参数
        """
        if self.final_output is None:
            raise ValueError("请先执行标准化流程")
        
        output_file = output_file or self.output_config['output_file']
        save_params = save_params if save_params is not None else self.output_config['save_transform_params']
        
        print("\n" + "=" * 60)
        print("保存结果")
        print("=" * 60)
        
        if self.output_config.get('save_to_excel', True):
            self.final_output.to_excel(output_file, index=False)
            print(f"标准化数据已保存到: {output_file}")
        
        if save_params:
            params_file = self.output_config['transform_params_file']
            all_params = {
                'pipeline': self.pipeline.get_all_params(),
                'column_info': {
                    'id_columns': self.data_loader.id_columns,
                    'numeric_columns': self.data_loader.numeric_columns,
                },
                'distribution_stats': self.distribution_analyzer.distribution_stats
            }
            
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return obj
            
            all_params_serializable = convert_to_serializable(all_params)
            
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(all_params_serializable, f, ensure_ascii=False, indent=2)
            
            print(f"变换参数已保存到: {params_file}")
        
        print("\n保存完成!")
    
    def load_params_and_transform(self, 
                                   params_file: str,
                                   new_data: Optional[pd.DataFrame] = None
                                   ) -> pd.DataFrame:
        """
        加载已保存的参数并对新数据进行变换
        
        参数:
            params_file: 参数字典文件路径
            new_data: 新数据 DataFrame，如未提供则使用原始数据
            
        返回:
            标准化后的 DataFrame
        """
        print(f"\n加载变换参数: {params_file}")
        
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        self.pipeline.load_all_params(params.get('pipeline', {}))
        
        if new_data is None:
            if self.original_data is None:
                raise ValueError("请提供新数据或先执行标准化流程")
            numeric_data = self.original_data[self.pipeline.all_numeric_columns]
        else:
            numeric_columns = params.get('column_info', {}).get('numeric_columns', [])
            numeric_data = new_data[numeric_columns]
        
        standardized = self.pipeline.zscore_scaler.transform(
            numeric_data, self.pipeline.all_numeric_columns
        )
        
        if params.get('pipeline', {}).get('skewed_columns'):
            for col in self.pipeline.skewed_columns:
                if col in standardized.columns:
                    pass
        
        return standardized


def main():
    """
    主函数
    提供命令行使用示例
    """
    print("=" * 60)
    print("经济数据标准化工具")
    print("=" * 60)
    
    custom_config = {
        'missing': {
            'strategy': 'interpolate',
        },
        'distribution': {
            'skewness_threshold': 0.5,
        },
        'output': {
            'output_file': '标准化数据.xlsx',
            'transform_params_file': '变换参数.json',
        }
    }
    
    standardizer = DataStandardizer(custom_config)
    
    print("\n选项:")
    print("  1. 执行完整流程（分布分析 + Box-Cox + Z-score）")
    print("  2. 仅执行 Z-score 标准化（跳过 Box-Cox）")
    print("  3. 仅查看数据分布（不执行标准化）")
    print("  4. 退出")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == '1':
        print("\n执行完整标准化流程...")
        result = standardizer.run_standardization_pipeline()
        standardizer.save_results()
        
    elif choice == '2':
        print("\n执行简单 Z-score 标准化...")
        result = standardizer.run_standardization_pipeline(skip_boxcox=True)
        standardizer.save_results()
        
    elif choice == '3':
        print("\n进行数据分布分析...")
        standardizer.data_loader.load_and_preprocess()
        numeric_data = standardizer.data_loader.get_numeric_data()
        standardizer.distribution_analyzer.calculate_distribution_stats(numeric_data)
        standardizer.distribution_analyzer.classify_columns()
        
    elif choice == '4':
        print("退出程序")
        return
    
    else:
        print("无效选择，使用默认选项 1")
        result = standardizer.run_standardization_pipeline()
        standardizer.save_results()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

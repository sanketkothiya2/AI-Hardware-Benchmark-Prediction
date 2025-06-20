#!/usr/bin/env python3
"""
Focused AI Hardware Performance ETL Pipeline
Extracts specific key metrics for AI hardware performance analysis and KPI prediction

Key Metrics Extracted:
- Latency (ms) - from MLPerf inference times
- Throughput (operations/sec) - from MLPerf results  
- FLOPS (operations/sec) - from ml_hardware.csv precision levels
- Memory Bandwidth (GB/s) - from GPU benchmarks and ml_hardware
- Energy Consumption (Watts) - TDP from hardware specs
- Precision Performance (FP32, FP16, INT8) - from ml_hardware.csv
- FPS (Frames per Second) - derived from graphics benchmarks
- TOPs/Watt - calculated efficiency metrics
- Memory Usage % - derived from memory specifications
- Network/Compute Density - derived ratios
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedAIHardwareETL:
    def __init__(self, raw_data_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self):
        """Load all raw CSV files"""
        try:
            logger.info("Loading raw datasets...")
            
            datasets = {}
            
            # Load GPU benchmarks
            gpu_bench_path = self.raw_data_dir / "GPU_benchmarks_v7.csv"
            if gpu_bench_path.exists():
                datasets['gpu_benchmarks'] = pd.read_csv(gpu_bench_path)
                logger.info(f"Loaded GPU benchmarks: {len(datasets['gpu_benchmarks'])} records")
            
            # Load GPU graphics APIs
            gpu_graphics_path = self.raw_data_dir / "GPU_scores_graphicsAPIs.csv"
            if gpu_graphics_path.exists():
                datasets['gpu_graphics'] = pd.read_csv(gpu_graphics_path)
                logger.info(f"Loaded GPU graphics: {len(datasets['gpu_graphics'])} records")
            
            # Load ML hardware
            ml_hw_path = self.raw_data_dir / "ml_hardware.csv"
            if ml_hw_path.exists():
                datasets['ml_hardware'] = pd.read_csv(ml_hw_path)
                logger.info(f"Loaded ML hardware: {len(datasets['ml_hardware'])} records")
            
            # Load MLPerf
            mlperf_path = self.raw_data_dir / "mlperf.csv"
            if mlperf_path.exists():
                datasets['mlperf'] = pd.read_csv(mlperf_path)
                logger.info(f"Loaded MLPerf: {len(datasets['mlperf'])} records")
                
            return datasets
                
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    
    def extract_key_metrics(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract the specific key metrics from all datasets"""
        logger.info("Extracting key metrics...")
        
        all_metrics = []
        
        # Process ML Hardware data (primary source for FLOPS and compute specs)
        if 'ml_hardware' in datasets:
            ml_hw = datasets['ml_hardware']
            for _, row in ml_hw.iterrows():
                try:
                    metrics = {
                        'hardware_name': self._clean_hardware_name(row.get('Hardware name', '')),
                        'manufacturer': row.get('Manufacturer', ''),
                        'type': row.get('Type', ''),
                        'release_year': self._extract_year(row.get('Release date', '')),
                        
                        # FLOPS metrics (converted to TFLOPS/TOPS)
                        'fp64_tflops': self._extract_flops(row.get('FP64 (double precision) performance (FLOP/s)', 0)) / 1e12,
                        'fp32_tflops': self._extract_flops(row.get('FP32 (single precision) performance (FLOP/s)', 0)) / 1e12,
                        'fp16_tflops': self._extract_flops(row.get('FP16 (half precision) performance (FLOP/s)', 0)) / 1e12,
                        'int8_tops': self._extract_flops(row.get('INT8 performance (OP/s)', 0)) / 1e12,
                        'int4_tops': self._extract_flops(row.get('INT4 performance (OP/s)', 0)) / 1e12,
                        
                        # Memory metrics
                        'memory_size_gb': self._extract_numeric(row.get('Memory size per board (Byte)', 0)) / 1e9,
                        'memory_bandwidth_gbps': self._extract_numeric(row.get('Memory bandwidth (byte/s)', 0)) / 1e9,
                        
                        # Energy consumption
                        'energy_consumption_watts': self._extract_numeric(row.get('TDP (W)', 0)),
                        
                        # Source identifier
                        'data_source': 'ml_hardware'
                    }
                    
                    if metrics['hardware_name']:  # Only add if we have a valid hardware name
                        all_metrics.append(metrics)
                        
                except Exception as e:
                    continue
        
        # Process GPU benchmarks data
        if 'gpu_benchmarks' in datasets:
            gpu_bench = datasets['gpu_benchmarks']
            for _, row in gpu_bench.iterrows():
                try:
                    hw_name = self._clean_hardware_name(row.get('gpuName', ''))
                    
                    # Check if we already have this hardware from ML data
                    existing = next((m for m in all_metrics if m['hardware_name'] == hw_name), None)
                    
                    if existing:
                        # Update existing entry with benchmark data
                        existing.update({
                            'g3d_mark_score': self._extract_numeric(row.get('G3Dmark', None)),
                            'g2d_mark_score': self._extract_numeric(row.get('G2Dmark', None)),
                            'price_usd': self._extract_numeric(row.get('price', None)),
                            'category': row.get('category', ''),
                            'test_date': row.get('testDate', '')
                        })
                        
                        # Calculate derived FPS indicator from G3D Mark
                        if existing.get('g3d_mark_score'):
                            existing['fps_indicator'] = existing['g3d_mark_score'] / 1000
                            
                        # Calculate performance per dollar if price available
                        if existing.get('g3d_mark_score') and existing.get('price_usd'):
                            existing['perf_per_dollar'] = existing['g3d_mark_score'] / existing['price_usd']
                    else:
                        # Create new entry from benchmark data
                        metrics = {
                            'hardware_name': hw_name,
                            'manufacturer': '',
                            'type': 'GPU',
                            'g3d_mark_score': self._extract_numeric(row.get('G3Dmark', None)),
                            'g2d_mark_score': self._extract_numeric(row.get('G2Dmark', None)),
                            'price_usd': self._extract_numeric(row.get('price', None)),
                            'energy_consumption_watts': self._extract_numeric(row.get('TDP', None)),
                            'category': row.get('category', ''),
                            'test_date': row.get('testDate', ''),
                            'data_source': 'gpu_benchmarks'
                        }
                        
                        # Calculate FPS indicator
                        if metrics.get('g3d_mark_score'):
                            metrics['fps_indicator'] = metrics['g3d_mark_score'] / 1000
                            
                        if hw_name:
                            all_metrics.append(metrics)
                            
                except Exception as e:
                    continue
        
        # Process GPU Graphics API data
        if 'gpu_graphics' in datasets:
            gpu_graphics = datasets['gpu_graphics']
            for _, row in gpu_graphics.iterrows():
                try:
                    hw_name = self._clean_hardware_name(row.get('Device', ''))
                    
                    # Find existing entry or create new one
                    existing = next((m for m in all_metrics if m['hardware_name'] == hw_name), None)
                    
                    graphics_data = {
                        'cuda_score': self._extract_numeric(row.get('CUDA', None)),
                        'metal_score': self._extract_numeric(row.get('Metal', None)),
                        'opencl_score': self._extract_numeric(row.get('OpenCL', None)),
                        'vulkan_score': self._extract_numeric(row.get('Vulkan', None))
                    }
                    
                    if existing:
                        existing.update(graphics_data)
                    elif hw_name:
                        graphics_data.update({
                            'hardware_name': hw_name,
                            'manufacturer': row.get('Manufacturer', ''),
                            'type': 'GPU',
                            'data_source': 'gpu_graphics'
                        })
                        all_metrics.append(graphics_data)
                        
                except Exception as e:
                    continue
        
        # Process MLPerf data for latency and throughput
        if 'mlperf' in datasets:
            mlperf = datasets['mlperf']
            
            # Aggregate MLPerf results by hardware
            mlperf_metrics = {}
            
            for _, row in mlperf.iterrows():
                try:
                    hw_name = self._clean_hardware_name(row.get('Accelerator', ''))
                    scenario = row.get('Scenario', '')
                    result = self._extract_numeric(row.get('Avg. Result at System Name', 0))
                    
                    if not hw_name or not result:
                        continue
                    
                    if hw_name not in mlperf_metrics:
                        mlperf_metrics[hw_name] = {
                            'throughput_measurements': [],
                            'latency_measurements': []
                        }
                    
                    # Server scenario typically provides latency-related data
                    if scenario == 'Server':
                        # Approximate latency from throughput (ms per operation)
                        if result > 0:
                            latency_ms = 1000 / result  # Convert throughput to latency
                            mlperf_metrics[hw_name]['latency_measurements'].append(latency_ms)
                    
                    # All scenarios provide throughput data
                    mlperf_metrics[hw_name]['throughput_measurements'].append(result)
                    
                except Exception as e:
                    continue
            
            # Add aggregated MLPerf data to metrics
            for hw_name, perf_data in mlperf_metrics.items():
                existing = next((m for m in all_metrics if m['hardware_name'] == hw_name), None)
                
                mlperf_summary = {}
                
                if perf_data['throughput_measurements']:
                    mlperf_summary['throughput_ops_sec'] = np.mean(perf_data['throughput_measurements'])
                    mlperf_summary['max_throughput_ops_sec'] = np.max(perf_data['throughput_measurements'])
                
                if perf_data['latency_measurements']:
                    mlperf_summary['latency_ms'] = np.mean(perf_data['latency_measurements'])
                    mlperf_summary['min_latency_ms'] = np.min(perf_data['latency_measurements'])
                
                if existing:
                    existing.update(mlperf_summary)
                elif hw_name and mlperf_summary:
                    mlperf_summary.update({
                        'hardware_name': hw_name,
                        'data_source': 'mlperf'
                    })
                    all_metrics.append(mlperf_summary)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        if df.empty:
            logger.warning("No metrics extracted from datasets")
            return df
        
        # Calculate derived efficiency metrics
        df = self._calculate_efficiency_metrics(df)
        
        # Remove entries with no useful data
        df = df.dropna(subset=['hardware_name'])
        df = df[df['hardware_name'].str.len() > 0]
        
        logger.info(f"Extracted metrics for {len(df)} hardware entries")
        return df
    
    def _calculate_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency and derived metrics"""
        logger.info("Calculating efficiency metrics...")
        
        # TOPs per Watt for different precisions
        for precision in ['fp32_tflops', 'fp16_tflops', 'int8_tops']:
            col_name = f"tops_per_watt_{precision.split('_')[0]}"
            df[col_name] = (df[precision] / df['energy_consumption_watts']).where(df['energy_consumption_watts'] > 0)
        
        # Memory efficiency ratios
        df['memory_compute_ratio'] = (df['memory_bandwidth_gbps'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
        
        # Precision scaling ratios  
        df['fp16_to_fp32_scaling'] = (df['fp16_tflops'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
        df['int8_to_fp32_scaling'] = (df['int8_tops'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
        
        # Memory usage estimation (percentage)
        df['memory_usage_percent_est'] = np.clip(
            (df['fp32_tflops'] * 4) / (df['memory_bandwidth_gbps'] * 1000) * 100, 0, 100
        )
        
        # Compute density (performance per GB of memory)
        df['compute_density_tflops_per_gb'] = (df['fp32_tflops'] / df['memory_size_gb']).where(df['memory_size_gb'] > 0)
        
        # Performance per watt from benchmarks
        df['perf_per_watt_g3d'] = (df['g3d_mark_score'] / df['energy_consumption_watts']).where(df['energy_consumption_watts'] > 0)
        
        return df
    
    def create_matrix_relations(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create specific matrix relations for KPI analysis"""
        logger.info("Creating performance matrices...")
        
        matrices = {}
        
        # 1. FLOPS Capability Matrix (Hardware x Precision Levels)
        flops_cols = ['fp64_tflops', 'fp32_tflops', 'fp16_tflops', 'int8_tops', 'int4_tops']
        flops_data = df[['hardware_name'] + flops_cols].dropna(subset=['hardware_name'])
        if not flops_data.empty:
            matrices['flops_capability_matrix'] = flops_data.set_index('hardware_name')
        
        # 2. Power Efficiency Matrix (Hardware x Efficiency Metrics)
        efficiency_cols = ['tops_per_watt_fp32', 'tops_per_watt_fp16', 'tops_per_watt_int8', 'perf_per_watt_g3d']
        efficiency_data = df[['hardware_name'] + efficiency_cols].dropna(subset=['hardware_name'])
        if not efficiency_data.empty:
            matrices['power_efficiency_matrix'] = efficiency_data.set_index('hardware_name')
        
        # 3. Memory Performance Matrix
        memory_cols = ['memory_size_gb', 'memory_bandwidth_gbps', 'memory_compute_ratio', 'compute_density_tflops_per_gb']
        memory_data = df[['hardware_name'] + memory_cols].dropna(subset=['hardware_name'])
        if not memory_data.empty:
            matrices['memory_performance_matrix'] = memory_data.set_index('hardware_name')
        
        # 4. Latency-Throughput Matrix
        perf_cols = ['latency_ms', 'throughput_ops_sec', 'max_throughput_ops_sec', 'min_latency_ms']
        perf_data = df[['hardware_name'] + perf_cols].dropna(subset=['hardware_name'])
        if not perf_data.empty:
            matrices['latency_throughput_matrix'] = perf_data.set_index('hardware_name')
        
        # 5. Graphics Performance Matrix (FPS and API scores)
        graphics_cols = ['fps_indicator', 'cuda_score', 'opencl_score', 'vulkan_score', 'g3d_mark_score']
        graphics_data = df[['hardware_name'] + graphics_cols].dropna(subset=['hardware_name'])
        if not graphics_data.empty:
            matrices['graphics_performance_matrix'] = graphics_data.set_index('hardware_name')
        
        # 6. Price-Performance Matrix
        price_cols = ['price_usd', 'perf_per_dollar', 'g3d_mark_score', 'fp32_tflops']
        price_data = df[['hardware_name'] + price_cols].dropna(subset=['hardware_name', 'price_usd'])
        if not price_data.empty:
            matrices['price_performance_matrix'] = price_data.set_index('hardware_name')
        
        return matrices
    
    def _clean_hardware_name(self, name: str) -> str:
        """Clean and standardize hardware names"""
        if pd.isna(name) or not isinstance(name, str):
            return ''
        
        # Basic cleaning
        name = str(name).strip()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove common suffixes that don't affect identity
        name = re.sub(r'\s*(PCIe|SXM\d*|OAM|NVL\d*)(\s+\d+GB)?$', '', name, flags=re.IGNORECASE)
        
        return name
    
    def _extract_flops(self, value) -> float:
        """Extract FLOPS value from various formats"""
        if pd.isna(value) or value == 0:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_numeric(self, value) -> Optional[float]:
        """Extract numeric value handling various formats"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value) if not pd.isna(value) else None
        if isinstance(value, str):
            # Extract numbers from string
            clean_value = re.sub(r'[^\d.-]', '', str(value))
            try:
                return float(clean_value) if clean_value else None
            except ValueError:
                return None
        return None
    
    def _extract_year(self, date_str) -> Optional[int]:
        """Extract year from date string"""
        if pd.isna(date_str):
            return None
        
        year_match = re.search(r'(\d{4})', str(date_str))
        if year_match:
            try:
                year = int(year_match.group(1))
                return year if 2000 <= year <= 2030 else None
            except ValueError:
                return None
        return None
    
    def run_etl_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Run the complete focused ETL pipeline"""
        logger.info("Starting Focused AI Hardware ETL Pipeline...")
        
        # Load raw data
        datasets = self.load_raw_data()
        
        if not datasets:
            logger.error("No datasets loaded")
            return {}
        
        # Extract key metrics
        metrics_df = self.extract_key_metrics(datasets)
        
        if metrics_df.empty:
            logger.error("No metrics extracted")
            return {}
        
        # Create matrix relations
        matrices = self.create_matrix_relations(metrics_df)
        
        # Add comprehensive dataset
        matrices['comprehensive_dataset'] = metrics_df
        
        # Save results
        self.save_results(matrices)
        
        # Generate summary
        self.generate_summary_report(matrices)
        
        logger.info("Focused ETL Pipeline completed successfully!")
        return matrices
    
    def save_results(self, matrices: Dict[str, pd.DataFrame]) -> None:
        """Save all matrices and datasets"""
        logger.info("Saving results...")
        
        for name, df in matrices.items():
            if df.empty:
                continue
            
            output_path = self.output_dir / f"focused_{name}.csv"
            df.to_csv(output_path, index=True)
            logger.info(f"Saved {name}: {len(df)} records to {output_path}")
    
    def generate_summary_report(self, matrices: Dict[str, pd.DataFrame]) -> None:
        """Generate summary report"""
        logger.info("Generating summary report...")
        
        report_lines = [
            "# Focused AI Hardware Performance ETL Report",
            f"Generated: {pd.Timestamp.now()}",
            "",
            "## Key Metrics Coverage:",
            ""
        ]
        
        main_df = matrices.get('comprehensive_dataset', pd.DataFrame())
        
        if not main_df.empty:
            # Metrics coverage
            metrics_coverage = {
                'Total Hardware Entries': len(main_df),
                'FLOPS Data (FP32)': main_df['fp32_tflops'].notna().sum(),
                'FLOPS Data (FP16)': main_df['fp16_tflops'].notna().sum(), 
                'FLOPS Data (INT8)': main_df['int8_tops'].notna().sum(),
                'Memory Bandwidth': main_df['memory_bandwidth_gbps'].notna().sum(),
                'Energy Consumption': main_df['energy_consumption_watts'].notna().sum(),
                'Latency Data': main_df['latency_ms'].notna().sum(),
                'Throughput Data': main_df['throughput_ops_sec'].notna().sum(),
                'Graphics Performance': main_df['fps_indicator'].notna().sum(),
                'Price Data': main_df['price_usd'].notna().sum()
            }
            
            for metric, count in metrics_coverage.items():
                pct = (count / len(main_df) * 100) if len(main_df) > 0 else 0
                report_lines.append(f"- {metric}: {count} ({pct:.1f}%)")
            
            report_lines.extend([
                "",
                "## Generated Matrices:",
                ""
            ])
            
            for name, df in matrices.items():
                if name != 'comprehensive_dataset' and not df.empty:
                    report_lines.append(f"- {name}: {df.shape[0]} × {df.shape[1]}")
            
            # Top performers
            report_lines.extend([
                "",
                "## Top Performers:",
                ""
            ])
            
            if 'fp32_tflops' in main_df.columns:
                top_compute = main_df.nlargest(3, 'fp32_tflops')['hardware_name'].tolist()
                report_lines.append(f"- Compute (FP32): {', '.join(top_compute)}")
            
            if 'tops_per_watt_fp32' in main_df.columns:
                top_efficiency = main_df.nlargest(3, 'tops_per_watt_fp32')['hardware_name'].tolist()
                report_lines.append(f"- Efficiency: {', '.join(top_efficiency)}")
        
        # Save report
        report_path = self.output_dir / "focused_etl_summary.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to {report_path}")

def main():
    """Run the focused ETL pipeline"""
    etl = FocusedAIHardwareETL()
    results = etl.run_etl_pipeline()
    
    if results:
        print("\n=== FOCUSED ETL PIPELINE COMPLETED ===")
        print(f"Generated {len(results)} datasets:")
        for name, df in results.items():
            print(f"  - {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        print("ETL Pipeline failed")

if __name__ == "__main__":
    main() 
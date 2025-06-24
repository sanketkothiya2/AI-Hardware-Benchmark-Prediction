#!/usr/bin/env python3
"""
AI Benchmark CSV to Normalized PostgreSQL Database Importer
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import logging
import sys
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'AI_BENCHMARK',
    'user': 'postgres',
    'password': 'admin',
    'port': 5432
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIBenchmarkImporter:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.conn = None
        self.df = None
        self.lookup_cache = {'manufacturers': {}, 'architectures': {}, 'categories': {}}
        
    def connect_to_database(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.autocommit = False
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_csv_data(self):
        """Load and preprocess CSV data"""
        try:
            logger.info(f"Loading CSV data from {self.csv_file_path}")
            self.df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            
            # Clean column names
            self.df.columns = [col.strip() for col in self.df.columns]
            
            # Handle missing values
            self.clean_data()
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def clean_data(self):
        """Clean and preprocess the data"""
        logger.info("Cleaning data...")
        
        # Replace empty strings and 'Unknown' with None
        self.df = self.df.replace(['', 'Unknown'], None)
        
        # Clean numeric columns
        numeric_columns = [
            'G3Dmark', 'G2Dmark', 'TDP', 'powerPerformance', 'FP32_Final',
            'testDate', 'price', 'FLOPS_per_Watt', 'gpuValue', 'Memory_GB',
            'Process size (nm)', 'CUDA', 'OpenCL', 'Vulkan', 'Metal',
            'PricePerformanceIndex', 'TOPs_per_Watt', 'Relative_Latency_Index',
            'Compute_Usage_Percent', 'GFLOPS_per_Watt', 'Performance_per_Dollar_per_Watt'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Handle boolean columns
        if 'IsLegacyLowPerf' in self.df.columns:
            self.df['IsLegacyLowPerf'] = self.df['IsLegacyLowPerf'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
        
        # Handle large numeric columns
        large_numeric_cols = [
            'FP16 (half precision) performance (FLOP/s)',
            'INT8 performance (OP/s)',
            'Memory size per board (Byte)',
            'Memory bandwidth (byte/s)',
            'FP16_Performance_Predicted',
            'INT8_Performance_Estimated'
        ]
        
        for col in large_numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        logger.info("Data cleaning completed")
    
    def get_or_create_lookup_id(self, table: str, name: str, manufacturer_id: int = None) -> int:
        """Get or create ID for lookup tables"""
        if name is None or str(name).strip() == '' or str(name).lower() == 'nan':
            return None
            
        name = str(name).strip()
        cache_key = f"{name}_{manufacturer_id}" if manufacturer_id else name
        
        if cache_key in self.lookup_cache[table]:
            return self.lookup_cache[table][cache_key]
        
        cursor = self.conn.cursor()
        
        try:
            if table == 'manufacturers':
                cursor.execute("SELECT manufacturer_id FROM manufacturers WHERE name = %s", (name,))
                result = cursor.fetchone()
                
                if result:
                    manufacturer_id = result[0]
                else:
                    cursor.execute("INSERT INTO manufacturers (name) VALUES (%s) RETURNING manufacturer_id", (name,))
                    manufacturer_id = cursor.fetchone()[0]
                
                self.lookup_cache[table][name] = manufacturer_id
                return manufacturer_id
            
            elif table == 'architectures':
                cursor.execute("SELECT architecture_id FROM architectures WHERE name = %s", (name,))
                result = cursor.fetchone()
                
                if result:
                    architecture_id = result[0]
                else:
                    cursor.execute("INSERT INTO architectures (name, manufacturer_id) VALUES (%s, %s) RETURNING architecture_id", (name, manufacturer_id))
                    architecture_id = cursor.fetchone()[0]
                
                self.lookup_cache[table][cache_key] = architecture_id
                return architecture_id
            
            elif table == 'categories':
                cursor.execute("SELECT category_id FROM categories WHERE name = %s", (name,))
                result = cursor.fetchone()
                
                if result:
                    category_id = result[0]
                else:
                    cursor.execute("INSERT INTO categories (name) VALUES (%s) RETURNING category_id", (name,))
                    category_id = cursor.fetchone()[0]
                
                self.lookup_cache[table][name] = category_id
                return category_id
        
        except Exception as e:
            logger.error(f"Error in get_or_create_lookup_id for {table}: {e}")
            raise
        finally:
            cursor.close()
    
    def import_gpus(self):
        """Import GPU data into the gpus table"""
        logger.info("Importing GPU data...")
        
        cursor = self.conn.cursor()
        gpu_data = []
        
        for idx, row in self.df.iterrows():
            try:
                manufacturer_id = self.get_or_create_lookup_id('manufacturers', row.get('Manufacturer'))
                architecture_id = self.get_or_create_lookup_id('architectures', row.get('Architecture'), manufacturer_id)
                category_id = self.get_or_create_lookup_id('categories', row.get('Category'))
                
                gpu_record = (
                    row.get('gpuName'),
                    manufacturer_id,
                    architecture_id,
                    category_id,
                    row.get('PerformanceCategory'),
                    row.get('GenerationCategory'),
                    row.get('PerformanceTier'),
                    row.get('Generation'),
                    row.get('EfficiencyClass'),
                    int(row['TDP']) if pd.notna(row.get('TDP')) else None,
                    int(row['Process size (nm)']) if pd.notna(row.get('Process size (nm)')) else None,
                    int(row['testDate']) if pd.notna(row.get('testDate')) else None,
                    float(row['price']) if pd.notna(row.get('price')) else None,
                    float(row['powerPerformance']) if pd.notna(row.get('powerPerformance')) else None,
                    float(row['gpuValue']) if pd.notna(row.get('gpuValue')) else None,
                    float(row['PricePerformanceIndex']) if pd.notna(row.get('PricePerformanceIndex')) else None,
                    bool(row['IsLegacyLowPerf']) if pd.notna(row.get('IsLegacyLowPerf')) else False,
                )
                
                gpu_data.append(gpu_record)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        insert_query = """
            INSERT INTO gpus (
                gpu_name, manufacturer_id, architecture_id, category_id,
                performance_category, generation_category, performance_tier, generation,
                efficiency_class, tdp, process_size_nm, test_date, price,
                power_performance, gpu_value, price_performance_index, is_legacy_low_perf
            ) VALUES %s
        """
        
        execute_values(cursor, insert_query, gpu_data, template=None, page_size=100)
        self.conn.commit()
        
        logger.info(f"Imported {len(gpu_data)} GPU records")
        cursor.close()
    
    def import_performance_metrics(self):
        """Import performance metrics data"""
        logger.info("Importing performance metrics...")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT gpu_id, gpu_name FROM gpus")
        gpu_mapping = {name: gpu_id for gpu_id, name in cursor.fetchall()}
        
        performance_data = []
        
        for idx, row in self.df.iterrows():
            gpu_name = row.get('gpuName')
            if gpu_name not in gpu_mapping:
                continue
                
            gpu_id = gpu_mapping[gpu_name]
            
            perf_record = (
                gpu_id,
                int(row['G3Dmark']) if pd.notna(row.get('G3Dmark')) else None,
                int(row['G2Dmark']) if pd.notna(row.get('G2Dmark')) else None,
                int(row['FP32_Final']) if pd.notna(row.get('FP32_Final')) else None,
                float(row['FLOPS_per_Watt']) if pd.notna(row.get('FLOPS_per_Watt')) else None,
                float(row['GFLOPS_per_Watt']) if pd.notna(row.get('GFLOPS_per_Watt')) else None,
                float(row['Performance_per_Dollar_per_Watt']) if pd.notna(row.get('Performance_per_Dollar_per_Watt')) else None,
                int(row['FP16 (half precision) performance (FLOP/s)']) if pd.notna(row.get('FP16 (half precision) performance (FLOP/s)')) else None,
                int(row['FP16_Performance_Predicted']) if pd.notna(row.get('FP16_Performance_Predicted')) else None,
                int(row['INT8 performance (OP/s)']) if pd.notna(row.get('INT8 performance (OP/s)')) else None,
                int(row['INT8_Performance_Estimated']) if pd.notna(row.get('INT8_Performance_Estimated')) else None,
                float(row['TOPs_per_Watt']) if pd.notna(row.get('TOPs_per_Watt')) else None,
                float(row['Relative_Latency_Index']) if pd.notna(row.get('Relative_Latency_Index')) else None,
                float(row['Compute_Usage_Percent']) if pd.notna(row.get('Compute_Usage_Percent')) else None,
                float(row['Throughput_ResNet50_ImageNet_fps']) if pd.notna(row.get('Throughput_ResNet50_ImageNet_fps')) else None,
                float(row['Throughput_BERT_Base_fps']) if pd.notna(row.get('Throughput_BERT_Base_fps')) else None,
                float(row['Throughput_GPT2_Small_fps']) if pd.notna(row.get('Throughput_GPT2_Small_fps')) else None,
                float(row['Throughput_MobileNetV2_fps']) if pd.notna(row.get('Throughput_MobileNetV2_fps')) else None,
                float(row['Throughput_EfficientNet_B0_fps']) if pd.notna(row.get('Throughput_EfficientNet_B0_fps')) else None,
                float(row['Avg_Throughput_fps']) if pd.notna(row.get('Avg_Throughput_fps')) else None,
            )
            
            performance_data.append(perf_record)
        
        insert_query = """
            INSERT INTO performance_metrics (
                gpu_id, g3d_mark, g2d_mark, fp32_final, flops_per_watt, gflops_per_watt,
                performance_per_dollar_per_watt, fp16_performance, fp16_performance_predicted,
                int8_performance, int8_performance_estimated, tops_per_watt,
                relative_latency_index, compute_usage_percent,
                throughput_resnet50_imagenet_fps, throughput_bert_base_fps,
                throughput_gpt2_small_fps, throughput_mobilenetv2_fps,
                throughput_efficientnet_b0_fps, avg_throughput_fps
            ) VALUES %s
        """
        
        execute_values(cursor, insert_query, performance_data, template=None, page_size=100)
        self.conn.commit()
        
        logger.info(f"Imported {len(performance_data)} performance metric records")
        cursor.close()
    
    def import_memory_specs(self):
        """Import memory specifications data"""
        logger.info("Importing memory specifications...")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT gpu_id, gpu_name FROM gpus")
        gpu_mapping = {name: gpu_id for gpu_id, name in cursor.fetchall()}
        
        memory_data = []
        
        for idx, row in self.df.iterrows():
            gpu_name = row.get('gpuName')
            if gpu_name not in gpu_mapping:
                continue
                
            gpu_id = gpu_mapping[gpu_name]
            
            memory_record = (
                gpu_id,
                int(row['Memory size per board (Byte)']) if pd.notna(row.get('Memory size per board (Byte)')) else None,
                float(row['Memory_GB']) if pd.notna(row.get('Memory_GB')) else None,
                row.get('MemoryTier'),
                int(row['Memory bandwidth (byte/s)']) if pd.notna(row.get('Memory bandwidth (byte/s)')) else None,
            )
            
            memory_data.append(memory_record)
        
        insert_query = """
            INSERT INTO memory_specs (
                gpu_id, memory_size_bytes, memory_gb, memory_tier, memory_bandwidth_bytes_per_sec
            ) VALUES %s
        """
        
        execute_values(cursor, insert_query, memory_data, template=None, page_size=100)
        self.conn.commit()
        
        logger.info(f"Imported {len(memory_data)} memory specification records")
        cursor.close()
    
    def import_process_technology(self):
        """Import process technology data"""
        logger.info("Importing process technology...")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT gpu_id, gpu_name FROM gpus")
        gpu_mapping = {name: gpu_id for gpu_id, name in cursor.fetchall()}
        
        process_data = []
        
        for idx, row in self.df.iterrows():
            gpu_name = row.get('gpuName')
            if gpu_name not in gpu_mapping:
                continue
                
            gpu_id = gpu_mapping[gpu_name]
            
            process_record = (
                gpu_id,
                int(row['CUDA']) if pd.notna(row.get('CUDA')) else None,
                int(row['OpenCL']) if pd.notna(row.get('OpenCL')) else None,
                int(row['Vulkan']) if pd.notna(row.get('Vulkan')) else None,
                int(row['Metal']) if pd.notna(row.get('Metal')) else None,
            )
            
            process_data.append(process_record)
        
        insert_query = """
            INSERT INTO process_technology (
                gpu_id, cuda_support, opencl_support, vulkan_support, metal_support
            ) VALUES %s
        """
        
        execute_values(cursor, insert_query, process_data, template=None, page_size=100)
        self.conn.commit()
        
        logger.info(f"Imported {len(process_data)} process technology records")
        cursor.close()
    
    def import_ai_performance(self):
        """Import AI performance data"""
        logger.info("Importing AI performance...")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT gpu_id, gpu_name FROM gpus")
        gpu_mapping = {name: gpu_id for gpu_id, name in cursor.fetchall()}
        
        ai_perf_data = []
        
        for idx, row in self.df.iterrows():
            gpu_name = row.get('gpuName')
            if gpu_name not in gpu_mapping:
                continue
                
            gpu_id = gpu_mapping[gpu_name]
            
            ai_perf_record = (
                gpu_id,
                row.get('AI_Efficiency_Tier'),
                row.get('AI_Performance_Category'),
            )
            
            ai_perf_data.append(ai_perf_record)
        
        insert_query = """
            INSERT INTO ai_performance (
                gpu_id, ai_efficiency_tier, ai_performance_category
            ) VALUES %s
        """
        
        execute_values(cursor, insert_query, ai_perf_data, template=None, page_size=100)
        self.conn.commit()
        
        logger.info(f"Imported {len(ai_perf_data)} AI performance records")
        cursor.close()
    
    def verify_import(self):
        """Verify the imported data"""
        logger.info("Verifying imported data...")
        
        cursor = self.conn.cursor()
        
        tables = ['manufacturers', 'architectures', 'categories', 'gpus', 
                 'performance_metrics', 'memory_specs', 'process_technology', 'ai_performance']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"{table}: {count} records")
        
        cursor.execute("SELECT COUNT(*) FROM gpu_complete_info")
        view_count = cursor.fetchone()[0]
        logger.info(f"gpu_complete_info view: {view_count} records")
        
        cursor.close()
    
    def run_import(self):
        """Run the complete import process"""
        try:
            self.connect_to_database()
            self.load_csv_data()
            
            logger.info("Starting database import process...")
            
            self.import_gpus()
            self.import_performance_metrics()
            self.import_memory_specs()
            self.import_process_technology()
            self.import_ai_performance()
            
            self.verify_import()
            
            logger.info("Import process completed successfully!")
            
        except Exception as e:
            logger.error(f"Import process failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Main function"""
    csv_file_path = "data/final/Ai-Benchmark-Final-enhanced-fixed.csv"
    
    if not Path(csv_file_path).exists():
        logger.error(f"CSV file not found: {csv_file_path}")
        sys.exit(1)
    
    importer = AIBenchmarkImporter(csv_file_path)
    importer.run_import()

if __name__ == "__main__":
    main()

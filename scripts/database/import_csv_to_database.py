#!/usr/bin/env python3
"""
AI Benchmark CSV to PostgreSQL Database Import Script
Purpose: Import data from Ai-Benchmark-Final-enhanced.csv to normalized PostgreSQL database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import logging
from datetime import datetime
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_import.log'),
        logging.StreamHandler()
    ]
)

class AIBenchmarkImporter:
    def __init__(self, db_config):
        """Initialize the importer with database configuration"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logging.info("Connected to PostgreSQL database successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect_database(self):
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("Disconnected from database")
    
    def load_csv_data(self, csv_path):
        """Load and prepare CSV data"""
        try:
            logging.info(f"Loading CSV data from: {csv_path}")
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Handle missing values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Convert boolean-like columns
            boolean_columns = ['IsLegacyLowPerf']
            for col in boolean_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
            
            return df
        except Exception as e:
            logging.error(f"Failed to load CSV data: {e}")
            return None
    
    def get_or_create_manufacturer(self, manufacturer_name):
        """Get manufacturer ID or create if not exists"""
        if pd.isna(manufacturer_name) or manufacturer_name.strip() == '':
            manufacturer_name = 'Unknown'
            
        self.cursor.execute(
            "SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = %s",
            (manufacturer_name,)
        )
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        else:
            # Create new manufacturer
            self.cursor.execute(
                "INSERT INTO manufacturers (manufacturer_name) VALUES (%s) RETURNING manufacturer_id",
                (manufacturer_name,)
            )
            return self.cursor.fetchone()[0]
    
    def get_or_create_architecture(self, architecture_name, manufacturer_id):
        """Get architecture ID or create if not exists"""
        if pd.isna(architecture_name) or architecture_name.strip() == '':
            architecture_name = 'Unknown'
            
        self.cursor.execute(
            "SELECT architecture_id FROM architectures WHERE architecture_name = %s",
            (architecture_name,)
        )
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        else:
            # Create new architecture
            self.cursor.execute(
                """INSERT INTO architectures (architecture_name, manufacturer_id) 
                   VALUES (%s, %s) RETURNING architecture_id""",
                (architecture_name, manufacturer_id)
            )
            return self.cursor.fetchone()[0]
    
    def get_lookup_id(self, table_name, column_name, value, id_column):
        """Generic function to get lookup table IDs"""
        if pd.isna(value) or value == '':
            value = 'Unknown'
            
        self.cursor.execute(
            f"SELECT {id_column} FROM {table_name} WHERE {column_name} = %s",
            (value,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def insert_gpu_device(self, row):
        """Insert GPU device record"""
        try:
            # Get foreign keys
            manufacturer_id = self.get_or_create_manufacturer(row.get('Manufacturer'))
            architecture_id = self.get_or_create_architecture(row.get('Architecture'), manufacturer_id)
            category_id = self.get_lookup_id('device_categories', 'category_name', row.get('Category'), 'category_id')
            performance_tier_id = self.get_lookup_id('performance_tiers', 'tier_name', row.get('PerformanceTier'), 'tier_id')
            efficiency_class_id = self.get_lookup_id('efficiency_classes', 'efficiency_class', row.get('EfficiencyClass'), 'efficiency_id')
            memory_tier_id = self.get_lookup_id('memory_tiers', 'tier_name', row.get('MemoryTier'), 'memory_tier_id')
            ai_category_id = self.get_lookup_id('ai_performance_categories', 'category_name', row.get('AI_Performance_Category'), 'ai_category_id')
            
            # Prepare device data
            device_data = {
                'gpu_name': row.get('gpuName'),
                'manufacturer_id': manufacturer_id,
                'architecture_id': architecture_id,
                'category_id': category_id,
                'performance_tier_id': performance_tier_id,
                'efficiency_class_id': efficiency_class_id,
                'memory_tier_id': memory_tier_id,
                'ai_category_id': ai_category_id,
                'generation_category': row.get('GenerationCategory'),
                'test_date': self._safe_int(row.get('testDate')),
                'price_usd': self._safe_float(row.get('price')),
                'tdp_watts': self._safe_float(row.get('TDP'), default=100.0),  # TDP is required
                'memory_size_gb': self._safe_float(row.get('Memory_GB')),
                'memory_size_bytes': self._safe_int(row.get('Memory size per board (Byte)')),
                'memory_bandwidth_bytes_per_sec': self._safe_int(row.get('Memory bandwidth (byte/s)')),
                'process_size_nm': self._safe_float(row.get('Process size (nm)'))
            }
            
            # Insert device
            insert_query = """
                INSERT INTO gpu_devices (
                    gpu_name, manufacturer_id, architecture_id, category_id, 
                    performance_tier_id, efficiency_class_id, memory_tier_id, ai_category_id,
                    generation_category, test_date, price_usd, tdp_watts,
                    memory_size_gb, memory_size_bytes, memory_bandwidth_bytes_per_sec, process_size_nm
                ) VALUES (
                    %(gpu_name)s, %(manufacturer_id)s, %(architecture_id)s, %(category_id)s,
                    %(performance_tier_id)s, %(efficiency_class_id)s, %(memory_tier_id)s, %(ai_category_id)s,
                    %(generation_category)s, %(test_date)s, %(price_usd)s, %(tdp_watts)s,
                    %(memory_size_gb)s, %(memory_size_bytes)s, %(memory_bandwidth_bytes_per_sec)s, %(process_size_nm)s
                ) RETURNING device_id
            """
            
            self.cursor.execute(insert_query, device_data)
            device_id = self.cursor.fetchone()[0]
            
            return device_id
            
        except Exception as e:
            logging.error(f"Failed to insert GPU device {row.get('gpuName')}: {e}")
            return None
    
    def insert_graphics_performance(self, device_id, row):
        """Insert graphics performance data"""
        try:
            perf_data = {
                'device_id': device_id,
                'g3d_mark': self._safe_int(row.get('G3Dmark')),
                'g2d_mark': self._safe_int(row.get('G2Dmark')),
                'power_performance': self._safe_float(row.get('powerPerformance')),
                'gpu_value': self._safe_float(row.get('gpuValue'))
            }
            
            insert_query = """
                INSERT INTO graphics_performance (device_id, g3d_mark, g2d_mark, power_performance, gpu_value)
                VALUES (%(device_id)s, %(g3d_mark)s, %(g2d_mark)s, %(power_performance)s, %(gpu_value)s)
            """
            
            self.cursor.execute(insert_query, perf_data)
            
        except Exception as e:
            logging.error(f"Failed to insert graphics performance for device {device_id}: {e}")
    
    def insert_ai_compute_performance(self, device_id, row):
        """Insert AI compute performance data"""
        try:
            ai_data = {
                'device_id': device_id,
                'fp32_flops': self._safe_int(row.get('FP32_Final'), default=1000000000),  # Required field
                'flops_per_watt': self._safe_float(row.get('FLOPS_per_Watt'), default=0.001),  # Required field
                'tops_per_watt': self._safe_float(row.get('TOPs_per_Watt'), default=0.001),  # Required field
                'gflops_per_watt': self._safe_float(row.get('GFLOPS_per_Watt'), default=1.0),  # Required field
                'fp16_flops': self._safe_int(row.get('FP16_Performance_Predicted')),
                'int8_ops': self._safe_int(row.get('INT8_Performance_Estimated')),
                'relative_latency_index': self._safe_float(row.get('Relative_Latency_Index')),
                'compute_usage_percent': self._safe_float(row.get('Compute_Usage_Percent')),
                'performance_per_dollar_per_watt': self._safe_float(row.get('Performance_per_Dollar_per_Watt'))
            }
            
            insert_query = """
                INSERT INTO ai_compute_performance (
                    device_id, fp32_flops, flops_per_watt, tops_per_watt, gflops_per_watt,
                    fp16_flops, int8_ops, relative_latency_index, compute_usage_percent,
                    performance_per_dollar_per_watt
                ) VALUES (
                    %(device_id)s, %(fp32_flops)s, %(flops_per_watt)s, %(tops_per_watt)s, %(gflops_per_watt)s,
                    %(fp16_flops)s, %(int8_ops)s, %(relative_latency_index)s, %(compute_usage_percent)s,
                    %(performance_per_dollar_per_watt)s
                )
            """
            
            self.cursor.execute(insert_query, ai_data)
            
        except Exception as e:
            logging.error(f"Failed to insert AI compute performance for device {device_id}: {e}")
    
    def insert_ai_model_throughput(self, device_id, row):
        """Insert AI model throughput data"""
        try:
            throughput_data = {
                'device_id': device_id,
                'resnet50_imagenet_fps': self._safe_float(row.get('Throughput_ResNet50_ImageNet_fps')),
                'bert_base_fps': self._safe_float(row.get('Throughput_BERT_Base_fps')),
                'gpt2_small_fps': self._safe_float(row.get('Throughput_GPT2_Small_fps')),
                'mobilenetv2_fps': self._safe_float(row.get('Throughput_MobileNetV2_fps')),
                'efficientnet_b0_fps': self._safe_float(row.get('Throughput_EfficientNet_B0_fps')),
                'avg_throughput_fps': self._safe_float(row.get('Avg_Throughput_fps'))
            }
            
            insert_query = """
                INSERT INTO ai_model_throughput (
                    device_id, resnet50_imagenet_fps, bert_base_fps, gpt2_small_fps,
                    mobilenetv2_fps, efficientnet_b0_fps, avg_throughput_fps
                ) VALUES (
                    %(device_id)s, %(resnet50_imagenet_fps)s, %(bert_base_fps)s, %(gpt2_small_fps)s,
                    %(mobilenetv2_fps)s, %(efficientnet_b0_fps)s, %(avg_throughput_fps)s
                )
            """
            
            self.cursor.execute(insert_query, throughput_data)
            
        except Exception as e:
            logging.error(f"Failed to insert AI model throughput for device {device_id}: {e}")
    
    def insert_api_support(self, device_id, row):
        """Insert API support data"""
        try:
            api_data = {
                'device_id': device_id,
                'cuda_support': self._safe_bool(row.get('CUDA')),
                'opencl_support': self._safe_bool(row.get('OpenCL')),
                'vulkan_support': self._safe_bool(row.get('Vulkan')),
                'metal_support': self._safe_bool(row.get('Metal'))
            }
            
            insert_query = """
                INSERT INTO api_support (device_id, cuda_support, opencl_support, vulkan_support, metal_support)
                VALUES (%(device_id)s, %(cuda_support)s, %(opencl_support)s, %(vulkan_support)s, %(metal_support)s)
            """
            
            self.cursor.execute(insert_query, api_data)
            
        except Exception as e:
            logging.error(f"Failed to insert API support for device {device_id}: {e}")
    
    def insert_device_classifications(self, device_id, row):
        """Insert device classifications"""
        try:
            class_data = {
                'device_id': device_id,
                'is_legacy_low_perf': self._safe_bool(row.get('IsLegacyLowPerf')),
                'price_performance_index': self._safe_float(row.get('PricePerformanceIndex')),
                'ai_efficiency_tier': row.get('AI_Efficiency_Tier'),
                'ai_performance_category': row.get('AI_Performance_Category')
            }
            
            insert_query = """
                INSERT INTO device_classifications (
                    device_id, is_legacy_low_perf, price_performance_index,
                    ai_efficiency_tier, ai_performance_category
                ) VALUES (
                    %(device_id)s, %(is_legacy_low_perf)s, %(price_performance_index)s,
                    %(ai_efficiency_tier)s, %(ai_performance_category)s
                )
            """
            
            self.cursor.execute(insert_query, class_data)
            
        except Exception as e:
            logging.error(f"Failed to insert device classifications for device {device_id}: {e}")
    
    def _safe_float(self, value, default=None):
        """Safely convert to float"""
        if pd.isna(value) or value == '' or value == 'NaN':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=None):
        """Safely convert to int"""
        if pd.isna(value) or value == '' or value == 'NaN':
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def _safe_bool(self, value, default=False):
        """Safely convert to bool"""
        if pd.isna(value) or value == '' or value == 'NaN':
            return default
        if isinstance(value, (bool, int)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y')
        return default
    
    def import_data(self, csv_path):
        """Main import function"""
        try:
            # Load CSV data
            df = self.load_csv_data(csv_path)
            if df is None:
                return False
            
            # Connect to database
            if not self.connect_database():
                return False
            
            logging.info("Starting data import...")
            success_count = 0
            error_count = 0
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Insert device
                    device_id = self.insert_gpu_device(row)
                    if device_id:
                        # Insert related data
                        self.insert_graphics_performance(device_id, row)
                        self.insert_ai_compute_performance(device_id, row)
                        self.insert_ai_model_throughput(device_id, row)
                        self.insert_api_support(device_id, row)
                        self.insert_device_classifications(device_id, row)
                        
                        success_count += 1
                        if success_count % 100 == 0:
                            self.conn.commit()
                            logging.info(f"Processed {success_count} devices...")
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing row {index}: {e}")
                    self.conn.rollback()
            
            # Final commit
            self.conn.commit()
            
            logging.info(f"Import completed: {success_count} successful, {error_count} errors")
            return True
            
        except Exception as e:
            logging.error(f"Import failed: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self.disconnect_database()

def main():
    """Main function"""
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'AI_BENCHMARK',
        'user': 'postgres',
        'password': 'admin',
        'port': 5432
    }
    
    # CSV file path
    csv_path = 'data/final/Ai-Benchmark-Final-enhanced.csv'
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Create importer and run
    importer = AIBenchmarkImporter(db_config)
    
    if importer.import_data(csv_path):
        logging.info("Data import completed successfully!")
    else:
        logging.error("Data import failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
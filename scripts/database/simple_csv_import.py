#!/usr/bin/env python3
"""
Simple CSV Import Script for AI Benchmark Database
Directly imports data from Ai-Benchmark-Final-enhanced.csv into normalized tables
"""

import pandas as pd
import psycopg2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleCSVImporter:
    def __init__(self):
        """Initialize with database connection"""
        self.db_config = {
            'host': 'localhost',
            'database': 'AI_BENCHMARK',
            'user': 'postgres',
            'password': 'admin',
            'port': 5432
        }
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logging.info("Connected to AI_BENCHMARK database")
            return True
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return False
    
    def safe_float(self, value, default=None):
        """Safely convert to float"""
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int(self, value, default=None):
        """Safely convert to int"""
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def safe_bool(self, value, default=False):
        """Safely convert to bool"""
        if pd.isna(value) or value == '':
            return default
        if isinstance(value, (bool, int)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y')
        return default
    
    def get_or_create_manufacturer(self, name):
        """Get or create manufacturer"""
        if pd.isna(name) or str(name).strip() == '':
            name = 'Unknown'
        
        self.cursor.execute("SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = %s", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        # Create new manufacturer
        self.cursor.execute("INSERT INTO manufacturers (manufacturer_name) VALUES (%s) RETURNING manufacturer_id", (name,))
        return self.cursor.fetchone()[0]
    
    def get_or_create_architecture(self, name, manufacturer_id):
        """Get or create architecture"""
        if pd.isna(name) or str(name).strip() == '':
            name = 'Unknown'
        
        self.cursor.execute("SELECT architecture_id FROM architectures WHERE architecture_name = %s", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        # Create new architecture
        self.cursor.execute(
            "INSERT INTO architectures (architecture_name, manufacturer_id) VALUES (%s, %s) RETURNING architecture_id",
            (name, manufacturer_id)
        )
        return self.cursor.fetchone()[0]
    
    def get_or_create_category(self, name):
        """Get or create device category"""
        if pd.isna(name) or str(name).strip() == '':
            name = 'Unknown'
        
        self.cursor.execute("SELECT category_id FROM device_categories WHERE category_name = %s", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        # Create new category
        self.cursor.execute(
            "INSERT INTO device_categories (category_name) VALUES (%s) RETURNING category_id",
            (name,)
        )
        return self.cursor.fetchone()[0]
    
    def import_csv_data(self, csv_path):
        """Main import function"""
        try:
            # Load CSV
            logging.info(f"Loading CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Replace inf with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            success_count = 0
            error_count = 0
            
            for index, row in df.iterrows():
                try:
                    # Get foreign keys
                    manufacturer_id = self.get_or_create_manufacturer(row.get('Manufacturer'))
                    architecture_id = self.get_or_create_architecture(row.get('Architecture'), manufacturer_id)
                    category_id = self.get_or_create_category(row.get('Category'))
                    
                    # Insert GPU device
                    device_query = """
                        INSERT INTO gpu_devices (
                            gpu_name, manufacturer_id, architecture_id, category_id,
                            generation_category, test_date, price_usd, tdp_watts,
                            memory_size_gb, memory_size_bytes, memory_bandwidth_bytes_per_sec, process_size_nm
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING device_id
                    """
                    
                    device_data = (
                        row.get('gpuName'),
                        manufacturer_id,
                        architecture_id,
                        category_id,
                        row.get('GenerationCategory'),
                        self.safe_int(row.get('testDate')),
                        self.safe_float(row.get('price')),
                        self.safe_float(row.get('TDP'), default=100.0),  # TDP required
                        self.safe_float(row.get('Memory_GB')),
                        self.safe_int(row.get('Memory size per board (Byte)')),
                        self.safe_int(row.get('Memory bandwidth (byte/s)')),
                        self.safe_float(row.get('Process size (nm)'))
                    )
                    
                    self.cursor.execute(device_query, device_data)
                    device_id = self.cursor.fetchone()[0]
                    
                    # Insert graphics performance
                    graphics_query = """
                        INSERT INTO graphics_performance (device_id, g3d_mark, g2d_mark, power_performance, gpu_value)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    graphics_data = (
                        device_id,
                        self.safe_int(row.get('G3Dmark')),
                        self.safe_int(row.get('G2Dmark')),
                        self.safe_float(row.get('powerPerformance')),
                        self.safe_float(row.get('gpuValue'))
                    )
                    self.cursor.execute(graphics_query, graphics_data)
                    
                    # Insert AI compute performance
                    ai_query = """
                        INSERT INTO ai_compute_performance (
                            device_id, fp32_flops, flops_per_watt, tops_per_watt, gflops_per_watt,
                            fp16_flops, int8_ops, relative_latency_index, compute_usage_percent,
                            performance_per_dollar_per_watt
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    ai_data = (
                        device_id,
                        self.safe_int(row.get('FP32_Final'), default=1000000000),  # Required
                        self.safe_float(row.get('FLOPS_per_Watt'), default=0.001),  # Required
                        self.safe_float(row.get('TOPs_per_Watt'), default=0.001),  # Required
                        self.safe_float(row.get('GFLOPS_per_Watt'), default=1.0),  # Required
                        self.safe_int(row.get('FP16_Performance_Predicted')),
                        self.safe_int(row.get('INT8_Performance_Estimated')),
                        self.safe_float(row.get('Relative_Latency_Index')),
                        self.safe_float(row.get('Compute_Usage_Percent')),
                        self.safe_float(row.get('Performance_per_Dollar_per_Watt'))
                    )
                    self.cursor.execute(ai_query, ai_data)
                    
                    # Insert AI model throughput
                    throughput_query = """
                        INSERT INTO ai_model_throughput (
                            device_id, resnet50_imagenet_fps, bert_base_fps, gpt2_small_fps,
                            mobilenetv2_fps, efficientnet_b0_fps, avg_throughput_fps
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    throughput_data = (
                        device_id,
                        self.safe_float(row.get('Throughput_ResNet50_ImageNet_fps')),
                        self.safe_float(row.get('Throughput_BERT_Base_fps')),
                        self.safe_float(row.get('Throughput_GPT2_Small_fps')),
                        self.safe_float(row.get('Throughput_MobileNetV2_fps')),
                        self.safe_float(row.get('Throughput_EfficientNet_B0_fps')),
                        self.safe_float(row.get('Avg_Throughput_fps'))
                    )
                    self.cursor.execute(throughput_query, throughput_data)
                    
                    # Insert API support
                    api_query = """
                        INSERT INTO api_support (device_id, cuda_support, opencl_support, vulkan_support, metal_support)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    api_data = (
                        device_id,
                        self.safe_bool(row.get('CUDA')),
                        self.safe_bool(row.get('OpenCL')),
                        self.safe_bool(row.get('Vulkan')),
                        self.safe_bool(row.get('Metal'))
                    )
                    self.cursor.execute(api_query, api_data)
                    
                    # Insert device classifications
                    class_query = """
                        INSERT INTO device_classifications (
                            device_id, is_legacy_low_perf, price_performance_index,
                            ai_efficiency_tier, ai_performance_category
                        ) VALUES (%s, %s, %s, %s, %s)
                    """
                    class_data = (
                        device_id,
                        self.safe_bool(row.get('IsLegacyLowPerf')),
                        self.safe_float(row.get('PricePerformanceIndex')),
                        row.get('AI_Efficiency_Tier'),
                        row.get('AI_Performance_Category')
                    )
                    self.cursor.execute(class_query, class_data)
                    
                    success_count += 1
                    
                    # Commit every 100 records
                    if success_count % 100 == 0:
                        self.conn.commit()
                        logging.info(f"Processed {success_count} devices...")
                        
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing row {index} ({row.get('gpuName', 'Unknown')}): {e}")
                    self.conn.rollback()
            
            # Final commit
            self.conn.commit()
            
            logging.info(f"Import completed! Success: {success_count}, Errors: {error_count}")
            
            # Verify import
            self.cursor.execute("SELECT COUNT(*) FROM gpu_devices")
            device_count = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM ai_compute_performance")
            ai_count = self.cursor.fetchone()[0]
            
            logging.info(f"Verification: {device_count} devices, {ai_count} AI performance records")
            
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Import failed: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("Database connection closed")

def main():
    """Main function"""
    csv_path = 'data/final/Ai-Benchmark-Final-enhanced.csv'
    
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    
    importer = SimpleCSVImporter()
    
    try:
        if importer.connect():
            if importer.import_csv_data(csv_path):
                logging.info("✅ CSV import completed successfully!")
                
                # Show summary
                importer.cursor.execute("""
                    SELECT m.manufacturer_name, COUNT(*) as device_count
                    FROM gpu_devices g 
                    JOIN manufacturers m ON g.manufacturer_id = m.manufacturer_id
                    GROUP BY m.manufacturer_name 
                    ORDER BY device_count DESC
                """)
                
                logging.info("📊 Import Summary by Manufacturer:")
                for row in importer.cursor.fetchall():
                    logging.info(f"   {row[0]}: {row[1]} devices")
                    
            else:
                logging.error("❌ CSV import failed!")
        else:
            logging.error("❌ Database connection failed!")
            
    finally:
        importer.close()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Database Verification Script for AI Benchmark Database
Purpose: Verify table structure, data integrity, normalization, and relationships
"""

import psycopg2
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseVerifier:
    def __init__(self):
        """Initialize database connection"""
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
            logging.info("✅ Connected to AI_BENCHMARK database")
            return True
        except Exception as e:
            logging.error(f"❌ Database connection failed: {e}")
            return False
    
    def verify_table_structure(self):
        """Verify all tables exist with correct structure"""
        print("\n" + "="*60)
        print("🏗️  TABLE STRUCTURE VERIFICATION")
        print("="*60)
        
        # Expected tables
        expected_tables = [
            'manufacturers', 'architectures', 'device_categories',
            'gpu_devices', 'graphics_performance', 'ai_compute_performance',
            'ai_model_throughput', 'api_support', 'device_classifications'
        ]
        
        # Check if all tables exist
        self.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        existing_tables = [row[0] for row in self.cursor.fetchall()]
        
        print(f"📋 Expected tables: {len(expected_tables)}")
        print(f"📋 Found tables: {len(existing_tables)}")
        
        # Check missing tables
        missing_tables = set(expected_tables) - set(existing_tables)
        extra_tables = set(existing_tables) - set(expected_tables)
        
        if missing_tables:
            print(f"❌ Missing tables: {list(missing_tables)}")
        else:
            print("✅ All expected tables found")
            
        if extra_tables:
            print(f"ℹ️  Extra tables: {list(extra_tables)}")
        
        # Show table details
        print(f"\n📊 Table Details:")
        for table in sorted(existing_tables):
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = self.cursor.fetchone()[0]
            
            # Get column count
            self.cursor.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """)
            col_count = self.cursor.fetchone()[0]
            
            print(f"   {table:25} | {count:6,} rows | {col_count:2} columns")
        
        return len(missing_tables) == 0
    
    def verify_data_integrity(self):
        """Verify data integrity and relationships"""
        print("\n" + "="*60)
        print("🔍 DATA INTEGRITY VERIFICATION")
        print("="*60)
        
        integrity_checks = []
        
        # 1. Check for orphaned records
        print("🔗 Checking for orphaned records...")
        
        # Graphics performance orphans
        self.cursor.execute("""
            SELECT COUNT(*) FROM graphics_performance gp
            LEFT JOIN gpu_devices gd ON gp.device_id = gd.device_id
            WHERE gd.device_id IS NULL
        """)
        orphaned_graphics = self.cursor.fetchone()[0]
        print(f"   Graphics performance orphans: {orphaned_graphics}")
        
        # AI compute performance orphans
        self.cursor.execute("""
            SELECT COUNT(*) FROM ai_compute_performance acp
            LEFT JOIN gpu_devices gd ON acp.device_id = gd.device_id
            WHERE gd.device_id IS NULL
        """)
        orphaned_ai = self.cursor.fetchone()[0]
        print(f"   AI compute performance orphans: {orphaned_ai}")
        
        # 2. Check for NULL values in required fields
        print("\n📋 Checking required field constraints...")
        
        # GPU devices must have manufacturer
        self.cursor.execute("SELECT COUNT(*) FROM gpu_devices WHERE manufacturer_id IS NULL")
        null_manufacturers = self.cursor.fetchone()[0]
        print(f"   GPU devices without manufacturer: {null_manufacturers}")
        
        # 3. Check for duplicates
        print("\n🔍 Checking for duplicates...")
        
        self.cursor.execute("""
            SELECT gpu_name, COUNT(*) as cnt 
            FROM gpu_devices 
            GROUP BY gpu_name 
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 5
        """)
        duplicates = self.cursor.fetchall()
        print(f"   Duplicate GPU names: {len(duplicates)}")
        if duplicates:
            for name, count in duplicates:
                print(f"      '{name}': {count} times")
        
        return True
    
    def verify_normalization(self):
        """Verify normalization is working correctly"""
        print("\n" + "="*60)
        print("📐 NORMALIZATION VERIFICATION")
        print("="*60)
        
        # 1. Check manufacturer normalization
        print("🏭 Manufacturer normalization...")
        self.cursor.execute("SELECT COUNT(DISTINCT manufacturer_name) FROM manufacturers")
        unique_manufacturers = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT m.manufacturer_name, COUNT(g.device_id) as device_count
            FROM manufacturers m
            LEFT JOIN gpu_devices g ON m.manufacturer_id = g.manufacturer_id
            GROUP BY m.manufacturer_id, m.manufacturer_name
            ORDER BY device_count DESC
        """)
        manufacturer_stats = self.cursor.fetchall()
        
        print(f"   Unique manufacturers: {unique_manufacturers}")
        print("   Top manufacturers by device count:")
        for name, count in manufacturer_stats[:5]:
            print(f"      {name}: {count:,} devices")
        
        # 2. Check architecture normalization
        print("\n🏗️  Architecture normalization...")
        self.cursor.execute("SELECT COUNT(DISTINCT architecture_name) FROM architectures")
        unique_architectures = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT a.architecture_name, m.manufacturer_name, COUNT(g.device_id) as device_count
            FROM architectures a
            LEFT JOIN manufacturers m ON a.manufacturer_id = m.manufacturer_id
            LEFT JOIN gpu_devices g ON a.architecture_id = g.architecture_id
            GROUP BY a.architecture_id, a.architecture_name, m.manufacturer_name
            ORDER BY device_count DESC
        """)
        architecture_stats = self.cursor.fetchall()
        
        print(f"   Unique architectures: {unique_architectures}")
        print("   Top architectures by device count:")
        for arch, manuf, count in architecture_stats[:5]:
            print(f"      {arch} ({manuf}): {count:,} devices")
        
        return True
    
    def verify_data_quality(self):
        """Verify data quality and ranges"""
        print("\n" + "="*60)
        print("📊 DATA QUALITY VERIFICATION")
        print("="*60)
        
        # 1. Check AI performance metrics
        print("🧠 AI Performance metrics...")
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                MIN(tops_per_watt) as min_tops_watt,
                MAX(tops_per_watt) as max_tops_watt,
                AVG(tops_per_watt) as avg_tops_watt
            FROM ai_compute_performance
            WHERE tops_per_watt IS NOT NULL AND tops_per_watt > 0
        """)
        
        stats = self.cursor.fetchone()
        total, min_tops, max_tops, avg_tops = stats
        
        print(f"   Records with AI metrics: {total:,}")
        print(f"   TOPs/Watt range: {min_tops:.6f} - {max_tops:.6f} (avg: {avg_tops:.6f})")
        
        # 2. Check power consumption
        print("\n⚡ Power consumption...")
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MIN(tdp_watts) as min_tdp,
                MAX(tdp_watts) as max_tdp,
                AVG(tdp_watts) as avg_tdp
            FROM gpu_devices
            WHERE tdp_watts IS NOT NULL AND tdp_watts > 0
        """)
        
        total, min_tdp, max_tdp, avg_tdp = self.cursor.fetchone()
        print(f"   Devices with TDP data: {total:,}")
        print(f"   TDP range: {min_tdp:.1f}W - {max_tdp:.1f}W (avg: {avg_tdp:.1f}W)")
        
        return True
    
    def run_full_verification(self):
        """Run complete database verification"""
        if not self.connect():
            return False
        
        print("🚀 Starting AI Benchmark Database Verification")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            structure_ok = self.verify_table_structure()
            integrity_ok = self.verify_data_integrity()
            normalization_ok = self.verify_normalization()
            quality_ok = self.verify_data_quality()
            
            print("\n" + "="*60)
            print("🏁 VERIFICATION RESULTS")
            print("="*60)
            
            checks = [
                ("Table Structure", structure_ok),
                ("Data Integrity", integrity_ok),
                ("Normalization", normalization_ok),
                ("Data Quality", quality_ok)
            ]
            
            for check_name, passed in checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {check_name:20} | {status}")
            
            all_passed = all([structure_ok, integrity_ok, normalization_ok, quality_ok])
            
            print("\n" + "="*60)
            if all_passed:
                print("🎉 ALL VERIFICATIONS PASSED!")
            else:
                print("⚠️  Some verifications need attention.")
            
            return all_passed
            
        except Exception as e:
            logging.error(f"Verification failed: {e}")
            return False
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    verifier = DatabaseVerifier()
    verifier.run_full_verification() 
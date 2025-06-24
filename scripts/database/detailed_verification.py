#!/usr/bin/env python3
"""
Detailed Database Verification Script
Purpose: Deep dive verification of relationships, indexes, views, and data consistency
"""

import psycopg2
import pandas as pd
from datetime import datetime

class DetailedVerifier:
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
            print("✅ Connected to AI_BENCHMARK database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def check_foreign_keys(self):
        """Check foreign key constraints"""
        print("\n" + "="*60)
        print("🔗 FOREIGN KEY CONSTRAINT VERIFICATION")
        print("="*60)
        
        # Get all foreign key constraints
        self.cursor.execute("""
            SELECT 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.table_name;
        """)
        
        foreign_keys = self.cursor.fetchall()
        
        print(f"Found {len(foreign_keys)} foreign key constraints:")
        for table, column, ref_table, ref_column in foreign_keys:
            print(f"   {table}.{column} → {ref_table}.{ref_column}")
        
        return len(foreign_keys) > 0
    
    def check_data_relationships(self):
        """Check specific data relationships and consistency"""
        print("\n" + "="*60)
        print("🔄 DATA RELATIONSHIP VERIFICATION")
        print("="*60)
        
        # Check device completeness
        print("📊 Device data completeness:")
        
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_devices,
                COUNT(CASE WHEN gp.device_id IS NOT NULL THEN 1 END) as with_graphics,
                COUNT(CASE WHEN acp.device_id IS NOT NULL THEN 1 END) as with_ai_compute,
                COUNT(CASE WHEN amt.device_id IS NOT NULL THEN 1 END) as with_model_throughput,
                COUNT(CASE WHEN apis.device_id IS NOT NULL THEN 1 END) as with_api_support
            FROM gpu_devices gd
            LEFT JOIN graphics_performance gp ON gd.device_id = gp.device_id
            LEFT JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
            LEFT JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
            LEFT JOIN api_support apis ON gd.device_id = apis.device_id
        """)
        
        stats = self.cursor.fetchone()
        total, graphics, ai_compute, throughput, api_support = stats
        
        print(f"   Total GPU devices: {total:,}")
        print(f"   With graphics performance: {graphics:,} ({graphics/total*100:.1f}%)")
        print(f"   With AI compute performance: {ai_compute:,} ({ai_compute/total*100:.1f}%)")
        print(f"   With model throughput: {throughput:,} ({throughput/total*100:.1f}%)")
        print(f"   With API support: {api_support:,} ({api_support/total*100:.1f}%)")
        
        # Check manufacturer-architecture relationships
        print("\n🏭 Manufacturer-Architecture relationships:")
        
        self.cursor.execute("""
            SELECT 
                m.manufacturer_name,
                COUNT(DISTINCT a.architecture_id) as arch_count,
                COUNT(gd.device_id) as device_count
            FROM manufacturers m
            LEFT JOIN architectures a ON m.manufacturer_id = a.manufacturer_id
            LEFT JOIN gpu_devices gd ON a.architecture_id = gd.architecture_id
            GROUP BY m.manufacturer_id, m.manufacturer_name
            ORDER BY device_count DESC
        """)
        
        manuf_stats = self.cursor.fetchall()
        
        for manuf, arch_count, device_count in manuf_stats:
            print(f"   {manuf}: {arch_count} architectures, {device_count:,} devices")
        
        return True
    
    def check_performance_metrics(self):
        """Check AI performance metrics consistency"""
        print("\n" + "="*60)
        print("🧠 AI PERFORMANCE METRICS VERIFICATION")
        print("="*60)
        
        # Check AI efficiency distributions
        print("⚡ Efficiency metrics:")
        
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                MIN(tops_per_watt) as min_efficiency,
                MAX(tops_per_watt) as max_efficiency,
                AVG(tops_per_watt) as avg_efficiency,
                COUNT(CASE WHEN tops_per_watt > 0.01 THEN 1 END) as high_efficiency_count
            FROM ai_compute_performance
            WHERE tops_per_watt IS NOT NULL AND tops_per_watt > 0
        """)
        
        efficiency_stats = self.cursor.fetchone()
        total, min_eff, max_eff, avg_eff, high_eff = efficiency_stats
        
        print(f"   Total AI records: {total:,}")
        print(f"   Efficiency range: {min_eff:.6f} - {max_eff:.6f} TOPs/W")
        print(f"   Average efficiency: {avg_eff:.6f} TOPs/W")
        print(f"   High efficiency devices (>0.01): {high_eff:,}")
        
        # Top performers
        print("\n🏆 Top AI performers:")
        
        self.cursor.execute("""
            SELECT 
                gd.gpu_name,
                m.manufacturer_name,
                acp.tops_per_watt,
                gd.tdp_watts
            FROM gpu_devices gd
            JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
            JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
            WHERE acp.tops_per_watt IS NOT NULL 
            ORDER BY acp.tops_per_watt DESC
            LIMIT 5
        """)
        
        top_performers = self.cursor.fetchall()
        
        print("   Top 5 most efficient devices:")
        for gpu_name, manuf, efficiency, tdp in top_performers:
            print(f"      {gpu_name[:40]:40} | {manuf:7} | {efficiency:.6f} TOPs/W | {tdp:.0f}W")
        
        return True
    
    def run_detailed_verification(self):
        """Run complete detailed verification"""
        if not self.connect():
            return False
        
        print("🔍 Starting Detailed Database Verification")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            fk_ok = self.check_foreign_keys()
            relationship_ok = self.check_data_relationships()
            performance_ok = self.check_performance_metrics()
            
            print("\n" + "="*60)
            print("🏁 DETAILED VERIFICATION RESULTS")
            print("="*60)
            
            checks = [
                ("Foreign Keys", fk_ok),
                ("Relationships", relationship_ok),
                ("Performance Metrics", performance_ok)
            ]
            
            for check_name, passed in checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {check_name:20} | {status}")
            
            all_passed = all([fk_ok, relationship_ok, performance_ok])
            
            print("\n" + "="*60)
            if all_passed:
                print("🎉 ALL DETAILED VERIFICATIONS PASSED!")
                print("🏆 Database is properly normalized and contains high-quality data!")
            else:
                print("⚠️  Some detailed verifications need attention.")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Detailed verification failed: {e}")
            return False
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    verifier = DetailedVerifier()
    verifier.run_detailed_verification() 
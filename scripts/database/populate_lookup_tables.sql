-- =====================================================================
-- POPULATE LOOKUP TABLES FOR AI BENCHMARK DATABASE
-- Purpose: Insert reference data for normalization
-- =====================================================================

\c AI_BENCHMARK;

-- =====================================================================
-- 1. POPULATE MANUFACTURERS
-- =====================================================================

INSERT INTO manufacturers (manufacturer_name, country, founded_year) VALUES
('NVIDIA', 'United States', 1993),
('AMD', 'United States', 1969),
('Intel', 'United States', 1968),
('Apple', 'United States', 1976),
('Qualcomm', 'United States', 1985),
('Broadcom', 'United States', 1991),
('Samsung', 'South Korea', 1938),
('ARM', 'United Kingdom', 1990),
('Imagination Technologies', 'United Kingdom', 1985),
('VIA', 'Taiwan', 1987),
('S3 Graphics', 'United States', 1989),
('Matrox', 'Canada', 1976),
('3dfx', 'United States', 1994),
('ATI', 'Canada', 1985),
('PowerVR', 'United Kingdom', 1985),
('Huawei', 'China', 1987),
('Google', 'United States', 1998),
('Unknown', NULL, NULL)
ON CONFLICT (manufacturer_name) DO NOTHING;

-- =====================================================================
-- 2. POPULATE DEVICE CATEGORIES
-- =====================================================================

INSERT INTO device_categories (category_name, category_description, target_market) VALUES
('Consumer', 'Gaming and consumer graphics cards', 'Gaming, Content Creation'),
('Professional', 'Workstation and datacenter GPUs', 'Professional Workloads, AI/ML'),
('Mobile', 'Laptop and mobile GPUs', 'Portable Computing'),
('Integrated', 'Integrated graphics solutions', 'Budget Computing'),
('Unknown', 'Category not specified', 'General')
ON CONFLICT (category_name) DO NOTHING;

-- =====================================================================
-- 3. POPULATE PERFORMANCE TIERS
-- =====================================================================

INSERT INTO performance_tiers (tier_name, tier_rank, min_flops, max_flops, description) VALUES
('Flagship', 1, 25000000000000, NULL, 'Highest performance tier'),
('High-End', 2, 15000000000000, 25000000000000, 'High performance tier'),
('Mid-Range', 3, 5000000000000, 15000000000000, 'Mainstream performance tier'),
('Entry-Level', 4, 1000000000000, 5000000000000, 'Budget performance tier'),
('Basic', 5, NULL, 1000000000000, 'Entry-level performance tier')
ON CONFLICT (tier_name) DO NOTHING;

-- =====================================================================
-- 4. POPULATE EFFICIENCY CLASSES
-- =====================================================================

INSERT INTO efficiency_classes (efficiency_class, class_rank, min_tops_per_watt, max_tops_per_watt, description) VALUES
('Excellent', 1, 0.08, NULL, 'Outstanding power efficiency'),
('Good', 2, 0.06, 0.08, 'Very good power efficiency'),
('Average', 3, 0.04, 0.06, 'Standard power efficiency'),
('Below Average', 4, 0.02, 0.04, 'Poor power efficiency'),
('Poor', 5, NULL, 0.02, 'Very poor power efficiency')
ON CONFLICT (efficiency_class) DO NOTHING;

-- =====================================================================
-- 5. POPULATE MEMORY TIERS
-- =====================================================================

INSERT INTO memory_tiers (tier_name, min_memory_gb, max_memory_gb, tier_rank) VALUES
('Ultra (24GB+)', 24.0, NULL, 1),
('High (16-24GB)', 16.0, 24.0, 2),
('Medium (8-16GB)', 8.0, 16.0, 3),
('Low (4-8GB)', 4.0, 8.0, 4),
('Minimal (<4GB)', NULL, 4.0, 5),
('Unknown', NULL, NULL, 6)
ON CONFLICT (tier_name) DO NOTHING;

-- =====================================================================
-- 6. POPULATE AI PERFORMANCE CATEGORIES
-- =====================================================================

INSERT INTO ai_performance_categories (category_name, category_rank, min_throughput_fps, max_throughput_fps, description) VALUES
('AI_Flagship', 1, 40000.0, NULL, 'Top-tier AI performance'),
('AI_High_End', 2, 20000.0, 40000.0, 'High-end AI performance'),
('AI_Mid_Range', 3, 10000.0, 20000.0, 'Mid-range AI performance'),
('AI_Entry', 4, 5000.0, 10000.0, 'Entry-level AI performance'),
('AI_Basic', 5, NULL, 5000.0, 'Basic AI performance')
ON CONFLICT (category_name) DO NOTHING;

-- =====================================================================
-- 7. POPULATE ARCHITECTURES (Common ones)
-- =====================================================================

-- NVIDIA Architectures
INSERT INTO architectures (architecture_name, manufacturer_id, release_year, process_node_nm, architecture_family) VALUES
-- Get manufacturer_id for NVIDIA
('Ampere', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2020, 8.0, 'Modern'),
('Turing', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2018, 12.0, 'Modern'),
('Pascal', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2016, 16.0, 'Modern'),
('Maxwell', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2014, 28.0, 'Legacy'),
('Kepler', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2012, 28.0, 'Legacy'),
('Volta', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2017, 12.0, 'Modern'),
('Ada Lovelace', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA'), 2022, 4.0, 'Latest')
ON CONFLICT (architecture_name) DO NOTHING;

-- AMD Architectures  
INSERT INTO architectures (architecture_name, manufacturer_id, release_year, process_node_nm, architecture_family) VALUES
('RDNA 2', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD'), 2020, 7.0, 'Modern'),
('RDNA', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD'), 2019, 7.0, 'Modern'),
('GCN (Vega)', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD'), 2017, 14.0, 'Modern'),
('GCN', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD'), 2012, 28.0, 'Legacy'),
('RDNA 3', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD'), 2022, 5.0, 'Latest')
ON CONFLICT (architecture_name) DO NOTHING;

-- Intel Architectures
INSERT INTO architectures (architecture_name, manufacturer_id, release_year, process_node_nm, architecture_family) VALUES
('Xe', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'Intel'), 2020, 10.0, 'Modern'),
('UHD Graphics', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'Intel'), 2017, 14.0, 'Modern')
ON CONFLICT (architecture_name) DO NOTHING;

-- Unknown/Generic
INSERT INTO architectures (architecture_name, manufacturer_id, release_year, process_node_nm, architecture_family) VALUES
('Unknown', NULL, NULL, NULL, 'Unknown')
ON CONFLICT (architecture_name) DO NOTHING;

-- =====================================================================
-- 8. VERIFICATION QUERIES
-- =====================================================================

-- Display populated data
SELECT 'Manufacturers', COUNT(*) as count FROM manufacturers
UNION ALL
SELECT 'Device Categories', COUNT(*) FROM device_categories
UNION ALL  
SELECT 'Performance Tiers', COUNT(*) FROM performance_tiers
UNION ALL
SELECT 'Efficiency Classes', COUNT(*) FROM efficiency_classes
UNION ALL
SELECT 'Memory Tiers', COUNT(*) FROM memory_tiers
UNION ALL
SELECT 'AI Performance Categories', COUNT(*) FROM ai_performance_categories
UNION ALL
SELECT 'Architectures', COUNT(*) FROM architectures;

-- Show sample data
SELECT 'Sample Manufacturers:' as info;
SELECT manufacturer_name, country FROM manufacturers LIMIT 5;

SELECT 'Sample Architectures:' as info;
SELECT a.architecture_name, m.manufacturer_name, a.release_year 
FROM architectures a 
LEFT JOIN manufacturers m ON a.manufacturer_id = m.manufacturer_id 
LIMIT 5;

-- =====================================================================
-- LOOKUP TABLES POPULATION COMPLETE
-- ===================================================================== 
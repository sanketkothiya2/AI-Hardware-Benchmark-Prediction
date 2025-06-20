-- =====================================================================
-- MANUAL SCHEMA CREATION FOR AI_BENCHMARK DATABASE
-- Run these commands manually in PostgreSQL
-- =====================================================================

-- Connect to AI_BENCHMARK database first
-- \c AI_BENCHMARK;

-- 1. MANUFACTURERS TABLE
CREATE TABLE IF NOT EXISTS manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    manufacturer_name VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. ARCHITECTURES TABLE
CREATE TABLE IF NOT EXISTS architectures (
    architecture_id SERIAL PRIMARY KEY,
    architecture_name VARCHAR(50) NOT NULL UNIQUE,
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. DEVICE CATEGORIES TABLE
CREATE TABLE IF NOT EXISTS device_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(30) NOT NULL UNIQUE,
    category_description TEXT
);

-- 4. MAIN GPU DEVICES TABLE
CREATE TABLE IF NOT EXISTS gpu_devices (
    device_id SERIAL PRIMARY KEY,
    gpu_name VARCHAR(100) NOT NULL UNIQUE,
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id),
    architecture_id INTEGER REFERENCES architectures(architecture_id),
    category_id INTEGER REFERENCES device_categories(category_id),
    
    -- Basic specifications
    generation_category VARCHAR(30),
    test_date INTEGER,
    price_usd DECIMAL(10,2),
    tdp_watts DECIMAL(6,2) NOT NULL,
    
    -- Memory specifications
    memory_size_gb DECIMAL(6,2),
    memory_size_bytes BIGINT,
    memory_bandwidth_bytes_per_sec BIGINT,
    
    -- Technical specifications
    process_size_nm DECIMAL(4,1),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. GRAPHICS PERFORMANCE TABLE
CREATE TABLE IF NOT EXISTS graphics_performance (
    performance_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES gpu_devices(device_id) ON DELETE CASCADE,
    g3d_mark INTEGER,
    g2d_mark INTEGER,
    power_performance DECIMAL(10,4),
    gpu_value DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. AI COMPUTE PERFORMANCE TABLE
CREATE TABLE IF NOT EXISTS ai_compute_performance (
    ai_performance_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES gpu_devices(device_id) ON DELETE CASCADE,
    
    -- Core AI metrics
    fp32_flops BIGINT NOT NULL,
    flops_per_watt DECIMAL(12,6) NOT NULL,
    tops_per_watt DECIMAL(8,6) NOT NULL,
    gflops_per_watt DECIMAL(10,4) NOT NULL,
    
    -- Precision performance
    fp16_flops BIGINT,
    int8_ops BIGINT,
    
    -- Efficiency metrics
    relative_latency_index DECIMAL(8,4),
    compute_usage_percent DECIMAL(5,2),
    performance_per_dollar_per_watt DECIMAL(12,6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. AI MODEL THROUGHPUT TABLE
CREATE TABLE IF NOT EXISTS ai_model_throughput (
    throughput_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES gpu_devices(device_id) ON DELETE CASCADE,
    
    -- Model-specific throughput (FPS)
    resnet50_imagenet_fps DECIMAL(10,2),
    bert_base_fps DECIMAL(10,2),
    gpt2_small_fps DECIMAL(10,2),
    mobilenetv2_fps DECIMAL(10,2),
    efficientnet_b0_fps DECIMAL(10,2),
    avg_throughput_fps DECIMAL(10,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. API SUPPORT TABLE
CREATE TABLE IF NOT EXISTS api_support (
    api_support_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES gpu_devices(device_id) ON DELETE CASCADE,
    cuda_support BOOLEAN DEFAULT FALSE,
    opencl_support BOOLEAN DEFAULT FALSE,
    vulkan_support BOOLEAN DEFAULT FALSE,
    metal_support BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. DEVICE CLASSIFICATIONS TABLE
CREATE TABLE IF NOT EXISTS device_classifications (
    classification_id SERIAL PRIMARY KEY,
    device_id INTEGER REFERENCES gpu_devices(device_id) ON DELETE CASCADE,
    is_legacy_low_perf BOOLEAN DEFAULT FALSE,
    price_performance_index DECIMAL(10,4),
    ai_efficiency_tier VARCHAR(20),
    ai_performance_category VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. CREATE INDEXES FOR PERFORMANCE
CREATE INDEX IF NOT EXISTS idx_gpu_devices_manufacturer ON gpu_devices(manufacturer_id);
CREATE INDEX IF NOT EXISTS idx_gpu_devices_architecture ON gpu_devices(architecture_id);
CREATE INDEX IF NOT EXISTS idx_gpu_devices_gpu_name ON gpu_devices(gpu_name);
CREATE INDEX IF NOT EXISTS idx_ai_compute_tops_per_watt ON ai_compute_performance(tops_per_watt);
CREATE INDEX IF NOT EXISTS idx_gpu_devices_tdp ON gpu_devices(tdp_watts);

-- 11. CREATE MAIN VIEW FOR EASY QUERYING
CREATE OR REPLACE VIEW v_complete_devices AS
SELECT 
    gd.device_id,
    gd.gpu_name,
    m.manufacturer_name,
    a.architecture_name,
    dc.category_name,
    gd.generation_category,
    gd.test_date,
    gd.price_usd,
    gd.tdp_watts,
    gd.memory_size_gb,
    
    -- AI Performance
    acp.fp32_flops,
    acp.tops_per_watt,
    acp.flops_per_watt,
    acp.relative_latency_index,
    acp.compute_usage_percent,
    
    -- Graphics Performance
    gp.g3d_mark,
    gp.g2d_mark,
    gp.power_performance,
    
    -- AI Model Throughput
    amt.avg_throughput_fps,
    amt.resnet50_imagenet_fps,
    amt.bert_base_fps,
    amt.gpt2_small_fps,
    
    -- API Support
    apis.cuda_support,
    apis.opencl_support,
    apis.vulkan_support,
    
    -- Classifications
    dcl.ai_efficiency_tier,
    dcl.ai_performance_category
    
FROM gpu_devices gd
LEFT JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
LEFT JOIN architectures a ON gd.architecture_id = a.architecture_id
LEFT JOIN device_categories dc ON gd.category_id = dc.category_id
LEFT JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
LEFT JOIN graphics_performance gp ON gd.device_id = gp.device_id
LEFT JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
LEFT JOIN api_support apis ON gd.device_id = apis.device_id
LEFT JOIN device_classifications dcl ON gd.device_id = dcl.device_id;

-- 12. POPULATE BASIC LOOKUP DATA
INSERT INTO manufacturers (manufacturer_name) VALUES 
('NVIDIA'), ('AMD'), ('Intel'), ('Apple'), ('Qualcomm'), ('Unknown')
ON CONFLICT (manufacturer_name) DO NOTHING;

INSERT INTO device_categories (category_name, category_description) VALUES 
('Consumer', 'Gaming and consumer graphics cards'),
('Professional', 'Workstation and datacenter GPUs'),
('Mobile', 'Laptop and mobile GPUs'),
('Integrated', 'Integrated graphics solutions'),
('Unknown', 'Category not specified')
ON CONFLICT (category_name) DO NOTHING;

-- Insert common architectures
INSERT INTO architectures (architecture_name, manufacturer_id) VALUES 
('Ampere', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA')),
('Turing', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA')),
('Pascal', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'NVIDIA')),
('RDNA 2', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD')),
('RDNA', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD')),
('GCN', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'AMD')),
('Xe', (SELECT manufacturer_id FROM manufacturers WHERE manufacturer_name = 'Intel')),
('Unknown', NULL)
ON CONFLICT (architecture_name) DO NOTHING;

-- SCHEMA CREATION COMPLETE
SELECT 'Schema created successfully! Tables:' as status;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name; 
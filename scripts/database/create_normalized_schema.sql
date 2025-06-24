-- AI Benchmark Normalized Database Schema
-- Created for PostgreSQL

-- Drop existing tables if they exist (in reverse order due to foreign key dependencies)
DROP TABLE IF EXISTS ai_performance CASCADE;
DROP TABLE IF EXISTS memory_specs CASCADE;
DROP TABLE IF EXISTS process_technology CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS gpus CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS architectures CASCADE;
DROP TABLE IF EXISTS manufacturers CASCADE;

-- Create Manufacturers lookup table
CREATE TABLE manufacturers (
    manufacturer_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Architectures lookup table
CREATE TABLE architectures (
    architecture_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Categories lookup table
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create main GPUs table
CREATE TABLE gpus (
    gpu_id SERIAL PRIMARY KEY,
    gpu_name VARCHAR(200) NOT NULL,
    manufacturer_id INTEGER REFERENCES manufacturers(manufacturer_id),
    architecture_id INTEGER REFERENCES architectures(architecture_id),
    category_id INTEGER REFERENCES categories(category_id),
    
    -- Classification fields
    performance_category VARCHAR(50),
    generation_category VARCHAR(50),
    performance_tier VARCHAR(50),
    generation VARCHAR(50),
    efficiency_class VARCHAR(50),
    
    -- Basic specifications
    tdp INTEGER,
    process_size_nm INTEGER,
    test_date INTEGER,
    price DECIMAL(10,2),
    
    -- Derived metrics
    power_performance DECIMAL(10,2),
    gpu_value DECIMAL(10,2),
    price_performance_index DECIMAL(10,2),
    is_legacy_low_perf BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Performance Metrics table
CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    gpu_id INTEGER REFERENCES gpus(gpu_id) ON DELETE CASCADE,
    
    -- Benchmark scores
    g3d_mark INTEGER,
    g2d_mark INTEGER,
    
    -- Performance calculations (using BIGINT for very large values)
    fp32_final BIGINT,
    flops_per_watt DECIMAL(10,6),
    gflops_per_watt DECIMAL(10,6),
    performance_per_dollar_per_watt DECIMAL(10,6),
    
    -- FP precision performance (using BIGINT for very large values)
    fp16_performance BIGINT,
    fp16_performance_predicted BIGINT,
    
    -- Integer performance (using BIGINT for very large values)
    int8_performance BIGINT,
    int8_performance_estimated BIGINT,
    
    -- AI-specific metrics
    tops_per_watt DECIMAL(10,6),
    relative_latency_index DECIMAL(10,2),
    compute_usage_percent DECIMAL(5,2),
    
    -- Throughput metrics (fps)
    throughput_resnet50_imagenet_fps DECIMAL(10,2),
    throughput_bert_base_fps DECIMAL(10,2),
    throughput_gpt2_small_fps DECIMAL(10,2),
    throughput_mobilenetv2_fps DECIMAL(10,2),
    throughput_efficientnet_b0_fps DECIMAL(10,2),
    avg_throughput_fps DECIMAL(10,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Memory Specifications table
CREATE TABLE memory_specs (
    memory_id SERIAL PRIMARY KEY,
    gpu_id INTEGER REFERENCES gpus(gpu_id) ON DELETE CASCADE,
    
    memory_size_bytes BIGINT,
    memory_gb DECIMAL(8,3),
    memory_tier VARCHAR(50),
    memory_bandwidth_bytes_per_sec BIGINT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Process Technology table
CREATE TABLE process_technology (
    process_id SERIAL PRIMARY KEY,
    gpu_id INTEGER REFERENCES gpus(gpu_id) ON DELETE CASCADE,
    
    -- API Support
    cuda_support INTEGER,
    opencl_support INTEGER,
    vulkan_support INTEGER,
    metal_support INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create AI Performance table
CREATE TABLE ai_performance (
    ai_perf_id SERIAL PRIMARY KEY,
    gpu_id INTEGER REFERENCES gpus(gpu_id) ON DELETE CASCADE,
    
    ai_efficiency_tier VARCHAR(50),
    ai_performance_category VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_gpus_manufacturer ON gpus(manufacturer_id);
CREATE INDEX idx_gpus_architecture ON gpus(architecture_id);
CREATE INDEX idx_gpus_category ON gpus(category_id);
CREATE INDEX idx_gpus_name ON gpus(gpu_name);
CREATE INDEX idx_performance_metrics_gpu ON performance_metrics(gpu_id);
CREATE INDEX idx_memory_specs_gpu ON memory_specs(gpu_id);
CREATE INDEX idx_process_technology_gpu ON process_technology(gpu_id);
CREATE INDEX idx_ai_performance_gpu ON ai_performance(gpu_id);

-- Create a view for complete GPU information
CREATE VIEW gpu_complete_info AS
SELECT 
    g.gpu_id,
    g.gpu_name,
    m.name as manufacturer,
    a.name as architecture,
    c.name as category,
    g.performance_category,
    g.generation_category,
    g.performance_tier,
    g.generation,
    g.efficiency_class,
    g.tdp,
    g.process_size_nm,
    g.test_date,
    g.price,
    g.power_performance,
    g.gpu_value,
    g.price_performance_index,
    g.is_legacy_low_perf,
    
    -- Performance metrics
    pm.g3d_mark,
    pm.g2d_mark,
    pm.fp32_final,
    pm.flops_per_watt,
    pm.gflops_per_watt,
    pm.performance_per_dollar_per_watt,
    pm.fp16_performance,
    pm.fp16_performance_predicted,
    pm.int8_performance,
    pm.int8_performance_estimated,
    pm.tops_per_watt,
    pm.relative_latency_index,
    pm.compute_usage_percent,
    pm.throughput_resnet50_imagenet_fps,
    pm.throughput_bert_base_fps,
    pm.throughput_gpt2_small_fps,
    pm.throughput_mobilenetv2_fps,
    pm.throughput_efficientnet_b0_fps,
    pm.avg_throughput_fps,
    
    -- Memory specifications
    ms.memory_size_bytes,
    ms.memory_gb,
    ms.memory_tier,
    ms.memory_bandwidth_bytes_per_sec,
    
    -- Process technology
    pt.cuda_support,
    pt.opencl_support,
    pt.vulkan_support,
    pt.metal_support,
    
    -- AI performance
    ap.ai_efficiency_tier,
    ap.ai_performance_category
    
FROM gpus g
LEFT JOIN manufacturers m ON g.manufacturer_id = m.manufacturer_id
LEFT JOIN architectures a ON g.architecture_id = a.architecture_id
LEFT JOIN categories c ON g.category_id = c.category_id
LEFT JOIN performance_metrics pm ON g.gpu_id = pm.gpu_id
LEFT JOIN memory_specs ms ON g.gpu_id = ms.gpu_id
LEFT JOIN process_technology pt ON g.gpu_id = pt.gpu_id
LEFT JOIN ai_performance ap ON g.gpu_id = ap.gpu_id;

-- Insert initial lookup data
INSERT INTO manufacturers (name) VALUES 
    ('NVIDIA'),
    ('AMD'),
    ('Intel');

INSERT INTO categories (name, description) VALUES 
    ('Consumer', 'Consumer-grade graphics cards'),
    ('Professional', 'Professional/Workstation graphics cards'),
    ('Mobile', 'Mobile/Laptop graphics cards');

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for gpus table
CREATE TRIGGER update_gpus_updated_at BEFORE UPDATE ON gpus
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMIT; 
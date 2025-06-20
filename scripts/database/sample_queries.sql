-- =====================================================================
-- SAMPLE QUERIES FOR AI BENCHMARK DATABASE
-- Purpose: Common queries for AI hardware analysis and prediction modeling
-- =====================================================================

\c AI_BENCHMARK;

-- =====================================================================
-- 1. BASIC DATA EXPLORATION
-- =====================================================================

-- Total devices in database
SELECT 
    'Total GPU Devices' as metric,
    COUNT(*) as count
FROM gpu_devices;

-- Devices by manufacturer
SELECT 
    m.manufacturer_name,
    COUNT(gd.device_id) as device_count,
    ROUND(COUNT(gd.device_id) * 100.0 / (SELECT COUNT(*) FROM gpu_devices), 2) as percentage
FROM manufacturers m
LEFT JOIN gpu_devices gd ON m.manufacturer_id = gd.manufacturer_id
GROUP BY m.manufacturer_id, m.manufacturer_name
ORDER BY device_count DESC;

-- Architecture distribution
SELECT 
    a.architecture_name,
    m.manufacturer_name,
    COUNT(gd.device_id) as device_count,
    MIN(gd.test_date) as first_release,
    MAX(gd.test_date) as latest_release
FROM architectures a
LEFT JOIN manufacturers m ON a.manufacturer_id = m.manufacturer_id
LEFT JOIN gpu_devices gd ON a.architecture_id = gd.architecture_id
GROUP BY a.architecture_id, a.architecture_name, m.manufacturer_name
ORDER BY device_count DESC;

-- =====================================================================
-- 2. AI PERFORMANCE ANALYSIS
-- =====================================================================

-- Top 20 AI performers by TOPs/Watt
SELECT 
    gd.gpu_name,
    m.manufacturer_name,
    a.architecture_name,
    acp.tops_per_watt,
    acp.fp32_flops,
    gd.tdp_watts,
    gd.price_usd,
    CASE 
        WHEN gd.price_usd IS NOT NULL 
        THEN ROUND(acp.tops_per_watt / gd.price_usd * 1000, 4)
        ELSE NULL 
    END as tops_per_watt_per_1k_usd
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN architectures a ON gd.architecture_id = a.architecture_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
WHERE acp.tops_per_watt IS NOT NULL
ORDER BY acp.tops_per_watt DESC
LIMIT 20;

-- AI efficiency comparison by manufacturer
SELECT 
    m.manufacturer_name,
    COUNT(gd.device_id) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_tops_per_watt,
    ROUND(MIN(acp.tops_per_watt), 6) as min_tops_per_watt,
    ROUND(MAX(acp.tops_per_watt), 6) as max_tops_per_watt,
    ROUND(STDDEV(acp.tops_per_watt), 6) as stddev_tops_per_watt,
    ROUND(AVG(gd.tdp_watts), 2) as avg_tdp_watts
FROM manufacturers m
JOIN gpu_devices gd ON m.manufacturer_id = gd.manufacturer_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
WHERE acp.tops_per_watt IS NOT NULL
GROUP BY m.manufacturer_id, m.manufacturer_name
ORDER BY avg_tops_per_watt DESC;

-- Power efficiency vs Performance correlation
SELECT 
    CASE 
        WHEN gd.tdp_watts <= 75 THEN 'Low Power (≤75W)'
        WHEN gd.tdp_watts <= 150 THEN 'Medium Power (76-150W)'
        WHEN gd.tdp_watts <= 250 THEN 'High Power (151-250W)'
        ELSE 'Very High Power (>250W)'
    END as power_category,
    COUNT(*) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(acp.fp32_flops::bigint), 0) as avg_fp32_flops,
    ROUND(AVG(gd.tdp_watts), 2) as avg_tdp
FROM gpu_devices gd
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
WHERE acp.tops_per_watt IS NOT NULL
GROUP BY 
    CASE 
        WHEN gd.tdp_watts <= 75 THEN 'Low Power (≤75W)'
        WHEN gd.tdp_watts <= 150 THEN 'Medium Power (76-150W)'
        WHEN gd.tdp_watts <= 250 THEN 'High Power (151-250W)'
        ELSE 'Very High Power (>250W)'
    END
ORDER BY avg_efficiency DESC;

-- =====================================================================
-- 3. AI MODEL THROUGHPUT ANALYSIS
-- =====================================================================

-- Best performers for different AI models
SELECT 
    'ResNet50' as model,
    gd.gpu_name,
    m.manufacturer_name,
    amt.resnet50_imagenet_fps as fps,
    gd.tdp_watts,
    ROUND(amt.resnet50_imagenet_fps / gd.tdp_watts, 2) as fps_per_watt
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
WHERE amt.resnet50_imagenet_fps IS NOT NULL
ORDER BY amt.resnet50_imagenet_fps DESC
LIMIT 10

UNION ALL

SELECT 
    'BERT Base' as model,
    gd.gpu_name,
    m.manufacturer_name,
    amt.bert_base_fps as fps,
    gd.tdp_watts,
    ROUND(amt.bert_base_fps / gd.tdp_watts, 2) as fps_per_watt
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
WHERE amt.bert_base_fps IS NOT NULL
ORDER BY amt.bert_base_fps DESC
LIMIT 10

UNION ALL

SELECT 
    'GPT-2 Small' as model,
    gd.gpu_name,
    m.manufacturer_name,
    amt.gpt2_small_fps as fps,
    gd.tdp_watts,
    ROUND(amt.gpt2_small_fps / gd.tdp_watts, 2) as fps_per_watt
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
WHERE amt.gpt2_small_fps IS NOT NULL
ORDER BY amt.gpt2_small_fps DESC
LIMIT 10;

-- Model throughput correlation matrix
SELECT 
    ROUND(CORR(amt.resnet50_imagenet_fps, amt.bert_base_fps), 4) as resnet_bert_corr,
    ROUND(CORR(amt.resnet50_imagenet_fps, amt.gpt2_small_fps), 4) as resnet_gpt2_corr,
    ROUND(CORR(amt.bert_base_fps, amt.gpt2_small_fps), 4) as bert_gpt2_corr,
    ROUND(CORR(amt.avg_throughput_fps, acp.tops_per_watt), 4) as throughput_efficiency_corr
FROM ai_model_throughput amt
JOIN ai_compute_performance acp ON amt.device_id = acp.device_id
WHERE amt.resnet50_imagenet_fps IS NOT NULL 
    AND amt.bert_base_fps IS NOT NULL 
    AND amt.gpt2_small_fps IS NOT NULL;

-- =====================================================================
-- 4. PRICE-PERFORMANCE ANALYSIS
-- =====================================================================

-- Best value GPUs for AI workloads
SELECT 
    gd.gpu_name,
    m.manufacturer_name,
    gd.price_usd,
    acp.tops_per_watt,
    ROUND(acp.tops_per_watt / gd.price_usd * 1000, 4) as tops_per_watt_per_1k_usd,
    amt.avg_throughput_fps,
    ROUND(amt.avg_throughput_fps / gd.price_usd, 2) as throughput_per_dollar,
    dc.ai_efficiency_tier,
    dc.ai_performance_category
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
JOIN device_classifications dc ON gd.device_id = dc.device_id
WHERE gd.price_usd IS NOT NULL 
    AND gd.price_usd > 0
    AND acp.tops_per_watt IS NOT NULL
ORDER BY tops_per_watt_per_1k_usd DESC
LIMIT 20;

-- Price categories analysis
SELECT 
    CASE 
        WHEN gd.price_usd <= 200 THEN 'Budget (<$200)'
        WHEN gd.price_usd <= 500 THEN 'Mid-Range ($200-500)'
        WHEN gd.price_usd <= 1000 THEN 'High-End ($500-1000)'
        ELSE 'Premium (>$1000)'
    END as price_category,
    COUNT(*) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(amt.avg_throughput_fps), 2) as avg_throughput,
    ROUND(AVG(gd.price_usd), 2) as avg_price
FROM gpu_devices gd
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
WHERE gd.price_usd IS NOT NULL AND gd.price_usd > 0
GROUP BY 
    CASE 
        WHEN gd.price_usd <= 200 THEN 'Budget (<$200)'
        WHEN gd.price_usd <= 500 THEN 'Mid-Range ($200-500)'
        WHEN gd.price_usd <= 1000 THEN 'High-End ($500-1000)'
        ELSE 'Premium (>$1000)'
    END
ORDER BY avg_price;

-- =====================================================================
-- 5. ARCHITECTURE AND GENERATION ANALYSIS
-- =====================================================================

-- Architecture performance evolution
SELECT 
    a.architecture_name,
    m.manufacturer_name,
    a.release_year,
    COUNT(gd.device_id) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(gd.tdp_watts), 2) as avg_tdp,
    ROUND(AVG(a.process_node_nm), 1) as avg_process_nm
FROM architectures a
JOIN manufacturers m ON a.manufacturer_id = m.manufacturer_id
JOIN gpu_devices gd ON a.architecture_id = gd.architecture_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
WHERE a.release_year IS NOT NULL
GROUP BY a.architecture_id, a.architecture_name, m.manufacturer_name, a.release_year
ORDER BY a.release_year DESC, avg_efficiency DESC;

-- Process node vs efficiency analysis
SELECT 
    ROUND(a.process_node_nm) as process_node_nm,
    COUNT(gd.device_id) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(gd.tdp_watts), 2) as avg_tdp_watts,
    ROUND(AVG(acp.fp32_flops::bigint / 1e12), 2) as avg_tflops
FROM architectures a
JOIN gpu_devices gd ON a.architecture_id = gd.architecture_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
WHERE a.process_node_nm IS NOT NULL
GROUP BY ROUND(a.process_node_nm)
ORDER BY process_node_nm;

-- =====================================================================
-- 6. MEMORY ANALYSIS
-- =====================================================================

-- Memory bandwidth vs AI performance
SELECT 
    CASE 
        WHEN gd.memory_bandwidth_bytes_per_sec IS NULL THEN 'Unknown'
        WHEN gd.memory_bandwidth_bytes_per_sec < 100e9 THEN 'Low (<100 GB/s)'
        WHEN gd.memory_bandwidth_bytes_per_sec < 500e9 THEN 'Medium (100-500 GB/s)'
        WHEN gd.memory_bandwidth_bytes_per_sec < 1000e9 THEN 'High (500-1000 GB/s)'
        ELSE 'Very High (>1000 GB/s)'
    END as bandwidth_category,
    COUNT(*) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(amt.avg_throughput_fps), 2) as avg_throughput,
    ROUND(AVG(gd.memory_bandwidth_bytes_per_sec / 1e9), 2) as avg_bandwidth_gb_s
FROM gpu_devices gd
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
GROUP BY 
    CASE 
        WHEN gd.memory_bandwidth_bytes_per_sec IS NULL THEN 'Unknown'
        WHEN gd.memory_bandwidth_bytes_per_sec < 100e9 THEN 'Low (<100 GB/s)'
        WHEN gd.memory_bandwidth_bytes_per_sec < 500e9 THEN 'Medium (100-500 GB/s)'
        WHEN gd.memory_bandwidth_bytes_per_sec < 1000e9 THEN 'High (500-1000 GB/s)'
        ELSE 'Very High (>1000 GB/s)'
    END
ORDER BY avg_efficiency DESC;

-- Memory size vs AI model performance
SELECT 
    mt.tier_name as memory_tier,
    COUNT(gd.device_id) as device_count,
    ROUND(AVG(gd.memory_size_gb), 2) as avg_memory_gb,
    ROUND(AVG(amt.resnet50_imagenet_fps), 2) as avg_resnet50_fps,
    ROUND(AVG(amt.bert_base_fps), 2) as avg_bert_fps,
    ROUND(AVG(amt.gpt2_small_fps), 2) as avg_gpt2_fps,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency
FROM memory_tiers mt
JOIN gpu_devices gd ON mt.memory_tier_id = gd.memory_tier_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
GROUP BY mt.memory_tier_id, mt.tier_name, mt.tier_rank
ORDER BY mt.tier_rank;

-- =====================================================================
-- 7. API SUPPORT ANALYSIS
-- =====================================================================

-- API support distribution and performance impact
SELECT 
    CASE 
        WHEN apis.cuda_support THEN 'CUDA'
        WHEN apis.opencl_support THEN 'OpenCL Only'
        WHEN apis.vulkan_support THEN 'Vulkan Only'
        WHEN apis.metal_support THEN 'Metal Only'
        ELSE 'Limited Support'
    END as primary_api,
    COUNT(*) as device_count,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(amt.avg_throughput_fps), 2) as avg_throughput
FROM api_support apis
JOIN ai_compute_performance acp ON apis.device_id = acp.device_id
JOIN ai_model_throughput amt ON apis.device_id = amt.device_id
GROUP BY 
    CASE 
        WHEN apis.cuda_support THEN 'CUDA'
        WHEN apis.opencl_support THEN 'OpenCL Only'
        WHEN apis.vulkan_support THEN 'Vulkan Only'
        WHEN apis.metal_support THEN 'Metal Only'
        ELSE 'Limited Support'
    END
ORDER BY avg_efficiency DESC;

-- =====================================================================
-- 8. TREND ANALYSIS
-- =====================================================================

-- Performance trends over time
SELECT 
    gd.test_date as test_year,
    COUNT(*) as devices_tested,
    ROUND(AVG(acp.tops_per_watt), 6) as avg_efficiency,
    ROUND(AVG(gd.tdp_watts), 2) as avg_tdp,
    ROUND(AVG(amt.avg_throughput_fps), 2) as avg_throughput,
    ROUND(AVG(gd.price_usd), 2) as avg_price
FROM gpu_devices gd
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
WHERE gd.test_date IS NOT NULL
GROUP BY gd.test_date
ORDER BY gd.test_date DESC;

-- =====================================================================
-- 9. OUTLIER DETECTION
-- =====================================================================

-- Identify performance outliers (devices with unusual efficiency)
WITH efficiency_stats AS (
    SELECT 
        AVG(tops_per_watt) as mean_efficiency,
        STDDEV(tops_per_watt) as std_efficiency
    FROM ai_compute_performance
    WHERE tops_per_watt IS NOT NULL
)
SELECT 
    gd.gpu_name,
    m.manufacturer_name,
    acp.tops_per_watt,
    ROUND((acp.tops_per_watt - es.mean_efficiency) / es.std_efficiency, 2) as z_score,
    CASE 
        WHEN ABS((acp.tops_per_watt - es.mean_efficiency) / es.std_efficiency) > 2 
        THEN 'Outlier'
        ELSE 'Normal'
    END as outlier_status
FROM gpu_devices gd
JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
CROSS JOIN efficiency_stats es
WHERE acp.tops_per_watt IS NOT NULL
ORDER BY ABS((acp.tops_per_watt - es.mean_efficiency) / es.std_efficiency) DESC
LIMIT 20;

-- =====================================================================
-- 10. EXPORT QUERIES FOR MACHINE LEARNING
-- =====================================================================

-- Complete feature matrix for ML modeling
SELECT 
    gd.gpu_name,
    m.manufacturer_name,
    a.architecture_name,
    gd.tdp_watts,
    gd.memory_size_gb,
    gd.memory_bandwidth_bytes_per_sec / 1e9 as memory_bandwidth_gb_s,
    a.process_node_nm,
    gd.price_usd,
    
    -- Target variables (AI performance metrics)
    acp.tops_per_watt,
    acp.fp32_flops,
    acp.flops_per_watt,
    amt.resnet50_imagenet_fps,
    amt.bert_base_fps,
    amt.gpt2_small_fps,
    amt.avg_throughput_fps,
    
    -- Binary features
    apis.cuda_support::int as cuda_support,
    apis.opencl_support::int as opencl_support,
    apis.vulkan_support::int as vulkan_support,
    
    -- Graphics performance
    gp.g3d_mark,
    gp.power_performance
    
FROM gpu_devices gd
LEFT JOIN manufacturers m ON gd.manufacturer_id = m.manufacturer_id
LEFT JOIN architectures a ON gd.architecture_id = a.architecture_id
LEFT JOIN ai_compute_performance acp ON gd.device_id = acp.device_id
LEFT JOIN ai_model_throughput amt ON gd.device_id = amt.device_id
LEFT JOIN api_support apis ON gd.device_id = apis.device_id
LEFT JOIN graphics_performance gp ON gd.device_id = gp.device_id
WHERE acp.tops_per_watt IS NOT NULL
ORDER BY acp.tops_per_watt DESC;

-- =====================================================================
-- END OF SAMPLE QUERIES
-- ===================================================================== 
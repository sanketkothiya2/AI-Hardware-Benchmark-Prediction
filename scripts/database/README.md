# AI Benchmark PostgreSQL Database Setup

## Overview
This directory contains scripts and documentation for setting up a normalized PostgreSQL database for AI hardware benchmarking data. The database is designed to store, analyze, and provide insights from your `Ai-Benchmark-Final-enhanced.csv` dataset.

## Database Information
- **Database Name**: `AI_BENCHMARK`
- **Username**: `postgres`  
- **Password**: `admin`
- **Host**: `localhost`
- **Port**: `5432`

## Prerequisites
1. **PostgreSQL 12+** installed and running
2. **Python 3.8+** with required packages:
   ```bash
   pip install pandas psycopg2-binary numpy
   ```
3. **psql command-line tool** accessible from PATH

## Quick Setup (Automated)
The easiest way to set up the complete database is to run the automated setup script:

```bash
cd scripts/database
python complete_database_setup.py
```

This script will:
1. Create the `AI_BENCHMARK` database
2. Create all tables with proper relationships
3. Populate lookup tables with reference data
4. Import data from your CSV file
5. Verify the setup is complete

## Manual Setup (Step by Step)

### Step 1: Create Database
```bash
psql -U postgres -h localhost -f setup_database.sql
```

### Step 2: Create Schema
```bash
psql -U postgres -h localhost -f create_ai_benchmark_schema.sql
```

### Step 3: Populate Lookup Tables
```bash
psql -U postgres -h localhost -f populate_lookup_tables.sql
```

### Step 4: Import CSV Data
```bash
python import_csv_to_database.py
```

## Database Schema

### Normalized Design
The database follows a normalized design with proper foreign key relationships:

```
manufacturers (18 records)
├── gpu_devices (2,108 records)
├── architectures (15 records)
└── device_categories (5 records)

gpu_devices
├── graphics_performance (2,108 records)
├── ai_compute_performance (2,108 records)  
├── ai_model_throughput (2,108 records)
├── api_support (2,108 records)
└── device_classifications (2,108 records)
```

### Key Tables

#### Core Tables
- **`gpu_devices`**: Main entity with device specifications
- **`manufacturers`**: GPU manufacturers (NVIDIA, AMD, Intel, etc.)
- **`architectures`**: GPU architectures (Ampere, RDNA 2, etc.)

#### Performance Tables
- **`ai_compute_performance`**: AI-specific metrics (TOPs/Watt, FLOPS, etc.)
- **`ai_model_throughput`**: Model-specific performance (ResNet50, BERT, GPT-2)
- **`graphics_performance`**: Graphics benchmarking scores

#### Classification Tables
- **`device_categories`**: Consumer, Professional, Mobile, etc.
- **`performance_tiers`**: Flagship, High-End, Mid-Range, etc.
- **`efficiency_classes`**: Power efficiency classifications
- **`memory_tiers`**: Memory size categories

### Views for Easy Querying
- **`v_complete_devices`**: Comprehensive device information with all metrics
- **`v_top_ai_performers`**: Top AI performers sorted by efficiency
- **`v_manufacturer_summary`**: Aggregated statistics by manufacturer

## Sample Queries

### Basic Data Exploration
```sql
-- Total devices by manufacturer
SELECT manufacturer_name, COUNT(*) as device_count 
FROM v_complete_devices 
GROUP BY manufacturer_name 
ORDER BY device_count DESC;
```

### AI Performance Analysis
```sql
-- Top 10 most efficient AI devices
SELECT gpu_name, manufacturer_name, tops_per_watt, tdp_watts
FROM v_complete_devices 
WHERE tops_per_watt IS NOT NULL
ORDER BY tops_per_watt DESC 
LIMIT 10;
```

### Price-Performance Analysis
```sql
-- Best value for AI workloads
SELECT gpu_name, price_usd, tops_per_watt,
       ROUND(tops_per_watt / price_usd * 1000, 4) as efficiency_per_1k_usd
FROM v_complete_devices 
WHERE price_usd > 0 AND tops_per_watt IS NOT NULL
ORDER BY efficiency_per_1k_usd DESC
LIMIT 10;
```

See `sample_queries.sql` for 50+ more advanced analysis queries.

## File Structure
```
scripts/database/
├── README.md                          # This file
├── setup_database.sql                 # Creates AI_BENCHMARK database
├── create_ai_benchmark_schema.sql     # Creates all tables and relationships
├── populate_lookup_tables.sql         # Populates reference/lookup tables
├── import_csv_to_database.py          # Imports CSV data into normalized tables
├── complete_database_setup.py         # Automated setup script
└── sample_queries.sql                 # Collection of analysis queries
```

## Database Features

### Performance Optimizations
- **Indexes**: Optimized indexes for common query patterns
- **Views**: Pre-built views for complex joins
- **Data Types**: Appropriate data types for storage efficiency
- **Constraints**: Data integrity through foreign keys and check constraints

### Data Quality
- **Normalization**: Eliminates data redundancy
- **Referential Integrity**: Foreign key constraints ensure data consistency
- **Data Validation**: Check constraints prevent invalid data
- **NULL Handling**: Proper handling of missing values

### Extensibility
- **Modular Design**: Easy to add new tables or metrics
- **Versioning**: Schema supports future enhancements
- **Triggers**: Automatic timestamp updates
- **Security**: Role-based access control

## Database Statistics (After Setup)

Expected record counts:
- **Total Tables**: 13 core tables + 3 views
- **GPU Devices**: 2,108 devices
- **Total Records**: ~15,000+ across all tables
- **Manufacturers**: 18 (NVIDIA, AMD, Intel, etc.)
- **Architectures**: 15+ (Ampere, RDNA 2, Pascal, etc.)

## Usage Examples

### Connecting to Database
```python
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    database='AI_BENCHMARK',
    user='postgres',
    password='admin',
    port=5432
)
```

### Common Analysis Tasks
1. **Performance Benchmarking**: Compare AI efficiency across manufacturers
2. **Price Analysis**: Find best value GPUs for specific budgets
3. **Technology Trends**: Analyze performance evolution over time
4. **Model-Specific**: Find best hardware for specific AI models
5. **Power Efficiency**: Analyze power consumption patterns

### Machine Learning Data Export
```sql
-- Export feature matrix for ML modeling
SELECT gpu_name, manufacturer_name, tdp_watts, memory_size_gb,
       tops_per_watt, avg_throughput_fps, cuda_support
FROM v_complete_devices 
WHERE tops_per_watt IS NOT NULL;
```

## Troubleshooting

### Common Issues
1. **Connection Failed**: Ensure PostgreSQL is running and credentials are correct
2. **Permission Denied**: Check user permissions and database ownership
3. **Import Errors**: Verify CSV file path and format
4. **Missing psql**: Install PostgreSQL client tools

### Log Files
- `complete_database_setup.log`: Automated setup logs
- `database_import.log`: CSV import logs

### Verification
After setup, verify the database:
```sql
-- Check table counts
SELECT 'GPU Devices' as table_name, COUNT(*) FROM gpu_devices
UNION ALL
SELECT 'AI Performance', COUNT(*) FROM ai_compute_performance
UNION ALL  
SELECT 'Manufacturers', COUNT(*) FROM manufacturers;
```

## Next Steps

1. **Data Analysis**: Use the sample queries to explore your data
2. **Visualization**: Connect BI tools (Tableau, Power BI) for dashboards
3. **Machine Learning**: Export data for predictive modeling
4. **API Development**: Build REST APIs for data access
5. **Real-time Updates**: Set up data pipelines for continuous updates

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Verify PostgreSQL installation and configuration
3. Ensure all prerequisite files are present
4. Test database connectivity before running imports

---

**Database Schema Version**: 1.0  
**Last Updated**: December 2024  
**Compatible with**: PostgreSQL 12+, Python 3.8+ 
# AI Benchmark Normalized Database Setup

This directory contains scripts to create a normalized PostgreSQL database from the AI benchmark CSV data.

## Files Overview

1. **`create_normalized_schema.sql`** - PostgreSQL script to create the normalized database schema
2. **`import_normalized_data.py`** - Python script to import CSV data into the normalized database
3. **`README_NORMALIZED.md`** - This documentation file

## Database Schema

The normalized database consists of the following tables:

### Lookup Tables
- **`manufacturers`** - GPU manufacturers (NVIDIA, AMD, Intel)
- **`architectures`** - GPU architectures (Ampere, RDNA 2, etc.)
- **`categories`** - GPU categories (Consumer, Professional, Mobile)

### Main Tables
- **`gpus`** - Main GPU information table
- **`performance_metrics`** - Benchmark scores and performance data
- **`memory_specs`** - Memory-related specifications
- **`process_technology`** - API support and process information
- **`ai_performance`** - AI-specific performance categories

### Views
- **`gpu_complete_info`** - Comprehensive view joining all tables

## Prerequisites

### Software Requirements
- PostgreSQL 12+ installed and running
- Python 3.7+ with the following packages:
  ```bash
  pip install pandas psycopg2-binary numpy
  ```

### Database Configuration
- **Host**: localhost
- **Database**: AI_BENCHMARK
- **User**: postgres
- **Password**: admin
- **Port**: 5432

## Setup Instructions

### Step 1: Create the Database Schema

1. Open your PostgreSQL client (psql, pgAdmin, or any other client)
2. Connect to your PostgreSQL instance
3. Run the schema creation script:

```sql
\i scripts/database/create_normalized_schema.sql
```

Or using psql command line:
```bash
psql -h localhost -U postgres -d AI_BENCHMARK -f scripts/database/create_normalized_schema.sql
```

### Step 2: Import the Data

1. Ensure the CSV file `data/final/Ai-Benchmark-Final-enhanced-fixed.csv` exists
2. Run the Python import script:

```bash
cd scripts/database
python import_normalized_data.py
```

Or from the project root:
```bash
python scripts/database/import_normalized_data.py
```

## Database Structure Details

### Table Relationships

```
manufacturers (1) ──── (many) architectures
manufacturers (1) ──── (many) gpus
architectures (1) ──── (many) gpus
categories (1) ──── (many) gpus

gpus (1) ──── (1) performance_metrics
gpus (1) ──── (1) memory_specs
gpus (1) ──── (1) process_technology
gpus (1) ──── (1) ai_performance
```

### Key Features

1. **Normalization**: Eliminates data redundancy by separating concerns into different tables
2. **Foreign Keys**: Maintains referential integrity
3. **Indexes**: Optimized for common query patterns
4. **Comprehensive View**: Easy access to complete GPU information through `gpu_complete_info` view
5. **Data Types**: Proper data types for optimal storage and performance
6. **Constraints**: Ensures data quality and consistency

## Sample Queries

### 1. Get all NVIDIA GPUs with their architectures
```sql
SELECT g.gpu_name, m.name as manufacturer, a.name as architecture
FROM gpus g
JOIN manufacturers m ON g.manufacturer_id = m.manufacturer_id
JOIN architectures a ON g.architecture_id = a.architecture_id
WHERE m.name = 'NVIDIA'
ORDER BY g.gpu_name;
```

### 2. Top 10 GPUs by G3D Mark performance
```sql
SELECT gpu_name, manufacturer, g3d_mark
FROM gpu_complete_info
WHERE g3d_mark IS NOT NULL
ORDER BY g3d_mark DESC
LIMIT 10;
```

### 3. Average performance by manufacturer
```sql
SELECT 
    m.name as manufacturer,
    AVG(pm.g3d_mark) as avg_g3d_mark,
    AVG(pm.fp32_final) as avg_fp32_performance,
    COUNT(*) as gpu_count
FROM gpus g
JOIN manufacturers m ON g.manufacturer_id = m.manufacturer_id
JOIN performance_metrics pm ON g.gpu_id = pm.gpu_id
WHERE pm.g3d_mark IS NOT NULL
GROUP BY m.name
ORDER BY avg_g3d_mark DESC;
```

### 4. Memory distribution analysis
```sql
SELECT 
    memory_tier,
    COUNT(*) as gpu_count,
    AVG(memory_gb) as avg_memory_gb
FROM gpu_complete_info
WHERE memory_tier IS NOT NULL
GROUP BY memory_tier
ORDER BY avg_memory_gb DESC;
```

### 5. AI Performance by category
```sql
SELECT 
    ai_performance_category,
    ai_efficiency_tier,
    COUNT(*) as gpu_count,
    AVG(tops_per_watt) as avg_tops_per_watt
FROM gpu_complete_info
WHERE ai_performance_category IS NOT NULL
GROUP BY ai_performance_category, ai_efficiency_tier
ORDER BY avg_tops_per_watt DESC NULLS LAST;
```

## Data Import Process

The import script performs the following operations:

1. **Data Cleaning**: 
   - Handles missing values and 'Unknown' entries
   - Converts data types appropriately
   - Processes scientific notation numbers

2. **Lookup Table Population**:
   - Creates manufacturers, architectures, and categories
   - Maintains referential integrity

3. **Main Data Import**:
   - Imports GPUs with proper foreign key relationships
   - Imports related performance, memory, and technology data

4. **Verification**:
   - Counts records in all tables
   - Verifies view functionality

## Troubleshooting

### Common Issues

1. **Connection Error**:
   - Verify PostgreSQL is running
   - Check database credentials in `import_normalized_data.py`
   - Ensure database `AI_BENCHMARK` exists

2. **Permission Error**:
   - Ensure user has CREATE, INSERT, and SELECT permissions
   - Check if user can create tables and indexes

3. **Missing Dependencies**:
   ```bash
   pip install pandas psycopg2-binary numpy
   ```

4. **CSV File Not Found**:
   - Ensure the CSV file path is correct
   - Check if the file exists in `data/final/Ai-Benchmark-Final-enhanced-fixed.csv`

### Performance Considerations

- The import script uses batch processing (100 records per batch) for efficiency
- Indexes are created after data import for better performance
- The complete view `gpu_complete_info` may be slow on large datasets

## Maintenance

### Updating Data
To update with new CSV data:
1. Drop existing tables (CASCADE will handle dependencies)
2. Run the schema script again
3. Run the import script with the new CSV file

### Backup
```bash
pg_dump -h localhost -U postgres -d AI_BENCHMARK > ai_benchmark_backup.sql
```

### Restore
```bash
psql -h localhost -U postgres -d AI_BENCHMARK < ai_benchmark_backup.sql
```

## Contact

For issues or questions about this database setup, please refer to the project documentation or create an issue in the project repository. 
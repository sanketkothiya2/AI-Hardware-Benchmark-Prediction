-- =====================================================================
-- AI BENCHMARK DATABASE SETUP SCRIPT
-- Purpose: Create database and prepare for schema creation
-- Database: AI_BENCHMARK
-- User: postgres
-- Password: admin
-- =====================================================================

-- Connect as superuser to create database
\c postgres

-- Drop database if exists (for fresh start)
DROP DATABASE IF EXISTS AI_BENCHMARK;

-- Create the AI_BENCHMARK database
CREATE DATABASE AI_BENCHMARK
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'English_Canada.1252'
    LC_CTYPE = 'English_Canada.1252'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Grant all privileges to postgres user
GRANT ALL PRIVILEGES ON DATABASE AI_BENCHMARK TO postgres;

-- Add comment to database
COMMENT ON DATABASE AI_BENCHMARK 
IS 'AI Hardware Benchmarking Database - Comprehensive performance analysis and prediction modeling';

-- Display success message
SELECT 'AI_BENCHMARK database created successfully!' as status;

-- Show database information
SELECT 
    datname as database_name,
    pg_encoding_to_char(encoding) as encoding,
    datcollate as collation,
    datctype as character_type,
    pg_size_pretty(pg_database_size(datname)) as size
FROM pg_database 
WHERE datname = 'AI_BENCHMARK'; 
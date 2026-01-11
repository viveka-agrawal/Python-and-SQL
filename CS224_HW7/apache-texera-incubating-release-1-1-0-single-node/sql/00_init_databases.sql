-- Create the user `texera` with `password` if it doesn't exist
DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT FROM pg_catalog.pg_roles WHERE rolname = 'texera'
        ) THEN
            CREATE ROLE texera LOGIN PASSWORD 'password';
        END IF;
    END
$$;

-- Create all required databases
CREATE DATABASE texera_db;
CREATE DATABASE texera_lakefs;
CREATE DATABASE texera_iceberg_catalog;

-- Grant privileges to texera user
GRANT ALL PRIVILEGES ON DATABASE texera_db TO texera;
GRANT ALL PRIVILEGES ON DATABASE texera_lakefs TO texera;
GRANT ALL PRIVILEGES ON DATABASE texera_iceberg_catalog TO texera;

-- Change ownership
ALTER DATABASE texera_db OWNER TO texera;
ALTER DATABASE texera_lakefs OWNER TO texera;
ALTER DATABASE texera_iceberg_catalog OWNER TO texera;
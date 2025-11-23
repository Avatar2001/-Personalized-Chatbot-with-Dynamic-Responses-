#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/models/db_schemes/myrag/
alembic upgrade head
cd /app

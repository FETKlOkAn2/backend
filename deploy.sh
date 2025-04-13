#!/bin/bash
set -e

# Ensure your environment variables are set
export DB_PATH="s3://your-bucket/database"  # Adjust to your database location

# Deploy with Serverless Framework
echo "Deploying function to AWS..."
serverless deploy --stage dev

echo "Deployment complete!"
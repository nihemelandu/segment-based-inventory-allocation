#!/usr/bin/env python3
"""
Data Loader Script for Google Cloud Platform
Loads data files to GCS bucket and provides BigQuery ingestion options
"""

import os
import argparse
import logging
from typing import Dict, List, Optional

from google.cloud import storage
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A comprehensive data loader for GCP that handles:
    1. Loading data files to Google Cloud Storage
    2. Creating BigQuery datasets and tables
    3. Loading data with schema validation
    """

    def __init__(self, project_id: str, bucket_name: str, dataset_id: str):
        """
        Initialize the DataLoader with GCP configurations
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.dataset_id = dataset_id
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
        # Get or create bucket
        self.bucket = self._get_or_create_bucket()
        
        # Get or create dataset
        self.dataset = self._get_or_create_dataset()

    def _get_or_create_bucket(self) -> storage.Bucket:
        """Get existing bucket or create new one"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            bucket.reload()
            logger.info(f"Using existing bucket: {self.bucket_name}")
            return bucket
        except NotFound:
            bucket = self.storage_client.create_bucket(self.bucket_name)
            logger.info(f"Created new bucket: {self.bucket_name}")
            return bucket

    def _get_or_create_dataset(self) -> bigquery.Dataset:
        """Get existing dataset or create new one"""
        dataset_id = f"{self.project_id}.{self.dataset_id}"
        try:
            dataset = self.bq_client.get_dataset(dataset_id)
            logger.info(f"Using existing dataset: {dataset_id}")
            return dataset
        except NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            logger.info(f"Created new dataset: {dataset_id}")
            return dataset

    def upload_to_gcs(self, local_file_path: str, gcs_blob_name: Optional[str] = None) -> str:
        """
        Upload a local file to Google Cloud Storage
        
        Args:
            local_file_path: Path to local file
            gcs_blob_name: Name for the blob in GCS (optional, uses filename if not provided)
            
        Returns:
            GCS URI of uploaded file
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        if gcs_blob_name is None:
            gcs_blob_name = os.path.basename(local_file_path)
        
        blob = self.bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_blob_name}"
        logger.info(f"Uploaded {local_file_path} to {gcs_uri}")
        
        return gcs_uri

    def load_single_table(self, gcs_uri: str, table_name: str, 
                         file_format: str = "CSV", 
                         write_disposition: str = "WRITE_TRUNCATE") -> str:
        """
        Load data from GCS to a single BigQuery table
        
        Args:
            gcs_uri: GCS URI of the data file
            table_name: Name for the BigQuery table
            file_format: File format (CSV, JSON, PARQUET)
            write_disposition: How to handle existing data
            
        Returns:
            Full table ID
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            autodetect=True,  # Auto-detect schema
        )
        
        # Configure based on file format
        if file_format.upper() == "CSV":
            job_config.source_format = bigquery.SourceFormat.CSV
            job_config.skip_leading_rows = 1  # Skip header row
        elif file_format.upper() == "JSON":
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        elif file_format.upper() == "PARQUET":
            job_config.source_format = bigquery.SourceFormat.PARQUET
        
        load_job = self.bq_client.load_table_from_uri(
            gcs_uri, table_id, job_config=job_config
        )
        
        load_job.result()  # Wait for the job to complete
        
        table = self.bq_client.get_table(table_id)
        logger.info(f"Loaded {table.num_rows} rows into {table_id}")
        
        return table_id

    def create_table_from_schema(self, table_name: str, schema: List[bigquery.SchemaField]) -> str:
        """
        Create a BigQuery table with specified schema
        
        Args:
            table_name: Name for the table
            schema: List of SchemaField objects
            
        Returns:
            Full table ID
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        table = bigquery.Table(table_id, schema=schema)
        
        try:
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Exception as e:
            if "Already Exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise e
        
        return table_id

    def load_with_schema_model(self, data_model_path: str, data_files: Dict[str, str]) -> Dict[str, str]:
        """
        Load data using a predefined data model/schema
        
        Args:
            data_model_path: Path to YAML file containing table schemas
            data_files: Dictionary mapping table names to GCS URIs
            
        Returns:
            Dictionary mapping table names to BigQuery table IDs
        """
        # Load data model
        with open(data_model_path, 'r') as f:
            data_model = yaml.safe_load(f)
        
        table_ids = {}
        
        for table_name, table_config in data_model['tables'].items():
            if table_name not in data_files:
                logger.warning(f"No data file provided for table: {table_name}")
                continue
            
            # Create schema from model
            schema = []
            for field in table_config['schema']:
                field_type = getattr(bigquery.SqlTypeNames, field['type'].upper())
                mode = field.get('mode', 'NULLABLE')
                schema_field = bigquery.SchemaField(
                    field['name'], 
                    field_type, 
                    mode=mode,
                    description=field.get('description', '')
                )
                schema.append(schema_field)
            
            # Create table
            table_id = self.create_table_from_schema(table_name, schema)
            
            # Load data
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE",
            )
            
            # Configure source format
            source_format = table_config.get('source_format', 'CSV')
            if source_format.upper() == "CSV":
                job_config.source_format = bigquery.SourceFormat.CSV
                job_config.skip_leading_rows = 1
            elif source_format.upper() == "JSON":
                job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            
            load_job = self.bq_client.load_table_from_uri(
                data_files[table_name], table_id, job_config=job_config
            )
            
            load_job.result()
            
            table = self.bq_client.get_table(table_id)
            logger.info(f"Loaded {table.num_rows} rows into {table_id}")
            
            table_ids[table_name] = table_id
        
        return table_ids

    def create_sample_data_model(self, output_path: str = "data_model.yaml"):
        """
        Create a sample data model YAML file for customer clustering project
        """
        sample_model = {
            "version": "1.0",
            "description": "Data model for customer clustering and demand forecasting",
            "tables": {
                "customers": {
                    "description": "Customer master data",
                    "source_format": "CSV",
                    "schema": [
                        {"name": "customer_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique customer identifier"},
                        {"name": "customer_name", "type": "STRING", "mode": "NULLABLE", "description": "Customer name"},
                        {"name": "email", "type": "STRING", "mode": "NULLABLE", "description": "Customer email"},
                        {"name": "registration_date", "type": "DATE", "mode": "NULLABLE", "description": "Customer registration date"},
                        {"name": "region", "type": "STRING", "mode": "NULLABLE", "description": "Customer region"},
                        {"name": "customer_segment", "type": "STRING", "mode": "NULLABLE", "description": "Customer segment"}
                    ]
                },
                "transactions": {
                    "description": "Customer transaction history",
                    "source_format": "CSV",
                    "schema": [
                        {"name": "transaction_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique transaction identifier"},
                        {"name": "customer_id", "type": "STRING", "mode": "REQUIRED", "description": "Customer identifier"},
                        {"name": "sku", "type": "STRING", "mode": "REQUIRED", "description": "Product SKU"},
                        {"name": "quantity", "type": "INTEGER", "mode": "REQUIRED", "description": "Quantity purchased"},
                        {"name": "unit_price", "type": "FLOAT", "mode": "REQUIRED", "description": "Unit price"},
                        {"name": "total_amount", "type": "FLOAT", "mode": "REQUIRED", "description": "Total transaction amount"},
                        {"name": "transaction_date", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Transaction timestamp"},
                        {"name": "fulfillment_center", "type": "STRING", "mode": "NULLABLE", "description": "Fulfillment center"}
                    ]
                },
                "products": {
                    "description": "Product catalog",
                    "source_format": "CSV",
                    "schema": [
                        {"name": "sku", "type": "STRING", "mode": "REQUIRED", "description": "Product SKU"},
                        {"name": "product_name", "type": "STRING", "mode": "REQUIRED", "description": "Product name"},
                        {"name": "category", "type": "STRING", "mode": "NULLABLE", "description": "Product category"},
                        {"name": "subcategory", "type": "STRING", "mode": "NULLABLE", "description": "Product subcategory"},
                        {"name": "brand", "type": "STRING", "mode": "NULLABLE", "description": "Product brand"},
                        {"name": "weight", "type": "FLOAT", "mode": "NULLABLE", "description": "Product weight"},
                        {"name": "dimensions", "type": "STRING", "mode": "NULLABLE", "description": "Product dimensions"}
                    ]
                },
                "fulfillment_centers": {
                    "description": "Fulfillment center information",
                    "source_format": "CSV",
                    "schema": [
                        {"name": "center_id", "type": "STRING", "mode": "REQUIRED", "description": "Fulfillment center ID"},
                        {"name": "center_name", "type": "STRING", "mode": "REQUIRED", "description": "Fulfillment center name"},
                        {"name": "region", "type": "STRING", "mode": "REQUIRED", "description": "Region"},
                        {"name": "latitude", "type": "FLOAT", "mode": "NULLABLE", "description": "Latitude"},
                        {"name": "longitude", "type": "FLOAT", "mode": "NULLABLE", "description": "Longitude"},
                        {"name": "capacity", "type": "INTEGER", "mode": "NULLABLE", "description": "Storage capacity"}
                    ]
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_model, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created sample data model at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Data Loader for Google Cloud Platform")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--bucket-name", required=True, help="GCS Bucket name")
    parser.add_argument("--dataset-id", required=True, help="BigQuery dataset ID")
    parser.add_argument("--local-file", help="Local file to upload")
    parser.add_argument("--gcs-blob-name", help="Name for GCS blob (optional)")
    parser.add_argument("--mode", choices=["single", "schema"], required=True, 
                       help="Loading mode: 'single' for single table, 'schema' for data model")
    parser.add_argument("--table-name", help="Table name for single mode")
    parser.add_argument("--file-format", default="CSV", choices=["CSV", "JSON", "PARQUET"],
                       help="File format")
    parser.add_argument("--data-model", help="Path to data model YAML file")
    parser.add_argument("--create-sample-model", action="store_true", 
                       help="Create sample data model file")
    
    args = parser.parse_args()
    
    # Initialize DataLoader
    loader = DataLoader(args.project_id, args.bucket_name, args.dataset_id)
    
    # Create sample model if requested
    if args.create_sample_model:
        loader.create_sample_data_model()
        return
    
    # Upload file to GCS if provided
    gcs_uri = None
    if args.local_file:
        gcs_uri = loader.upload_to_gcs(args.local_file, args.gcs_blob_name)
    
    # Load data based on mode
    if args.mode == "single":
        if not gcs_uri:
            raise ValueError("GCS URI required for single mode")
        if not args.table_name:
            raise ValueError("Table name required for single mode")
        
        table_id = loader.load_single_table(gcs_uri, args.table_name, args.file_format)
        print(f"Data loaded to: {table_id}")
    
    elif args.mode == "schema":
        if not args.data_model:
            raise ValueError("Data model file required for schema mode")
        
        # For demo purposes, assume single file maps to first table in model
        if gcs_uri:
            with open(args.data_model, 'r') as f:
                model = yaml.safe_load(f)
            first_table = list(model['tables'].keys())[0]
            data_files = {first_table: gcs_uri}
        else:
            data_files = {}
        
        table_ids = loader.load_with_schema_model(args.data_model, data_files)
        print("Data loaded to tables:")
        for table, table_id in table_ids.items():
            print(f"  {table}: {table_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data Cleaning and Preprocessing Script
Pulls data from BigQuery, cleans it, and saves cleaned data back to BigQuery
"""

import pandas as pd
from typing import Dict, List, Optional
import argparse
import logging
from datetime import datetime, timedelta

from google.cloud import bigquery
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaning and preprocessing for customer clustering project
    Handles pulling data from BigQuery, cleaning, and saving back
    """

    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize DataCleaner
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_client = bigquery.Client(project=project_id)
        
        # Store cleaned datasets
        self.cleaned_data = {}
        self.cleaning_stats = {}

    def pull_data_from_bigquery(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Pull data from BigQuery table into pandas DataFrame
        
        Args:
            table_name: Name of the BigQuery table
            limit: Optional limit on number of rows to pull
            
        Returns:
            pandas DataFrame with the data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Pulling data from {self.project_id}.{self.dataset_id}.{table_name}")
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Successfully pulled {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error pulling data from {table_name}: {str(e)}")
            raise

    def clean_customers_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customers table
        
        Args:
            df: Raw customers DataFrame
            
        Returns:
            Cleaned customers DataFrame
        """
        logger.info("Cleaning customers table...")
        original_rows = len(df)
        stats = {'original_rows': original_rows}
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Handle missing values
        # Remove rows where customer_id is missing (critical field)
        missing_customer_id = cleaned_df['customer_id'].isnull().sum()
        cleaned_df = cleaned_df.dropna(subset=['customer_id'])
        stats['missing_customer_id_removed'] = missing_customer_id
        
        # Fill missing customer names with 'Unknown'
        missing_names = cleaned_df['customer_name'].isnull().sum()
        cleaned_df['customer_name'].fillna('Unknown', inplace=True)
        stats['missing_names_filled'] = missing_names
        
        # Fill missing regions with 'Unknown'
        missing_regions = cleaned_df['region'].isnull().sum()
        cleaned_df['region'].fillna('Unknown', inplace=True)
        stats['missing_regions_filled'] = missing_regions
        
        # 2. Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'])
        duplicates_removed = initial_rows - len(cleaned_df)
        stats['duplicates_removed'] = duplicates_removed
        
        # 3. Standardize data formats
        # Clean customer_id format (remove whitespace, convert to uppercase)
        cleaned_df['customer_id'] = cleaned_df['customer_id'].astype(str).str.strip().str.upper()
        
        # Standardize region names
        cleaned_df['region'] = cleaned_df['region'].astype(str).str.strip().str.title()
        
        # Clean email addresses
        if 'email' in cleaned_df.columns:
            cleaned_df['email'] = cleaned_df['email'].astype(str).str.strip().str.lower()
            # Remove invalid email formats
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = ~cleaned_df['email'].str.match(email_pattern, na=False)
            cleaned_df.loc[invalid_emails, 'email'] = None
            stats['invalid_emails_cleared'] = invalid_emails.sum()
        
        # 4. Handle registration dates
        if 'registration_date' in cleaned_df.columns:
            # Convert to datetime
            cleaned_df['registration_date'] = pd.to_datetime(cleaned_df['registration_date'], errors='coerce')
            
            # Remove future dates (data quality issue)
            future_dates = cleaned_df['registration_date'] > datetime.now()
            cleaned_df.loc[future_dates, 'registration_date'] = None
            stats['future_dates_cleared'] = future_dates.sum()
        
        stats['final_rows'] = len(cleaned_df)
        stats['rows_removed'] = original_rows - len(cleaned_df)
        
        self.cleaning_stats['customers'] = stats
        logger.info(f"Customers table cleaned: {original_rows} -> {len(cleaned_df)} rows")
        
        return cleaned_df

    def clean_transactions_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean transactions table
        
        Args:
            df: Raw transactions DataFrame
            
        Returns:
            Cleaned transactions DataFrame
        """
        logger.info("Cleaning transactions table...")
        original_rows = len(df)
        stats = {'original_rows': original_rows}
        
        cleaned_df = df.copy()
        
        # 1. Remove rows with missing critical fields
        critical_fields = ['transaction_id', 'customer_id', 'sku', 'quantity', 'unit_price']
        before_critical = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=critical_fields)
        stats['missing_critical_fields_removed'] = before_critical - len(cleaned_df)
        
        # 2. Handle data type conversions
        cleaned_df['quantity'] = pd.to_numeric(cleaned_df['quantity'], errors='coerce')
        cleaned_df['unit_price'] = pd.to_numeric(cleaned_df['unit_price'], errors='coerce')
        cleaned_df['total_amount'] = pd.to_numeric(cleaned_df['total_amount'], errors='coerce')
        
        # Remove rows where numeric conversions failed
        before_numeric = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['quantity', 'unit_price'])
        stats['invalid_numeric_removed'] = before_numeric - len(cleaned_df)
        
        # 3. Remove business rule violations
        # Remove negative quantities or prices
        invalid_business_rules = (
            (cleaned_df['quantity'] <= 0) |
            (cleaned_df['unit_price'] <= 0) |
            (cleaned_df['quantity'] > 1000) |  # Unreasonably high quantity
            (cleaned_df['unit_price'] > 10000)  # Unreasonably high price
        )
        cleaned_df = cleaned_df[~invalid_business_rules]
        stats['business_rule_violations_removed'] = invalid_business_rules.sum()
        
        # 4. Calculate/fix total_amount
        # Recalculate total_amount to ensure consistency
        cleaned_df['total_amount'] = cleaned_df['quantity'] * cleaned_df['unit_price']
        
        # 5. Handle transaction dates
        if 'transaction_date' in cleaned_df.columns:
            cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'], errors='coerce')
            
            # Remove future dates
            future_dates = cleaned_df['transaction_date'] > datetime.now()
            cleaned_df = cleaned_df[~future_dates]
            stats['future_transaction_dates_removed'] = future_dates.sum()
            
            # Remove very old dates (beyond business logic - e.g., before company started)
            cutoff_date = datetime.now() - timedelta(days=3650)  # 10 years
            very_old = cleaned_df['transaction_date'] < cutoff_date
            cleaned_df = cleaned_df[~very_old]
            stats['very_old_transactions_removed'] = very_old.sum()
        
        # 6. Remove duplicate transactions
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['transaction_id'])
        stats['duplicate_transactions_removed'] = initial_rows - len(cleaned_df)
        
        # 7. Standardize text fields
        cleaned_df['customer_id'] = cleaned_df['customer_id'].astype(str).str.strip().str.upper()
        cleaned_df['sku'] = cleaned_df['sku'].astype(str).str.strip().str.upper()
        
        if 'fulfillment_center' in cleaned_df.columns:
            cleaned_df['fulfillment_center'] = cleaned_df['fulfillment_center'].astype(str).str.strip().str.upper()
            # Fill missing fulfillment centers
            missing_fc = cleaned_df['fulfillment_center'].isnull().sum()
            cleaned_df['fulfillment_center'].fillna('UNKNOWN', inplace=True)
            stats['missing_fulfillment_centers_filled'] = missing_fc
        
        stats['final_rows'] = len(cleaned_df)
        stats['rows_removed'] = original_rows - len(cleaned_df)
        
        self.cleaning_stats['transactions'] = stats
        logger.info(f"Transactions table cleaned: {original_rows} -> {len(cleaned_df)} rows")
        
        return cleaned_df

    def clean_products_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean products table
        
        Args:
            df: Raw products DataFrame
            
        Returns:
            Cleaned products DataFrame
        """
        logger.info("Cleaning products table...")
        original_rows = len(df)
        stats = {'original_rows': original_rows}
        
        cleaned_df = df.copy()
        
        # 1. Remove rows with missing SKU (critical field)
        missing_sku = cleaned_df['sku'].isnull().sum()
        cleaned_df = cleaned_df.dropna(subset=['sku'])
        stats['missing_sku_removed'] = missing_sku
        
        # 2. Remove duplicate SKUs
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['sku'])
        stats['duplicate_skus_removed'] = initial_rows - len(cleaned_df)
        
        # 3. Standardize text fields
        cleaned_df['sku'] = cleaned_df['sku'].astype(str).str.strip().str.upper()
        
        # Clean product names
        if 'product_name' in cleaned_df.columns:
            cleaned_df['product_name'] = cleaned_df['product_name'].astype(str).str.strip()
            missing_names = cleaned_df['product_name'].isnull().sum()
            cleaned_df['product_name'].fillna('Unknown Product', inplace=True)
            stats['missing_product_names_filled'] = missing_names
        
        # Standardize categories
        text_fields = ['category', 'subcategory', 'brand']
        for field in text_fields:
            if field in cleaned_df.columns:
                missing_count = cleaned_df[field].isnull().sum()
                cleaned_df[field] = cleaned_df[field].astype(str).str.strip().str.title()
                cleaned_df[field].fillna('Unknown', inplace=True)
                stats[f'missing_{field}_filled'] = missing_count
        
        # 4. Handle numeric fields
        if 'weight' in cleaned_df.columns:
            cleaned_df['weight'] = pd.to_numeric(cleaned_df['weight'], errors='coerce')
            # Remove negative weights
            negative_weights = cleaned_df['weight'] < 0
            cleaned_df.loc[negative_weights, 'weight'] = None
            stats['negative_weights_cleared'] = negative_weights.sum()
        
        stats['final_rows'] = len(cleaned_df)
        stats['rows_removed'] = original_rows - len(cleaned_df)
        
        self.cleaning_stats['products'] = stats
        logger.info(f"Products table cleaned: {original_rows} -> {len(cleaned_df)} rows")
        
        return cleaned_df

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            df: DataFrame
            column: Column name to check for outliers
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)

    def save_cleaned_data_to_bigquery(self, df: pd.DataFrame, table_name: str, 
                                     suffix: str = "_cleaned") -> str:
        """
        Save cleaned DataFrame back to BigQuery
        
        Args:
            df: Cleaned DataFrame
            table_name: Original table name
            suffix: Suffix to add to cleaned table name
            
        Returns:
            Full table ID of saved table
        """
        cleaned_table_name = f"{table_name}{suffix}"
        table_id = f"{self.project_id}.{self.dataset_id}.{cleaned_table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",  # Overwrite existing data
            autodetect=True
        )
        
        logger.info(f"Saving cleaned data to {table_id}")
        
        job = self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for job to complete
        
        logger.info(f"Successfully saved {len(df)} rows to {cleaned_table_name}")
        return table_id

    def generate_cleaning_report(self) -> Dict:
        """
        Generate a comprehensive cleaning report
        
        Returns:
            Dictionary containing cleaning statistics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'tables_processed': list(self.cleaning_stats.keys()),
            'summary': {},
            'detailed_stats': self.cleaning_stats
        }
        
        # Calculate summary statistics
        for table, stats in self.cleaning_stats.items():
            original = stats.get('original_rows', 0)
            final = stats.get('final_rows', 0)
            removed = stats.get('rows_removed', 0)
            
            report['summary'][table] = {
                'original_rows': original,
                'final_rows': final,
                'rows_removed': removed,
                'retention_rate': round((final / original * 100), 2) if original > 0 else 0
            }
        
        return report

    def clean_all_tables(self, tables: List[str], limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Clean all specified tables
        
        Args:
            tables: List of table names to clean
            limit: Optional limit on rows to pull from each table
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned_tables = {}
        
        for table in tables:
            try:
                # Pull data
                df = self.pull_data_from_bigquery(table, limit)
                
                # Clean based on table type
                if table == 'customers':
                    cleaned_df = self.clean_customers_table(df)
                elif table == 'transactions':
                    cleaned_df = self.clean_transactions_table(df)
                elif table == 'products':
                    cleaned_df = self.clean_products_table(df)
                else:
                    logger.warning(f"No specific cleaning logic for table: {table}")
                    cleaned_df = df  # Return as-is
                
                cleaned_tables[table] = cleaned_df
                
                # Save cleaned data back to BigQuery
                self.save_cleaned_data_to_bigquery(cleaned_df, table)
                
            except Exception as e:
                logger.error(f"Error cleaning table {table}: {str(e)}")
                continue
        
        return cleaned_tables


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning and Preprocessing")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--dataset-id", required=True, help="BigQuery dataset ID")
    parser.add_argument("--tables", nargs='+', required=True, 
                       help="Tables to clean (e.g., customers transactions products)")
    parser.add_argument("--limit", type=int, help="Limit rows per table (for testing)")
    parser.add_argument("--output-report", help="Path to save cleaning report (JSON)")
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = DataCleaner(args.project_id, args.dataset_id)
    
    # Clean all tables
    cleaned_data = cleaner.clean_all_tables(args.tables, args.limit)
    
    # Generate and display report
    report = cleaner.generate_cleaning_report()
    
    print("\n" + "="*50)
    print("DATA CLEANING REPORT")
    print("="*50)
    
    for table, summary in report['summary'].items():
        print(f"\n{table.upper()}:")
        print(f"  Original rows: {summary['original_rows']:,}")
        print(f"  Final rows: {summary['final_rows']:,}")
        print(f"  Rows removed: {summary['rows_removed']:,}")
        print(f"  Retention rate: {summary['retention_rate']}%")
    
    # Save report if requested
    if args.output_report:
        import json
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.output_report}")


if __name__ == "__main__":
    main()

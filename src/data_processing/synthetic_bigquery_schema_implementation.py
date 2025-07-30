"""
BigQuery Schema Implementation for Fulfillment Optimization Project
================================================================

This script creates the complete data model schema in BigQuery, including:
- Table creation with proper data types and constraints
- Indexes and clustering for performance optimization
- Data validation rules and business logic
- Synthetic data loading capabilities
- Schema documentation and validation

Prerequisites:
- Google Cloud SDK installed and authenticated
- BigQuery API enabled
- Appropriate IAM permissions (BigQuery Admin or Data Editor)

Usage:
python bigquery_schema_implementation.py --project_id your-project-id --dataset_id fulfillment_data
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BigQuerySchemaImplementation:
    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize BigQuery client and dataset configuration
        
        Args:
            project_id: Google Cloud Project ID
            dataset_id: BigQuery dataset name
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_ref = self.client.dataset(dataset_id)
        
        # Table schemas definition
        self.table_schemas = self._define_table_schemas()
        
    def _define_table_schemas(self) -> Dict[str, List[bigquery.SchemaField]]:
        """Define BigQuery schemas for all tables"""
        
        schemas = {
            'customers': [
                bigquery.SchemaField('customer_id', 'STRING', mode='REQUIRED', 
                                    description='Unique customer identifier, format: C######'),
                bigquery.SchemaField('registration_date', 'DATE', mode='REQUIRED',
                                    description='Account creation date (UTC)'),
                bigquery.SchemaField('customer_type', 'STRING', mode='REQUIRED',
                                    description='Customer classification: individual, business, wholesale'),
                bigquery.SchemaField('city', 'STRING', mode='NULLABLE',
                                    description='Primary shipping city (may be NULL for privacy)'),
                bigquery.SchemaField('state', 'STRING', mode='NULLABLE',
                                    description='State/Province code (US: 2-char, International: varies)'),
                bigquery.SchemaField('zip_code', 'STRING', mode='NULLABLE',
                                    description='Primary ZIP/Postal code'),
                bigquery.SchemaField('customer_segment', 'STRING', mode='REQUIRED',
                                    description='ML-derived behavior segment: high_frequency, medium_frequency, low_frequency, seasonal'),
                bigquery.SchemaField('total_orders', 'INTEGER', mode='REQUIRED',
                                    description='Count of completed orders (updated nightly, excludes cancelled)')
            ],
            
            'products': [
                bigquery.SchemaField('sku_id', 'STRING', mode='REQUIRED',
                                    description='Unique product identifier, format: SKU######'),
                bigquery.SchemaField('product_name', 'STRING', mode='REQUIRED',
                                    description='Product display name'),
                bigquery.SchemaField('category', 'STRING', mode='REQUIRED',
                                    description='Primary product category: Electronics, Clothing, Home & Garden, Sports, Books'),
                bigquery.SchemaField('subcategory', 'STRING', mode='NULLABLE',
                                    description='Product subcategory (varies by category)'),
                bigquery.SchemaField('brand', 'STRING', mode='NULLABLE',
                                    description='Brand/Manufacturer name (may be NULL for generic)'),
                bigquery.SchemaField('price', 'NUMERIC', mode='REQUIRED',
                                    description='Current selling price in USD (must be positive)'),
                bigquery.SchemaField('weight_lbs', 'NUMERIC', mode='REQUIRED',
                                    description='Product weight in pounds (for shipping calculations)'),
                bigquery.SchemaField('dimensions_cubic_in', 'NUMERIC', mode='REQUIRED',
                                    description='Volume in cubic inches (for storage optimization)'),
                bigquery.SchemaField('launch_date', 'DATE', mode='REQUIRED',
                                    description='Product introduction date'),
                bigquery.SchemaField('seasonality_factor', 'NUMERIC', mode='REQUIRED',
                                    description='Seasonal demand multiplier (0.5-2.0, used for forecasting)'),
                bigquery.SchemaField('base_demand', 'INTEGER', mode='REQUIRED',
                                    description='Weekly baseline demand units (historical average)')
            ],
            
            'fulfillment_centers': [
                bigquery.SchemaField('fc_id', 'STRING', mode='REQUIRED',
                                    description='Unique FC identifier, format: FC_SS## (SS=State code)'),
                bigquery.SchemaField('fc_name', 'STRING', mode='REQUIRED',
                                    description='Facility display name'),
                bigquery.SchemaField('city', 'STRING', mode='REQUIRED',
                                    description='FC city location'),
                bigquery.SchemaField('state', 'STRING', mode='REQUIRED',
                                    description='FC state/province (2-character US state codes)'),
                bigquery.SchemaField('zip_code', 'STRING', mode='REQUIRED',
                                    description='FC ZIP code (for distance calculations)'),
                bigquery.SchemaField('total_capacity', 'INTEGER', mode='REQUIRED',
                                    description='Maximum storage units (physical capacity limit)'),
                bigquery.SchemaField('current_utilization', 'NUMERIC', mode='REQUIRED',
                                    description='Capacity utilization ratio (0.0-1.0, updated weekly)'),
                bigquery.SchemaField('storage_cost_per_unit', 'NUMERIC', mode='REQUIRED',
                                    description='Monthly storage cost per unit in USD'),
                bigquery.SchemaField('labor_cost_per_hour', 'NUMERIC', mode='REQUIRED',
                                    description='Average hourly labor rate in USD'),
                bigquery.SchemaField('operational_since', 'DATE', mode='REQUIRED',
                                    description='FC opening date')
            ],
            
            'orders': [
                bigquery.SchemaField('order_id', 'STRING', mode='REQUIRED',
                                    description='Unique order identifier, format: ORD########'),
                bigquery.SchemaField('customer_id', 'STRING', mode='REQUIRED',
                                    description='References customers.customer_id'),
                bigquery.SchemaField('order_date', 'DATE', mode='REQUIRED',
                                    description='Order placement date (UTC timezone)'),
                bigquery.SchemaField('order_total', 'NUMERIC', mode='REQUIRED',
                                    description='Total order value in USD (sum of line items, excludes shipping)'),
                bigquery.SchemaField('shipping_cost', 'NUMERIC', mode='REQUIRED',
                                    description='Shipping charges in USD (0.00 for free shipping)'),
                bigquery.SchemaField('shipping_method', 'STRING', mode='REQUIRED',
                                    description='Delivery method: standard, expedited, standard_free'),
                bigquery.SchemaField('fulfillment_center', 'STRING', mode='NULLABLE',
                                    description='Assigned FC for fulfillment (NULL = pending assignment)'),
                bigquery.SchemaField('order_status', 'STRING', mode='REQUIRED',
                                    description='Current order state: pending, processing, shipped, delivered, cancelled'),
                bigquery.SchemaField('total_weight', 'NUMERIC', mode='REQUIRED',
                                    description='Total order weight in pounds (sum of item weights × quantities)'),
                bigquery.SchemaField('customer_zip', 'STRING', mode='REQUIRED',
                                    description='Delivery ZIP code (for shipping cost calculation)')
            ],
            
            'order_items': [
                bigquery.SchemaField('order_id', 'STRING', mode='REQUIRED',
                                    description='References orders.order_id'),
                bigquery.SchemaField('sku_id', 'STRING', mode='REQUIRED',
                                    description='References products.sku_id'),
                bigquery.SchemaField('quantity', 'INTEGER', mode='REQUIRED',
                                    description='Units ordered (must be positive)'),
                bigquery.SchemaField('unit_price', 'NUMERIC', mode='REQUIRED',
                                    description='Price per unit at order time (historical pricing)'),
                bigquery.SchemaField('line_total', 'NUMERIC', mode='REQUIRED',
                                    description='Extended line amount (quantity × unit_price)')
            ],
            
            'inventory_current': [
                bigquery.SchemaField('fc_id', 'STRING', mode='REQUIRED',
                                    description='References fulfillment_centers.fc_id'),
                bigquery.SchemaField('sku_id', 'STRING', mode='REQUIRED',
                                    description='References products.sku_id'),
                bigquery.SchemaField('on_hand_qty', 'INTEGER', mode='REQUIRED',
                                    description='Physical inventory count (never negative)'),
                bigquery.SchemaField('committed_qty', 'INTEGER', mode='REQUIRED',
                                    description='Reserved for pending orders (subset of on_hand_qty)'),
                bigquery.SchemaField('available_qty', 'INTEGER', mode='REQUIRED',
                                    description='Available for sale (on_hand_qty - committed_qty)'),
                bigquery.SchemaField('reorder_point', 'INTEGER', mode='REQUIRED',
                                    description='Reorder trigger level (when to replenish stock)'),
                bigquery.SchemaField('max_stock_level', 'INTEGER', mode='REQUIRED',
                                    description='Maximum inventory target (storage capacity constraint)'),
                bigquery.SchemaField('last_updated', 'TIMESTAMP', mode='REQUIRED',
                                    description='Last inventory update timestamp (for freshness tracking)')
            ],
            
            'inventory_movements': [
                bigquery.SchemaField('movement_id', 'STRING', mode='REQUIRED',
                                    description='Unique movement identifier, format: MOV######'),
                bigquery.SchemaField('fc_id', 'STRING', mode='REQUIRED',
                                    description='Source/destination FC (references fulfillment_centers.fc_id)'),
                bigquery.SchemaField('sku_id', 'STRING', mode='REQUIRED',
                                    description='Product moved (references products.sku_id)'),
                bigquery.SchemaField('movement_type', 'STRING', mode='REQUIRED',
                                    description='Type: receipt, shipment, transfer_in, transfer_out, adjustment'),
                bigquery.SchemaField('quantity', 'INTEGER', mode='REQUIRED',
                                    description='Units moved (positive=increase, negative=decrease)'),
                bigquery.SchemaField('movement_date', 'DATE', mode='REQUIRED',
                                    description='Transaction date (UTC timezone)'),
                bigquery.SchemaField('reference_id', 'STRING', mode='NULLABLE',
                                    description='Related transaction ID (order ID, PO number, etc.)')
            ],
            
            'shipping_costs': [
                bigquery.SchemaField('fc_id', 'STRING', mode='REQUIRED',
                                    description='Origin FC (references fulfillment_centers.fc_id)'),
                bigquery.SchemaField('destination_zip', 'STRING', mode='REQUIRED',
                                    description='Delivery ZIP code (3-digit or 5-digit)'),
                bigquery.SchemaField('standard_cost', 'NUMERIC', mode='REQUIRED',
                                    description='Standard shipping rate in USD (base delivery method)'),
                bigquery.SchemaField('expedited_cost', 'NUMERIC', mode='REQUIRED',
                                    description='Expedited shipping rate in USD (faster delivery)'),
                bigquery.SchemaField('avg_transit_days', 'INTEGER', mode='REQUIRED',
                                    description='Average delivery time in business days (standard shipping)'),
                bigquery.SchemaField('carrier', 'STRING', mode='REQUIRED',
                                    description='Primary carrier: UPS, FedEx, USPS')
            ]
        }
        
        return schemas
    
    def create_dataset(self, location: str = 'US', description: str = None) -> None:
        """
        Create BigQuery dataset if it doesn't exist
        
        Args:
            location: Dataset location (default: 'US')
            description: Dataset description
        """
        try:
            dataset = self.client.get_dataset(self.dataset_ref)
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = location
            dataset.description = description or f"Fulfillment Optimization ML Dataset - Created {datetime.now().isoformat()}"
            
            # Set dataset labels for organization
            dataset.labels = {
                'project_type': 'ml_fulfillment',
                'environment': 'development',
                'created_by': 'data_science_team'
            }
            
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self.dataset_id}")
    
    def create_table(self, table_name: str, schema: List[bigquery.SchemaField], 
                    clustering_fields: List[str] = None, partition_field: str = None) -> None:
        """
        Create a BigQuery table with schema and optimizations
        
        Args:
            table_name: Name of the table to create
            schema: List of SchemaField objects
            clustering_fields: Fields to cluster on for performance
            partition_field: Field to partition on (usually date/timestamp)
        """
        table_ref = self.dataset_ref.table(table_name)
        
        try:
            table = self.client.get_table(table_ref)
            logger.info(f"Table {table_name} already exists")
            return
        except NotFound:
            pass
        
        # Create table with schema
        table = bigquery.Table(table_ref, schema=schema)
        
        # Add partitioning if specified
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field
            )
            logger.info(f"Table {table_name} will be partitioned by {partition_field}")
        
        # Add clustering if specified
        if clustering_fields:
            table.clustering_fields = clustering_fields
            logger.info(f"Table {table_name} will be clustered by {clustering_fields}")
        
        # Set table description and labels
        table.description = f"Fulfillment optimization data - {table_name} table"
        table.labels = {
            'table_type': 'ml_dataset',
            'business_domain': 'fulfillment'
        }
        
        # Create the table
        table = self.client.create_table(table, timeout=30)
        logger.info(f"Created table {table_name} with {len(schema)} columns")
    
    def create_all_tables(self) -> None:
        """Create all tables with appropriate optimizations"""
        
        # Table creation order (respects dependencies)
        table_configs = {
            'customers': {
                'clustering_fields': ['customer_segment', 'state'],
                'partition_field': None
            },
            'products': {
                'clustering_fields': ['category', 'brand'],
                'partition_field': None
            },
            'fulfillment_centers': {
                'clustering_fields': ['state'],
                'partition_field': None
            },
            'orders': {
                'clustering_fields': ['customer_id', 'fulfillment_center'],
                'partition_field': 'order_date'
            },
            'order_items': {
                'clustering_fields': ['sku_id'],
                'partition_field': None
            },
            'inventory_current': {
                'clustering_fields': ['fc_id', 'sku_id'],
                'partition_field': None
            },
            'inventory_movements': {
                'clustering_fields': ['fc_id', 'sku_id'],
                'partition_field': 'movement_date'
            },
            'shipping_costs': {
                'clustering_fields': ['fc_id'],
                'partition_field': None
            }
        }
        
        logger.info("Creating all tables...")
        for table_name, config in table_configs.items():
            schema = self.table_schemas[table_name]
            self.create_table(
                table_name=table_name,
                schema=schema,
                clustering_fields=config['clustering_fields'],
                partition_field=config['partition_field']
            )
    
    def create_views(self) -> None:
        """Create useful analytical views"""
        
        views = {
            'customer_summary': """
            SELECT 
                c.customer_id,
                c.customer_type,
                c.customer_segment,
                c.state,
                c.total_orders,
                COUNT(o.order_id) as actual_orders,
                SUM(o.order_total) as total_spent,
                AVG(o.order_total) as avg_order_value,
                MIN(o.order_date) as first_order_date,
                MAX(o.order_date) as last_order_date,
                DATE_DIFF(MAX(o.order_date), MIN(o.order_date), DAY) as customer_lifetime_days
            FROM `{project}.{dataset}.customers` c
            LEFT JOIN `{project}.{dataset}.orders` o ON c.customer_id = o.customer_id
            WHERE o.order_status != 'cancelled'
            GROUP BY c.customer_id, c.customer_type, c.customer_segment, c.state, c.total_orders
            """,
            
            'product_performance': """
            SELECT 
                p.sku_id,
                p.product_name,
                p.category,
                p.brand,
                p.price,
                COUNT(oi.order_id) as total_orders,
                SUM(oi.quantity) as total_units_sold,
                SUM(oi.line_total) as total_revenue,
                AVG(oi.unit_price) as avg_selling_price,
                p.price - AVG(oi.unit_price) as price_variance
            FROM `{project}.{dataset}.products` p
            LEFT JOIN `{project}.{dataset}.order_items` oi ON p.sku_id = oi.sku_id
            GROUP BY p.sku_id, p.product_name, p.category, p.brand, p.price
            """,
            
            'inventory_health': """
            SELECT 
                ic.fc_id,
                fc.fc_name,
                ic.sku_id,
                p.category,
                ic.on_hand_qty,
                ic.available_qty,
                ic.reorder_point,
                CASE 
                    WHEN ic.available_qty <= ic.reorder_point THEN 'REORDER_NEEDED'
                    WHEN ic.available_qty = 0 THEN 'STOCKOUT'
                    WHEN ic.available_qty > ic.max_stock_level THEN 'OVERSTOCK'
                    ELSE 'HEALTHY'
                END as inventory_status,
                ic.last_updated
            FROM `{project}.{dataset}.inventory_current` ic
            JOIN `{project}.{dataset}.fulfillment_centers` fc ON ic.fc_id = fc.fc_id
            JOIN `{project}.{dataset}.products` p ON ic.sku_id = p.sku_id
            """,
            
            'shipping_analysis': """
            SELECT 
                o.fulfillment_center,
                LEFT(o.customer_zip, 3) as zip_3digit,
                COUNT(*) as shipment_count,
                AVG(o.shipping_cost) as avg_shipping_cost,
                AVG(o.total_weight) as avg_weight,
                AVG(sc.avg_transit_days) as avg_transit_days
            FROM `{project}.{dataset}.orders` o
            LEFT JOIN `{project}.{dataset}.shipping_costs` sc 
                ON o.fulfillment_center = sc.fc_id 
                AND LEFT(o.customer_zip, 5) = sc.destination_zip
            WHERE o.order_status IN ('shipped', 'delivered')
                AND o.fulfillment_center IS NOT NULL
            GROUP BY o.fulfillment_center, LEFT(o.customer_zip, 3)
            """
        }
        
        logger.info("Creating analytical views...")
        for view_name, view_sql in views.items():
            formatted_sql = view_sql.format(
                project=self.project_id,
                dataset=self.dataset_id
            )
            
            view_ref = self.dataset_ref.table(view_name)
            view = bigquery.Table(view_ref)
            view.view_query = formatted_sql
            
            try:
                view = self.client.create_table(view)
                logger.info(f"Created view: {view_name}")
            except Conflict:
                # Update existing view
                view = self.client.update_table(view, ['view_query'])
                logger.info(f"Updated view: {view_name}")
    
    def load_synthetic_data(self, data_directory: str = 'synthetic_data') -> None:
        """
        Load synthetic data from CSV files into BigQuery tables
        
        Args:
            data_directory: Directory containing CSV files
        """
        if not os.path.exists(data_directory):
            logger.error(f"Data directory {data_directory} not found")
            return
        
        # Define load order (respects foreign key dependencies)
        load_order = [
            'customers', 'products', 'fulfillment_centers',
            'orders', 'order_items', 'inventory_current', 
            'inventory_movements', 'shipping_costs'
        ]
        
        logger.info(f"Loading synthetic data from {data_directory}...")
        
        for table_name in load_order:
            csv_file = os.path.join(data_directory, f'{table_name}.csv')
            
            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                continue
            
            # Read CSV to validate data
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loading {len(df)} rows into {table_name}")
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
                continue
            
            # Configure load job
            table_ref = self.dataset_ref.table(table_name)
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,  # Skip header row
                autodetect=False,  # Use predefined schema
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Replace data
                schema=self.table_schemas[table_name]
            )
            
            # Load data
            try:
                with open(csv_file, 'rb') as source_file:
                    job = self.client.load_table_from_file(
                        source_file, table_ref, job_config=job_config
                    )
                
                job.result()  # Wait for completion
                logger.info(f"Successfully loaded {table_name}: {job.output_rows} rows")
                
            except Exception as e:
                logger.error(f"Error loading {table_name}: {e}")
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Run data quality checks on loaded data"""
        
        validation_queries = {
            'row_counts': """
            SELECT 
                'customers' as table_name, COUNT(*) as row_count FROM `{project}.{dataset}.customers`
            UNION ALL
            SELECT 'products', COUNT(*) FROM `{project}.{dataset}.products`
            UNION ALL  
            SELECT 'orders', COUNT(*) FROM `{project}.{dataset}.orders`
            UNION ALL
            SELECT 'order_items', COUNT(*) FROM `{project}.{dataset}.order_items`
            ORDER BY table_name
            """,
            
            'data_integrity': """
            SELECT 
                'customers_with_orders' as check_name,
                COUNT(DISTINCT c.customer_id) as count
            FROM `{project}.{dataset}.customers` c
            JOIN `{project}.{dataset}.orders` o ON c.customer_id = o.customer_id
            
            UNION ALL
            
            SELECT 
                'orders_with_valid_customers',
                COUNT(*)
            FROM `{project}.{dataset}.orders` o
            JOIN `{project}.{dataset}.customers` c ON o.customer_id = c.customer_id
            
            UNION ALL
            
            SELECT 
                'order_items_with_valid_products',
                COUNT(*)
            FROM `{project}.{dataset}.order_items` oi  
            JOIN `{project}.{dataset}.products` p ON oi.sku_id = p.sku_id
            """,
            
            'business_rules': """
            SELECT 
                'free_shipping_orders' as rule_name,
                COUNT(*) as count
            FROM `{project}.{dataset}.orders` 
            WHERE shipping_cost = 0 AND (order_total >= 75 OR customer_id IN (
                SELECT customer_id FROM `{project}.{dataset}.customers` 
                WHERE customer_type = 'business'
            ))
            
            UNION ALL
            
            SELECT 
                'inventory_availability_consistency',
                COUNT(*)
            FROM `{project}.{dataset}.inventory_current`
            WHERE available_qty = on_hand_qty - committed_qty
            """
        }
        
        logger.info("Running data quality validation...")
        results = {}
        
        for check_name, query in validation_queries.items():
            formatted_query = query.format(project=self.project_id, dataset=self.dataset_id)
            
            try:
                query_job = self.client.query(formatted_query)
                results[check_name] = [dict(row) for row in query_job]
                logger.info(f"Validation check '{check_name}' completed")
            except Exception as e:
                logger.error(f"Validation check '{check_name}' failed: {e}")
                results[check_name] = {'error': str(e)}
        
        return results
    
    def generate_documentation(self) -> str:
        """Generate dataset documentation"""
        
        doc = f"""
# BigQuery Dataset Documentation: {self.dataset_id}

**Project:** {self.project_id}  
**Created:** {datetime.now().isoformat()}  
**Purpose:** Fulfillment Optimization ML Dataset

## Tables Created:

"""
        
        for table_name, schema in self.table_schemas.items():
            doc += f"### {table_name.upper()}\n"
            doc += f"**Columns:** {len(schema)}\n\n"
            
            for field in schema:
                doc += f"- **{field.name}** ({field.field_type}): {field.description}\n"
            doc += "\n"
        
        doc += """
## Views Created:
- **customer_summary**: Customer analytics with order history
- **product_performance**: Product sales metrics
- **inventory_health**: Current inventory status
- **shipping_analysis**: Shipping cost and performance analysis

## Usage Examples:

```sql
-- Customer segmentation analysis
SELECT customer_segment, COUNT(*), AVG(total_spent)
FROM `{project}.{dataset}.customer_summary`
GROUP BY customer_segment;

-- Inventory reorder alerts
SELECT fc_name, COUNT(*) as items_need_reorder
FROM `{project}.{dataset}.inventory_health`
WHERE inventory_status = 'REORDER_NEEDED'
GROUP BY fc_name;
```

## Next Steps:
1. Run EDA queries to understand data patterns
2. Build customer clustering models
3. Develop demand forecasting pipelines
4. Create fulfillment optimization algorithms
""".format(project=self.project_id, dataset=self.dataset_id)
        
        return doc

def main():
    parser = argparse.ArgumentParser(description='Implement Fulfillment Optimization schema in BigQuery')
    parser.add_argument('--project_id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--dataset_id', default='fulfillment_optimization', help='BigQuery dataset name')
    parser.add_argument('--location', default='US', help='Dataset location')
    parser.add_argument('--load_data', action='store_true', help='Load synthetic data from CSV files')
    parser.add_argument('--data_dir', default='synthetic_data', help='Directory containing CSV files')
    parser.add_argument('--validate', action='store_true', help='Run data quality validation')
    
    args = parser.parse_args()
    
    # Initialize schema implementation
    schema_impl = BigQuerySchemaImplementation(args.project_id, args.dataset_id)
    
    try:
        # Create dataset
        logger.info(f"Creating dataset {args.dataset_id} in project {args.project_id}")
        schema_impl.create_dataset(location=args.location)
        
        # Create tables
        schema_impl.create_all_tables()
        
        # Create views
        #schema_impl.create_views()
        
        # Load synthetic data if requested
        if args.load_data:
            schema_impl.load_synthetic_data(args.data_dir)
        
        # Run validation if requested
        if args.validate:
            validation_results = schema_impl.validate_data_quality()
            logger.info("Data quality validation results:")
            print(json.dumps(validation_results, indent=2, default=str))
        
        # Generate documentation
        documentation = schema_impl.generate_documentation()
        doc_file = f"{args.dataset_id}_documentation.md"
        with open(doc_file, 'w') as f:
            f.write(documentation)
        logger.info(f"Documentation saved to {doc_file}")
        
        logger.info("Schema implementation completed successfully!")
        
    except Exception as e:
        logger.error(f"Schema implementation failed: {e}")
        raise

if __name__ == "__main__":
    main()

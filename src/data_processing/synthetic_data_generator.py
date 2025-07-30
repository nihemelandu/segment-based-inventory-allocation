"""
Synthetic Data Generator for Fulfillment Optimization Project
============================================================

This script generates realistic synthetic data that mirrors the structure and patterns
expected from real e-commerce and fulfillment systems. The data includes realistic
business logic, seasonal patterns, and data quality issues found in real systems.

Tables Generated:
1. customers - Customer master data
2. products - Product/SKU catalog 
3. fulfillment_centers - FC locations and capabilities
4. orders - Customer orders
5. order_items - Line items for each order
6. inventory - Current and historical inventory levels
7. shipping_costs - Cost matrix between FCs and ZIP codes
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from faker import Faker
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

class FulfillmentDataGenerator:
    def __init__(self, base_date=datetime(2022, 1, 1)):
        self.base_date = base_date
        self.end_date = datetime(2024, 12, 31)
        
        # Business parameters
        self.seasonal_multipliers = {
            'Q1': 0.8,  # Jan-Mar: Post-holiday slowdown
            'Q2': 0.9,  # Apr-Jun: Spring increase
            'Q3': 1.0,  # Jul-Sep: Summer steady
            'Q4': 1.4   # Oct-Dec: Holiday surge
        }
        
        # Product categories with different demand patterns
        self.product_categories = {
            'Electronics': {'base_demand': 100, 'seasonality': 1.2, 'avg_price': 250},
            'Clothing': {'base_demand': 150, 'seasonality': 1.5, 'avg_price': 45},
            'Home & Garden': {'base_demand': 80, 'seasonality': 1.1, 'avg_price': 85},
            'Sports': {'base_demand': 60, 'seasonality': 1.3, 'avg_price': 120},
            'Books': {'base_demand': 40, 'seasonality': 0.9, 'avg_price': 15}
        }
        
        # Customer behavior segments
        self.customer_segments = {
            'high_frequency': {'order_rate': 24, 'avg_order_value': 180, 'price_sensitivity': 0.3},
            'medium_frequency': {'order_rate': 8, 'avg_order_value': 95, 'price_sensitivity': 0.6},
            'low_frequency': {'order_rate': 2, 'avg_order_value': 65, 'price_sensitivity': 0.8},
            'seasonal': {'order_rate': 4, 'avg_order_value': 120, 'price_sensitivity': 0.4}
        }
        
    def generate_customers(self, n_customers=10000):
        """Generate customer master data with realistic demographics"""
        print(f"Generating {n_customers} customers...")
        
        customers = []
        us_states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        customer_types = ['individual', 'business', 'wholesale']
        
        for i in range(n_customers):
            # Assign customer to behavioral segment
            segment = np.random.choice(list(self.customer_segments.keys()), 
                                     p=[0.15, 0.45, 0.30, 0.10])
            
            # Generate registration date (weighted toward more recent)
            days_ago = np.random.exponential(365)  # Exponential distribution
            reg_date = self.end_date - timedelta(days=min(days_ago, 1095))  # Max 3 years
            
            customer = {
                'customer_id': f'C{i+1:06d}',
                'registration_date': reg_date.date(),
                'customer_type': np.random.choice(customer_types, p=[0.85, 0.12, 0.03]),
                'city': fake.city(),
                'state': np.random.choice(us_states),
                'zip_code': fake.zipcode(),
                'customer_segment': segment,
                # Add some data quality issues
                'total_orders': None if random.random() < 0.02 else 0,  # 2% missing initially
            }
            customers.append(customer)
            
        return pd.DataFrame(customers)
    
    def generate_products(self, n_products=2000):
        """Generate product catalog with categories and attributes"""
        print(f"Generating {n_products} products...")
        
        products = []
        
        for i in range(n_products):
            category = np.random.choice(list(self.product_categories.keys()))
            category_info = self.product_categories[category]
            
            # Generate price with some variation
            base_price = category_info['avg_price']
            price = np.random.gamma(2, base_price/2)  # Gamma distribution for realistic price spread
            
            product = {
                'sku_id': f'SKU{i+1:06d}',
                'product_name': fake.catch_phrase(),
                'category': category,
                'subcategory': fake.word().title(),
                'brand': fake.company(),
                'price': round(price, 2),
                'weight_lbs': np.random.gamma(2, 1),  # Most items light, some heavy
                'dimensions_cubic_in': np.random.gamma(3, 50),
                'launch_date': fake.date_between(start_date='-2y', end_date='today'),
                # Seasonality indicator for demand forecasting
                'seasonality_factor': category_info['seasonality'],
                'base_demand': category_info['base_demand']
            }
            products.append(product)
            
        return pd.DataFrame(products)
    
    def generate_fulfillment_centers(self):
        """Generate fulfillment center master data"""
        print("Generating fulfillment centers...")
        
        # Strategic locations for realistic coverage
        fc_locations = [
            {'name': 'West Coast Hub', 'city': 'Los Angeles', 'state': 'CA', 'capacity': 100000},
            {'name': 'Pacific Northwest', 'city': 'Seattle', 'state': 'WA', 'capacity': 75000},
            {'name': 'Southwest Hub', 'city': 'Dallas', 'state': 'TX', 'capacity': 85000},
            {'name': 'Southeast Hub', 'city': 'Atlanta', 'state': 'GA', 'capacity': 90000},
            {'name': 'Northeast Hub', 'city': 'Newark', 'state': 'NJ', 'capacity': 120000},
            {'name': 'Midwest Hub', 'city': 'Chicago', 'state': 'IL', 'capacity': 95000},
            {'name': 'Florida Hub', 'city': 'Miami', 'state': 'FL', 'capacity': 70000}
        ]
        
        fulfillment_centers = []
        for i, fc in enumerate(fc_locations):
            center = {
                'fc_id': f'FC_{fc["state"]}{i+1:02d}',
                'fc_name': fc['name'],
                'city': fc['city'],
                'state': fc['state'],
                'zip_code': fake.zipcode(),
                'total_capacity': fc['capacity'],
                'current_utilization': np.random.uniform(0.6, 0.9),  # 60-90% utilized
                'storage_cost_per_unit': np.random.uniform(0.50, 1.20),  # Varies by location
                'labor_cost_per_hour': np.random.uniform(15, 25),
                'operational_since': fake.date_between(start_date='-5y', end_date='-1y')
            }
            fulfillment_centers.append(center)
            
        return pd.DataFrame(fulfillment_centers)
    
    def generate_orders_and_items(self, customers_df, products_df, fc_df, n_orders=50000):
        """Generate realistic orders with seasonal patterns and customer behavior"""
        print(f"Generating {n_orders} orders with line items...")
        
        orders = []
        order_items = []
        order_id_counter = 1
        
        # Create date range with proper weighting
        total_days = (self.end_date - self.base_date).days
        dates = pd.date_range(start=self.base_date, end=self.end_date, freq='D')
        
        for _ in range(n_orders):
            # Select customer based on segment behavior
            customer = customers_df.sample(1).iloc[0]
            segment_info = self.customer_segments[customer['customer_segment']]
            
            # Generate order date with seasonal patterns
            random_date = pd.Timestamp(np.random.choice(dates))
            quarter = f'Q{(random_date.month - 1) // 3 + 1}'
            seasonal_boost = self.seasonal_multipliers[quarter]
            
            # Skip this order if it's before customer registration
            if random_date.date() < customer['registration_date']:
                continue
                
            # Generate order details
            order_id = f'ORD{order_id_counter:08d}'
            order_id_counter += 1
            
            # Number of items in order (most orders are small)
            n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            order_total = 0
            order_weight = 0
            
            # Select fulfillment center (geographic preference)
            if customer['state'] in ['CA', 'WA', 'OR']:
                preferred_fcs = fc_df[fc_df['state'].isin(['CA', 'WA'])]['fc_id'].tolist()
            elif customer['state'] in ['NY', 'NJ', 'CT', 'MA']:
                preferred_fcs = fc_df[fc_df['state'] == 'NJ']['fc_id'].tolist()
            elif customer['state'] in ['TX', 'AZ', 'NM']:
                preferred_fcs = fc_df[fc_df['state'] == 'TX']['fc_id'].tolist()
            else:
                preferred_fcs = fc_df['fc_id'].tolist()
            
            assigned_fc = np.random.choice(preferred_fcs)
            
            # Generate line items
            selected_products = products_df.sample(n_items)
            
            for idx, product in selected_products.iterrows():
                quantity = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
                item_total = product['price'] * quantity
                order_total += item_total
                order_weight += product['weight_lbs'] * quantity
                
                order_item = {
                    'order_id': order_id,
                    'sku_id': product['sku_id'],
                    'quantity': quantity,
                    'unit_price': product['price'],
                    'line_total': item_total
                }
                order_items.append(order_item)
            
            # Apply seasonal and segment adjustments to order value
            adjusted_total = order_total * seasonal_boost
            
            # Determine shipping method based on order value and customer type
            if adjusted_total > 75 or customer['customer_type'] == 'business':
                shipping_method = 'standard_free'
                shipping_cost = 0
            else:
                shipping_method = np.random.choice(['standard', 'expedited'], p=[0.8, 0.2])
                shipping_cost = 8.99 if shipping_method == 'standard' else 15.99
            
            # Order status (most are delivered, some recent ones processing)
            days_since_order = (self.end_date - random_date).days
            if days_since_order < 3:
                status = np.random.choice(['pending', 'processing', 'shipped'], p=[0.3, 0.4, 0.3])
            elif days_since_order < 7:
                status = np.random.choice(['shipped', 'delivered'], p=[0.3, 0.7])
            else:
                status = np.random.choice(['delivered', 'cancelled'], p=[0.95, 0.05])
            
            order = {
                'order_id': order_id,
                'customer_id': customer['customer_id'],
                'order_date': random_date.date(),
                'order_total': round(adjusted_total, 2),
                'shipping_cost': shipping_cost,
                'shipping_method': shipping_method,
                'fulfillment_center': assigned_fc,
                'order_status': status,
                'total_weight': round(order_weight, 2),
                'customer_zip': customer['zip_code']
            }
            orders.append(order)
        
        return pd.DataFrame(orders), pd.DataFrame(order_items)
    
    def generate_inventory_data(self, products_df, fc_df):
        """Generate current inventory levels and historical movements"""
        print("Generating inventory data...")
        
        inventory_current = []
        inventory_movements = []
        
        # Current inventory snapshot
        for _, fc in fc_df.iterrows():
            # Each FC stocks about 60-80% of all products
            stocked_products = products_df.sample(frac=np.random.uniform(0.6, 0.8))
            
            for _, product in stocked_products.iterrows():
                # Inventory levels based on product demand and seasonality
                base_stock = product['base_demand'] * 7  # 1 week of demand
                seasonal_adj = product['seasonality_factor']
                
                on_hand = max(0, np.random.poisson(base_stock * seasonal_adj))
                committed = min(on_hand, np.random.poisson(on_hand * 0.3))
                available = on_hand - committed
                
                inventory = {
                    'fc_id': fc['fc_id'],
                    'sku_id': product['sku_id'],
                    'on_hand_qty': on_hand,
                    'committed_qty': committed,
                    'available_qty': available,
                    'reorder_point': int(base_stock * 0.2),
                    'max_stock_level': int(base_stock * 2),
                    'last_updated': fake.date_between(start_date='-7d', end_date='today')
                }
                inventory_current.append(inventory)
        
        # Generate some historical movements
        movement_types = ['receipt', 'shipment', 'transfer_in', 'transfer_out', 'adjustment']
        
        for _ in range(len(inventory_current) * 5):  # 5 movements per current inventory record
            current_inv = np.random.choice(range(len(inventory_current)))
            inv_record = inventory_current[current_inv]
            
            movement = {
                'movement_id': f'MOV{fake.random_int(min=100000, max=999999)}',
                'fc_id': inv_record['fc_id'],
                'sku_id': inv_record['sku_id'],
                'movement_type': np.random.choice(movement_types),
                'quantity': np.random.randint(1, 50),
                'movement_date': fake.date_between(start_date='-90d', end_date='today'),
                'reference_id': f'REF{fake.random_int(min=10000, max=99999)}'
            }
            inventory_movements.append(movement)
        
        return pd.DataFrame(inventory_current), pd.DataFrame(inventory_movements)
    
    def generate_shipping_costs(self, fc_df):
        """Generate shipping cost matrix between FCs and ZIP codes"""
        print("Generating shipping costs...")
        
        # Generate representative ZIP codes for major metro areas
        major_zips = [
            '90210', '10001', '60601', '30309', '77001', '98101', '33101', 
            '19101', '85001', '80201', '97201', '84101', '89101', '37201'
        ]
        
        shipping_costs = []
        
        for _, fc in fc_df.iterrows():
            for zip_code in major_zips:
                # Distance-based shipping cost with some randomness
                base_cost = np.random.uniform(5, 25)  # Base shipping cost
                
                # Add zone-based pricing
                if fc['state'] == 'CA' and zip_code.startswith(('9', '8')):
                    zone_multiplier = 0.8  # Same zone discount
                elif fc['state'] == 'NJ' and zip_code.startswith(('0', '1')):
                    zone_multiplier = 0.8
                else:
                    zone_multiplier = 1.2  # Cross-country premium
                
                cost_record = {
                    'fc_id': fc['fc_id'],
                    'destination_zip': zip_code,
                    'standard_cost': round(base_cost * zone_multiplier, 2),
                    'expedited_cost': round(base_cost * zone_multiplier * 1.8, 2),
                    'avg_transit_days': np.random.randint(1, 8),
                    'carrier': np.random.choice(['UPS', 'FedEx', 'USPS'], p=[0.4, 0.35, 0.25])
                }
                shipping_costs.append(cost_record)
        
        return pd.DataFrame(shipping_costs)
    
    def add_data_quality_issues(self, dataframes):
        """Add realistic data quality issues found in production systems"""
        print("Adding realistic data quality issues...")
        
        customers_df, products_df, fc_df, orders_df, order_items_df, inventory_df, movements_df, shipping_df = dataframes
        
        # Customer data issues
        # Some customers missing city (2%)
        mask = np.random.random(len(customers_df)) < 0.02
        customers_df.loc[mask, 'city'] = None
        
        # Some inconsistent state formats
        mask = np.random.random(len(customers_df)) < 0.01
        customers_df.loc[mask, 'state'] = customers_df.loc[mask, 'state'].str.lower()
        
        # Product data issues
        # Some products missing brand (5%)
        mask = np.random.random(len(products_df)) < 0.05
        products_df.loc[mask, 'brand'] = None
        
        # Some negative weights (data entry errors)
        mask = np.random.random(len(products_df)) < 0.001
        products_df.loc[mask, 'weight_lbs'] = -products_df.loc[mask, 'weight_lbs']
        
        # Order data issues
        # Some orders missing fulfillment center (pending assignment)
        mask = np.random.random(len(orders_df)) < 0.15
        orders_df.loc[mask, 'fulfillment_center'] = None
        
        # Some duplicate order IDs (rare system error)
        if len(orders_df) > 100:
            duplicate_indices = np.random.choice(orders_df.index, size=3, replace=False)
            orders_df.loc[duplicate_indices[1:], 'order_id'] = orders_df.loc[duplicate_indices[0], 'order_id']
        
        return customers_df, products_df, fc_df, orders_df, order_items_df, inventory_df, movements_df, shipping_df
    
    def update_customer_totals(self, customers_df, orders_df):
        """Update customer total_orders based on generated orders"""
        print("Updating customer order totals...")
        
        order_counts = orders_df.groupby('customer_id').size().reset_index(name='order_count')
        
        # Merge and update
        customers_df = customers_df.set_index('customer_id')
        order_counts = order_counts.set_index('customer_id')
        
        customers_df.update(order_counts.rename(columns={'order_count': 'total_orders'}))
        customers_df['total_orders'] = customers_df['total_orders'].fillna(0).astype(int)
        
        return customers_df.reset_index()
    
    def generate_all_data(self, output_dir='synthetic_data'):
        """Generate all datasets and save to files"""
        print("=" * 60)
        print("GENERATING SYNTHETIC FULFILLMENT DATA")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all datasets
        customers_df = self.generate_customers(10000)
        products_df = self.generate_products(2000)
        fc_df = self.generate_fulfillment_centers()
        orders_df, order_items_df = self.generate_orders_and_items(customers_df, products_df, fc_df, 50000)
        inventory_df, movements_df = self.generate_inventory_data(products_df, fc_df)
        shipping_df = self.generate_shipping_costs(fc_df)
        
        # Add realistic data quality issues
        all_dfs = self.add_data_quality_issues([
            customers_df, products_df, fc_df, orders_df, order_items_df, 
            inventory_df, movements_df, shipping_df
        ])
        customers_df, products_df, fc_df, orders_df, order_items_df, inventory_df, movements_df, shipping_df = all_dfs
        
        # Update customer totals
        customers_df = self.update_customer_totals(customers_df, orders_df)
        
        # Save all datasets
        datasets = {
            'customers': customers_df,
            'products': products_df,
            'fulfillment_centers': fc_df,
            'orders': orders_df,
            'order_items': order_items_df,
            'inventory_current': inventory_df,
            'inventory_movements': movements_df,
            'shipping_costs': shipping_df
        }
        
        print("\nSaving datasets...")
        for name, df in datasets.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"  {name}: {len(df)} records -> {filepath}")
        
        # Generate data summary
        self.generate_data_summary(datasets, output_dir)
        
        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE!")
        print(f"All files saved to: {output_dir}/")
        print("=" * 60)
        
        return datasets
    
    def generate_data_summary(self, datasets, output_dir):
        """Generate data quality and summary report"""
        summary_report = []
        summary_report.append("SYNTHETIC DATA GENERATION SUMMARY")
        summary_report.append("=" * 50)
        summary_report.append(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_report.append(f"Data Period: {self.base_date.date()} to {self.end_date.date()}")
        summary_report.append("")
        
        for name, df in datasets.items():
            summary_report.append(f"{name.upper()}")
            summary_report.append("-" * 30)
            summary_report.append(f"Records: {len(df):,}")
            summary_report.append(f"Columns: {len(df.columns)}")
            
            # Data quality metrics
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                summary_report.append("Missing Data:")
                for col, count in missing_data[missing_data > 0].items():
                    pct = (count / len(df)) * 100
                    summary_report.append(f"  {col}: {count} ({pct:.1f}%)")
            else:
                summary_report.append("No missing data")
            
            summary_report.append("")
        
        # Business logic validation
        summary_report.append("BUSINESS LOGIC VALIDATION")
        summary_report.append("-" * 30)
        summary_report.append(f"Customer Segments: {datasets['customers']['customer_segment'].value_counts().to_dict()}")
        summary_report.append(f"Product Categories: {datasets['products']['category'].value_counts().to_dict()}")
        summary_report.append(f"Order Statuses: {datasets['orders']['order_status'].value_counts().to_dict()}")
        summary_report.append("")
        
        # Save summary report
        with open(os.path.join(output_dir, 'data_summary.txt'), 'w') as f:
            f.write('\n'.join(summary_report))

if __name__ == "__main__":
    # Generate synthetic data
    generator = FulfillmentDataGenerator()
    datasets = generator.generate_all_data()
    
    # Display sample data
    print("\nSAMPLE DATA PREVIEW:")
    print("-" * 40)
    for name, df in datasets.items():
        print(f"\n{name.upper()} (first 3 rows):")
        print(df.head(3).to_string(index=False))
        if len(df.columns) > 8:  # Truncate if too many columns
            print("  ... (additional columns not shown)")

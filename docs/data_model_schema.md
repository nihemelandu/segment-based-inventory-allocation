# Data Model Schema - Fulfillment Optimization Project

## Overview
This document defines the complete data model schema for the synthetic fulfillment optimization dataset, including table structures, relationships, constraints, and business rules.

## Entity Relationship Diagram (Conceptual)

```
CUSTOMERS (1) ──────── (M) ORDERS (1) ──────── (M) ORDER_ITEMS
    │                       │                          │
    │                       │                          │
    │                       └── (M) FULFILLMENT_CENTERS │
    │                                     │             │
    │                                     │             │
    └─────────────────────────────────────┘             │
                                                        │
PRODUCTS (1) ────────────────────────────────────────────┘
    │
    │
    └── (M) INVENTORY_CURRENT ── (1) FULFILLMENT_CENTERS
    │
    └── (M) INVENTORY_MOVEMENTS ── (1) FULFILLMENT_CENTERS

FULFILLMENT_CENTERS (1) ──── (M) SHIPPING_COSTS
```

---

## Table Schemas

### 1. CUSTOMERS
**Description:** Master data for all customers (individuals and businesses)  
**Business Owner:** Customer Success Team  
**Update Frequency:** Real-time (new registrations), Daily (aggregated metrics)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| customer_id | VARCHAR | 10 | PRIMARY KEY, NOT NULL | Unique customer identifier | Format: C######, Sequential, Never reused |
| registration_date | DATE | - | NOT NULL | Account creation date | Must be <= current date, UTC timezone |
| customer_type | VARCHAR | 20 | NOT NULL | Customer classification | ENUM: 'individual', 'business', 'wholesale' |
| city | VARCHAR | 50 | NULLABLE | Primary shipping city | May be NULL for privacy, International formats vary |
| state | VARCHAR | 20 | NULLABLE | State/Province code | US: 2-char code, International: varies |
| zip_code | VARCHAR | 10 | NULLABLE | Primary ZIP/Postal code | US: 5 or 9 digits, International: varies |
| customer_segment | VARCHAR | 20 | NOT NULL | ML-derived behavior segment | ENUM: 'high_frequency', 'medium_frequency', 'low_frequency', 'seasonal' |
| total_orders | INTEGER | - | DEFAULT 0 | Count of completed orders | Updated nightly, Excludes cancelled orders |

**Data Quality Issues:**
- 2% missing city values (privacy preferences)
- 1% inconsistent state formatting (mixed case)
- total_orders may be 0 for new customers

**Indexes:**
- PRIMARY: customer_id
- INDEX: customer_segment, registration_date
- INDEX: state, zip_code (for geographic analysis)

---

### 2. PRODUCTS
**Description:** Product catalog with SKU details and attributes  
**Business Owner:** Product Management Team  
**Update Frequency:** Daily (price updates), Weekly (new products)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| sku_id | VARCHAR | 10 | PRIMARY KEY, NOT NULL | Unique product identifier | Format: SKU######, Never reused |
| product_name | VARCHAR | 200 | NOT NULL | Display name | Marketing-friendly description |
| category | VARCHAR | 50 | NOT NULL | Primary product category | ENUM: 'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books' |
| subcategory | VARCHAR | 50 | NULLABLE | Product subcategory | Varies by category |
| brand | VARCHAR | 100 | NULLABLE | Brand/Manufacturer name | May be NULL for generic/private label |
| price | DECIMAL | 10,2 | NOT NULL, CHECK > 0 | Current selling price (USD) | Must be positive, 2 decimal places |
| weight_lbs | DECIMAL | 8,3 | NOT NULL, CHECK > 0 | Product weight in pounds | For shipping calculations, 3 decimal precision |
| dimensions_cubic_in | DECIMAL | 10,2 | NOT NULL, CHECK > 0 | Volume in cubic inches | For storage/packing optimization |
| launch_date | DATE | - | NOT NULL | Product introduction date | Must be <= current date |
| seasonality_factor | DECIMAL | 3,2 | NOT NULL, DEFAULT 1.0 | Seasonal demand multiplier | Range: 0.5 - 2.0, Used for demand forecasting |
| base_demand | INTEGER | - | NOT NULL, CHECK > 0 | Weekly baseline demand units | Historical average, Updated quarterly |

**Data Quality Issues:**
- 5% missing brand values (generic products)
- <0.1% negative weights (data entry errors)
- Some discontinued products still in catalog

**Indexes:**
- PRIMARY: sku_id
- INDEX: category, subcategory
- INDEX: price (for price range queries)
- INDEX: launch_date

---

### 3. FULFILLMENT_CENTERS
**Description:** Warehouse and distribution center master data  
**Business Owner:** Supply Chain Operations  
**Update Frequency:** Monthly (capacity updates), Quarterly (cost updates)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| fc_id | VARCHAR | 10 | PRIMARY KEY, NOT NULL | Unique FC identifier | Format: FC_SS##, SS=State code |
| fc_name | VARCHAR | 100 | NOT NULL | Facility display name | Business-friendly name |
| city | VARCHAR | 50 | NOT NULL | FC city location | For geographic routing |
| state | VARCHAR | 20 | NOT NULL | FC state/province | 2-character US state codes |
| zip_code | VARCHAR | 10 | NOT NULL | FC ZIP code | For precise distance calculations |
| total_capacity | INTEGER | - | NOT NULL, CHECK > 0 | Maximum storage units | Physical capacity limit |
| current_utilization | DECIMAL | 3,2 | NOT NULL, CHECK 0-1 | Capacity utilization ratio | 0.0 = empty, 1.0 = full, Updated weekly |
| storage_cost_per_unit | DECIMAL | 6,2 | NOT NULL, CHECK > 0 | Monthly storage cost per unit | USD, Varies by location/facility type |
| labor_cost_per_hour | DECIMAL | 6,2 | NOT NULL, CHECK > 0 | Average hourly labor rate | USD, For operational cost modeling |
| operational_since | DATE | - | NOT NULL | FC opening date | Must be < current date |

**Business Rules:**
- Each state should have at least one FC
- Utilization should not exceed 0.95 (95%) for operational efficiency
- Labor costs vary by geographic market

**Indexes:**
- PRIMARY: fc_id
- INDEX: state (for regional analysis)
- INDEX: current_utilization

---

### 4. ORDERS
**Description:** Customer order header information  
**Business Owner:** Order Management Team  
**Update Frequency:** Real-time

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| order_id | VARCHAR | 15 | PRIMARY KEY, NOT NULL | Unique order identifier | Format: ORD########, Sequential |
| customer_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References customers.customer_id | Must exist in customers table |
| order_date | DATE | - | NOT NULL | Order placement date | UTC timezone, Must be >= customer registration |
| order_total | DECIMAL | 10,2 | NOT NULL, CHECK >= 0 | Total order value (USD) | Sum of line items, Excludes shipping |
| shipping_cost | DECIMAL | 8,2 | NOT NULL, CHECK >= 0 | Shipping charges (USD) | 0.00 for free shipping orders |
| shipping_method | VARCHAR | 20 | NOT NULL | Delivery method selected | ENUM: 'standard', 'expedited', 'standard_free' |
| fulfillment_center | VARCHAR | 10 | FOREIGN KEY, NULLABLE | Assigned FC for fulfillment | References fulfillment_centers.fc_id, NULL = pending |
| order_status | VARCHAR | 20 | NOT NULL | Current order state | ENUM: 'pending', 'processing', 'shipped', 'delivered', 'cancelled' |
| total_weight | DECIMAL | 8,2 | NOT NULL, CHECK >= 0 | Total order weight (lbs) | Sum of item weights × quantities |
| customer_zip | VARCHAR | 10 | NOT NULL | Delivery ZIP code | For shipping cost calculation |

**Data Quality Issues:**
- 15% NULL fulfillment_center (pending assignment)
- Some duplicate order_ids (rare system errors)
- Historical orders may have different status values

**Business Rules:**
- Free shipping for orders > $75 OR business customers
- Orders must be assigned to FC within 24 hours
- Cancelled orders retain original order_total for analysis

**Indexes:**
- PRIMARY: order_id
- FOREIGN KEY: customer_id, fulfillment_center
- INDEX: order_date, order_status
- INDEX: customer_id, order_date (for customer analysis)

---

### 5. ORDER_ITEMS
**Description:** Individual line items within each order  
**Update Frequency:** Real-time (with order creation)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| order_id | VARCHAR | 15 | FOREIGN KEY, NOT NULL | References orders.order_id | Part of composite key |
| sku_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References products.sku_id | Part of composite key |
| quantity | INTEGER | - | NOT NULL, CHECK > 0 | Units ordered | Must be positive integer |
| unit_price | DECIMAL | 10,2 | NOT NULL, CHECK > 0 | Price per unit at order time | Historical pricing, May differ from current |
| line_total | DECIMAL | 10,2 | NOT NULL, CHECK >= 0 | Extended line amount | quantity × unit_price |

**Business Rules:**
- unit_price captures historical pricing at time of order
- line_total must equal quantity × unit_price
- Same SKU can appear multiple times per order (different configurations)

**Indexes:**
- PRIMARY: (order_id, sku_id)
- FOREIGN KEY: order_id, sku_id
- INDEX: sku_id (for product analysis)

---

### 6. INVENTORY_CURRENT
**Description:** Current inventory levels by SKU and fulfillment center  
**Update Frequency:** Real-time (with transactions), Batch reconciliation (nightly)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| fc_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References fulfillment_centers.fc_id | Part of composite key |
| sku_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References products.sku_id | Part of composite key |
| on_hand_qty | INTEGER | - | NOT NULL, CHECK >= 0 | Physical inventory count | Never negative |
| committed_qty | INTEGER | - | NOT NULL, CHECK >= 0 | Reserved for pending orders | Subset of on_hand_qty |
| available_qty | INTEGER | - | NOT NULL, CHECK >= 0 | Available for sale | on_hand_qty - committed_qty |
| reorder_point | INTEGER | - | NOT NULL, CHECK >= 0 | Reorder trigger level | When to replenish stock |
| max_stock_level | INTEGER | - | NOT NULL, CHECK > 0 | Maximum inventory target | Storage capacity constraint |
| last_updated | TIMESTAMP | - | NOT NULL | Last inventory update | For data freshness tracking |

**Business Rules:**
- available_qty = on_hand_qty - committed_qty
- committed_qty <= on_hand_qty
- Reorder triggered when available_qty <= reorder_point
- Not all SKUs stocked at all FCs (60-80% coverage)

**Indexes:**
- PRIMARY: (fc_id, sku_id)
- INDEX: available_qty (for stockout analysis)
- INDEX: last_updated (for data freshness)

---

### 7. INVENTORY_MOVEMENTS
**Description:** Historical inventory transactions and movements  
**Update Frequency:** Real-time (with each transaction)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| movement_id | VARCHAR | 15 | PRIMARY KEY, NOT NULL | Unique movement identifier | Format: MOV######, Sequential |
| fc_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References fulfillment_centers.fc_id | Source/destination FC |
| sku_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References products.sku_id | Product moved |
| movement_type | VARCHAR | 20 | NOT NULL | Type of inventory movement | ENUM: 'receipt', 'shipment', 'transfer_in', 'transfer_out', 'adjustment' |
| quantity | INTEGER | - | NOT NULL, CHECK != 0 | Units moved | Positive = increase, Negative = decrease |
| movement_date | DATE | - | NOT NULL | Transaction date | UTC timezone |
| reference_id | VARCHAR | 15 | NULLABLE | Related transaction ID | Order ID, PO number, etc. |

**Business Rules:**
- Positive quantity = inventory increase (receipt, transfer_in, positive adjustment)
- Negative quantity = inventory decrease (shipment, transfer_out, negative adjustment)
- reference_id links to related business transaction

**Indexes:**
- PRIMARY: movement_id
- INDEX: (fc_id, sku_id, movement_date)
- INDEX: movement_type, movement_date

---

### 8. SHIPPING_COSTS
**Description:** Shipping cost matrix between fulfillment centers and destinations  
**Update Frequency:** Weekly (rate updates), Monthly (new routes)

| Column Name | Data Type | Length | Constraints | Description | Business Rules |
|-------------|-----------|---------|-------------|-------------|----------------|
| fc_id | VARCHAR | 10 | FOREIGN KEY, NOT NULL | References fulfillment_centers.fc_id | Origin FC |
| destination_zip | VARCHAR | 10 | NOT NULL | Delivery ZIP code | 3-digit or 5-digit ZIP |
| standard_cost | DECIMAL | 6,2 | NOT NULL, CHECK > 0 | Standard shipping rate (USD) | Base delivery method |
| expedited_cost | DECIMAL | 6,2 | NOT NULL, CHECK > 0 | Expedited shipping rate (USD) | Faster delivery option |
| avg_transit_days | INTEGER | - | NOT NULL, CHECK > 0 | Average delivery time | Business days for standard shipping |
| carrier | VARCHAR | 20 | NOT NULL | Primary carrier | ENUM: 'UPS', 'FedEx', 'USPS' |

**Business Rules:**
- expedited_cost > standard_cost (typically 1.5-2x)
- Costs vary by distance and carrier zones
- Major metropolitan ZIP codes covered for all FCs

**Indexes:**
- PRIMARY: (fc_id, destination_zip)
- INDEX: destination_zip (for customer location lookups)

---

## Referential Integrity

### Foreign Key Relationships:
```sql
orders.customer_id → customers.customer_id
orders.fulfillment_center → fulfillment_centers.fc_id
order_items.order_id → orders.order_id
order_items.sku_id → products.sku_id
inventory_current.fc_id → fulfillment_centers.fc_id
inventory_current.sku_id → products.sku_id
inventory_movements.fc_id → fulfillment_centers.fc_id
inventory_movements.sku_id → products.sku_id
shipping_costs.fc_id → fulfillment_centers.fc_id
```

### Data Validation Rules:
```sql
-- Order totals must match line item sums
CHECK: orders.order_total = SUM(order_items.line_total) WHERE order_items.order_id = orders.order_id

-- Inventory availability constraints
CHECK: inventory_current.available_qty = inventory_current.on_hand_qty - inventory_current.committed_qty
CHECK: inventory_current.committed_qty <= inventory_current.on_hand_qty

-- Date consistency
CHECK: orders.order_date >= customers.registration_date
CHECK: products.launch_date <= orders.order_date (for products in orders)
```

---

## Business Rules Summary

### Customer Segmentation:
- **high_frequency:** >12 orders/year, >$150 AOV
- **medium_frequency:** 4-12 orders/year, $50-150 AOV  
- **low_frequency:** <4 orders/year, <$100 AOV
- **seasonal:** Concentrated ordering in Q4, moderate AOV

### Operational Rules:
- Free shipping threshold: $75+ OR business customers
- FC assignment: Geographic preference with capacity constraints
- Inventory reorder: Triggered at reorder_point, max at max_stock_level
- Seasonal demand: Q4 peak (1.4x), Q1 trough (0.8x)

### Data Quality Expectations:
- **Customer data:** 98% complete for core fields
- **Product data:** 95% complete (5% missing brands acceptable)
- **Order data:** 100% complete except FC assignment (15% pending normal)
- **Inventory data:** Real-time accuracy within 95%

This schema provides the foundation for customer clustering, demand forecasting, and fulfillment optimization algorithms while maintaining data quality standards typical of production e-commerce systems.
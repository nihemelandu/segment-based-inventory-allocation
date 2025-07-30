erDiagram
    CUSTOMERS {
        string customer_id PK "C######"
        date registration_date
        string customer_type "individual|business|wholesale"
        string city
        string state
        string zip_code
        string customer_segment "high|medium|low|seasonal"
        int total_orders "calculated field"
    }

    PRODUCTS {
        string sku_id PK "SKU######"
        string product_name
        string category "Electronics|Clothing|etc"
        string subcategory
        string brand
        decimal price "USD"
        decimal weight_lbs
        decimal dimensions_cubic_in
        date launch_date
        decimal seasonality_factor "0.5-2.0"
        int base_demand "weekly units"
    }

    FULFILLMENT_CENTERS {
        string fc_id PK "FC_SS##"
        string fc_name
        string city
        string state
        string zip_code
        int total_capacity
        decimal current_utilization "0.0-1.0"
        decimal storage_cost_per_unit "USD/month"
        decimal labor_cost_per_hour "USD"
        date operational_since
    }

    ORDERS {
        string order_id PK "ORD########"
        string customer_id FK
        date order_date
        decimal order_total "USD"
        decimal shipping_cost "USD"
        string shipping_method "standard|expedited|standard_free"
        string fulfillment_center FK "nullable"
        string order_status "pending|processing|shipped|delivered|cancelled"
        decimal total_weight "lbs"
        string customer_zip
    }

    ORDER_ITEMS {
        string order_id PK,FK
        string sku_id PK,FK
        int quantity
        decimal unit_price "historical price"
        decimal line_total "quantity × unit_price"
    }

    INVENTORY_CURRENT {
        string fc_id PK,FK
        string sku_id PK,FK
        int on_hand_qty
        int committed_qty
        int available_qty "on_hand - committed"
        int reorder_point
        int max_stock_level
        timestamp last_updated
    }

    INVENTORY_MOVEMENTS {
        string movement_id PK "MOV######"
        string fc_id FK
        string sku_id FK
        string movement_type "receipt|shipment|transfer_in|transfer_out|adjustment"
        int quantity "positive=increase, negative=decrease"
        date movement_date
        string reference_id "order_id, PO, etc"
    }

    SHIPPING_COSTS {
        string fc_id PK,FK
        string destination_zip PK
        decimal standard_cost "USD"
        decimal expedited_cost "USD"
        int avg_transit_days
        string carrier "UPS|FedEx|USPS"
    }

    %% Relationships
    CUSTOMERS ||--o{ ORDERS : "places"
    ORDERS ||--o{ ORDER_ITEMS : "contains"
    PRODUCTS ||--o{ ORDER_ITEMS : "ordered_as"
    FULFILLMENT_CENTERS ||--o{ ORDERS : "fulfills"
    FULFILLMENT_CENTERS ||--o{ INVENTORY_CURRENT : "stocks"
    PRODUCTS ||--o{ INVENTORY_CURRENT : "stocked_as"
    FULFILLMENT_CENTERS ||--o{ INVENTORY_MOVEMENTS : "moves_inventory"
    PRODUCTS ||--o{ INVENTORY_MOVEMENTS : "inventory_moved"
    FULFILLMENT_CENTERS ||--o{ SHIPPING_COSTS : "ships_from"

    %% Business Rule Annotations
    CUSTOMERS {
        note "Segments: high_frequency(15%), medium_frequency(45%), low_frequency(30%), seasonal(10%)"
    }
    
    ORDERS {
        note "15% have NULL fulfillment_center (pending assignment)"
        note "Free shipping: order_total > $75 OR customer_type = 'business'"
    }
    
    INVENTORY_CURRENT {
        note "Not all SKUs in all FCs (60-80% coverage per FC)"
        note "available_qty = on_hand_qty - committed_qty"
    }
    
    PRODUCTS {
        note "Seasonality: Q4=1.4x, Q3=1.0x, Q2=0.9x, Q1=0.8x"
    }
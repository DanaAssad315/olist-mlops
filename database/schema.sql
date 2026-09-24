
-- 1. Customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50) NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city VARCHAR(100) NOT NULL,
    customer_state VARCHAR(10) NOT NULL
);


-- 2. Sellers
CREATE TABLE IF NOT EXISTS sellers (
    seller_id VARCHAR(50) PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city VARCHAR(100) NOT NULL,
    seller_state VARCHAR(10) NOT NULL
);


-- 3. Products
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_lenght INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty INTEGER,
    product_weight_g DOUBLE PRECISION,
    product_length_cm DOUBLE PRECISION,
    product_height_cm DOUBLE PRECISION,
    product_width_cm DOUBLE PRECISION
);


-- 4. Category Translation
CREATE TABLE IF NOT EXISTS category_translation (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100) NOT NULL
);


-- 5. Geolocation
CREATE TABLE IF NOT EXISTS geolocation (
    geolocation_zip_code_prefix INTEGER NOT NULL,
    geolocation_lat DOUBLE PRECISION NOT NULL,
    geolocation_lng DOUBLE PRECISION NOT NULL,
    geolocation_city VARCHAR(100) NOT NULL,
    geolocation_state VARCHAR(10) NOT NULL
);


-- 6. Orders
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    order_status VARCHAR(30) NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP NOT NULL,

    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- 7. Order Items
CREATE TABLE IF NOT EXISTS order_items (
    order_id VARCHAR(50) NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    seller_id VARCHAR(50) NOT NULL,
    shipping_limit_date TIMESTAMP NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    freight_value DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (order_id, order_item_id),

    CONSTRAINT fk_order_items_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_order_items_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id),

    CONSTRAINT fk_order_items_seller
        FOREIGN KEY (seller_id)
        REFERENCES sellers(seller_id)
);


-- 8. Order Payments
CREATE TABLE IF NOT EXISTS order_payments (
    order_id VARCHAR(50) NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type VARCHAR(30) NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (order_id, payment_sequential),

    CONSTRAINT fk_order_payments_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
);


-- 9. Order Reviews
CREATE TABLE IF NOT EXISTS order_reviews (
    review_id VARCHAR(50) NOT NULL,
    order_id VARCHAR(50) NOT NULL,
    review_score INTEGER NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP NOT NULL,
    review_answer_timestamp TIMESTAMP NOT NULL,

    PRIMARY KEY (review_id, order_id),

    CONSTRAINT fk_order_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
);

-- ============================================
-- Indexes
-- ============================================

CREATE INDEX IF NOT EXISTS idx_orders_customer_id
    ON orders(customer_id);

CREATE INDEX IF NOT EXISTS idx_order_items_product_id
    ON order_items(product_id);

CREATE INDEX IF NOT EXISTS idx_order_items_seller_id
    ON order_items(seller_id);

CREATE INDEX IF NOT EXISTS idx_order_payments_order_id
    ON order_payments(order_id);

CREATE INDEX IF NOT EXISTS idx_order_reviews_order_id
    ON order_reviews(order_id);

CREATE INDEX IF NOT EXISTS idx_geolocation_zip_code
    ON geolocation(geolocation_zip_code_prefix);
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(r"C:\Users\dell 7300\OneDrive\Desktop\data")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "olist",
    "user": "olist_user",
    "password": "olist_password",
}

BATCH_SIZE = 5000


# ============================================================
# Database connection
# ============================================================


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# Helper: Convert NaN to None
# ============================================================


def clean_value(value):
    if pd.isna(value):
        return None
    return value


# ============================================================
# Generic batch insert
# ============================================================


def insert_dataframe(conn, df, table_name, columns):
    records = [
        tuple(clean_value(value) for value in row)
        for row in df[columns].itertuples(index=False, name=None)
    ]

    query = f"""
        INSERT INTO {table_name} ({", ".join(columns)})
        VALUES %s
    """

    with conn.cursor() as cursor:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]

            execute_values(cursor, query, batch, page_size=BATCH_SIZE)

            conn.commit()

            print(
                f"  {min(start + BATCH_SIZE, len(records)):,}"
                f"/{len(records):,} rows inserted"
            )


# ============================================================
# Customers
# ============================================================


def load_customers(conn):
    print("\nLoading customers...")

    file_path = DATA_DIR / "olist_customers_dataset.csv"

    df = pd.read_csv(file_path)

    columns = [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ]

    insert_dataframe(conn, df, "customers", columns)


# ============================================================
# Sellers
# ============================================================


def load_sellers(conn):
    print("\nLoading sellers...")

    file_path = DATA_DIR / "olist_sellers_dataset.csv"

    df = pd.read_csv(file_path)

    columns = [
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ]

    insert_dataframe(conn, df, "sellers", columns)


# ============================================================
# Products
# ============================================================


def load_products(conn):
    print("\nLoading products...")

    file_path = DATA_DIR / "olist_products_dataset.csv"

    df = pd.read_csv(file_path)

    columns = [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    insert_dataframe(conn, df, "products", columns)


# ============================================================
# Category Translation
# ============================================================


def load_category_translation(conn):
    print("\nLoading category translation...")

    file_path = DATA_DIR / "product_category_name_translation.csv"

    df = pd.read_csv(file_path)

    columns = [
        "product_category_name",
        "product_category_name_english",
    ]

    insert_dataframe(conn, df, "category_translation", columns)


# ============================================================
# Geolocation
# ============================================================


def load_geolocation(conn):
    print("\nLoading geolocation...")

    file_path = DATA_DIR / "olist_geolocation_dataset.csv"

    df = pd.read_csv(file_path)

    columns = [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ]

    insert_dataframe(conn, df, "geolocation", columns)


# ============================================================
# Orders
# ============================================================


def load_orders(conn):
    print("\nLoading orders...")

    file_path = DATA_DIR / "olist_orders_dataset.csv"

    df = pd.read_csv(file_path)

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    columns = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    insert_dataframe(conn, df, "orders", columns)


# ============================================================
# Order Items
# ============================================================


def load_order_items(conn):
    print("\nLoading order items...")

    file_path = DATA_DIR / "olist_order_items_dataset.csv"

    df = pd.read_csv(file_path)

    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"], errors="coerce"
    )

    columns = [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ]

    insert_dataframe(conn, df, "order_items", columns)


# ============================================================
# Order Payments
# ============================================================


def load_order_payments(conn):
    print("\nLoading order payments...")

    file_path = DATA_DIR / "olist_order_payments_dataset.csv"

    df = pd.read_csv(file_path)

    columns = [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ]

    insert_dataframe(conn, df, "order_payments", columns)


# ============================================================
# Order Reviews
# ============================================================


def load_order_reviews(conn):
    print("\nLoading order reviews...")

    file_path = DATA_DIR / "olist_order_reviews_dataset.csv"

    df = pd.read_csv(file_path)

    date_columns = [
        "review_creation_date",
        "review_answer_timestamp",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    columns = [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ]

    insert_dataframe(conn, df, "order_reviews", columns)


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 60)
    print("OLIST DATA INGESTION")
    print("=" * 60)

    conn = None

    try:
        conn = get_connection()

        print("\nConnected to PostgreSQL successfully!")

        # Order matters because of Foreign Keys
        load_customers(conn)
        load_sellers(conn)
        load_products(conn)
        load_category_translation(conn)
        load_geolocation(conn)
        load_orders(conn)
        load_order_items(conn)
        load_order_payments(conn)
        load_order_reviews(conn)

        print("\n" + "=" * 60)
        print("DATA INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as error:
        print("\nERROR:")
        print(error)

        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    conn = None

    try:
        conn = get_connection()

        print("Connected to PostgreSQL successfully!")

        load_order_reviews(conn)

        print("\nOrder reviews ingestion completed successfully!")

    except Exception as error:
        print("\nERROR:")
        print(error)

        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

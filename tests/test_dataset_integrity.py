import pytest
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, KB_DIR, PRACTICE_DIR

EXPECTED_ROW_COUNTS = {
    "customers.csv": 300,
    "orders.csv": 1000,
    "order_items.csv": 1783,
    "products.csv": 70,
    "pincodes.csv": 36,
    "holidays.csv": 4,
    "support_logs.csv": 244,
}

def test_csv_file_existence_and_row_counts():
    for filename, expected_count in EXPECTED_ROW_COUNTS.items():
        filepath = DATA_DIR / filename
        assert filepath.exists(), f"Missing CSV table: {filename}"
        df = pd.read_csv(filepath, dtype=str)
        assert len(df) == expected_count, f"{filename} row count mismatch: expected {expected_count}, got {len(df)}"

def test_practice_files_existence_and_counts():
    questions_path = PRACTICE_DIR / "questions.csv"
    conversations_path = PRACTICE_DIR / "conversations.csv"
    
    assert questions_path.exists(), "Missing practice/questions.csv"
    assert conversations_path.exists(), "Missing practice/conversations.csv"
    
    df_q = pd.read_csv(questions_path, dtype=str)
    assert len(df_q) == 52, f"Expected 52 questions, got {len(df_q)}"
    
    df_c = pd.read_csv(conversations_path, dtype=str)
    assert len(df_c) == 96, f"Expected 96 conversation turns, got {len(df_c)}"
    assert df_c["conversation_id"].nunique() == 23, f"Expected 23 unique conversations, got {df_c['conversation_id'].nunique()}"

def test_kb_markdown_files_count_and_headers():
    kb_files = list(KB_DIR.glob("*.md"))
    assert len(kb_files) == 21, f"Expected 21 KB markdown files, found {len(kb_files)}"
    
    section_ids_found = 0
    for kb_file in kb_files:
        content = kb_file.read_text(encoding="utf-8")
        assert content.startswith("---"), f"{kb_file.name} missing YAML frontmatter"
        assert "doc_id:" in content, f"{kb_file.name} missing doc_id"
        assert "doc_type:" in content, f"{kb_file.name} missing doc_type"
        assert "status:" in content, f"{kb_file.name} missing status"
        if "## [" in content:
            section_ids_found += content.count("## [")
    
    assert section_ids_found > 0, "No section IDs found in KB files"

def test_relational_integrity():
    customers = pd.read_csv(DATA_DIR / "customers.csv", dtype=str)
    orders = pd.read_csv(DATA_DIR / "orders.csv", dtype=str)
    order_items = pd.read_csv(DATA_DIR / "order_items.csv", dtype=str)
    products = pd.read_csv(DATA_DIR / "products.csv", dtype=str)
    pincodes = pd.read_csv(DATA_DIR / "pincodes.csv", dtype=str)
    
    # 1. Every order's customer_id must exist in customers
    assert orders["customer_id"].isin(customers["customer_id"]).all(), "Orphan customer_id in orders.csv"
    
    # 2. Every order_item's order_id must exist in orders
    assert order_items["order_id"].isin(orders["order_id"]).all(), "Orphan order_id in order_items.csv"
    
    # 3. Every order_item's sku must exist in products
    assert order_items["sku"].isin(products["sku"]).all(), "Orphan sku in order_items.csv"
    
    # 4. Every order's delivery_pincode must exist in pincodes
    assert orders["delivery_pincode"].isin(pincodes["pincode"]).all(), "Invalid delivery_pincode in orders.csv"

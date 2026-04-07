from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR  / "store_sales.csv"
DB_FILE = BASE_DIR / "store_sales.db"

def main():
    df = pd.read_csv(CSV_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    # 基础清洗
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store_id"] = df["store_id"].astype(str).str.strip()

    numeric_cols = ["sales", "orders", "customers", "gross_margin", "stockout_rate"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "store_id", *numeric_cols])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_FILE)
    try:
        df.to_sql("store_sales", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_store_date ON store_sales(store_id, date)")
        conn.commit()
        print(f"SQLite 数据库已生成: {DB_FILE}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
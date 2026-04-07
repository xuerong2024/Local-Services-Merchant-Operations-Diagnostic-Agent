import json
import sqlite3
from pathlib import Path
from datetime import datetime

from qwen_agent.tools.base import BaseTool, register_tool

# =========================
# 运行时日志管理
# =========================
_RUNTIME_LOGS = []
_CURRENT_LOG_FILE = None

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "store_sales.db"


def set_log_file(log_file_path: str):
    global _CURRENT_LOG_FILE
    _CURRENT_LOG_FILE = log_file_path


def clear_runtime_logs():
    global _RUNTIME_LOGS
    _RUNTIME_LOGS = []


def get_runtime_logs():
    return _RUNTIME_LOGS.copy()


def append_runtime_log(message: str):
    global _RUNTIME_LOGS, _CURRENT_LOG_FILE
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    _RUNTIME_LOGS.append(line)

    if _CURRENT_LOG_FILE:
        log_path = Path(_CURRENT_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


@register_tool("analyze_store_sales")
class AnalyzeStoreSales(BaseTool):
    description = (
        "分析门店经营数据。输入门店ID和开始/结束日期，"
        "从 SQLite 数据库中查询并返回营收、订单、客单价、毛利率、缺货率等摘要。"
    )

    parameters = [
        {
            "name": "store_id",
            "type": "string",
            "description": "门店ID，例如 S001",
            "required": True,
        },
        {
            "name": "start_date",
            "type": "string",
            "description": "开始日期，例如 2026-03-01",
            "required": True,
        },
        {
            "name": "end_date",
            "type": "string",
            "description": "结束日期，例如 2026-03-07",
            "required": True,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        append_runtime_log('Start calling tool "analyze_store_sales" ...')
        append_runtime_log(params)

        try:
            args = json.loads(params)
            store_id = str(args["store_id"]).strip()
            start_date = args["start_date"]
            end_date = args["end_date"]

            if not DB_PATH.exists():
                err = f"数据库不存在: {DB_PATH}"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps({"success": False, "error": err}, ensure_ascii=False)

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            try:
                sql = """
                SELECT date, store_id, sales, orders, customers, gross_margin, stockout_rate
                FROM store_sales
                WHERE store_id = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
                """
                rows = conn.execute(sql, (store_id, start_date, end_date)).fetchall()
            finally:
                conn.close()

            if not rows:
                err = f"未找到门店 {store_id} 在 {start_date} 到 {end_date} 的数据"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps({"success": False, "error": err}, ensure_ascii=False)

            sales_list = [float(r["sales"]) for r in rows]
            orders_list = [int(r["orders"]) for r in rows]
            customers_list = [int(r["customers"]) for r in rows]
            margin_list = [float(r["gross_margin"]) for r in rows]
            stockout_list = [float(r["stockout_rate"]) for r in rows]

            total_sales = sum(sales_list)
            total_orders = sum(orders_list)
            total_customers = sum(customers_list)
            avg_margin = sum(margin_list) / len(margin_list)
            avg_stockout = sum(stockout_list) / len(stockout_list)
            avg_ticket = total_sales / total_orders if total_orders > 0 else 0

            first_sales = sales_list[0]
            last_sales = sales_list[-1]
            sales_change = last_sales - first_sales
            sales_change_pct = (sales_change / first_sales) if first_sales else 0

            first_orders = orders_list[0]
            last_orders = orders_list[-1]
            orders_change_pct = ((last_orders - first_orders) / first_orders) if first_orders else 0

            result = {
                "success": True,
                "data": {
                    "store_id": store_id,
                    "period": [start_date, end_date],
                    "days": len(rows),
                    "total_sales": round(total_sales, 2),
                    "total_orders": total_orders,
                    "total_customers": total_customers,
                    "avg_margin": round(avg_margin, 4),
                    "avg_stockout_rate": round(avg_stockout, 4),
                    "avg_ticket": round(avg_ticket, 2),
                    "sales_trend": {
                        "first_day_sales": round(first_sales, 2),
                        "last_day_sales": round(last_sales, 2),
                        "change": round(sales_change, 2),
                        "change_pct": round(sales_change_pct, 4),
                    },
                    "orders_trend": {
                        "first_day_orders": first_orders,
                        "last_day_orders": last_orders,
                        "change_pct": round(orders_change_pct, 4),
                    }
                }
            }

            append_runtime_log(
                f"Tool result summary: total_sales={result['data']['total_sales']}, "
                f"total_orders={result['data']['total_orders']}, "
                f"avg_margin={result['data']['avg_margin']}, "
                f"avg_stockout_rate={result['data']['avg_stockout_rate']}"
            )
            append_runtime_log('Finished tool calling.')
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            append_runtime_log(f"Tool exception: {str(e)}")
            append_runtime_log('Finished tool calling.')
            return json.dumps(
                {"success": False, "error": f"工具执行失败: {str(e)}"},
                ensure_ascii=False
            )
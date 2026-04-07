import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from qwen_agent.tools.base import BaseTool, register_tool

# =========================
# 运行时日志管理
# =========================
_RUNTIME_LOGS = []
_CURRENT_LOG_FILE = None


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
        "分析门店经营数据。输入用户上传的数据文件路径、门店ID和开始/结束日期，"
        "返回营收、订单、客单价、毛利率、缺货率等摘要。"
    )

    parameters = [
        {
            "name": "file_path",
            "type": "string",
            "description": "用户上传的数据文件路径，支持 CSV、XLSX、XLS",
            "required": True,
        },
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
            file_path = args["file_path"]
            store_id = args["store_id"]
            start_date = args["start_date"]
            end_date = args["end_date"]

            p = Path(file_path)
            if not p.exists():
                err = f"文件不存在: {file_path}"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps(
                    {"success": False, "error": err},
                    ensure_ascii=False
                )

            suffix = p.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(p)
            elif suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(p)
            else:
                err = f"暂不支持的文件类型: {suffix}，请上传 CSV 或 Excel 文件"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps(
                    {"success": False, "error": err},
                    ensure_ascii=False
                )

            df.columns = [str(c).strip() for c in df.columns]

            required_cols = [
                "date",
                "store_id",
                "sales",
                "orders",
                "customers",
                "gross_margin",
                "stockout_rate",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                err = f"缺少必要字段: {missing}。当前列名为: {list(df.columns)}"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps(
                    {"success": False, "error": err},
                    ensure_ascii=False
                )

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["store_id"] = df["store_id"].astype(str).str.strip()

            for col in ["sales", "orders", "customers", "gross_margin", "stockout_rate"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(
                subset=["date", "sales", "orders", "customers", "gross_margin", "stockout_rate"]
            )

            sub = df[
                (df["store_id"] == str(store_id).strip())
                & (df["date"] >= pd.to_datetime(start_date))
                & (df["date"] <= pd.to_datetime(end_date))
            ].copy()

            if sub.empty:
                err = f"未找到门店 {store_id} 在 {start_date} 到 {end_date} 的数据"
                append_runtime_log(f"Tool error: {err}")
                append_runtime_log('Finished tool calling.')
                return json.dumps(
                    {"success": False, "error": err},
                    ensure_ascii=False
                )

            sub = sub.sort_values("date")

            total_sales = float(sub["sales"].sum())
            total_orders = int(sub["orders"].sum())
            total_customers = int(sub["customers"].sum())
            avg_margin = float(sub["gross_margin"].mean())
            avg_stockout = float(sub["stockout_rate"].mean())
            avg_ticket = total_sales / total_orders if total_orders > 0 else 0

            first_sales = float(sub.iloc[0]["sales"])
            last_sales = float(sub.iloc[-1]["sales"])
            sales_change = last_sales - first_sales
            sales_change_pct = (sales_change / first_sales) if first_sales else 0

            first_orders = float(sub.iloc[0]["orders"])
            last_orders = float(sub.iloc[-1]["orders"])
            orders_change_pct = ((last_orders - first_orders) / first_orders) if first_orders else 0

            result = {
                "success": True,
                "data": {
                    "store_id": store_id,
                    "period": [start_date, end_date],
                    "days": int(len(sub)),
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
                        "first_day_orders": int(first_orders),
                        "last_day_orders": int(last_orders),
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
                {
                    "success": False,
                    "error": f"工具执行失败: {str(e)}"
                },
                ensure_ascii=False
            )
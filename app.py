import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEMO_FILE = BASE_DIR / "data" / "store_sales.csv"

load_dotenv()

# 只要 import，就会触发 register_tool
import tools.store_tools as store_tools

print("API KEY exists:", bool(os.getenv("DASHSCOPE_API_KEY")))

# =========================
# 日志目录
# =========================
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "workspace_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

llm_cfg = {
    "model": "qwen-max-latest",
    "model_type": "qwen_dashscope",
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "generate_cfg": {
        "top_p": 0.8
    }
}

system_message = """
你是一个门店经营分析助手。

你的职责：
1. 理解用户提出的经营问题；
2. 必要时调用工具读取和分析门店数据；
3. 给出清晰的诊断结论；
4. 提供可执行的经营建议。

非常重要的规则：
- 用户会上传一个 CSV 或 Excel 文件。
- 当你调用 analyze_store_sales 工具时，必须把当前上传文件路径作为 file_path 参数传入。
- 如果工具返回 success=false 或 error，不要编造结论，要明确说明错误原因。
- 如果用户只问“最近一周经营表现怎么样”，你不能使用现实世界时间。
- 你必须基于上传数据中的实际日期范围理解“最近一周”：
  - 若用户说“最近一周”，请以数据中的最晚日期为基准，向前取7天。
  - 若数据不足7天，则基于已有日期范围分析。
- 如果用户明确给出日期范围，则优先使用用户给定范围。
- 输出请尽量采用以下结构：
  - 问题判断
  - 关键发现
  - 可能原因
  - 建议动作
"""

bot = Assistant(
    llm=llm_cfg,
    system_message=system_message,
    function_list=["analyze_store_sales"]
)


def normalize_history(history):
    messages = []
    if not history:
        return messages

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": str(bot_msg)})

    return messages


def extract_text_from_response(resp):
    if resp is None:
        return "没有拿到模型输出。"

    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        content = resp.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("text"):
                        texts.append(str(c["text"]))
                    elif c.get("content"):
                        texts.append(str(c["content"]))
                else:
                    texts.append(str(c))
            return "\n".join(texts).strip() or json.dumps(resp, ensure_ascii=False)
        return json.dumps(resp, ensure_ascii=False)

    if isinstance(resp, list):
        if not resp:
            return "没有拿到模型输出。"
        last = resp[-1]
        return extract_text_from_response(last)

    return str(resp)


def get_data_date_hint(file_path: str) -> str:
    """
    从上传文件中读取日期范围，告诉模型“最近一周”应该如何理解。
    """
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return "无法识别文件类型，无法预读取日期范围。"

        df.columns = [str(c).strip() for c in df.columns]
        if "date" not in df.columns:
            return f"文件缺少 date 列。当前列名：{list(df.columns)}"

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        if df.empty:
            return "文件中没有可用的日期数据。"

        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        return (
            f"当前上传数据的日期范围是 {min_date} 到 {max_date}。"
            f"如果用户说“最近一周”，请以 {max_date} 为结束日期向前取7天；"
            f"如果数据不足7天，则基于已有日期范围分析。"
        )
    except Exception as e:
        return f"预读取日期范围失败：{str(e)}"


def save_app_log(log_file: Path, message: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def format_tool_logs_for_display(logs):
    if not logs:
        return ""

    lines = ["[工具调用日志]"]
    for line in logs:
        cleaned = line
        idx = cleaned.find("] ")
        if idx != -1:
            cleaned = cleaned[idx + 2:]
        lines.append(cleaned)

    return "\n".join(lines)


def run_agent_with_upload(message, history, uploaded_file):
    if not uploaded_file:
        return "请先上传一个 CSV 或 Excel 文件，再提问。"

    file_path = str(uploaded_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"agent_run_{timestamp}.log"

    # 初始化本次请求的工具日志
    store_tools.clear_runtime_logs()
    store_tools.set_log_file(str(log_file))

    data_date_hint = get_data_date_hint(file_path)

    merged_system_message = f"""
{system_message}

补充上下文：
- 当前用户已上传数据文件，路径为：{file_path}
- 当你调用 analyze_store_sales 工具时，必须把这个路径作为 file_path 参数传入
- {data_date_hint}
"""

    messages = [
        {"role": "system", "content": merged_system_message},
    ]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    # 写入应用级日志
    save_app_log(log_file, f"[APP] 用户问题: {message}")
    save_app_log(log_file, f"[APP] 上传文件: {file_path}")
    save_app_log(log_file, f"[APP] 日期提示: {data_date_hint}")

    try:
        final_resp = None
        for resp in bot.run(messages=messages):
            final_resp = resp

        answer = extract_text_from_response(final_resp)
        tool_logs = store_tools.get_runtime_logs()
        display_logs = format_tool_logs_for_display(tool_logs)

        save_app_log(log_file, f"[APP] 最终回答:\n{answer}")

        if display_logs:
            final_text = (
                f"{display_logs}\n\n"
                f"[日志文件]\n{log_file}\n\n"
                f"{answer}"
            )
        else:
            final_text = (
                f"[日志文件]\n{log_file}\n\n"
                f"{answer}"
            )

        return final_text

    except Exception as e:
        err_msg = f"运行失败：{str(e)}"
        save_app_log(log_file, f"[APP] 异常: {err_msg}")

        tool_logs = store_tools.get_runtime_logs()
        display_logs = format_tool_logs_for_display(tool_logs)

        if display_logs:
            return f"{display_logs}\n\n[日志文件]\n{log_file}\n\n{err_msg}"
        return f"[日志文件]\n{log_file}\n\n{err_msg}"


import gradio as gr

def use_demo_file():
    return str(DEMO_FILE)

with gr.Blocks() as demo:
    gr.Markdown("## 门店经营分析 Agent")

    gr.Markdown("### 1️⃣ 上传数据 或 使用示例数据")

    file_input = gr.File(
        label="上传门店经营数据（CSV / Excel）",
        file_count="single",
        type="filepath"
    )

    demo_btn = gr.Button("👉 使用示例数据（store_sales.csv）")

    # 点击按钮时，把示例文件路径写入 file_input
    demo_btn.click(
        fn=use_demo_file,
        inputs=[],
        outputs=file_input
    )

    gr.Markdown("### 2️⃣ 提问")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="例如：S001 在 2026-03-01 到 2026-03-03 的经营表现怎么样？")
    send_btn = gr.Button("发送")

    def chat_fn(message, history, uploaded_file):
        response = run_agent_with_upload(message, history, uploaded_file)

        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return history, ""

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, chatbot, file_input],
        outputs=[chatbot, msg]
    )

    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot, file_input],
        outputs=[chatbot, msg]
    )

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    debug=True
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
    )
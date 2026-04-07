import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from pathlib import Path
import tools.rag_tools as rag_tools

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
2. 必要时调用 analyze_store_sales 查询门店经营数据；
3. 必要时调用 retrieve_operation_knowledge 检索运营知识；
4. 基于数据和知识给出清晰诊断与可执行建议。

非常重要的规则：
- analyze_store_sales 用于获取门店经营指标与趋势。
- retrieve_operation_knowledge 用于检索经营优化经验、规则和案例。
- 如果用户的问题既需要看数据，也需要给建议，优先结合两个工具。
- 如果工具返回 success=false 或 error，不要编造结论，要明确说明错误原因。
- 输出尽量采用以下结构：
  - 问题判断
  - 关键发现
  - 可能原因
  - 建议动作
  - 参考依据
"""

bot = Assistant(
    llm=llm_cfg,
    system_message=system_message,
    function_list=[
        "analyze_store_sales",
        "retrieve_operation_knowledge",
    ]
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


def run_agent(message, history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"agent_run_{timestamp}.log"

    store_tools.clear_runtime_logs()
    rag_tools.clear_runtime_logs()

    store_tools.set_log_file(str(log_file))
    rag_tools.set_log_file(str(log_file))

    messages = [
        {"role": "system", "content": system_message},
    ]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    save_app_log(log_file, f"[APP] 用户问题: {message}")

    try:
        final_resp = None
        for resp in bot.run(messages=messages):
            final_resp = resp

        answer = extract_text_from_response(final_resp)
        tool_logs = store_tools.get_runtime_logs()
        display_logs = format_tool_logs_for_display(tool_logs)
        for resp in bot.run(messages=messages):
            print("RAW RESP:", resp)
            final_resp = resp

        save_app_log(log_file, f"[APP] 最终回答:\n{answer}")

        if display_logs:
            return f"{display_logs}\n\n[日志文件]\n{log_file}\n\n{answer}"
        return f"[日志文件]\n{log_file}\n\n{answer}"

    except Exception as e:
        err_msg = f"运行失败：{str(e)}"
        save_app_log(log_file, f"[APP] 异常: {err_msg}")
        return f"[日志文件]\n{log_file}\n\n{err_msg}"
import gradio as gr

def use_demo_file():
    return str(DEMO_FILE)

with gr.Blocks() as demo:
    gr.Markdown("## 门店经营分析 Agent（SQLite版）")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="例如：S001 在 2026-03-01 到 2026-03-03 的经营表现怎么样？")
    send_btn = gr.Button("发送")

    def chat_fn(message, history):
        response = run_agent(message, history)
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
# demo.launch(
#     server_name="127.0.0.1",
#     server_port=7860,
#     debug=True
# )
demo.launch(share=True)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
    )
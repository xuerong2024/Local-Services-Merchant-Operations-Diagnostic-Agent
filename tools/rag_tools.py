import json
from datetime import datetime
from pathlib import Path

from qwen_agent.tools.base import BaseTool, register_tool
from rag.retriever import retrieve

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


def need_web_fallback(query: str, local_results: list) -> bool:
    freshness_keywords = [
        "最新", "近期", "最近", "今年", "本月", "本周",
        "2025", "2026", "政策", "新闻", "竞品", "行业动态"
    ]

    if any(k in query for k in freshness_keywords):
        return True

    if not local_results:
        return True

    best_score = local_results[0].get("score", 0)
    if len(local_results) < 2 or best_score < 0.65:
        return True

    return False


from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")

def search_web(query: str, top_k: int = 3) -> list:
    append_runtime_log(f'Executing web search for query: {query}')

    try:
        res = tavily_client.search(query=query)

        results = []
        for item in res.get("results", [])[:top_k]:
            results.append({
                "source": "web",
                "title": item.get("title"),
                "url": item.get("url"),
                "text": item.get("content"),
                "score": 0.7
            })

        return results

    except Exception as e:
        append_runtime_log(f"Web search error: {str(e)}")
        return []


def normalize_local_results(local_results: list) -> list:
    normalized = []
    for r in local_results:
        normalized.append({
            "source": "local",
            "doc_name": r.get("doc_name", ""),
            "chunk_id": r.get("chunk_id", -1),
            "score": r.get("score", 0),
            "text": r.get("text", "")
        })
    return normalized


@register_tool("retrieve_operation_knowledge")
class RetrieveOperationKnowledge(BaseTool):
    description = (
        "检索门店经营与运营优化知识库。优先查询内部知识库，"
        "在内部信息不足时可联网补充检索。"
    )

    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "要检索的经营问题，例如 转化率下降如何优化",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "integer",
            "description": "返回的知识片段数量，默认3",
            "required": False,
        },
        {
            "name": "enable_web",
            "type": "boolean",
            "description": "是否允许联网补充检索，默认 true",
            "required": False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        append_runtime_log('Start calling tool "retrieve_operation_knowledge" ...')
        append_runtime_log(params)

        try:
            args = json.loads(params)
            query = str(args["query"]).strip()
            top_k = int(args.get("top_k", 3))
            enable_web = bool(args.get("enable_web", True))

            local_results = retrieve(query=query, top_k=top_k)
            append_runtime_log(f"Local retrieved: {len(local_results)} items")

            merged_results = normalize_local_results(local_results)

            if enable_web and need_web_fallback(query, local_results):
                append_runtime_log("Trigger web retrieval...")
                web_results = search_web(query=query, top_k=top_k)
                append_runtime_log(f"Web retrieved: {len(web_results)} items")
                merged_results.extend(web_results)
            else:
                append_runtime_log("Skip web retrieval.")

            if not merged_results:
                append_runtime_log("No relevant knowledge found.")
                append_runtime_log("Finished tool calling.")
                return json.dumps({"success": True, "results": []}, ensure_ascii=False)

            append_runtime_log(
                "Retrieved knowledge: " +
                "; ".join([
                    f'{r.get("source", "unknown")}::{r.get("doc_name", r.get("title", "unknown"))}'
                    for r in merged_results
                ])
            )
            append_runtime_log("Finished tool calling.")

            return json.dumps({
                "success": True,
                "results": merged_results
            }, ensure_ascii=False)

        except Exception as e:
            append_runtime_log(f"Tool exception: {str(e)}")
            append_runtime_log("Finished tool calling.")
            return json.dumps({
                "success": False,
                "error": f"知识检索失败: {str(e)}"
            }, ensure_ascii=False)
"""Web search tool — uses Perplexity Sonar via OpenRouter."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def _web_search(ctx: ToolContext, query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # Fallback: try OpenAI direct
        api_key_oai = os.environ.get("OPENAI_API_KEY", "")
        if not api_key_oai:
            return json.dumps({"error": "No API key for web search (OPENROUTER_API_KEY or OPENAI_API_KEY)."})
        return _web_search_openai(query, api_key_oai)

    return _web_search_openrouter(query, api_key)


def _web_search_openrouter(query: str, api_key: str) -> str:
    """Use Perplexity Sonar via OpenRouter for web search."""
    try:
        import httpx
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "perplexity/sonar",
                "messages": [{"role": "user", "content": query}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            citations = data.get("citations", [])
            result: Dict[str, Any] = {"answer": answer}
            if citations:
                result["sources"] = citations[:5]
            return json.dumps(result, ensure_ascii=False, indent=2)
        return json.dumps({"error": "Unexpected response", "raw": str(data)[:500]})
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def _web_search_openai(query: str, api_key: str) -> str:
    """Fallback: use OpenAI Responses API (requires billing)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            tool_choice="auto",
            input=query,
        )
        d = resp.model_dump()
        text = ""
        for item in d.get("output", []) or []:
            if item.get("type") == "message":
                for block in item.get("content", []) or []:
                    if block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
        return json.dumps({"answer": text or "(no answer)"}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": "Search the web via OpenAI Responses API. Returns JSON with answer + sources.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]

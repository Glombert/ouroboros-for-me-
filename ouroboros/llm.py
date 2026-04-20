"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-2.5-flash-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}




# ─────────────────────────────────────────────────────────
# Google AI Studio fallback (uses OpenAI-compatible API)
# ─────────────────────────────────────────────────────────
# Uses the OpenAI-compatible endpoint:
# https://generativelanguage.googleapis.com/v1beta/openai/
# No extra SDK needed — just openai library.
# Model name mapping: OpenRouter model -> Google model name

GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

GOOGLE_MODEL_MAP: Dict[str, str] = {
    "anthropic/claude-sonnet-4.6": "gemini-2.0-flash",
    "anthropic/claude-opus-4.6": "gemini-2.5-pro-preview-05-06",
    "openai/gpt-4o": "gemini-2.0-flash",
    "openai/gpt-4o-mini": "gemini-2.0-flash-lite",
    "google/gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
    "google/gemini-2.0-flash": "gemini-2.0-flash",
}


def _map_to_google_model(openrouter_model: str) -> str:
    """Map OpenRouter model name to Google AI Studio model name."""
    if openrouter_model in GOOGLE_MODEL_MAP:
        return GOOGLE_MODEL_MAP[openrouter_model]
    # If it already looks like a Google model, use as-is
    if openrouter_model.startswith("gemini"):
        return openrouter_model
    # Extract model family from openrouter format (provider/model)
    if "/" in openrouter_model:
        _, model_part = openrouter_model.split("/", 1)
        # Try to match by model family
        for key, val in GOOGLE_MODEL_MAP.items():
            if model_part.split("-")[0] in key:
                return val
    return "gemini-2.0-flash"  # safe default

# ─────────────────────────────────────────────────────────
# Anthropic Direct API client (no OpenRouter middleman)
# ─────────────────────────────────────────────────────────
ANTHROPIC_MODEL_MAP: Dict[str, str] = {
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-5",
    "anthropic/claude-opus-4.6": "claude-opus-4-5",
    "anthropic/claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
    "anthropic/claude-opus-4": "claude-opus-4-5",
    "anthropic/claude-sonnet-4": "claude-sonnet-4-5",
}

_anthropic_client_cache = None


def get_anthropic_client() -> Optional["AnthropicDirectClient"]:
    """Return cached AnthropicDirectClient instance, or None if unavailable."""
    global _anthropic_client_cache
    if _anthropic_client_cache is not None:
        return _anthropic_client_cache
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        _anthropic_client_cache = AnthropicDirectClient(api_key=api_key)
        return _anthropic_client_cache
    except Exception as e:
        log.warning(f"Could not create AnthropicDirectClient: {e}")
        return None


def _map_to_anthropic_model(openrouter_model: str) -> str:
    """Map OpenRouter model name to native Anthropic model ID."""
    if openrouter_model in ANTHROPIC_MODEL_MAP:
        return ANTHROPIC_MODEL_MAP[openrouter_model]
    if openrouter_model.startswith("claude-"):
        return openrouter_model
    return "claude-sonnet-4-5"  # safe default



# ─────────────────────────────────────────────────────────
# DeepSeek Direct API client (OpenAI-compatible)
# ─────────────────────────────────────────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_MODEL_MAP: Dict[str, str] = {
    "deepseek/deepseek-chat": "deepseek-chat",
    "deepseek/deepseek-coder": "deepseek-coder",
    "anthropic/claude-sonnet-4.6": "deepseek-chat",
    "anthropic/claude-opus-4.6": "deepseek-chat",
    "openai/gpt-4o": "deepseek-chat",
    "openai/gpt-4o-mini": "deepseek-chat",
    "google/gemini-2.5-flash": "deepseek-chat",
}

DEEPSEEK_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),   # input/output per 1M tokens
    "deepseek-coder": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

_deepseek_client_cache = None

# Global flag: set True when OpenRouter returns 402 (insufficient funds)
# Cleared automatically when balance is restored.
_openrouter_payment_failed: bool = False


def get_deepseek_client():
    """Return cached DeepSeekClient or None if unavailable."""
    global _deepseek_client_cache
    if _deepseek_client_cache is not None:
        return _deepseek_client_cache
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    try:
        _deepseek_client_cache = DeepSeekClient(api_key=api_key)
        return _deepseek_client_cache
    except Exception as e:
        log.warning(f"Could not create DeepSeekClient: {e}")
        return None


def _map_to_deepseek_model(openrouter_model: str) -> str:
    """Map OpenRouter model name to DeepSeek model ID."""
    if openrouter_model in DEEPSEEK_MODEL_MAP:
        return DEEPSEEK_MODEL_MAP[openrouter_model]
    if openrouter_model.startswith("deepseek-"):
        return openrouter_model
    return "deepseek-chat"  # safe default


class DeepSeekClient:
    """
    Direct DeepSeek API client. OpenAI-compatible endpoint.
    Bypasses OpenRouter completely.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None:
            try:
                import openai
                self._openai_client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=DEEPSEEK_BASE_URL,
                )
            except ImportError:
                raise RuntimeError("openai package required for DeepSeekClient")
        return self._openai_client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-chat",
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 4096,
        tool_choice: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send chat to DeepSeek API. Returns (message_dict, usage_dict)."""
        native_model = _map_to_deepseek_model(model)
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": native_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage_raw = resp_dict.get("usage") or {}

        # Calculate cost from pricing table
        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        pricing = DEEPSEEK_MODEL_PRICING.get(native_model, (0.27, 1.10))
        cost = (prompt_tokens * pricing[0] + completion_tokens * pricing[1]) / 1_000_000

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
            "cached_tokens": int((usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "cost": cost,
        }

        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        return msg, usage

class AnthropicDirectClient:
    """
    Direct Anthropic API client. Bypasses OpenRouter completely.
    Supports tool_use, prompt caching (cache_control).
    Uses native anthropic SDK.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")
        return self._client

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-format messages to Anthropic format.
        Returns (system_prompt: str, messages: list).
        """
        import json as _json
        system_parts = []
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    ant_content = []
                    if content:
                        ant_content.append({"type": "text", "text": str(content)})
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = _json.loads(args)
                            except Exception:
                                args = {}
                        ant_content.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"tool_{len(ant_content)}"),
                            "name": func.get("name", ""),
                            "input": args,
                        })
                    anthropic_messages.append({"role": "assistant", "content": ant_content})
                else:
                    if isinstance(content, list):
                        anthropic_messages.append({"role": "assistant", "content": content})
                    else:
                        anthropic_messages.append({"role": "assistant", "content": str(content) if content else ""})
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                result_content = content
                if not isinstance(result_content, list):
                    result_content = [{"type": "text", "text": str(result_content) if result_content else ""}]
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result_content,
                }
                if (anthropic_messages and
                        anthropic_messages[-1]["role"] == "user" and
                        isinstance(anthropic_messages[-1]["content"], list)):
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result_block]})
                continue

            if role == "user":
                if isinstance(content, list):
                    anthropic_messages.append({"role": "user", "content": content})
                else:
                    anthropic_messages.append({"role": "user", "content": str(content) if content else ""})

        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            ant_tool: Dict[str, Any] = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            if "cache_control" in tool:
                ant_tool["cache_control"] = tool["cache_control"]
            anthropic_tools.append(ant_tool)
        return anthropic_tools

    def _convert_response(self, resp) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert native Anthropic response to OpenAI-compatible format."""
        import json as _json

        content_blocks = resp.content or []
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": _json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                        },
                    })

        msg = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls

        usage_obj = resp.usage
        prompt_tokens = getattr(usage_obj, "input_tokens", 0) or 0
        completion_tokens = getattr(usage_obj, "output_tokens", 0) or 0
        cached_tokens = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        cache_write_tokens = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0

        MODEL_PRICING_ANTHROPIC = {
            "claude-sonnet-4-5": (3.0, 0.3, 15.0),
            "claude-opus-4-5": (15.0, 1.5, 75.0),
            "claude-3-5-sonnet-20241022": (3.0, 0.3, 15.0),
            "claude-3-5-haiku-20241022": (0.8, 0.08, 4.0),
            "claude-3-haiku-20240307": (0.25, 0.03, 1.25),
        }
        model_name = resp.model or ""
        pricing = None
        for key, val in MODEL_PRICING_ANTHROPIC.items():
            if model_name.startswith(key):
                pricing = val
                break
        if pricing is None:
            pricing = (3.0, 0.3, 15.0)

        input_price, cached_price, output_price = pricing
        cost = (
            (prompt_tokens - cached_tokens) / 1_000_000 * input_price
            + cached_tokens / 1_000_000 * cached_price
            + cache_write_tokens / 1_000_000 * input_price * 1.25
            + completion_tokens / 1_000_000 * output_price
        )

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost": round(cost, 6),
            "provider": "anthropic_direct",
        }

        log.info(
            f"AnthropicDirect: {model_name} -> in={prompt_tokens} out={completion_tokens} "
            f"cached={cached_tokens} cost=${cost:.4f}"
        )
        return msg, usage

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Chat via Anthropic direct API."""
        client = self._get_client()
        anthropic_model = _map_to_anthropic_model(model)

        system_prompt, ant_messages = self._convert_messages(messages)
        ant_tools = self._convert_tools(tools) if tools else None

        kwargs: Dict[str, Any] = {
            "model": anthropic_model,
            "max_tokens": min(max_tokens, 16384),
            "messages": ant_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if ant_tools:
            kwargs["tools"] = ant_tools
            if tool_choice == "auto":
                kwargs["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                kwargs["tool_choice"] = {"type": "any"}

        resp = client.messages.create(**kwargs)
        return self._convert_response(resp)



_google_client_cache = None


def get_google_client() -> Optional["GoogleAIClient"]:
    """Return cached GoogleAIClient instance, or None if unavailable."""
    global _google_client_cache
    if _google_client_cache is not None:
        return _google_client_cache
    api_key = os.environ.get("GOOGLE_AI_API_KEY", "")
    if not api_key:
        return None
    try:
        _google_client_cache = GoogleAIClient(api_key=api_key)
        return _google_client_cache
    except Exception as e:
        log.warning(f"Could not create GoogleAIClient: {e}")
        return None


class GoogleAIClient:
    """
    Google AI Studio fallback using OpenAI-compatible API.
    Uses https://generativelanguage.googleapis.com/v1beta/openai/
    No special SDK required — works via openai library.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_AI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not set")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=GOOGLE_OPENAI_BASE,
                api_key=self._api_key,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Chat via Google AI Studio OpenAI-compatible API."""
        client = self._get_client()
        google_model = _map_to_google_model(model)

        # Clean messages: Google doesn't support cache_control, clean it out
        clean_messages = []
        for m in messages:
            cm = {k: v for k, v in m.items() if k != "cache_control"}
            if isinstance(cm.get("content"), list):
                # Clean cache_control from content blocks too
                new_content = []
                for block in cm["content"]:
                    if isinstance(block, dict):
                        new_content.append({k: v for k, v in block.items() if k != "cache_control"})
                    else:
                        new_content.append(block)
                cm["content"] = new_content
            clean_messages.append(cm)

        # Clean tools too
        clean_tools = None
        if tools:
            clean_tools = []
            for t in tools:
                ct = {k: v for k, v in t.items() if k != "cache_control"}
                clean_tools.append(ct)

        kwargs: Dict[str, Any] = {
            "model": google_model,
            "messages": clean_messages,
            "max_tokens": min(max_tokens, 8192),  # Google free tier limit
        }
        if clean_tools:
            kwargs["tools"] = clean_tools
            kwargs["tool_choice"] = tool_choice

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                log.warning("Google AI Studio quota exhausted")
            raise

        resp_dict = resp.model_dump()
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        usage = resp_dict.get("usage") or {}
        # Google doesn't return cost, set to 0 (it's free)
        usage["cost"] = 0.0
        usage["provider"] = "google_ai_studio"

        log.info(f"GoogleAIClient: {google_model} -> tokens in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")
        return msg, usage


class LLMClient:
    """OpenRouter API wrapper. All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _call_direct_apis(
        self,
        messages,
        model: str,
        tools=None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ):
        """
        Route call to a direct API (Anthropic / Google AI Studio / DeepSeek).
        Priority:
          1. Anthropic Direct — if model is an Anthropic model
          2. DeepSeek — lightweight everyday tasks (cheap)
          3. Google AI Studio — as secondary backup
          Raises the last exception if all fail.
        """
        last_err = None

        # 1. Anthropic Direct — best for Anthropic models, also good for code tasks
        if model.startswith("anthropic/"):
            anthropic_cl = get_anthropic_client()
            if anthropic_cl is not None:
                log.info("Direct API → Anthropic")
                try:
                    return anthropic_cl.chat(
                        messages=messages, model=model, tools=tools,
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens, tool_choice=tool_choice,
                    )
                except Exception as e:
                    log.error(f"Anthropic Direct failed: {e}")
                    last_err = e

        # 2. DeepSeek Direct — cheap, handles general tasks well
        deepseek_cl = get_deepseek_client()
        if deepseek_cl is not None:
            log.info("Direct API → DeepSeek")
            try:
                return deepseek_cl.chat(
                    messages=messages, model=model, tools=tools,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens, tool_choice=tool_choice,
                )
            except Exception as e:
                log.error(f"DeepSeek Direct failed: {e}")
                last_err = e

        # 3. Google AI Studio — fallback of fallbacks
        google_cl = get_google_client()
        if google_cl is not None:
            log.warning("Direct API → Google AI Studio")
            try:
                return google_cl.chat(
                    messages=messages, model=model, tools=tools,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens, tool_choice=tool_choice,
                )
            except Exception as e:
                log.error(f"Google AI Studio failed: {e}")
                last_err = e

        raise RuntimeError(
            "All direct API providers failed. Check your API keys and balances."
            + (f" Last error: {last_err}" if last_err else "")
        )

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching
        if model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            # Add cache_control to last tool for Anthropic prompt caching
            # This caches all tool schemas (they never change between calls)
            tools_with_cache = [t for t in tools]  # shallow copy
            if tools_with_cache:
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        global _openrouter_payment_failed

        # If OpenRouter is known to be out of funds — skip it entirely
        if _openrouter_payment_failed:
            log.warning("OpenRouter payment failed flag set — skipping to direct APIs")
            return self._call_direct_apis(
                messages=messages, model=model, tools=tools,
                reasoning_effort=reasoning_effort, max_tokens=max_tokens,
                tool_choice=tool_choice,
            )

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as openrouter_err:
            err_str = str(openrouter_err)
            # Detect 402 Payment Required — no point retrying, switch to direct APIs
            is_payment_error = ("402" in err_str or
                                "insufficient" in err_str.lower() or
                                "payment" in err_str.lower() or
                                "credit" in err_str.lower() and "balance" in err_str.lower())
            if is_payment_error:
                _openrouter_payment_failed = True
                log.warning(f"OpenRouter 402 payment error — switching to direct APIs permanently: {err_str[:120]}")
            else:
                log.warning(f"OpenRouter failed: {err_str[:120]}")

            return self._call_direct_apis(
                messages=messages, model=model, tools=tools,
                reasoning_effort=reasoning_effort, max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models

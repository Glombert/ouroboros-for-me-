"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().

Fallback chain: OpenRouter → Anthropic Direct → DeepSeek → Google AI Studio
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"

# Global flag: set to True when OpenRouter returns 402/401 or any failure
# Forces all subsequent calls to use direct APIs
_openrouter_payment_failed = False


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


# ---------------------------------------------------------------------------
# Direct API helpers (called when OpenRouter is unavailable)
# ---------------------------------------------------------------------------

def _call_anthropic_direct(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call Anthropic API directly using the anthropic library."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    client = anthropic.Anthropic(api_key=api_key)

    # Convert messages: separate system message
    system_msg = ""
    converted: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "system":
            if isinstance(content, list):
                system_msg = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            else:
                system_msg = str(content)
        elif role in ("user", "assistant"):
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        if p.get("type") == "text":
                            parts.append({"type": "text", "text": p.get("text", "")})
                        elif p.get("type") == "tool_result":
                            parts.append({
                                "type": "tool_result",
                                "tool_use_id": p.get("tool_use_id", ""),
                                "content": str(p.get("content", "")),
                            })
                    else:
                        parts.append({"type": "text", "text": str(p)})
                converted.append({"role": role, "content": parts})
            elif m.get("tool_calls"):
                # Assistant message with tool calls
                parts: List[Dict[str, Any]] = []
                if content:
                    parts.append({"type": "text", "text": str(content)})
                for tc in m.get("tool_calls", []):
                    import json
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    parts.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": parts})
            else:
                converted.append({"role": role, "content": str(content) if content else ""})
        elif role == "tool":
            # Tool result
            tool_result = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": str(m.get("content", "")),
            }
            if converted and converted[-1]["role"] == "user":
                if isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(tool_result)
                else:
                    converted[-1]["content"] = [
                        {"type": "text", "text": converted[-1]["content"]},
                        tool_result,
                    ]
            else:
                converted.append({"role": "user", "content": [tool_result]})

    # Convert tools to Anthropic format
    anthropic_tools = []
    if tools:
        for t in tools:
            if t.get("type") == "function":
                func = t.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

    kwargs: Dict[str, Any] = {
        "model": "claude-sonnet-4-5",
        "max_tokens": min(max_tokens, 8096),
        "messages": converted,
    }
    if system_msg:
        kwargs["system"] = system_msg
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    response = client.messages.create(**kwargs)

    # Convert response to OpenAI-compatible format
    import json as _json
    content_text = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": _json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                },
            })

    msg: Dict[str, Any] = {"role": "assistant", "content": content_text}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        # claude-sonnet-4-5: $3/1M input, $15/1M output
        "cost": (response.usage.input_tokens * 3.0 + response.usage.output_tokens * 15.0) / 1_000_000,
    }

    return msg, usage


def _call_deepseek_direct(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call DeepSeek API directly (OpenAI-compatible)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # Strip cache_control from tools (DeepSeek doesn't support it)
    import copy
    clean_tools = None
    if tools:
        clean_tools = []
        for t in tools:
            ct = copy.deepcopy(t)
            ct.pop("cache_control", None)
            clean_tools.append(ct)

    kwargs: Dict[str, Any] = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": min(max_tokens, 4096),
    }
    if clean_tools:
        kwargs["tools"] = clean_tools
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**kwargs)
    resp_dict = resp.model_dump()
    choices = resp_dict.get("choices") or [{}]
    msg = (choices[0] if choices else {}).get("message") or {}
    usage_raw = resp_dict.get("usage") or {}

    prompt_t = usage_raw.get("prompt_tokens", 0)
    completion_t = usage_raw.get("completion_tokens", 0)
    usage = {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": prompt_t + completion_t,
        # deepseek-chat: $0.14/1M input, $0.28/1M output
        "cost": (prompt_t * 0.14 + completion_t * 0.28) / 1_000_000,
    }

    return msg, usage


def _call_google_direct(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call Google AI Studio API (OpenAI-compatible endpoint)."""
    api_key = os.environ.get("GOOGLE_AI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_AI_API_KEY not set")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Strip cache_control from tools
    import copy
    clean_tools = None
    if tools:
        clean_tools = []
        for t in tools:
            ct = copy.deepcopy(t)
            ct.pop("cache_control", None)
            clean_tools.append(ct)

    kwargs: Dict[str, Any] = {
        "model": "gemini-2.0-flash",
        "messages": messages,
        "max_tokens": min(max_tokens, 4096),
    }
    if clean_tools:
        kwargs["tools"] = clean_tools
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**kwargs)
    resp_dict = resp.model_dump()
    choices = resp_dict.get("choices") or [{}]
    msg = (choices[0] if choices else {}).get("message") or {}
    usage_raw = resp_dict.get("usage") or {}

    prompt_t = usage_raw.get("prompt_tokens", 0)
    completion_t = usage_raw.get("completion_tokens", 0)
    usage = {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": prompt_t + completion_t,
        "cost": 0.0,  # free tier
    }

    return msg, usage


# ---------------------------------------------------------------------------
# Main LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenRouter API wrapper with direct API fallback."""

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
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
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
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost).

        Fallback chain: OpenRouter → Anthropic Direct → DeepSeek → Google AI Studio
        """
        global _openrouter_payment_failed
        effort = normalize_reasoning_effort(reasoning_effort)

        # ── Try OpenRouter first (unless we know it's unavailable) ──────────
        if not _openrouter_payment_failed:
            try:
                client = self._get_client()
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
                    tools_with_cache = [t for t in tools]
                    if tools_with_cache:
                        last_tool = {**tools_with_cache[-1]}
                        last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                        tools_with_cache[-1] = last_tool
                    kwargs["tools"] = tools_with_cache
                    kwargs["tool_choice"] = tool_choice

                resp = client.chat.completions.create(**kwargs)
                resp_dict = resp.model_dump()
                usage = resp_dict.get("usage") or {}
                choices = resp_dict.get("choices") or [{}]
                msg = (choices[0] if choices else {}).get("message") or {}

                # Treat empty response as failure → fallback
                if not msg or (not msg.get("content") and not msg.get("tool_calls")):
                    raise RuntimeError("OpenRouter returned empty response")

                # Extract cached_tokens from prompt_tokens_details if available
                if not usage.get("cached_tokens"):
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                        usage["cached_tokens"] = int(prompt_details["cached_tokens"])

                if not usage.get("cache_write_tokens"):
                    prompt_details_for_write = usage.get("prompt_tokens_details") or {}
                    if isinstance(prompt_details_for_write, dict):
                        cache_write = (prompt_details_for_write.get("cache_write_tokens")
                                      or prompt_details_for_write.get("cache_creation_tokens")
                                      or prompt_details_for_write.get("cache_creation_input_tokens"))
                        if cache_write:
                            usage["cache_write_tokens"] = int(cache_write)

                if not usage.get("cost"):
                    gen_id = resp_dict.get("id") or ""
                    if gen_id:
                        cost = self._fetch_generation_cost(gen_id)
                        if cost is not None:
                            usage["cost"] = cost

                return msg, usage

            except Exception as e:
                err_str = str(e)
                if "402" in err_str or "insufficient" in err_str.lower() or "payment" in err_str.lower():
                    log.warning("OpenRouter payment required (402) — switching to direct APIs permanently")
                else:
                    log.warning(f"OpenRouter failed ({err_str[:120]}) — switching to direct APIs")
                _openrouter_payment_failed = True

        # ── Direct API fallback chain ────────────────────────────────────────
        log.info("Using direct API fallback: Anthropic → DeepSeek → Google")
        errors: List[str] = []

        try:
            log.info("Trying Anthropic Direct...")
            msg, usage = _call_anthropic_direct(messages, tools=tools, max_tokens=max_tokens)
            if msg and (msg.get("content") or msg.get("tool_calls")):
                usage["provider"] = "anthropic_direct"
                log.info("✓ Anthropic Direct: success")
                return msg, usage
        except Exception as e:
            errors.append(f"Anthropic: {str(e)[:100]}")
            log.warning(f"Anthropic Direct failed: {e}")

        try:
            log.info("Trying DeepSeek Direct...")
            msg, usage = _call_deepseek_direct(messages, tools=tools, max_tokens=max_tokens)
            if msg and (msg.get("content") or msg.get("tool_calls")):
                usage["provider"] = "deepseek_direct"
                log.info("✓ DeepSeek Direct: success")
                return msg, usage
        except Exception as e:
            errors.append(f"DeepSeek: {str(e)[:100]}")
            log.warning(f"DeepSeek Direct failed: {e}")

        try:
            log.info("Trying Google AI Studio Direct...")
            msg, usage = _call_google_direct(messages, tools=tools, max_tokens=max_tokens)
            if msg and (msg.get("content") or msg.get("tool_calls")):
                usage["provider"] = "google_direct"
                log.info("✓ Google AI Studio: success")
                return msg, usage
        except Exception as e:
            errors.append(f"Google: {str(e)[:100]}")
            log.warning(f"Google Direct failed: {e}")

        raise RuntimeError(f"All providers failed. Errors: {'; '.join(errors)}")

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

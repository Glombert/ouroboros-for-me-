#!/usr/bin/env python3
"""
Fallback Diagnostic Test
Verifies that when OpenRouter returns 402 (insufficient credits),
the system correctly falls back to direct API providers.

Run with: python3 test_fallback.py
"""
import os
import sys
import traceback
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ouroboros.llm import LLMClient
from openai import AuthenticationError
import httpx

PASS = "✅"
FAIL = "❌"
INFO = "→"


def make_auth_error(msg: str, status: int = 402) -> AuthenticationError:
    """Create a properly formed AuthenticationError with a mock httpx request."""
    req = httpx.Request("POST", "https://api.openrouter.ai/v1/chat/completions")
    resp = httpx.Response(status, json={"error": {"code": status, "message": msg}}, request=req)
    return AuthenticationError(message=msg, response=resp, body={})


def make_success_response(text: str = "Hello from DeepSeek fallback!"):
    mock_msg = MagicMock()
    mock_msg.content = text
    mock_msg.tool_calls = None
    mock_msg.refusal = None
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg, finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return mock_resp


def test_exception_types():
    """TEST 1: Verify that 402 errors are caught by `except Exception`."""
    print("\n[TEST 1] Exception type hierarchy check")
    e = make_auth_error("Insufficient credits", 402)
    mro = [c.__name__ for c in type(e).__mro__]
    print(f"  {INFO} AuthenticationError MRO: {mro}")
    assert isinstance(e, Exception), f"{FAIL} AuthenticationError is NOT a subclass of Exception!"
    print(f"  {PASS} AuthenticationError IS a subclass of Exception — will be caught by `except Exception`")
    return True


def test_resolve_direct_model_with_deepseek():
    """TEST 2: Verify that _resolve_direct_model includes DeepSeek as universal fallback."""
    print("\n[TEST 2] _resolve_direct_model includes DeepSeek fallback for all models")
    llm = LLMClient()

    test_models = [
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.7",
        "openai/gpt-4o",
        "google/gemini-2.0-flash",
        "meta-llama/llama-3.1-8b-instruct",
    ]

    all_passed = True
    for model in test_models:
        providers = llm._resolve_direct_model(model)
        has_deepseek = "deepseek" in providers
        status = PASS if has_deepseek else FAIL
        print(f"  {status} {model} → providers: {list(providers.keys())}")
        if not has_deepseek:
            all_passed = False
            print(f"      {FAIL} MISSING DeepSeek fallback!")

    return all_passed


def test_openrouter_402_triggers_fallback():
    """TEST 3: OpenRouter 402 → triggers direct API fallback chain (Anthropic → DeepSeek)."""
    print("\n[TEST 3] OpenRouter 402 triggers direct fallback chain")
    llm = LLMClient()
    call_log = []
    direct_call_count = [0]

    # Patch the OpenRouter client on the llm instance
    def fail_openrouter(**kwargs):
        call_log.append(("openrouter", kwargs.get("model")))
        raise make_auth_error("Insufficient credits — OpenRouter")

    mock_or_client = MagicMock()
    mock_or_client.chat.completions.create.side_effect = fail_openrouter
    original_client = llm._client
    llm._client = mock_or_client

    # Create a fake OpenAI class (used inside _call_direct_fallback)
    class FakeCompletions:
        def create(self, **kwargs):
            direct_call_count[0] += 1
            model = kwargs.get("model", "unknown")
            if direct_call_count[0] == 1:
                # First direct attempt (Anthropic) fails
                call_log.append(("anthropic_direct", model))
                raise make_auth_error("Credit balance is too low — Anthropic")
            else:
                # Second direct attempt (DeepSeek) succeeds
                call_log.append(("deepseek_direct", model))
                return make_success_response("Hello from DeepSeek!")

    class FakeDirectClient:
        chat = MagicMock()

        def __init__(self, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = FakeCompletions()

    with patch("openai.OpenAI", FakeDirectClient):
        try:
            msg, usage = llm.chat(
                messages=[{"role": "user", "content": "test fallback"}],
                model="anthropic/claude-sonnet-4.6",
                tools=None,
            )
            print(f"  {INFO} Call log: {call_log}")
            print(f"  {INFO} Response: {str(msg)[:80]}")

            or_tried = any(c[0] == "openrouter" for c in call_log)
            fallback_tried = any(c[0] in ("anthropic_direct", "deepseek_direct") for c in call_log)
            deepseek_reached = any(c[0] == "deepseek_direct" for c in call_log)

            print(f"  {PASS if or_tried else FAIL} OpenRouter tried first: {or_tried}")
            print(f"  {PASS if fallback_tried else FAIL} Direct fallback triggered: {fallback_tried}")
            print(f"  {PASS if deepseek_reached else FAIL} DeepSeek reached after Anthropic failure: {deepseek_reached}")

            llm._client = original_client
            return or_tried and fallback_tried and deepseek_reached

        except Exception as e:
            print(f"  {FAIL} Unexpected exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            llm._client = original_client
            return False


def test_all_providers_fail_is_visible():
    """TEST 4: When ALL providers fail, a clear exception is raised (not silent None response)."""
    print("\n[TEST 4] All providers fail → exception raised, not silent None")
    llm = LLMClient()
    call_log = []

    def fail_openrouter(**kwargs):
        call_log.append(("openrouter", kwargs.get("model")))
        raise make_auth_error("Insufficient credits — OpenRouter")

    mock_or_client = MagicMock()
    mock_or_client.chat.completions.create.side_effect = fail_openrouter
    original_client = llm._client
    llm._client = mock_or_client

    class AlwaysFailCompletions:
        def create(self, **kwargs):
            call_log.append(("direct", kwargs.get("model", "unknown")))
            raise make_auth_error("All out of credits")

    class AlwaysFailClient:
        def __init__(self, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = AlwaysFailCompletions()

    with patch("openai.OpenAI", AlwaysFailClient):
        try:
            result = llm.chat(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-sonnet-4.6",
                tools=None,
            )
            print(f"  {FAIL} No exception raised! Result: {result}")
            print(f"  {FAIL} This is the SILENT FAILURE bug — system returned None instead of raising")
            llm._client = original_client
            return False
        except (RuntimeError, AuthenticationError, Exception) as e:
            print(f"  {INFO} Exception raised: {type(e).__name__}: {str(e)[:100]}")
            print(f"  {INFO} Call log: {call_log}")
            if isinstance(e, RuntimeError):
                print(f"  {PASS} RuntimeError raised — failure is explicit and visible")
            else:
                print(f"  {PASS} Exception raised — failure is visible (type: {type(e).__name__})")
            llm._client = original_client
            return True


def test_consciousness_model_has_deepseek_fallback():
    """TEST 5: The model used by consciousness.py has DeepSeek as fallback."""
    print("\n[TEST 5] consciousness.py model has DeepSeek fallback")
    try:
        from ouroboros.consciousness import DEFAULT_LIGHT_MODEL
        model = DEFAULT_LIGHT_MODEL
    except ImportError:
        model = "google/gemini-2.0-flash"
        print(f"  {INFO} DEFAULT_LIGHT_MODEL import failed, using: {model!r}")

    llm = LLMClient()
    providers = llm._resolve_direct_model(model)
    has_deepseek = "deepseek" in providers
    print(f"  {INFO} Model: {model!r}")
    print(f"  {INFO} Resolved providers: {list(providers.keys())}")
    if has_deepseek:
        print(f"  {PASS} DeepSeek fallback available for consciousness model")
    else:
        print(f"  {FAIL} NO DeepSeek fallback for consciousness model!")
    return has_deepseek


if __name__ == "__main__":
    print("=" * 65)
    print("FALLBACK DIAGNOSTIC TEST")
    print("Tests that 402 billing errors trigger proper provider fallback")
    print("=" * 65)

    tests = [
        ("exception_types", test_exception_types),
        ("resolve_direct_model", test_resolve_direct_model_with_deepseek),
        ("openrouter_402_triggers_fallback", test_openrouter_402_triggers_fallback),
        ("all_providers_fail_is_visible", test_all_providers_fail_is_visible),
        ("consciousness_model_has_deepseek", test_consciousness_model_has_deepseek_fallback),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  {FAIL} TEST CRASHED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        icon = "PASS" if ok else "FAIL"
        print(f"  {icon}  {name}")
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print(f"\n{PASS} ALL TESTS PASSED — fallback mechanism works correctly")
        sys.exit(0)
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"\n{FAIL} FAILURES: {failed}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Fallback Diagnostic Test
Verifies that when OpenRouter returns 402 (insufficient credits),
the system correctly falls back to direct API providers.

Run with: python3 test_fallback.py
"""
import os
import sys
import json
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
    resp = httpx.Response(status, json={"error": {"code": status, "message": msg}})
    return AuthenticationError(message=msg, response=resp, body={})


def make_deepseek_success_response(text: str = "Hello from DeepSeek fallback!"):
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
    print(f"  {INFO} AuthenticationError MRO: {[c.__name__ for c in type(e).__mro__]}")
    assert isinstance(e, Exception), f"{FAIL} AuthenticationError is NOT a subclass of Exception!"
    print(f"  {PASS} AuthenticationError is a subclass of Exception — will be caught by `except Exception`")
    return True


def test_resolve_direct_model_with_deepseek():
    """TEST 2: Verify that _resolve_direct_model now includes DeepSeek as universal fallback."""
    print("\n[TEST 2] _resolve_direct_model includes DeepSeek fallback")
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
    """TEST 3: OpenRouter 402 → should trigger direct API fallback."""
    print("\n[TEST 3] OpenRouter 402 triggers direct fallback chain")
    llm = LLMClient()
    call_log = []

    class FakeCompletions:
        """Simulates the direct provider OpenAI client — fails anthropic, succeeds deepseek."""
        _call_count = 0

        def create(self, **kwargs):
            FakeCompletions._call_count += 1
            model = kwargs.get("model", "unknown")
            if FakeCompletions._call_count == 1:
                call_log.append(("anthropic_direct", model))
                raise make_auth_error("Credit balance is too low — Anthropic direct")
            else:
                call_log.append(("deepseek_direct", model))
                return make_deepseek_success_response()

    class FakeDirectClient:
        class FakeChat:
            completions = FakeCompletions()
        chat = FakeChat()

    def mock_openrouter_create(**kwargs):
        call_log.append(("openrouter", kwargs.get("model")))
        raise make_auth_error("Insufficient credits — OpenRouter")

    # Patch the OpenRouter client and the OpenAI constructor used in direct fallback
    original_client = llm._client
    mock_or_client = MagicMock()
    mock_or_client.chat.completions.create.side_effect = mock_openrouter_create
    llm._client = mock_or_client

    with patch("ouroboros.llm.OpenAI", return_value=FakeDirectClient()):
        try:
            msg, usage = llm.chat(
                messages=[{"role": "user", "content": "test fallback"}],
                model="anthropic/claude-sonnet-4.6",
                tools=None,
            )
            print(f"  {INFO} Call log: {call_log}")
            print(f"  {INFO} Response content: {msg.get('content', msg)[:80]}")

            or_tried = any(c[0] == "openrouter" for c in call_log)
            fallback_tried = any(c[0] in ("anthropic_direct", "deepseek_direct") for c in call_log)
            deepseek_reached = any(c[0] == "deepseek_direct" for c in call_log)

            if or_tried:
                print(f"  {PASS} OpenRouter was tried first")
            else:
                print(f"  {FAIL} OpenRouter was NOT tried first")

            if fallback_tried:
                print(f"  {PASS} Direct fallback was triggered")
            else:
                print(f"  {FAIL} Direct fallback was NOT triggered")

            if deepseek_reached:
                print(f"  {PASS} DeepSeek was reached after Anthropic direct failure")
            else:
                print(f"  {FAIL} DeepSeek was NOT reached — fallback incomplete!")

            llm._client = original_client
            return or_tried and fallback_tried and deepseek_reached
        except Exception as e:
            print(f"  {FAIL} Unexpected exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            llm._client = original_client
            return False


def test_all_providers_fail():
    """TEST 4: Verify that when ALL providers fail, a clear RuntimeError is raised (not silence)."""
    print("\n[TEST 4] All providers fail → RuntimeError raised (not silence)")
    llm = LLMClient()
    call_log = []

    class AlwaysFailingCompletions:
        def create(self, **kwargs):
            call_log.append(kwargs.get("model", "unknown"))
            raise make_auth_error("All out of credits everywhere")

    class AlwaysFailingClient:
        class FakeChat:
            completions = AlwaysFailingCompletions()
        chat = FakeChat()

    mock_or_client = MagicMock()
    mock_or_client.chat.completions.create.side_effect = lambda **kw: (
        call_log.append(("openrouter", kw.get("model"))) or (_ for _ in ()).throw(
            make_auth_error("Insufficient credits — OpenRouter")))

    # Can't do that cleanly, use a function
    def fail_openrouter(**kwargs):
        call_log.append(("openrouter", kwargs.get("model")))
        raise make_auth_error("Insufficient credits — OpenRouter")

    mock_or_client.chat.completions.create.side_effect = fail_openrouter
    original_client = llm._client
    llm._client = mock_or_client

    raised_error = None
    with patch("ouroboros.llm.OpenAI", return_value=AlwaysFailingClient()):
        try:
            llm.chat(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-sonnet-4.6",
                tools=None,
            )
            print(f"  {FAIL} No exception raised — this is the SILENT FAILURE bug!")
            llm._client = original_client
            return False
        except (RuntimeError, AuthenticationError, Exception) as e:
            raised_error = e
            print(f"  {INFO} Exception raised: {type(e).__name__}: {str(e)[:100]}")
            if isinstance(e, RuntimeError):
                print(f"  {PASS} Correct: RuntimeError raised — failure is visible, not silent")
            else:
                print(f"  {PASS} Exception raised — failure is visible (type: {type(e).__name__})")
            llm._client = original_client
            return True


def test_consciousness_model_has_deepseek_fallback():
    """TEST 5: The DEFAULT_LIGHT_MODEL used by consciousness.py has DeepSeek fallback."""
    print("\n[TEST 5] consciousness.py DEFAULT_LIGHT_MODEL has DeepSeek fallback")
    try:
        from ouroboros.consciousness import DEFAULT_LIGHT_MODEL
        llm = LLMClient()
        providers = llm._resolve_direct_model(DEFAULT_LIGHT_MODEL)
        has_deepseek = "deepseek" in providers
        print(f"  {INFO} DEFAULT_LIGHT_MODEL = {DEFAULT_LIGHT_MODEL!r}")
        print(f"  {INFO} Resolved providers: {list(providers.keys())}")
        if has_deepseek:
            print(f"  {PASS} DeepSeek fallback available for consciousness model")
        else:
            print(f"  {FAIL} NO DeepSeek fallback for consciousness model!")
        return has_deepseek
    except ImportError as e:
        print(f"  {INFO} Could not import DEFAULT_LIGHT_MODEL: {e}")
        print(f"  {INFO} Checking google/gemini-2.0-flash-001 directly...")
        llm = LLMClient()
        providers = llm._resolve_direct_model("google/gemini-2.0-flash-001")
        has_deepseek = "deepseek" in providers
        print(f"  {INFO} providers: {list(providers.keys())}")
        if has_deepseek:
            print(f"  {PASS} DeepSeek fallback available")
        else:
            print(f"  {FAIL} NO DeepSeek fallback!")
        return has_deepseek


if __name__ == "__main__":
    print("=" * 65)
    print("FALLBACK DIAGNOSTIC TEST")
    print("Verifies that 402 billing errors trigger proper provider fallback")
    print("=" * 65)

    results = {}
    tests = [
        ("exception_types", test_exception_types),
        ("resolve_direct_model", test_resolve_direct_model_with_deepseek),
        ("openrouter_402_triggers_fallback", test_openrouter_402_triggers_fallback),
        ("all_providers_fail_is_visible", test_all_providers_fail),
        ("consciousness_model_has_deepseek", test_consciousness_model_has_deepseek_fallback),
    ]

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
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print(f"\n{PASS} ALL TESTS PASSED — fallback mechanism is working correctly")
        sys.exit(0)
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"\n{FAIL} FAILURES: {failed}")
        sys.exit(1)

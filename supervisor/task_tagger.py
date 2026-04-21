"""
Task Tagger — assigns semantic tags to tasks for model routing.

Tags are orthogonal to task type (which is structural).
  task_tag  = WHAT kind of thinking is needed.
  task_type = HOW the task is routed/handled.

Usage:
    from supervisor.task_tagger import tag_task
    task_tag, clean_text, source = tag_task(user_message)
"""
from __future__ import annotations
import re
import logging

log = logging.getLogger(__name__)

# Valid semantic tags
VALID_TAGS = frozenset({
    "code",      # Writing/editing code, debugging, refactoring
    "research",  # Analysis, investigation, log reading
    "chat",      # Casual conversation, questions, brainstorming
    "test",      # Running tests, linters, validators
    "evolution", # Autonomous self-improvement cycles
    "diagnose",  # Root-cause analysis, tracing bugs
    "admin",     # Server setup, configuration, installs
    "design",    # Architecture, system design, spec-writing
})

DEFAULT_TAG = "chat"

# Manual tag pattern: [tag] at start of message (case insensitive)
_TAG_PATTERN = re.compile(r"^\[([a-zA-Z]+)\]\s*", re.IGNORECASE)

# Keyword heuristics for free/offline fallback
_TAG_KEYWORDS: dict[str, list[str]] = {
    "code":      ["исправь", "напиши", "рефакторинг", "баг", "fix", "write", "refactor",
                  "implement", "edit", "код", "code", "функци", "class ", "module",
                  "def ", "import ", "python", "js ", "typescript"],
    "test":      ["тест", "проверь", "запусти тест", "run tests", "run test", "lint",
                  "validate", "pytest", "unittest", "coverage"],
    "research":  ["изучи", "найди", "проанализируй", "research", "analyze", "investigate",
                  "поищи", "что такое", "what is", "how does", "explain", "расскажи"],
    "diagnose":  ["почему", "что сломалось", "не работает", "debug", "why", "root cause",
                  "сломалось", "error", "exception", "traceback", "failed", "сломан"],
    "admin":     ["установи", "настрой", "сервер", "install", "setup", "systemd",
                  "config", "деплой", "deploy", "сервис", "service", "server", "docker"],
    "design":    ["спроектируй", "архитектур", "design", "architect", "план",
                  "plan", "как устроен", "придумай", "предложи систему", "как сделать"],
    "evolution": ["evolv", "эволюц", "/evolve"],
    # "chat": [] — default fallback, no keywords needed
}

# Model routing matrix: tag → preferred model chain
# Each entry: primary model + fallback chain (ordered by preference).
# The first AVAILABLE model in the chain is used.
TAG_MODEL_ROUTING: dict[str, dict] = {
    # Heavy thinking: Claude is best for code/architecture/diagnosis
    "code":      {"primary": "anthropic/claude-sonnet-4.6",
                  "fallbacks": ["deepseek/deepseek-chat", "google/gemini-2.0-flash"]},
    "diagnose":  {"primary": "anthropic/claude-sonnet-4.6",
                  "fallbacks": ["deepseek/deepseek-chat", "google/gemini-2.0-flash"]},
    "design":    {"primary": "anthropic/claude-sonnet-4.6",
                  "fallbacks": ["deepseek/deepseek-chat", "google/gemini-2.0-flash"]},
    "research":  {"primary": "anthropic/claude-sonnet-4.6",
                  "fallbacks": ["deepseek/deepseek-chat", "google/gemini-2.0-flash"]},
    "evolution": {"primary": "anthropic/claude-sonnet-4.6",
                  "fallbacks": ["deepseek/deepseek-chat", "google/gemini-2.0-flash"]},
    # Conversational: DeepSeek is great for chat (cheap, fast, fluent)
    "chat":      {"primary": "deepseek/deepseek-chat",
                  "fallbacks": ["google/gemini-2.0-flash", "anthropic/claude-sonnet-4.6"]},
    "admin":     {"primary": "deepseek/deepseek-chat",
                  "fallbacks": ["google/gemini-2.0-flash", "anthropic/claude-sonnet-4.6"]},
    # Mechanical/validation: Gemini Flash is fast and cheap
    "test":      {"primary": "google/gemini-2.0-flash",
                  "fallbacks": ["deepseek/deepseek-chat", "anthropic/claude-sonnet-4.6"]},
}


def extract_manual_tag(text: str) -> tuple[str | None, str]:
    """Extract manual [tag] prefix from message text.

    Returns (tag, cleaned_text):
    - tag is None if not found or invalid.
    - cleaned_text has the [tag] prefix removed (same as input if no tag).
    """
    m = _TAG_PATTERN.match(text.strip())
    if m:
        candidate = m.group(1).lower()
        if candidate in VALID_TAGS:
            return candidate, text[m.end():].strip()
    return None, text


def infer_tag_from_keywords(text: str) -> str:
    """Fast keyword-based tag inference. Free, works offline.

    Returns a tag from VALID_TAGS. Falls back to DEFAULT_TAG ("chat").
    """
    text_lower = text.lower()
    for tag, keywords in _TAG_KEYWORDS.items():
        if keywords and any(kw in text_lower for kw in keywords):
            return tag
    return DEFAULT_TAG


def tag_task(text: str) -> tuple[str, str, str]:
    """Determine tag, clean text, and source for a task message.

    Priority: manual [tag] > keyword heuristic > default

    Returns (tag, cleaned_text, source) where:
    - tag: str from VALID_TAGS
    - cleaned_text: text with manual [tag] prefix removed if any
    - source: "manual" | "inferred"
    """
    # 1. Manual override (highest priority)
    tag, clean = extract_manual_tag(text)
    if tag:
        log.debug("Manual tag detected: [%s]", tag)
        return tag, clean, "manual"

    # 2. Keyword heuristic (fast, free, no LLM call needed)
    tag = infer_tag_from_keywords(text)
    return tag, text, "inferred"

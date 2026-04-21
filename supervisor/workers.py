"""Worker functions for handling tasks and direct chat."""
import datetime
import json
import os
import re
import threading
import time
import traceback
import uuid
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from supervisor.git_ops import git_pull, git_status
from supervisor.queue import enqueue_task, _queue_lock
from supervisor.state import DRIVE_ROOT, append_jsonl, load_state, save_state
from supervisor.telegram import get_tg

# Import agent lazily to avoid circular imports
_agent_cache = None
_agent_lock = threading.Lock()

# Event queue for supervisor communication
_EVENT_Q = None


def _get_ctx():
    """Get multiprocessing context (for compatibility)."""
    import multiprocessing as mp
    return mp.get_context("spawn")


def get_event_q():
    """Get the current EVENT_Q, creating if needed."""
    global _EVENT_Q
    if _EVENT_Q is None:
        _EVENT_Q = _get_ctx().Queue()
    return _EVENT_Q


def _get_chat_agent():
    """Lazy load the chat agent."""
    global _agent_cache
    if _agent_cache is None:
        with _agent_lock:
            if _agent_cache is None:
                from ouroboros.agent import Agent
                _agent_cache = Agent()
    return _agent_cache


def handle_chat_direct(chat_id: int, text: str, image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None) -> None:
    """Handle a direct chat message from the owner."""
    try:
        # Extract task tag from text if present
        task_type = None
        cleaned_text = text
        
        # Look for [tag] at the beginning of the message
        tag_match = re.match(r'^\s*\[(\w+)\]\s*(.*)', text)
        if tag_match:
            tag = tag_match.group(1).lower()
            cleaned_text = tag_match.group(2).strip()
            
            # Map tag to task_type
            valid_tags = {
                'code': 'code',
                'research': 'research',
                'chat': 'chat',
                'test': 'test',
                'evolve': 'evolve',
                'diagnose': 'diagnose',
                'admin': 'admin',
            }
            
            if tag in valid_tags:
                task_type = valid_tags[tag]
        
        agent = _get_chat_agent()
        task = {
            "id": uuid.uuid4().hex[:8],
            "type": "task",
            "chat_id": chat_id,
            "text": cleaned_text,
            "_is_direct_chat": True,
        }
        
        # Add task_type if extracted
        if task_type:
            task["task_type"] = task_type
        
        if image_data:
            # image_data is (base64, mime) or (base64, mime, caption)
            task["image_base64"] = image_data[0]
            task["image_mime"] = image_data[1]
            if len(image_data) > 2 and image_data[2]:
                task["image_caption"] = image_data[2]
                # Prefer caption as task text if text is empty
                if not cleaned_text:
                    task["text"] = image_data[2]
        # Fallback for truly empty messages
        if not task["text"]:
            task["text"] = "(image attached)" if image_data else ""
        events = agent.handle_task(task)
        for e in events:
            get_event_q().put(e)
    except Exception as e:
        import traceback
        err_msg = f"⚠️ Error: {type(e).__name__}: {e}"
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "direct_chat_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        try:
            from supervisor.telegram import get_tg
            tg = get_tg()
            tg.send_message(chat_id, err_msg)
        except Exception:
            pass


def handle_worker_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle a worker task (non-direct chat)."""
    try:
        agent = _get_chat_agent()
        return agent.handle_task(task)
    except Exception as e:
        import traceback
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "worker_task_error",
                "task_id": task.get("id", "unknown"),
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        # Re-raise so worker can handle it
        raise


def handle_restart_request(reason: str) -> None:
    """Handle a restart request."""
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "restart_request",
            "reason": reason,
        },
    )
    # Signal supervisor to restart
    get_event_q().put({"type": "restart"})


def handle_promote_to_stable(reason: str) -> None:
    """Handle a promote-to-stable request."""
    try:
        from supervisor.git_ops import git_promote_to_stable
        git_promote_to_stable(reason)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "promote_to_stable",
                "reason": reason,
            },
        )
    except Exception as e:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "promote_to_stable_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        raise


def handle_switch_model(model: Optional[str] = None, effort: Optional[str] = None) -> None:
    """Handle a model switch request."""
    try:
        agent = _get_chat_agent()
        agent.switch_model(model=model, effort=effort)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "switch_model",
                "model": model,
                "effort": effort,
            },
        )
    except Exception as e:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "switch_model_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        raise


def handle_toggle_evolution(enabled: bool) -> None:
    """Handle evolution toggle request."""
    st = load_state()
    st["evolution_mode_enabled"] = enabled
    if not enabled:
        st["evolution_consecutive_failures"] = 0
    save_state(st)
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "toggle_evolution",
            "enabled": enabled,
        },
    )


def handle_toggle_consciousness(enabled: bool) -> None:
    """Handle background consciousness toggle request."""
    st = load_state()
    st["background_consciousness_enabled"] = enabled
    save_state(st)
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "toggle_consciousness",
            "enabled": enabled,
        },
    )


def handle_forward_to_worker(task_id: str, text: str) -> None:
    """Forward a message to a specific worker task."""
    try:
        # Create a task for the worker
        task = {
            "id": uuid.uuid4().hex[:8],
            "type": "worker_message",
            "original_task_id": task_id,
            "text": text,
            "_is_direct_chat": False,
        }
        enqueue_task(task)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "forward_to_worker",
                "task_id": task_id,
                "forwarded_task_id": task["id"],
            },
        )
    except Exception as e:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "forward_to_worker_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        raise


# Worker management structures
WORKERS: Dict[int, Any] = {}
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
CRASH_TS: List[float] = []
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}


def get_running_task_ids() -> List[str]:
    """Return list of task IDs currently being processed by workers."""
    return [w.busy_task_id for w in WORKERS.values() if w.busy_task_id]


class Worker:
    """Worker process wrapper."""
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.process = None
        self.busy_task_id = None
        self.busy_since = None
        self.last_heartbeat = None
        self.heartbeat_lag = 0.0
        self.terminate_requested = False
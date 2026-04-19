import asyncio
import json
import os
from typing import Dict, List, Optional

from ouroboros.tools.tool_execution import execute_tool
from ouroboros.tools.tool_handling import handle_tool_results
from ouroboros.llm import call_llm


tasks_file = "/content/drive/MyDrive/Ouroboros/tasks.json"


async def background_heartbeat(interval: int = 300):
    """
    Background task that periodically checks for pending tasks and executes them.
    :param interval: Time interval between checks (in seconds).
    """
    while True:
        await asyncio.sleep(interval)
        await process_pending_tasks()


async def process_pending_tasks():
    """
    Check the tasks.json file for pending tasks and execute them.
    """
    if not os.path.exists(tasks_file):
        return

    with open(tasks_file, "r") as f:
        tasks = json.load(f)

    for task in tasks:
        if task.get("status") == "pending":
            await execute_task(task)


async def execute_task(task: Dict):
    """
    Execute a single task from the queue.
    :param task: Task dictionary containing details like tool name, parameters, etc.
    """
    try:
        # Update task status to "in_progress"
        task["status"] = "in_progress"
        update_tasks_file(task)

        # Execute the tool or call the LLM
        if "tool" in task:
            result = await execute_tool(task["tool"], task.get("params", {}))
            await handle_tool_results(result)
        elif "llm_prompt" in task:
            response = await call_llm(task["llm_prompt"])
            # Handle the LLM response as needed

        # Update task status to "completed"
        task["status"] = "completed"
        task["result"] = result if "result" in locals() else response
        update_tasks_file(task)

    except Exception as e:
        # Update task status to "failed"
        task["status"] = "failed"
        task["error"] = str(e)
        update_tasks_file(task)


def update_tasks_file(updated_task: Dict):
    """
    Update the tasks.json file with the new task status.
    :param updated_task: Task dictionary with updated status.
    """
    if not os.path.exists(tasks_file):
        tasks = []
    else:
        with open(tasks_file, "r") as f:
            tasks = json.load(f)

    # Find and update the task in the list
    for i, task in enumerate(tasks):
        if task.get("id") == updated_task.get("id"):
            tasks[i] = updated_task
            break

    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)


async def main():
    """
    Start the autonomous loop.
    """
    # Start the background heartbeat
    asyncio.create_task(background_heartbeat())

    # Keep the main loop running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
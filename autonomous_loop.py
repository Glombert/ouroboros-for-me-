import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutonomousLoop:
    def __init__(self, tasks_file: str = "tasks.json", interval: int = 300):
        """
        Инициализация автономного цикла.
        
        :param tasks_file: Путь к файлу с задачами (JSON).
        :param interval: Интервал проверки задач в секундах.
        """
        self.tasks_file = tasks_file
        self.interval = interval
        self.tasks: List[Dict] = []

    async def load_tasks(self) -> None:
        """Загружает задачи из файла."""
        try:
            with open(self.tasks_file, "r") as f:
                self.tasks = json.load(f)
            logger.info(f"Loaded {len(self.tasks)} tasks from {self.tasks_file}")
        except FileNotFoundError:
            logger.warning(f"Tasks file {self.tasks_file} not found. Starting with empty task list.")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {self.tasks_file}. Starting with empty task list.")

    async def save_tasks(self) -> None:
        """Сохраняет задачи в файл."""
        with open(self.tasks_file, "w") as f:
            json.dump(self.tasks, f, indent=2)
        logger.info(f"Saved {len(self.tasks)} tasks to {self.tasks_file}")

    async def run(self) -> None:
        """Запускает автономный цикл."""
        await self.load_tasks()
        while True:
            logger.info("Checking for tasks...")
            for task in self.tasks:
                if not task.get("completed", False):
                    logger.info(f"Executing task: {task.get('description', 'No description')}")
                    # Здесь будет логика выполнения задачи
                    task["completed"] = True
            await self.save_tasks()
            await asyncio.sleep(self.interval)

if __name__ == "__main__":
    loop = AutonomousLoop()
    asyncio.run(loop.run())
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
        """Загрузка задач из файла."""
        try:
            with open(self.tasks_file, "r") as f:
                self.tasks = json.load(f)
            logger.info(f"Загружено {len(self.tasks)} задач из {self.tasks_file}")
        except FileNotFoundError:
            logger.warning(f"Файл {self.tasks_file} не найден. Создан новый список задач.")
            self.tasks = []
        except json.JSONDecodeError:
            logger.error(f"Ошибка декодирования JSON в файле {self.tasks_file}. Создан новый список задач.")
            self.tasks = []

    async def save_tasks(self) -> None:
        """Сохранение задач в файл."""
        with open(self.tasks_file, "w") as f:
            json.dump(self.tasks, f, indent=2)
        logger.info(f"Сохранено {len(self.tasks)} задач в {self.tasks_file}")

    async def process_tasks(self) -> None:
        """Обработка задач."""
        for task in self.tasks:
            if not task.get("completed", False):
                logger.info(f"Обработка задачи: {task.get('description', 'Без описания')}")
                # Здесь будет логика выполнения задачи
                task["completed"] = True

    async def run(self) -> None:
        """Запуск автономного цикла."""
        await self.load_tasks()
        while True:
            await self.process_tasks()
            await self.save_tasks()
            await asyncio.sleep(self.interval)

if __name__ == "__main__":
    loop = AutonomousLoop()
    asyncio.run(loop.run())
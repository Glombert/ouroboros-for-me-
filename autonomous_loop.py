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
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Загружает задачи из файла."""
        try:
            with open(self.tasks_file, "r") as f:
                self.tasks = json.load(f)
            logger.info(f"Загружено {len(self.tasks)} задач из {self.tasks_file}")
        except FileNotFoundError:
            logger.warning(f"Файл {self.tasks_file} не найден. Создан новый список задач.")
            self.tasks = []
        except json.JSONDecodeError:
            logger.error(f"Ошибка декодирования {self.tasks_file}. Создан новый список задач.")
            self.tasks = []

    def _save_tasks(self) -> None:
        """Сохраняет задачи в файл."""
        with open(self.tasks_file, "w") as f:
            json.dump(self.tasks, f, indent=2)
        logger.info(f"Сохранено {len(self.tasks)} задач в {self.tasks_file}")

    async def _process_tasks(self) -> None:
        """Обрабатывает задачи в цикле."""
        while True:
            if self.tasks:
                task = self.tasks.pop(0)
                logger.info(f"Обработка задачи: {task.get('description', 'Без описания')}")
                # Здесь будет логика выполнения задачи
                self._save_tasks()
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        """Запускает автономный цикл."""
        logger.info("Автономный цикл запущен")
        await self._process_tasks()

if __name__ == "__main__":
    loop = AutonomousLoop()
    asyncio.run(loop.start())
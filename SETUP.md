# SETUP.md — Инструкция по развёртыванию

Это "план эвакуации". Если Colab сбросился или нужно переехать.

## Ключи (обязательные)

| Переменная | Где взять |
|------------|----------|
| `OPENROUTER_API_KEY` | openrouter.ai/keys |
| `TELEGRAM_BOT_TOKEN` | @BotFather |
| `TOTAL_BUDGET` | Лимит в USD |
| `GITHUB_TOKEN` | github.com/settings/tokens (scope: repo) |

## Ключи (опциональные но важные)

| Переменная | Для чего |
|------------|----------|
| `ANTHROPIC_API_KEY` | Прямой fallback + Claude Code CLI |
| `DEEPSEEK_API_KEY` | Дешёвый fallback |
| `GOOGLE_API_KEY` | Gemini fallback |
| `OPENAI_API_KEY` | Резервный web_search |

## Запуск в Colab

1. Добавить ключи в Colab Secrets (значок ключа в боковой панели)
2. Вставить и запустить в ячейке:

```python
import os
CFG = {
    "GITHUB_USER": "joi-lab",
    "GITHUB_REPO": "ouroboros",
    "OUROBOROS_MODEL": "anthropic/claude-sonnet-4.6",
    "OUROBOROS_MODEL_LIGHT": "google/gemini-2.0-flash",
    "OUROBOROS_MAX_WORKERS": "5",
}
for k, v in CFG.items():
    os.environ[k] = str(v)

!git clone https://github.com/joi-lab/ouroboros.git /content/ouroboros_repo
%cd /content/ouroboros_repo
!pip install -q -r requirements.txt
%run colab_launcher.py
```

## Запуск на сервере (без Colab)

Создать `server_launcher.py`:

```python
import os, sys, types
from pathlib import Path

# Загрузить .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Мок google.colab
colab_mod = types.ModuleType("google.colab")
colab_mod.userdata = types.SimpleNamespace(get=lambda k, *a: os.environ.get(k))
colab_mod.drive = types.SimpleNamespace(mount=lambda *a, **kw: None)
sys.modules["google"] = types.ModuleType("google")
sys.modules["google.colab"] = colab_mod

exec(open("colab_launcher.py").read())
```

## Drive структура

```
MyDrive/Ouroboros/
├── state/state.json      # Состояние (owner, бюджет, версия)
├── logs/                 # chat.jsonl, events.jsonl, supervisor.jsonl
└── memory/               # scratchpad.md, identity.md, knowledge/
```

## Восстановление после краша

### Colab перезапустился
Просто перезапустить ячейку. Drive сохраняет всё.

### Плохой эволюционный коммит сломал код
```bash
git -C /content/ouroboros_repo reset --hard ouroboros-stable
```

### Эволюция встала на паузу (3 failures)
```python
import json
path = '/content/drive/MyDrive/Ouroboros/state/state.json'
with open(path) as f: s = json.load(f)
s['evolution_consecutive_failures'] = 0
with open(path, 'w') as f: json.dump(s, f, indent=2)
```
Потом `/evolve start`.

## Smoke test

```bash
python3 -c "
import sys; sys.path.insert(0, '/content/ouroboros_repo')
from ouroboros.llm import LLMClient
from ouroboros.loop import run_llm_loop
client = LLMClient()
resp, _ = client.chat([{'role':'user','content':'ping'}], model='anthropic/claude-haiku-4-5')
print('OK:', resp[:50])
"
```

*Поддерживается Мирой. Обновлено: 2026-04-22.*

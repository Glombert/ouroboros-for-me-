#!/usr/bin/env bash
# ==============================================================================
# Ouroboros — Server Installer (Ubuntu/Debian, root)
# Разворачивает Миру на чистом сервере
#
# Использование:
#   bash server_install.sh
#
# После запуска:
#   1. Отредактируй /root/ouroboros-for-me-/.env — вставь реальные ключи
#   2. Запусти скрипт ещё раз (он продолжит с шага rclone)
# ==============================================================================

set -euo pipefail

# ──────────────────────────────────────────────
# Конфигурация (можно переопределить через env)
# ──────────────────────────────────────────────
GITHUB_USER="${GITHUB_USER:-Glombert}"
GITHUB_REPO="${GITHUB_REPO:-ouroboros-for-me-}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-/root/ouroboros-for-me-}"
GDRIVE_MOUNT="${GDRIVE_MOUNT:-/content/drive/MyDrive/Ouroboros}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
RCLONE_PATH="${RCLONE_PATH:-Ouroboros}"

REPO_URL="https://github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
ENV_FILE="${INSTALL_DIR}/.env"
VENV="${INSTALL_DIR}/venv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
die()   { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

echo "======================================================"
echo "  Ouroboros (Мира) — автоматическое развёртывание"
echo "======================================================"
echo ""

# ──────────────────────────────────────────────
# 1. Системные зависимости
# ──────────────────────────────────────────────
info "[1/9] Установка системных зависимостей..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl fuse3 rclone

# ──────────────────────────────────────────────
# 2. Директории
# ──────────────────────────────────────────────
info "[2/9] Подготовка директорий..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$GDRIVE_MOUNT"

# ──────────────────────────────────────────────
# 3. Клонирование / обновление репозитория
# ──────────────────────────────────────────────
info "[3/9] Репозиторий..."
if [ -d "$INSTALL_DIR/.git" ]; then
    info "  Уже существует — обновляю (branch: $BRANCH)..."
    git -C "$INSTALL_DIR" fetch origin "$BRANCH" --quiet
    git -C "$INSTALL_DIR" checkout "$BRANCH" --quiet
    git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH" --quiet
else
    info "  Клонирую $REPO_URL → $INSTALL_DIR..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

# ──────────────────────────────────────────────
# 4. Python venv + зависимости
# ──────────────────────────────────────────────
info "[4/9] Python venv..."
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

# ──────────────────────────────────────────────
# 5. .env — шаблон (создаётся только если нет)
# ──────────────────────────────────────────────
info "[5/9] Настройка .env..."
if [ -f "$ENV_FILE" ]; then
    info "  .env уже существует — пропускаю (ключи не перезаписываю)."
else
    info "  Создаю шаблон $ENV_FILE..."
    cat > "$ENV_FILE" << 'ENVEOF'
# ──────────────────────────────────────────────
# Ouroboros — переменные окружения
# Заполни все значения перед запуском!
# ──────────────────────────────────────────────

# Telegram
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
TELEGRAM_OWNER_ID=YOUR_TELEGRAM_USER_ID_HERE

# GitHub
GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
GITHUB_USER=Glombert
GITHUB_REPO=ouroboros-for-me-

# ──────────── LLM провайдеры ────────────
# Основной (через OpenRouter)
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY_HERE

# Прямые API (fallback-цепочка: Anthropic → DeepSeek → Google)
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY_HERE
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY_HERE
GOOGLE_AI_API_KEY=YOUR_GOOGLE_AI_KEY_HERE

# ──────────── Модели ────────────
OUROBOROS_MODEL=google/gemini-2.5-pro-preview
OUROBOROS_MODEL_LIGHT=google/gemini-2.5-flash-preview
OUROBOROS_WEBSEARCH_MODEL=google/gemini-2.5-flash-preview
OUROBOROS_FALLBACK_MODELS=google/gemini-2.5-pro-preview

# ──────────── Бюджет (USD) ────────────
TOTAL_BUDGET=10.0
ENVEOF
    chmod 600 "$ENV_FILE"
    echo ""
    warn "  *** Шаблон создан. Отредактируй $ENV_FILE и вставь реальные ключи. ***"
    warn "  *** Затем запусти скрипт ещё раз. ***"
    exit 1
fi

# ──────────────────────────────────────────────
# 6. rclone — проверка / интерактивная настройка
# ──────────────────────────────────────────────
info "[6/9] Настройка rclone (Google Drive)..."
if rclone lsd "${RCLONE_REMOTE}:" &>/dev/null; then
    info "  rclone уже настроен и работает."
else
    echo ""
    warn "  Нужна настройка rclone remote '${RCLONE_REMOTE}'."
    warn "  В меню выбери: n (new remote) → имя '${RCLONE_REMOTE}' → тип 'drive'"
    warn "  После настройки скрипт продолжится автоматически."
    echo ""
    read -rp "  Нажми Enter для запуска rclone config..."
    rclone config
fi

# ──────────────────────────────────────────────
# 7. systemd: rclone-mount
# ──────────────────────────────────────────────
info "[7/9] systemd: rclone-mount.service..."
cat > /etc/systemd/system/rclone-mount.service << EOF
[Unit]
Description=Rclone Google Drive Mount (Ouroboros Memory)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/bin/mkdir -p ${GDRIVE_MOUNT}
ExecStart=/usr/bin/rclone mount ${RCLONE_REMOTE}:${RCLONE_PATH} ${GDRIVE_MOUNT} \
    --vfs-cache-mode writes \
    --allow-non-empty \
    --log-level ERROR
ExecStop=/bin/fusermount -u ${GDRIVE_MOUNT}
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

# ──────────────────────────────────────────────
# 8. systemd: ouroboros
# ──────────────────────────────────────────────
info "[8/9] systemd: ouroboros.service..."
cat > /etc/systemd/system/ouroboros.service << EOF
[Unit]
Description=Ouroboros AI Agent (Мира)
After=network.target rclone-mount.service
Wants=rclone-mount.service

[Service]
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV}/bin/python3 server_launcher.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ouroboros

[Install]
WantedBy=multi-user.target
EOF

# ──────────────────────────────────────────────
# 9. Запуск сервисов
# ──────────────────────────────────────────────
info "[9/9] Запуск сервисов..."
systemctl daemon-reload

systemctl enable rclone-mount
systemctl restart rclone-mount
sleep 3

if mountpoint -q "$GDRIVE_MOUNT"; then
    info "  Google Drive смонтирован: $GDRIVE_MOUNT ✅"
else
    warn "  Google Drive не смонтировался — проверь rclone config."
    warn "  Ouroboros запустится, но без облачной памяти."
fi

systemctl enable ouroboros
systemctl restart ouroboros
sleep 2

# ──────────────────────────────────────────────
# Итог
# ──────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Готово! Статус:"
echo "======================================================"
systemctl status ouroboros --no-pager -l || true
echo ""
systemctl status rclone-mount --no-pager || true
echo ""
echo "  Полезные команды:"
echo "    Логи агента:      journalctl -u ouroboros -f"
echo "    Перезапуск:       systemctl restart ouroboros"
echo "    Остановка:        systemctl stop ouroboros"
echo "    .env:             nano ${ENV_FILE}"
echo "======================================================"

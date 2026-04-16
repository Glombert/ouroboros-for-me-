#!/usr/bin/env bash
# ============================================================
# Ouroboros — Server installer (Ubuntu/Linux)
# ============================================================
# Run this script once on your server to set up Ouroboros.
# Prerequisites: Ubuntu 20.04+, Python 3.8+, git
#
# Usage:
#   bash server_install.sh
#
# After running:
#   1. Edit ~/.env and fill in your API keys
#   2. systemctl --user enable --now ouroboros
# ============================================================

set -e
REPO_DIR="${OUROBOROS_REPO_DIR:-$HOME/ouroboros_repo}"
DRIVE_ROOT="${OUROBOROS_DRIVE_ROOT:-$HOME/ouroboros_data}"
GITHUB_USER="${GITHUB_USER:-igtip}"
GITHUB_REPO="${GITHUB_REPO:-ouroboros}"
BRANCH="${BRANCH:-ouroboros}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }

# ----------------------------
# 1) Create local Drive dirs
# ----------------------------
info "Creating local Drive directories at $DRIVE_ROOT ..."
mkdir -p "$DRIVE_ROOT"/{state,logs,memory,index,locks,archive}

# ----------------------------
# 2) Clone repo (if not present)
# ----------------------------
if [ -d "$REPO_DIR/.git" ]; then
    info "Repo already present at $REPO_DIR, pulling latest ..."
    git -C "$REPO_DIR" fetch origin "$BRANCH" --quiet
    git -C "$REPO_DIR" checkout "$BRANCH" --quiet
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH" --quiet
else
    info "Cloning repo to $REPO_DIR ..."
    git clone --branch "$BRANCH" \
        "https://github.com/$GITHUB_USER/$GITHUB_REPO.git" \
        "$REPO_DIR"
fi

# ----------------------------
# 3) Install Python deps
# ----------------------------
info "Installing Python dependencies ..."
pip3 install --user --quiet -r "$REPO_DIR/requirements.txt"
pip3 install --user --quiet python-dotenv

# Check for optional playwright (not critical)
pip3 install --user --quiet playwright playwright-stealth 2>/dev/null || \
    warn "playwright not installed (optional, browser automation won't work)"

# ----------------------------
# 4) Create .env template (only if missing)
# ----------------------------
ENV_FILE="$HOME/.env"
if [ ! -f "$ENV_FILE" ]; then
    info "Creating .env template at $ENV_FILE ..."
    cat > "$ENV_FILE" << 'EOF'
# Ouroboros — environment secrets
# Fill in your values and save.

TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
GITHUB_TOKEN=your_github_token_here
GITHUB_USER=igtip
GITHUB_REPO=ouroboros
TOTAL_BUDGET=10.0

# Optional
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OUROBOROS_MODEL=anthropic/claude-sonnet-4.6
OUROBOROS_MAX_WORKERS=5
EOF
    chmod 600 "$ENV_FILE"
    warn ".env template created. Edit it now: nano ~/.env"
else
    info ".env already exists at $ENV_FILE — skipping template creation."
fi

# ----------------------------
# 5) Create systemd user service
# ----------------------------
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"

SERVICE_FILE="$SERVICE_DIR/ouroboros.service"
info "Creating systemd user service at $SERVICE_FILE ..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Ouroboros AI Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$(which python3) $REPO_DIR/server_launcher.py
EnvironmentFile=$HOME/.env
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# Enable lingering so service runs without active login session
loginctl enable-linger "$(whoami)" 2>/dev/null || \
    warn "Could not enable linger (may need root). Service will only run while logged in."

systemctl --user daemon-reload
info "systemd service created."

# ----------------------------
# Done
# ----------------------------
echo ""
echo "========================================================"
echo "  Ouroboros server setup complete!"
echo "========================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1) Fill in API keys:      nano ~/.env"
echo ""
echo "  2) Start the service:     systemctl --user enable --now ouroboros"
echo ""
echo "  3) Check logs:            journalctl --user -u ouroboros -f"
echo ""
echo "  4) Stop:                  systemctl --user stop ouroboros"
echo "  5) Restart:               systemctl --user restart ouroboros"
echo ""
echo "  Data directory:           $DRIVE_ROOT"
echo "  Repo directory:           $REPO_DIR"
echo "========================================================"

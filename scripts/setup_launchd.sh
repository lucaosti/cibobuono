#!/usr/bin/env bash
# setup_launchd.sh — Installa il LaunchAgent macOS per la pipeline CiboBuono.
#
# Uso:
#   bash scripts/setup_launchd.sh           # installa con --skip-push (default sicuro)
#   bash scripts/setup_launchd.sh --push    # installa con git push automatico
#   bash scripts/setup_launchd.sh --unload  # rimuove il LaunchAgent
#
# La pipeline girerà in background ogni 30 minuti, ripartendo al login.

set -euo pipefail

LABEL="com.cibobuono.pipeline"
PLIST_DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"

# ── Risolvi percorsi ────────────────────────────────────────────────────────

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Trova il Python del venv (cerca nell'ordine: .venv, venv, env, fallback a which python3)
for candidate in "$REPO/.venv" "$REPO/venv" "$REPO/env"; do
    if [[ -x "$candidate/bin/python" ]]; then
        VENV="$candidate"
        break
    fi
done
if [[ -z "${VENV:-}" ]]; then
    PY="$(which python3 2>/dev/null || true)"
    if [[ -z "$PY" ]]; then
        echo "ERRORE: nessun Python trovato. Crea un venv con: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        exit 1
    fi
    VENV="$(dirname "$(dirname "$PY")")"
fi

echo "Repo:  $REPO"
echo "Venv:  $VENV"
echo "Plist: $PLIST_DEST"

# ── Unload ──────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--unload" ]]; then
    if launchctl list | grep -q "$LABEL"; then
        launchctl unload "$PLIST_DEST" 2>/dev/null || true
        echo "LaunchAgent rimosso."
    else
        echo "LaunchAgent non era caricato."
    fi
    exit 0
fi

# ── Determina flag push ──────────────────────────────────────────────────────

if [[ "${1:-}" == "--push" ]]; then
    PUSH_FLAG=""
    echo "Modalità: push automatico abilitato"
else
    PUSH_FLAG="--skip-push"
    echo "Modalità: --skip-push (push manuale). Usa --push per abilitare il push automatico."
fi

# ── Crea logs/ ──────────────────────────────────────────────────────────────

mkdir -p "$REPO/logs"

# ── Genera il plist ──────────────────────────────────────────────────────────

cat > "$PLIST_DEST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${VENV}/bin/python</string>
        <string>-m</string>
        <string>scripts.run_pipeline</string>
        <string>--watch</string>
        <string>--poll-interval</string>
        <string>1800</string>
PLIST

if [[ -n "$PUSH_FLAG" ]]; then
cat >> "$PLIST_DEST" <<PLIST
        <string>--skip-push</string>
PLIST
fi

cat >> "$PLIST_DEST" <<PLIST
    </array>

    <key>WorkingDirectory</key>
    <string>${REPO}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${VENV}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ThrottleInterval</key>
    <integer>60</integer>

    <key>StandardOutPath</key>
    <string>${REPO}/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${REPO}/logs/launchd_stderr.log</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
PLIST

# ── Carica il LaunchAgent ────────────────────────────────────────────────────

# Rimuovi eventuale versione precedente
if launchctl list | grep -q "$LABEL"; then
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

launchctl load "$PLIST_DEST"

echo ""
echo "LaunchAgent installato e avviato."
echo ""
echo "Comandi utili:"
echo "  Stato:    launchctl list | grep cibobuono"
echo "  Log:      tail -f $REPO/logs/launchd_stdout.log"
echo "  Errori:   tail -f $REPO/logs/launchd_stderr.log"
echo "  Stop:     bash scripts/setup_launchd.sh --unload"

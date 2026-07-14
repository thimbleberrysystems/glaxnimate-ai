#!/usr/bin/env bash
#
# Install the live AI Bridge plugin into Glaxnimate.
#
# Usage:  bash scripts/install_plugin.sh
#
# Needs sudo only for PyQt6. Everything else lands in your home directory.

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/plugin/AiBridge"
SHARE="${XDG_DATA_HOME:-$HOME/.local/share}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# Ask the app where it looks, rather than guessing. Qt's AppDataLocation is
# <share>/<organization>/<app>, and Glaxnimate's organization is "stalefiles" —
# not "glaxnimate". Guess wrong and the plugin installs somewhere the app never
# scans: it simply never appears in the menu, with nothing to tell you why.
find_data_dir() {
    find "$SHARE" -maxdepth 2 -iname glaxnimate -type d 2>/dev/null | head -1
}

APP_DIR="$(find_data_dir)"
if [ -z "$APP_DIR" ]; then
    # Never launched: run it once offscreen so it creates its own directories.
    timeout 15 env QT_QPA_PLATFORM=offscreen glaxnimate >/dev/null 2>&1 || true
    APP_DIR="$(find_data_dir)"
fi
[ -n "$APP_DIR" ] || die "cannot find Glaxnimate's data dir under $SHARE — run the app once, then retry"

DEST="$APP_DIR/plugins/AiBridge"

say "1/2  Qt bindings"
# PyQt6 from apt links the SAME system Qt6 that Glaxnimate already has loaded, so
# there is one Qt in the process.
#
# Do NOT substitute `pip install PySide6`: it bundles its own copy of Qt6, and two
# Qts in one address space is undefined behaviour — it will crash, eventually, in a
# way that looks like anything but the real cause.
if python3 -c "import PyQt6.QtNetwork" 2>/dev/null; then
    ok "python3-pyqt6 already present"
else
    sudo apt-get install -y python3-pyqt6 python3-pyqt6.qtnetwork
    ok "installed python3-pyqt6"
fi

say "2/2  Plugin"
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -r "$SRC" "$DEST"
ok "installed to $DEST"

cat <<'EOF'

Now, in Glaxnimate:

  1. Open (or create) a document
  2. Plugins > Start AI Bridge      -> "listening on 127.0.0.1:9123"

Then from Claude:

  gui_live_status()                 -> confirms the link
  gui_live_run("comp.width = 1280") -> edits the document you are looking at

Every AI edit is a single undo step, so Ctrl+Z always puts it back.
EOF

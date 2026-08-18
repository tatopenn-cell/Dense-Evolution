#!/usr/bin/env bash
# Dense-Evolution Composer -- uninstaller (macOS / Linux). Removes every
# icon install-composer.sh could have created (Desktop, applications menu,
# autostart, both Online/Offline variants), the offline copy, and the
# launcher folder. The dense-evolution Python package itself is left alone
# unless you explicitly say yes below -- you may be using it for other
# things besides Composer.
set -e

INSTALL_DIR="$HOME/.dense-evolution-composer"
OS_NAME=$(uname -s)

echo "Disinstallazione di Dense-Evolution Composer"
echo

remove_if_exists() {
    if [ -e "$1" ]; then
        rm -rf "$1"
        echo "Rimossa: $1"
    fi
}

if [ "$OS_NAME" = "Darwin" ]; then
    remove_if_exists "$HOME/Desktop/Dense-Evolution Composer (Online).command"
    remove_if_exists "$HOME/Desktop/Dense-Evolution Composer (Offline).command"
    PLIST="$HOME/Library/LaunchAgents/com.dense-evolution.composer.plist"
    if [ -f "$PLIST" ]; then
        launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        echo "Rimosso LaunchAgent di avvio automatico."
    fi
else
    remove_if_exists "$HOME/Desktop/Dense-Evolution Composer (Online).desktop"
    remove_if_exists "$HOME/Desktop/Dense-Evolution Composer (Offline).desktop"
    remove_if_exists "$HOME/.local/share/applications/dense-evolution-composer-online.desktop"
    remove_if_exists "$HOME/.local/share/applications/dense-evolution-composer-offline.desktop"
    remove_if_exists "$HOME/.config/autostart/dense-evolution-composer.desktop"
fi

if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Cartella $INSTALL_DIR rimossa (inclusa la copia offline, se presente)."
else
    echo "Nessuna cartella di lancio trovata."
fi
echo

PYTHON_BIN=""
for candidate in python3 python; do
    command -v "$candidate" >/dev/null 2>&1 && { PYTHON_BIN="$candidate"; break; }
done

read -r -p "Disinstallare anche il pacchetto Python dense-evolution? [s/N] " REMOVE_PKG || true
if [[ "$REMOVE_PKG" =~ ^[Ss]$ ]] && [ -n "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" -m pip uninstall -y dense-evolution
else
    echo "Pacchetto dense-evolution lasciato installato."
fi

echo
echo "Fatto."

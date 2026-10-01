#!/usr/bin/env bash
# Dense-Evolution Dashboard (Streamlit) -- uninstaller (macOS / Linux).
set -e

INSTALL_DIR="$HOME/.dense-evolution-dashboard"

echo "Disinstallazione di Dense-Evolution Dashboard (Streamlit)"
for f in "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).command" "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).desktop"; do
    if [ -e "$f" ]; then
        rm -f "$f"
        echo "Rimossa: $f"
    fi
done
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Cartella $INSTALL_DIR rimossa."
fi

read -r -p "Disinstallare anche il pacchetto Python dense-evolution? [s/N] " REMOVE_PKG || true
if [[ "$REMOVE_PKG" =~ ^[Ss]$ ]]; then
    python3 -m pip uninstall -y dense-evolution || python -m pip uninstall -y dense-evolution
fi
echo "Fatto."

#!/usr/bin/env bash
# Dense-Evolution Dashboard (Streamlit) -- installer (macOS / Linux).
# See uninstall-dashboard.sh to undo everything this creates.
set -e

APP_URL="https://raw.githubusercontent.com/tatopenn-cell/Dense-Evolution/main/tools/dashboard/app.py"
INSTALL_DIR="$HOME/.dense-evolution-dashboard"
LAUNCHER="$INSTALL_DIR/launch-dashboard.sh"
OS_NAME=$(uname -s)

ask() {
    local prompt="$1" default="$2" reply
    read -r -p "$prompt " reply || true
    reply=${reply:-$default}
    [[ "$reply" =~ ^[Nn]$ ]] && return 1 || return 0
}

echo "============================================================"
echo " Dense-Evolution Dashboard (Streamlit) -- installazione"
echo "============================================================"
echo
echo "Licenza del pacchetto: Business Source License 1.1"
echo "  https://github.com/tatopenn-cell/Dense-Evolution/blob/main/LICENSE.md"
echo
if ! ask "Hai letto e accetti i termini della licenza? [s/N]" N; then
    echo "Installazione annullata."
    exit 0
fi

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3 non trovato. Installalo da https://www.python.org/downloads/ e rilancia questo script."
    exit 1
fi

echo "Installo/aggiorno dense-evolution[dashboard]..."
"$PYTHON_BIN" -m pip install --upgrade "dense-evolution[dashboard]"

mkdir -p "$INSTALL_DIR"
echo "Scarico l'app Dashboard..."
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$APP_URL" -o "$INSTALL_DIR/app.py"
else
    "$PYTHON_BIN" -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$APP_URL" "$INSTALL_DIR/app.py"
fi

cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec "$PYTHON_BIN" -m streamlit run app.py
EOF
chmod +x "$LAUNCHER"

if ask "Icona sul Desktop? [S/n]" S; then
    mkdir -p "$HOME/Desktop"
    if [ "$OS_NAME" = "Darwin" ]; then
        printf '#!/usr/bin/env bash\nexec "%s"\n' "$LAUNCHER" > "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).command"
        chmod +x "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).command"
    else
        printf '[Desktop Entry]\nType=Application\nName=Dense-Evolution Dashboard (Streamlit)\nExec=%s\nTerminal=true\nCategories=Science;\n' "$LAUNCHER" > "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).desktop"
        chmod +x "$HOME/Desktop/Dense-Evolution Dashboard (Streamlit).desktop"
    fi
fi

if ask "Avviare ora la Dashboard? [S/n]" S; then
    exec "$LAUNCHER"
fi
echo "Avviala quando vuoi con: $LAUNCHER"

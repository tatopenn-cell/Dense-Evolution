#!/usr/bin/env bash
# Dense-Evolution Composer -- local kernel installer (macOS / Linux).
# See uninstall-composer.sh to undo everything this creates.
set -e

COMPOSER_URL="https://tatopenn-cell.github.io/Dense-Evolution/composer/"
INSTALL_DIR="$HOME/.dense-evolution-composer"
OFFLINE_DIR="$INSTALL_DIR/offline"
LAUNCHER_ONLINE="$INSTALL_DIR/launch-composer-online.sh"
LAUNCHER_OFFLINE="$INSTALL_DIR/launch-composer-offline.sh"
OS_NAME=$(uname -s)

ask() {
    # ask "prompt" "S|N" -> 0 (yes) or 1 (no); default is whichever of S/N is uppercase
    local prompt="$1" default="$2" reply
    read -r -p "$prompt " reply || true
    reply=${reply:-$default}
    [[ "$reply" =~ ^[Nn]$ ]] && return 1 || return 0
}

echo "============================================================"
echo " Dense-Evolution Composer -- installazione del kernel locale"
echo "============================================================"
echo
echo "Questo script:"
echo "  1. Installa/aggiorna il pacchetto Python \"dense-evolution[composer]\""
echo "     da PyPI (dense_evolution stesso + JAX + fastapi/uvicorn/pydantic,"
echo "     solo per l'esecuzione locale dei circuiti -- nessun altro dato"
echo "     lascia questo PC)."
echo "  2. Scarica (opzionale) una copia offline della pagina Composer."
echo "  3. Crea (a tua scelta) icone di avvio -- Desktop, menu applicazioni,"
echo "     avvio automatico al login -- e puoi rimuovere tutto in qualsiasi"
echo "     momento con uninstall-composer.sh."
echo
echo "Licenza del pacchetto: Business Source License 1.1"
echo "  https://github.com/tatopenn-cell/Dense-Evolution/blob/main/LICENSE.md"
echo
echo "Nessun passo qui sotto parte da solo: hai scaricato ed eseguito questo"
echo "script di persona -- un sito web non puo' farlo al posto tuo."
echo
if ! ask "Continuare con l'installazione? [S/n]" S; then
    echo "Installazione annullata."
    exit 0
fi
echo

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3 non trovato su questo sistema."
    echo "Installalo da https://www.python.org/downloads/ (o dal package manager del tuo OS) e rilancia questo script."
    exit 1
fi
echo "Trovato $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print("Python %d.%d" % sys.version_info[:2])'))."
echo

echo "Installo/aggiorno dense-evolution[composer]..."
"$PYTHON_BIN" -m pip install --upgrade "dense-evolution[composer]"
echo

mkdir -p "$INSTALL_DIR"

cat > "$LAUNCHER_ONLINE" <<EOF
#!/usr/bin/env bash
if command -v open >/dev/null 2>&1; then open "$COMPOSER_URL" >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$COMPOSER_URL" >/dev/null 2>&1 &
fi
exec "$PYTHON_BIN" -m dense_evolution.cli serve
EOF
chmod +x "$LAUNCHER_ONLINE"

HAVE_OFFLINE=0
if ask "Scaricare anche una copia offline della pagina Composer, per usarla senza internet? [S/n]" S; then
    echo "Scarico la copia offline..."
    if "$PYTHON_BIN" -m dense_evolution.cli offline-composer "$OFFLINE_DIR" && [ -f "$OFFLINE_DIR/composer/index.html" ]; then
        HAVE_OFFLINE=1
        cat > "$LAUNCHER_OFFLINE" <<EOF
#!/usr/bin/env bash
if command -v open >/dev/null 2>&1; then open "$OFFLINE_DIR/composer/index.html" >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$OFFLINE_DIR/composer/index.html" >/dev/null 2>&1 &
fi
exec "$PYTHON_BIN" -m dense_evolution.cli serve
EOF
        chmod +x "$LAUNCHER_OFFLINE"
        echo "Copia offline pronta in $OFFLINE_DIR."
    else
        echo "Download della copia offline non riuscito (serve internet la prima volta) -- puoi riprovare dopo con:"
        echo "  $PYTHON_BIN -m dense_evolution.cli offline-composer \"$OFFLINE_DIR\""
    fi
fi
echo

make_launcher_icon() {
    # make_launcher_icon <dest-no-ext> <label> <launcher-path>
    local dest="$1" label="$2" target="$3"
    if [ "$OS_NAME" = "Darwin" ]; then
        cat > "$dest.command" <<EOF
#!/usr/bin/env bash
exec "$target"
EOF
        chmod +x "$dest.command"
        echo "  creata: $dest.command"
    else
        cat > "$dest.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$label
Comment=Avvia il kernel locale di Dense-Evolution Composer
Exec=$target
Terminal=true
Categories=Science;
EOF
        chmod +x "$dest.desktop"
        echo "  creata: $dest.desktop"
    fi
}

echo "Dove vuoi le icone di avvio? Puoi scegliere piu' di un posto: verranno"
echo "create un'icona \"(Online)\" e, se scaricata sopra, una \"(Offline)\" --"
echo "scegli tu quale usare ogni volta, nessuna delle due sostituisce l'altra."
echo

if ask "Icona sul Desktop? [S/n]" S; then
    mkdir -p "$HOME/Desktop"
    make_launcher_icon "$HOME/Desktop/Dense-Evolution Composer (Online)" "Dense-Evolution Composer (Online)" "$LAUNCHER_ONLINE"
    [ "$HAVE_OFFLINE" = "1" ] && make_launcher_icon "$HOME/Desktop/Dense-Evolution Composer (Offline)" "Dense-Evolution Composer (Offline)" "$LAUNCHER_OFFLINE"
fi

if [ "$OS_NAME" != "Darwin" ] && ask "Voce nel menu applicazioni? [S/n]" S; then
    mkdir -p "$HOME/.local/share/applications"
    make_launcher_icon "$HOME/.local/share/applications/dense-evolution-composer-online" "Dense-Evolution Composer (Online)" "$LAUNCHER_ONLINE"
    [ "$HAVE_OFFLINE" = "1" ] && make_launcher_icon "$HOME/.local/share/applications/dense-evolution-composer-offline" "Dense-Evolution Composer (Offline)" "$LAUNCHER_OFFLINE"
fi

if ask "Avviare automaticamente il kernel all'accensione/login? [s/N]" N; then
    if [ "$OS_NAME" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.dense-evolution.composer.plist"
        mkdir -p "$HOME/Library/LaunchAgents"
        cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>Label</key><string>com.dense-evolution.composer</string>
    <key>ProgramArguments</key><array><string>$LAUNCHER_ONLINE</string></array>
    <key>RunAtLoad</key><true/>
</dict></plist>
EOF
        launchctl load "$PLIST" 2>/dev/null || true
        echo "Avvio automatico attivato (LaunchAgent, usa la pagina online)."
    else
        mkdir -p "$HOME/.config/autostart"
        make_launcher_icon "$HOME/.config/autostart/dense-evolution-composer" "Dense-Evolution Composer" "$LAUNCHER_ONLINE"
        echo "Avvio automatico attivato (XDG autostart, usa la pagina online)."
    fi
fi
echo

if ask "Aprire ora Composer e avviare il kernel? [S/n]" S; then
    exec "$LAUNCHER_ONLINE"
else
    echo "Puoi avviare Composer in qualsiasi momento dalle icone create sopra,"
    echo "oppure eseguendo \"$LAUNCHER_ONLINE\"."
fi

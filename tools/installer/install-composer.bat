@echo off
setlocal enabledelayedexpansion
rem Dense-Evolution Composer -- local kernel installer (Windows).
rem No chcp 65001 -- it breaks set /p input reading; the script text is
rem plain ASCII on purpose so it does not need it.
rem See uninstall-composer.bat to undo everything this creates.

set "COMPOSER_URL=https://tatopenn-cell.github.io/Dense-Evolution/composer/"
set "ICON_URL=https://tatopenn-cell.github.io/Dense-Evolution/assets/dense-evolution.ico"
set "LICENSE_URL=https://github.com/tatopenn-cell/Dense-Evolution/blob/main/LICENSE.md"
set "INSTALL_DIR=%USERPROFILE%\DenseEvolutionComposer"
set "OFFLINE_DIR=%INSTALL_DIR%\offline"
set "ICON_FILE=%INSTALL_DIR%\dense-evolution.ico"
set "LAUNCHER_ONLINE=%INSTALL_DIR%\launch-composer-online.bat"
set "LAUNCHER_OFFLINE=%INSTALL_DIR%\launch-composer-offline.bat"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
set "STARTUP=%STARTMENU%\Startup"

echo ============================================================
echo  Dense-Evolution Composer -- installazione del kernel locale
echo ============================================================
echo.
echo Questo script:
echo   1. Ti fa leggere e accettare la licenza del pacchetto.
echo   2. Installa/aggiorna il pacchetto Python "dense-evolution[composer]"
echo      da PyPI (dense_evolution stesso + JAX + fastapi/uvicorn/pydantic,
echo      solo per l'esecuzione locale dei circuiti -- nessun altro dato
echo      lascia questo PC).
echo   3. Scarica (opzionale) una copia offline della pagina Composer.
echo   4. Crea (a tua scelta) icone di avvio -- Desktop, menu Start,
echo      avvio automatico all'accensione -- e puoi rimuovere tutto in
echo      qualsiasi momento con uninstall-composer.bat.
echo.
echo Nessun passo qui sotto parte da solo: hai scaricato ed eseguito questo
echo file di persona -- un sito web non puo' farlo al posto tuo.
echo.

rem ---- Consenso esplicito alla licenza (obbligatorio, nessun default) ----
echo ------------------------------------------------------------
echo  Licenza
echo ------------------------------------------------------------
echo Software: Dense Evolution
echo Licenza:  Business Source License 1.1
echo   Uso consentito: strettamente non commerciale (limiti aggiuntivi
echo   per uso in produzione commerciale/industriale).
echo   Testo completo: %LICENSE_URL%
echo.
set "ACCEPT_LICENSE="
set /p "ACCEPT_LICENSE=Hai letto e accetti i termini della licenza? [s/N] "
if /i not "!ACCEPT_LICENSE!"=="s" (
    echo.
    echo Devi accettare la licenza per continuare. Installazione annullata.
    pause
    exit /b 0
)
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python non trovato su questo sistema.
    echo Installalo da https://www.python.org/downloads/ ^(spunta "Add python.exe to PATH"^) e rilancia questo script.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Trovato Python %PYVER%.
echo.

echo Installo/aggiorno dense-evolution[composer]...
python -m pip install --upgrade "dense-evolution[composer]"
if errorlevel 1 (
    echo.
    echo Installazione fallita -- controlla i messaggi sopra.
    pause
    exit /b 1
)
echo.

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

rem ---- Icona (stessa usata dal sito, scaricata una volta sola) ----
echo Scarico l'icona...
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%ICON_URL%' -OutFile '%ICON_FILE%' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
if not exist "%ICON_FILE%" (
    echo   non riuscito ^(le icone create sotto useranno l'icona di default^).
)
echo.

rem Launcher scripts: the one real place that opens a page and starts the
rem kernel, so every shortcut (Desktop/Start Menu/Startup) just calls one
rem of these and can never drift from what "running it by hand" does.
(
    echo @echo off
    echo start "" "%COMPOSER_URL%"
    echo python -m dense_evolution.cli serve
    echo pause
) > "%LAUNCHER_ONLINE%"

set "GET_OFFLINE=S"
set /p "GET_OFFLINE=Scaricare anche una copia offline della pagina Composer, per usarla senza internet? [S/n] "
set "HAVE_OFFLINE=0"
if /i not "!GET_OFFLINE!"=="n" (
    echo Scarico la copia offline...
    python -m dense_evolution.cli offline-composer "%OFFLINE_DIR%"
    if exist "%OFFLINE_DIR%\composer\index.html" (
        set "HAVE_OFFLINE=1"
        (
            echo @echo off
            echo start "" "%OFFLINE_DIR%\composer\index.html"
            echo python -m dense_evolution.cli serve
            echo pause
        ) > "%LAUNCHER_OFFLINE%"
        echo Copia offline pronta in "%OFFLINE_DIR%".
    ) else (
        echo Download della copia offline non riuscito ^(serve internet la prima volta^) -- puoi riprovare dopo con:
        echo   python -m dense_evolution.cli offline-composer "%OFFLINE_DIR%"
    )
)
echo.

echo Dove vuoi le icone di avvio? Puoi scegliere piu' di un posto. Verranno
echo create fino a due icone separate -- una per ogni modo di avviare
echo Composer che hai scelto sopra (Online, Offline) --
echo tutte con la stessa icona del progetto, nessuna sostituisce le altre.
echo.

set "WANT_DESKTOP=S"
set /p "WANT_DESKTOP=Icone sul Desktop? [S/n] "
if /i not "!WANT_DESKTOP!"=="n" (
    call :create_shortcut "%USERPROFILE%\Desktop\Dense-Evolution Composer (Online).lnk" "%LAUNCHER_ONLINE%" "Avvia Composer (pagina online, sempre aggiornata)"
    if "!HAVE_OFFLINE!"=="1" call :create_shortcut "%USERPROFILE%\Desktop\Dense-Evolution Composer (Offline).lnk" "%LAUNCHER_OFFLINE%" "Avvia Composer (copia offline, non serve internet)"
)

set "WANT_STARTMENU=S"
set /p "WANT_STARTMENU=Voci nel menu Start? [S/n] "
if /i not "!WANT_STARTMENU!"=="n" (
    call :create_shortcut "%STARTMENU%\Dense-Evolution Composer (Online).lnk" "%LAUNCHER_ONLINE%" "Avvia Composer (pagina online, sempre aggiornata)"
    if "!HAVE_OFFLINE!"=="1" call :create_shortcut "%STARTMENU%\Dense-Evolution Composer (Offline).lnk" "%LAUNCHER_OFFLINE%" "Avvia Composer (copia offline, non serve internet)"
)

set "WANT_STARTUP=N"
set /p "WANT_STARTUP=Avviare automaticamente il kernel Composer all'accensione del PC? [s/N] "
if /i "!WANT_STARTUP!"=="s" (
    call :create_shortcut "%STARTUP%\Dense-Evolution Composer.lnk" "%LAUNCHER_ONLINE%" "Avvia automaticamente il kernel Composer all'accesso"
    echo Avvio automatico attivato ^(usa la pagina online^).
)
echo.

set "RUN_NOW=S"
set /p "RUN_NOW=Aprire ora Composer e avviare il kernel? [S/n] "
if /i not "!RUN_NOW!"=="n" (
    call "%LAUNCHER_ONLINE%"
) else (
    echo Puoi avviare Dense-Evolution in qualsiasi momento dalle icone create sopra,
    echo oppure eseguendo "%LAUNCHER_ONLINE%".
    pause
)
exit /b 0

:create_shortcut
set "DEST=%~1"
set "TARGET=%~2"
set "DESC=%~3"
if exist "%ICON_FILE%" (
    powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%DEST%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = '%DESC%'; $s.IconLocation = '%ICON_FILE%'; $s.Save()"
) else (
    powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%DEST%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = '%DESC%'; $s.Save()"
)
if exist "%DEST%" (
    echo   creata: "%DEST%"
) else (
    echo   non riuscita: "%DEST%"
)
exit /b 0

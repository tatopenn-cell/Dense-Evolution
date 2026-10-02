@echo off
setlocal enabledelayedexpansion
rem Dense-Evolution Dashboard (Streamlit) -- installer (Windows).
rem See uninstall-dashboard.bat to undo everything this creates.

set "LICENSE_URL=https://github.com/tatopenn-cell/Dense-Evolution/blob/main/LICENSE.md"
set "ICON_URL=https://tatopenn-cell.github.io/Dense-Evolution/assets/dense-evolution.ico"
set "APP_URL=https://raw.githubusercontent.com/tatopenn-cell/Dense-Evolution/main/tools/dashboard/app.py"
set "INSTALL_DIR=%USERPROFILE%\DenseEvolutionDashboard"
set "ICON_FILE=%INSTALL_DIR%\dense-evolution.ico"
set "LAUNCHER=%INSTALL_DIR%\launch-dashboard.bat"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

echo ============================================================
echo  Dense-Evolution Dashboard (Streamlit) -- installazione
echo ============================================================
echo.
echo Questo script:
echo   1. Ti fa leggere e accettare la licenza del pacchetto.
echo   2. Installa/aggiorna il pacchetto Python "dense-evolution[dashboard,qmmm,pennylane]"
echo      da PyPI (dense_evolution + Streamlit + Qiskit).
echo   3. Scarica l'app Dashboard da GitHub in "%INSTALL_DIR%".
echo   4. Crea (a tua scelta) icone di avvio -- Desktop, menu Start --
echo      e puoi rimuovere tutto con uninstall-dashboard.bat.
echo.

echo ------------------------------------------------------------
echo  Licenza
echo ------------------------------------------------------------
echo Software: Dense Evolution
echo Licenza:  Business Source License 1.1
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

echo Installo/aggiorno dense-evolution[dashboard]...
python -m pip install --upgrade "dense-evolution[dashboard,qmmm,pennylane]"
if errorlevel 1 (
    echo.
    echo Installazione fallita -- controlla i messaggi sopra.
    pause
    exit /b 1
)
echo.

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo Scarico l'app Dashboard...
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%APP_URL%' -OutFile '%INSTALL_DIR%\app.py' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%APP_URL:app.py=sota_views.py%' -OutFile '%INSTALL_DIR%\sota_views.py' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%APP_URL:app.py=engine_explorer.py%' -OutFile '%INSTALL_DIR%\engine_explorer.py' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%APP_URL:app.py=circuit_editor.py%' -OutFile '%INSTALL_DIR%\circuit_editor.py' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
if not exist "%INSTALL_DIR%\circuit_editor.py" (
    echo Download non riuscito ^(serve internet^).
    pause
    exit /b 1
)
if not exist "%INSTALL_DIR%\app.py" (
    echo Download non riuscito ^(serve internet^).
    pause
    exit /b 1
)
echo App Dashboard pronta in "%INSTALL_DIR%".
echo.

powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%ICON_URL%' -OutFile '%ICON_FILE%' -UseBasicParsing } catch { exit 1 }" >nul 2>&1

(
    echo @echo off
    echo cd /d "%INSTALL_DIR%"
    echo python -m streamlit run app.py
    echo pause
) > "%LAUNCHER%"

set "WANT_DESKTOP=S"
set /p "WANT_DESKTOP=Icona sul Desktop? [S/n] "
if /i not "!WANT_DESKTOP!"=="n" (
    call :create_shortcut "%USERPROFILE%\Desktop\Dense-Evolution Dashboard (Streamlit).lnk" "%LAUNCHER%" "Avvia la Dashboard Streamlit"
)

set "WANT_STARTMENU=S"
set /p "WANT_STARTMENU=Voce nel menu Start? [S/n] "
if /i not "!WANT_STARTMENU!"=="n" (
    call :create_shortcut "%STARTMENU%\Dense-Evolution Dashboard (Streamlit).lnk" "%LAUNCHER%" "Avvia la Dashboard Streamlit"
)
echo.

set "RUN_NOW=S"
set /p "RUN_NOW=Avviare ora la Dashboard? [S/n] "
if /i not "!RUN_NOW!"=="n" (
    call "%LAUNCHER%"
) else (
    echo Puoi avviarla dalle icone create sopra, oppure eseguendo "%LAUNCHER%".
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

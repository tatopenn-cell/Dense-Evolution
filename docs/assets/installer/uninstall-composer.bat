@echo off
setlocal enabledelayedexpansion
rem No chcp 65001 -- it breaks "set /p" input reading (see
rem install-composer.bat's comment on this); this script's text is
rem plain ASCII on purpose so it doesn't need it.
rem Dense-Evolution Composer -- uninstaller (Windows). Removes every icon
rem install-composer.bat could have created (Desktop, Start Menu, Startup,
rem both Online/Offline variants), the offline copy, and the launcher
rem folder. The dense-evolution Python package itself is left alone unless
rem you explicitly say yes below -- you may be using it for other things
rem besides Composer.

set "INSTALL_DIR=%USERPROFILE%\DenseEvolutionComposer"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
set "STARTUP=%STARTMENU%\Startup"

echo Disinstallazione di Dense-Evolution Composer
echo.

for %%L in (
    "%USERPROFILE%\Desktop\Dense-Evolution Composer (Online).lnk"
    "%USERPROFILE%\Desktop\Dense-Evolution Composer (Offline).lnk"
    "%USERPROFILE%\Desktop\Dense-Evolution Dashboard (Streamlit).lnk"
    "%USERPROFILE%\Desktop\Dense-Evolution Composer.lnk"
    "%STARTMENU%\Dense-Evolution Composer (Online).lnk"
    "%STARTMENU%\Dense-Evolution Composer (Offline).lnk"
    "%STARTMENU%\Dense-Evolution Dashboard (Streamlit).lnk"
    "%STARTUP%\Dense-Evolution Composer.lnk"
) do (
    if exist %%L (
        del %%L
        echo Rimossa: %%L
    )
)

if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo Cartella "%INSTALL_DIR%" rimossa ^(inclusa la copia offline, se presente^).
) else (
    echo Nessuna cartella di lancio trovata.
)
echo.

set "REMOVE_PKG=N"
set /p "REMOVE_PKG=Disinstallare anche il pacchetto Python dense-evolution? [s/N] "
if /i "!REMOVE_PKG!"=="s" (
    python -m pip uninstall -y dense-evolution
) else (
    echo Pacchetto dense-evolution lasciato installato.
)

echo.
echo Fatto.
pause

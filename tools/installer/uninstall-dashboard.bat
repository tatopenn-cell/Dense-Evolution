@echo off
setlocal enabledelayedexpansion
rem Dense-Evolution Dashboard (Streamlit) -- uninstaller (Windows). Removes the
rem icons install-dashboard.bat created and the app folder. The dense-evolution
rem Python package is left installed unless you say yes below.

set "INSTALL_DIR=%USERPROFILE%\DenseEvolutionDashboard"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

echo Disinstallazione di Dense-Evolution Dashboard (Streamlit)
echo.

for %%L in (
    "%USERPROFILE%\Desktop\Dense-Evolution Dashboard (Streamlit).lnk"
    "%STARTMENU%\Dense-Evolution Dashboard (Streamlit).lnk"
) do (
    if exist %%L (
        del %%L
        echo Rimossa: %%L
    )
)

if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo Cartella "%INSTALL_DIR%" rimossa.
) else (
    echo Nessuna cartella trovata.
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

@echo off
setlocal

cd /d C:\LuxCoreCustom\source\LuxCore-wheels-v2.11.2-HERO

set "PY313=C:\Users\Mathias\AppData\Local\Programs\Python\Python313"
set "PATH=%PY313%;%PY313%\Scripts;%PATH%"

set "LUX_BUILD_TYPE=Release"

echo.
echo ============================================
echo   LuxCore 2.11.2 - Python 3.13 HERO Wheel
echo ============================================
echo.

python --version
echo Build Type: %LUX_BUILD_TYPE%

echo.
echo Checking repairwheel...
python -m repairwheel -V
if errorlevel 1 (
    python -m pip install repairwheel
    if errorlevel 1 goto :error
)

echo.
echo Building pyluxcore...
make pyluxcore
if errorlevel 1 goto :error

echo.
echo Building wheel...
make wheel-test
if errorlevel 1 goto :error

echo.
echo ============================================
echo   BUILD ERFOLGREICH
echo ============================================
echo.

dir /s /b *.whl

pause
exit /b 0

:error
echo.
echo ============================================
echo   BUILD FEHLGESCHLAGEN
echo ============================================
echo.
pause
exit /b 1
@echo off
cd %~dp0
set "opencvVersion=4.10.0"
set "installPath=%CD%\..\OpenCL\OpenCV"
set "buildPath=%CD%\build-opencv"

if not exist %installPath% (
    echo Zmieniono miejsce wykonywania skryptu
	set "installPath=%CD%\OpenCV_TO_COPY"
	echo.
	echo OpenCv zostanie zainstalowane w:
	echo %CD%\OpenCV_TO_COPY
)

where mingw32-make>nul 2>&1
if errorlevel 1 (
	echo ERROR: brak Mingw lub nie zostala dodana do zmiennych systemowych
	exit /b 1
)

where cmake>nul 2>&1
if errorlevel 1 (
	echo ERROR: brak cMake lub nie zostala dodana do zmiennych systemowych
	exit /b 1
)

REM === Pobierz źródła ===
git clone --branch %opencvVersion% https://github.com/opencv/opencv.git
git clone --branch %opencvVersion% https://github.com/opencv/opencv_contrib.git

REM === Utwórz katalog build ===
mkdir "%buildPath%"
cd /d "%buildPath%"

REM === Konfiguracja CMake (STATYCZNA LINKOWANA) ===
cmake ^
  ../opencv ^
  -G "MinGW Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%installPath%" ^
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DBUILD_TESTS=OFF ^
  -DBUILD_PERF_TESTS=OFF ^
  -DBUILD_EXAMPLES=OFF

REM === Kompilacja oraz Instalacja pakietu OpenCV===
mingw32-make -j8
mingw32-make install

echo.
echo OpenCV %opencvVersion% zbudowany statycznie i zainstalowany w: %installPath%
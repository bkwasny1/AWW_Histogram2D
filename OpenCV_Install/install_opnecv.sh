#!/bin/bash

set -e
opencvVersion="4.10.0"
currentDir="$(pwd)"
installPath="$currentDir/../OpenCL/OpenCV"
buildPath="$currentDir/build-opencv"

command -v cmake >/dev/null 2>&1 || { echo >&2 "ERROR: cmake nie jest zainstalowany."; exit 1; }
command -v make >/dev/null 2>&1 || { echo >&2 "ERROR: make nie jest zainstalowany."; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo >&2 "ERROR: g++ nie jest zainstalowany."; exit 1; }


if [ ! -d "$installPath" ]; then
    echo "Zmieniono miejsce instalacji OpenCV:"
    installPath="$currentDir/OpenCV_TO_COPY"
    echo "OpenCV zostanie zainstalowane w: $installPath"
fi

git clone --branch "$opencvVersion" https://github.com/opencv/opencv.git
git clone --branch "$opencvVersion" https://github.com/opencv/opencv_contrib.git

mkdir -p "$buildPath"
cd "$buildPath"

cmake ../opencv \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$installPath" \
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF

make -j"$(nproc)"
make install

echo
echo "OpenCV $opencvVersion zostało zbudowane statycznie i zainstalowane w: $installPath"

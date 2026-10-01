#!/usr/bin/env bash
set -euo pipefail
version=6.1.2
out=dense_evolution/native_hf/_libcint
src="${RUNNER_TEMP:-/tmp}/libcint-src"
rm -rf "$src"
git clone --depth 1 --branch "v$version" https://github.com/sunqm/libcint.git "$src"
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_QUADMATH=ON -DWITH_FORTRAN=OFF -DENABLE_EXAMPLE=OFF -DENABLE_TEST=OFF "$@"
cmake --build "$src/build" --config Release -j 4
mkdir -p "$out"
inc=(-I"$src/include" -I"$src/build/include")
drv=dense_evolution/native_hf/csrc/cint_driver.c
case "$(uname -s)" in
  Linux*) cc -O3 -shared -fPIC "${inc[@]}" "$drv" "$src/build/libcint.a" -lm -o "$out/libdecint.so" ;;
  Darwin*) cc -O3 -dynamiclib -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 "${inc[@]}" "$drv" "$src/build/libcint.a" -o "$out/libdecint.dylib" ;;
  *) gcc -O3 -shared -static "${inc[@]}" "$drv" "$src/build/libcint.a" -o "$out/libdecint.dll" ;;
esac
cp "$src/LICENSE" "$out/LICENSE-libcint"
ls -l "$out"

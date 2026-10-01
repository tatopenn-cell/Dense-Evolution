#!/usr/bin/env bash
set -euo pipefail
version=6.1.2
out=dense_evolution/native_hf/_libcint
src="${RUNNER_TEMP:-/tmp}/libcint-src"
rm -rf "$src"
git clone --depth 1 --branch "v$version" https://github.com/sunqm/libcint.git "$src"
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DWITH_FORTRAN=OFF -DENABLE_EXAMPLE=OFF -DENABLE_TEST=OFF "$@"
cmake --build "$src/build" --config Release -j 4
mkdir -p "$out"
case "$(uname -s)" in
  Linux*) cp -L "$src/build/libcint.so" "$out/libcint.so" ;;
  Darwin*) cp -L "$src/build/libcint.dylib" "$out/libcint.dylib" ;;
  *) cp "$src/build/libcint.dll" "$out/libcint.dll" ;;
esac
cp "$src/LICENSE" "$out/LICENSE-libcint"
ls -l "$out"

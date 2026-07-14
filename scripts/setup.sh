#!/usr/bin/env bash
#
# Build Glaxnimate from source: the GUI app *and* the Python bindings.
#
# The PyPI `glaxnimate` wheel is unusable here — it was built on openSUSE
# against SUSE's patched ffmpeg and demands versioned symbols
# (LIBAVCODEC_61.19_SUSE) that no other ffmpeg build exports. Compiling
# against this machine's own Qt6/KF6/ffmpeg removes that problem entirely.
#
# Usage:  bash scripts/setup.sh
# Only the apt/install steps use sudo; the build runs as you.

set -euo pipefail

SRC="${GLAXNIMATE_SRC:-$HOME/src/glaxnimate}"
BUILD="$SRC/build"
JOBS="$(nproc)"
PY="/usr/bin/python3"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. deps
say "1/5  Installing build dependencies (sudo will prompt)"

# pybind11 is a bundled submodule, so it is not listed here. potrace is NOT —
# despite external/potrace existing, src/trace does find_package(Potrace) and
# wants the system library, so libpotrace-dev is required.
sudo apt-get update
sudo apt-get install -y \
    cmake ninja-build extra-cmake-modules \
    qt6-base-dev qt6-base-private-dev qt6-tools-dev qt6-tools-dev-tools qt6-svg-dev \
    libkf6i18n-dev libkf6coreaddons-dev libkf6widgetsaddons-dev libkf6xmlgui-dev \
    libkf6archive-dev libkf6completion-dev libkf6iconthemes-dev libkf6config-dev \
    libkf6crash-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libpotrace-dev zlib1g-dev python3-dev

ok "dependencies installed"

# ---------------------------------------------------------------- 2. source
say "2/5  Fetching source"

if [ -d "$SRC/.git" ]; then
    git -C "$SRC" pull --ff-only || true
    git -C "$SRC" submodule update --init --recursive
else
    mkdir -p "$(dirname "$SRC")"
    git clone --recurse-submodules https://github.com/KDE/glaxnimate.git "$SRC"
fi

# The bundled submodules are required; a shallow/partial clone silently breaks the build.
for sub in external/pybind11 external/Qt-Color-Widgets; do
    [ -n "$(ls -A "$SRC/$sub" 2>/dev/null)" ] || die "submodule $sub is empty — run: git -C $SRC submodule update --init --recursive"
done
ok "source at $SRC ($(git -C "$SRC" log --oneline -1))"

# ---------------------------------------------------------------- 3. configure
say "3/5  Configuring (CMake + Ninja)"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPython3_EXECUTABLE="$PY"

ok "configured"

# ---------------------------------------------------------------- 4. build
say "4/5  Building — GUI app + Python bindings (this takes a while)"

# The app. Ubuntu 26.04 ships ffmpeg 8 while Glaxnimate targets 7, so the video
# module is the one part that may not compile. If it fails, STOP and report —
# do not silently continue with a partial build.
cmake --build "$BUILD" -j "$JOBS" || die "app build failed (see output above — likely the ffmpeg 8 vs 7 video module)"
ok "GUI app built"

# The bindings are EXCLUDE_FROM_ALL, so they are NOT covered by the line above
# and must be named explicitly. This is easy to miss.
cmake --build "$BUILD" -j "$JOBS" --target glaxnimate_python || die "python bindings build failed"
ok "Python bindings built"

# ---------------------------------------------------------------- 5. install
say "5/5  Installing + verifying"

sudo cmake --install "$BUILD"
sudo ldconfig

MODULE="$(find "$BUILD" -name 'glaxnimate*.so' -newer "$BUILD/CMakeCache.txt" | head -1)"
[ -n "$MODULE" ] || MODULE="$(find "$BUILD" -name 'glaxnimate*.so' | head -1)"
[ -n "$MODULE" ] || die "built .so not found"

echo "  module: $MODULE"
echo "  app:    $(command -v glaxnimate || echo 'NOT ON PATH')"

# Prove the binding actually imports and can render — the whole project depends
# on render_image() returning a PIL image.
PYTHONPATH="$(dirname "$MODULE")" "$PY" - <<'EOF'
import glaxnimate
print("  import OK ->", glaxnimate.__file__)
with glaxnimate.environment.Headless():
    d = glaxnimate.model.Document("")
    d.main.width, d.main.height = 320, 240
    lay = d.main.add_shape("Layer")
    lay.add_shape("Fill").color.value = "#e33"
    e = lay.add_shape("Ellipse")
    e.size.value = glaxnimate.utils.Size(80, 80)
    e.position.set_keyframe(0, glaxnimate.utils.Point(40, 120))
    e.position.set_keyframe(30, glaxnimate.utils.Point(280, 120))
    img = d.render_image(15)
    print("  render_image(15) ->", type(img).__name__, img.size)
EOF

ok "Glaxnimate built, installed and verified"
echo
echo "Python module lives at: $(dirname "$MODULE")"
echo "Tell Claude it's done and it will wire it into the project venv."

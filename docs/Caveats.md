# Caveats

Veshiru is a Windows-exclusive emulator. This page only covers Windows builds and their quirks.

<!-- TOC -->
- [MSYS2](#msys2)
<!-- /TOC -->

## MSYS2

Only the `MINGW64` environment is tested (or `CLANGARM64` on ARM); however, all of the others should work (in theory) sans `MINGW32`.

When packaging an MSYS2 build that is NOT fully static, you will need to copy all dependent DLLs recursively alongside the `windeployqt6`; for example:

```sh
# MSYS_TOOLCHAIN is typically just mingw64
# since Windows is case-insensitive, you can set this to $MSYSTEM
export PATH="/${MSYS_TOOLCHAIN}/bin:$PATH"

# grab deps of a dll or exe and place them in the current dir
deps() {
    # string parsing is fun
    objdump -p "$1" | awk '/DLL Name:/ {print $3}' | while read -r dll; do
        [ -z "$dll" ] && continue

        # bin directory is used for DLLs, so we can do a quick "hack"
        # and use command to find the path of the DLL
        dllpath=$(command -v "$dll" 2>/dev/null || true)

        [ -z "$dllpath" ] && continue

        # explicitly exclude system32/syswow64 deps
        # these aren't needed to be bundled, as all systems already have them
        case "$dllpath" in
            *System32* | *SysWOW64*) continue ;;
        esac

        # avoid copying deps multiple times
        if [ ! -f "$dll" ]; then
            echo "$dllpath"
            cp "$dllpath" "$dll"

            # also grab the dependencies of the dependent DLL; e.g.
            # double-conversion is a dep of Qt6Core.dll but NOT eden.exe
            deps "$dllpath"
        fi
    done
}

# NB: must be done in a directory containing eden.exe
deps eden.exe

# deploy Qt plugins and such
windeployqt6 --no-compiler-runtime --no-opengl-sw --no-system-dxc-compiler \
    --no-system-d3d-compiler eden.exe

# grab deps for Qt plugins
find ./*/ -name "*.dll" | while read -r dll; do deps "$dll"; done
```

# Dependencies

Veshiru is a Windows-exclusive emulator: Windows 10/11 x86-64 is the supported platform. You MUST have a C++ compiler.

* On Windows, we support:
  * **[MSVC](https://visualstudio.microsoft.com/downloads/)** (default)
    * It's STRONGLY RECOMMENDED to use the **Community** option and **Visual Studio 2022**
    * You need to install: **[Desktop development with C++](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)**
  * **[clang-cl](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-180)**
    * You need to install: **C++ Clang tools for Windows**
  * **[MSYS2](https://www.msys2.org)**

The following additional tools are also required:

* **[CMake](https://www.cmake.org/)** 3.31+ - already included with the Android SDK
* **[Git](https://git-scm.com/)** for version control
  * **[Windows installer](https://gitforwindows.org)**
* **[Python3](https://www.python.org/downloads/)** 3.10+ - necessary to download external repositories
* On Windows, you must install the **[Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)** as well
  * *A convenience script to install the latest SDK is provided in:*
    * `tools/windows/install-vulkan-sdk.ps1` (for PowerShell 5+)
    * `tools/windows/install-vulkan-sdk.sh` (for Git Bash, etc)

If you are on desktop and plan to use the Qt frontend, you *must* install Qt 6, and optionally Qt Creator (the **RECOMMENDED** IDE for building)

* MSVC/clang-cl users on Windows must install through the official [Qt](https://www.qt.io/download-qt-installer-oss) installer
* MSYS2 can also install Qt 6 via the package manager

* For help setting up Qt Creator, run `./install.sh -h qtcreator`

* If you're using clang-cl and want to still use MSVC
  * Check the option to add "C++ clang compiler for Windows" on Visual Studio installer and uncheck "x64/x86 build tool for MSVC" while selecting "C++ desktop developement tools" and change Visual Studio to 2026, from 2022.
  * At qt creator section generator tab change Visual Studio 17 2022 to 2026.
  * Finally, to use clang-cl: `cmake -S . -B build -G "Visual Studio 17 2026" -T ClangCL`

If you are on **Windows** and building with **MSVC** or **clang-cl**, you may go [back home](Build.md) and continue.

## Externals

The following are handled by Eden's externals:

* [FFmpeg](https://ffmpeg.org/) (should use `-DVESHIRU_USE_EXTERNAL_FFMPEG=ON`)
* [SDL3](https://www.libsdl.org/download-2.0.php) 3.2.10+ (Use `-DVESHIRU_USE_BUNDLED_SDL2=ON` to reduce compile time)

All other dependencies will be downloaded and built by [CPM](https://github.com/cpm-cmake/CPM.cmake/) if `VESHIRU_USE_CPM` is on, but will always use system dependencies if available (UNIX-like only):

* [Boost](https://www.boost.org/users/download/) 1.57.0+
* [Catch2](https://github.com/catchorg/Catch2) 3.0.1 if `VESHIRU_TESTS` or `DYNARMIC_TESTS` are on
* [fmt](https://fmt.dev/) 8.0.1+
* [lz4](http://www.lz4.org)
* [nlohmann\_json](https://github.com/nlohmann/json) 3.8+
* [OpenSSL](https://www.openssl.org/source/) 3+
* [ZLIB](https://www.zlib.net/) 1.2+
* [zstd](https://facebook.github.io/zstd/) 1.5+
* [enet](http://enet.bespin.org/) 1.3+
* [Opus](https://opus-codec.org/) 1.3+

Vulkan 1.3.274+ is also needed:

* [VulkanUtilityLibraries](https://github.com/KhronosGroup/Vulkan-Utility-Libraries)
* [VulkanHeaders](https://github.com/KhronosGroup/Vulkan-Headers)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)
* [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers)

Certain other dependencies will be fetched by CPM regardless. System packages *can* be used for these libraries, but many are either not packaged by most distributions OR have issues when used by the system:

* [SimpleIni](https://github.com/brofield/simpleini)
* [DiscordRPC](https://github.com/eden-emulator/discord-rpc)
* [cubeb](https://github.com/mozilla/cubeb)
* [libusb](https://github.com/libusb/libusb)
* [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
* [sirit](https://github.com/eden-emulator/sirit)
* [httplib](https://github.com/yhirose/cpp-httplib) - if `ENABLE_UPDATE_CHECKER` or `ENABLE_WEB_SERVICE` are on
* [cpp-jwt](https://github.com/arun11299/cpp-jwt) 1.4+ - if `ENABLE_WEB_SERVICE` is on

On amd64:

* [xbyak](https://github.com/herumi/xbyak) - 7.22 or earlier is recommended

On aarch64 OR if `DYNARMIC_TESTS` is on:

* [oaknut](https://github.com/merryhime/oaknut) 2.0.1+

On riscv64:

* [biscuit](https://github.com/lioncash/biscuit) 0.9.1+

## Commands

No additional system packages are required on Windows beyond the tools above. Qt 6.8.3 with the Qt Charts module is what CI uses:

```powershell
python -m pip install aqtinstall
python -m aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 -m qtcharts -O C:\Qt
```

Then configure with `-DVESHIRU_USE_BUNDLED_QT=OFF -DQt6_DIR=C:/Qt/6.8.3/msvc2022_64/lib/cmake/Qt6`.

## All Done

You may now return to the **[root build guide](Build.md)**.

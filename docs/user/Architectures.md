# User Handbook - Architectures and Platforms

Notes and caveats for the architectures and platforms Veshiru supports.

Veshiru is a Windows-exclusive emulator: it is developed, built and tested for 64-bit Windows 10 and 11 only, and only Windows packages are published. Other platforms and architectures are not supported and receive no fixes or support.

# Architectures

## amd64

AMD64, aka x86_64, is the only supported architecture. This is the most tested and supported target for desktop Windows.

### Caveats

AMD64 systems are almost always limited by the CPU. For example, you will often hit max CPU usage before the GPU ever reaches 70% usage, with minimal exceptions (that tend to pop up only at >200fps). JIT is slow!

## aarch64

ARM64, aka aarch64, is not supported. Windows on ARM is not tested and is unlikely to work. NCE (Native Code Execution) is not available on Windows.

## riscv64

RISC-V, aka riscv64, is not supported. Windows/riscv64 does not exist, and there is no JIT backend for it.

## Other

Other architectures, such as SPARC, MIPS, PowerPC, Loong, and all 32-bit architectures are completely unsupported, as there is no JIT backend or emitter thereof.

IA-64 (Itanium) support is completely unknown and untestable: existing amd64 packages will not run on IA-64.

# Platforms

## Windows

Windows 10 and 11 (x86-64) are supported. Anything below, or Windows on ARM, is unsupported.

To run Veshiru, you'll need to install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

Neither AMD/NVIDIA drivers work nearly as well as the AMD RADV drivers found on Linux, but Linux is not a supported platform for Veshiru. Compatibility on Windows is still largely fine, but performance and some hard-to-run games may suffer compared to what Linux achieves.

### Windows on ARM

Unsupported and untested. Use the x86-64 build under emulation at your own risk, or better, a machine with an x86-64 CPU.

## Everything else

Linux, macOS, the BSDs, Solaris, HaikuOS and Android are not supported platforms for Veshiru. No packages are published for them, CI does not build for them, and issues specific to them are out of scope.

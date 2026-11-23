<!--
# SPDX-FileCopyrightText: Copyright 2026 FUKKO contributors
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
-->

# VTX-HLE, the high-level NVN graphics path

VTX-HLE by Vortex727 (https://github.com/Vortex727/FUKKO)

Switch 2 games draw through NVN2, the console's graphics library. Eglantine can run their graphics
in two ways, chosen per game with **Graphics → Graphics path**:

- **Emulated GPU** (the default). The game's own NVN2 library runs as it does on the console and
  turns every call into the GPU's hardware command stream, which the emulator decodes and draws.
- **VTX-HLE (NVN)**. The emulator answers the game's NVN2 calls itself, records the command
  buffers the game builds with them, and replays the recorded commands on Vulkan. The library code
  that writes hardware commands never runs.

## Switching

- Configure → Graphics → Graphics path, or the same setting in a game's properties. It takes
  effect at the next game start.
- The first VTX-HLE launch after Emulated GPU has built a shader cache clears the game's shader
  caches. Later VTX-HLE launches can reuse the cache built on the new path.
- `EDEN_NX2_NVN_HLE=1` forces VTX-HLE on and `EDEN_NX2_NVN_HLE=0` forces it off for any game, over
  the setting.
- While VTX-HLE is selected, the settings it does not apply are greyed out, with the reason in the
  label and the tooltip (`src/common/vtx_hle_settings.cpp`). Their saved values stay untouched and
  the Emulated GPU path applies them as before.
- With VTX-HLE selected the page also shows its own options: occlusion queries inside render
  passes (on), fast-linked pipelines (off) and fewer GPU syncs between render passes (on).
- **Refresh shaders every minute** (off by default) starts with fresh shaders, then drops shader
  translations and compiled pipelines every 60 seconds at a frame boundary. Draws wait for their
  new pipelines, which can briefly stutter. Use it to test black areas developing during continued
  play. It preserves textures and GPU results. The periodic refresh runs entirely in memory.
  Enable it before starting the game.
- The game list's **Remove Vulkan Pipeline Cache** removes both graphics paths' caches, including
  VTX-HLE's translation-key files and driver cache. **Remove All Pipeline Caches** also includes them.

VTX-HLE needs the Vulkan renderer and a device with buffer device addresses and 64-bit shader
integers. On other devices, with another renderer, or if the replay cannot start, the game runs on
the emulated GPU and the log says why (`[nvn] graphics path: ...`). Switch 1 games always use the
emulated GPU.

## How it works

- **Bridge** (`src/core/hle/nvn/`): the NVN2 functions the game looks up are answered with the
  emulator's own implementations (memory pools, buffers, textures, samplers, programs, command
  buffers, queues, syncs, events, windows, the game's DLSS calls).
- **Recorder** (`src/video_core/nvn/`): command buffers are recorded into a compact record format.
- **Replay** (`src/video_core/renderer_vulkan/nvn/`): a replay thread turns the records into
  Vulkan command buffers and presents through the renderer's present manager. The game's compiled
  shaders are lowered by Eglantine's Ampere frontend and translated by the shader recompiler, with
  untracked global memory reached through buffer device addresses.
- **Page tracker** (`src/core/memory/page_tracker*`): which guest pages the CPU wrote since the
  replay took their contents, and traps on the next CPU access for the lazy write-back of GPU
  results. VTX-HLE turns it on when it hooks a game; with the emulated GPU it stays off and
  tracks nothing.

The design is documented for developers in
[`src/video_core/renderer_vulkan/nvn/DESIGN.md`](../src/video_core/renderer_vulkan/nvn/DESIGN.md).

## What this port does not include

Compared with FUKKO, Eglantine's VTX-HLE leaves out the parts that rely on FUKKO features Eglantine
does not have:

- NVIDIA DLSS / DLAA / DLSS neural rendering through FUKKO's Streamline loader. A game's own DLSS
  network is replaced by FSR1 (cooperative-matrix lowering of the network is not available).
- Frame generation and Eglantine's presentation passes (window filters past bilinear, FXAA/SMAA,
  post-processing) on the replay's frames: the replay presents with its plain blit. These settings
  are greyed out while VTX-HLE is selected.
- HDR output.
- FUKKO's performance HUD, GPU command capture and native shader registry. The MCP server is
  ported separately with a subset of its tools (see [`tools/mcp/README.md`](../tools/mcp/README.md)).

## Credits

- VTX-HLE by Vortex727 (https://github.com/Vortex727/FUKKO). FUKKO by Vortex727
  (https://github.com/Vortex727/FUKKO). See [NOTICE](NOTICE), which has to stay with any copy of
  this code.
- MCP server from FUKKO by Vortex727 (https://github.com/Vortex727/FUKKO), originally written by
  quixomatic (`src/mcp`, `src/yuzu/mcp_host_qt.*`, `src/yuzu_cmd/mcp_host_cli.*`, `tools/mcp`).
  The same [NOTICE](NOTICE) applies to it.
- ShiftX2 fixes by Duck912: the uniform snapshot at submit that fixes Ocarina of Time's vertex
  explosions, the CPU-written-page uniform snapshot, keeping stale GPU contents off textures the
  CPU rewrote, the overlap source rule, raw global memory helpers kept as calls, and presenting
  suboptimal images on the unpaced acquire path.
- The bridge is informed by [nxnative](https://github.com/seedds/nxnative) by seedds (MIT). Files
  that contain code ported from it carry seedds's copyright notice and the MIT licence next to the
  GPL (`SPDX-License-Identifier: GPL-3.0-or-later AND MIT`).
- The BC7 encoder of the ASTC stand-ins (`src/video_core/host_shaders/bc7_encoder.comp`) is ported
  from `BC7Encode.hlsl` of [DirectXTex](https://github.com/microsoft/DirectXTex), Copyright (c)
  Microsoft Corporation, MIT License. The file carries Microsoft's copyright notice and the MIT
  licence next to the GPL (`SPDX-License-Identifier: GPL-3.0-or-later AND MIT`).

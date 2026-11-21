# Bananza on a GTX 1650 laptop: what one 2.1.0 log shows

A review of one owner-supplied `veshiru_log.txt.old` for 2.2.0, and the one change taken from it.
No game was run for this review, and no frame rate was measured before or after the change.

## The session

Veshiru 2.1.0 (`451f273`), Windows 11, i7-10750H (6 cores, 12 threads), 31.8 GiB RAM, GeForce
GTX 1650 with 3935 MiB, NVIDIA driver 621.24. Donkey Kong Bananza `040074A01BF12000` v3.2.0 from a
NUP container, docked, 1x, for 833 seconds. Settings away from the defaults: asynchronous shaders,
`native_async_draws` Bounded, asynchronous presentation, frame generation (Open-Frame-Gen towards
60), "Use the game's own DLSS", GPU accuracy Low, texture cache limit 3935 MiB.

The log is at the default level. It has no `Nx2Profile` zones, no stutter trace and no read
trace, so it says where frames wait but not which code takes the time.

## What the log shows

**1. The shader cache was empty.** `Shader cache v2: 0 bytes, 0 live records`: every pipeline of
the session was built while playing. 924 were built at run time, with 3.3 s of translation on the
GPU thread, 1.9 s of the GPU thread waiting for stage translation on workers, and **48.5 s of
draws held for pipelines a worker was still building** (4.8 s and 5.2 s of two 10 s windows just
after gameplay began). A second run of the same places starts with these pipelines cached.

**2. Driver builds on workers were several times slower once gameplay began.** Of the first 400
builds over 2 ms, those at the title screen (45 to 50 s) averaged 32 ms. From 105 s on, 5 s
windows average 70 to 280 ms, including windows with only two to eight builds, where the two
driver workers cannot have been queueing behind each other. The log does not say why: larger
shaders, the driver's own locks against the render thread, or the workers getting little CPU are
all possible.

**3. Gameplay ran at a median of 14 FPS, and that is not shader stutter.** Menus ran at 48 to 60
FPS (57 to 97 s, and 810 s on). From 117 to 801 s the 62 ten-second windows range from 3.5 to 20.6
game FPS. Between 600 and 690 s there are almost no hitches and one to four pipelines per window,
and the game still runs at 19 to 20 FPS. In that stretch a present takes 50 to 57 ms, of which 31
to 39 ms is the guest waiting for the GPU fence. In the menus it is 17 to 19 ms and under 2 ms.
This matches the earlier profile in [bananza-profile-fixes.md](bananza-profile-fixes.md), where
the GPU command and back threads each used 47 to 53 ms of CPU per frame. This log adds no zone
data to that, so it selects no change there.

**4. Guest reads of GPU-written memory wait for the host GPU.** In gameplay there are about 29
such reads a second, 4 KiB each, at 12 ms a read on average: roughly 355 ms of a guest thread
blocked in every second. One read took 2.6 s (104 to 107 s), when both channels also logged a
2 s wait on a semaphore while the first compute pipelines were building.

**5. Small audio files sometimes take a long time to read.** 299 NUP decompressions over 2 ms were
logged, 8.6 s in total. 36 of them are clips of at most 0.25 MiB that took 70 ms or more, and five
of those took 327 to 339 ms. Decoding 100 KiB of zstd takes well under a millisecond, so the time
is somewhere else in the read (the disk, the file handle, or the thread not running). Only one of
the five falls next to a save commit. The log cannot tell these apart.

**6. Not costs.** About half of the 2998 lines are per-build and per-read Info lines, each capped
at 400 or limited to slow cases. Frame generation ran with and without output at the same game
FPS (590 s against 600 s), so it is not what holds the game back here. The stage cache used 34 of
256 MiB and never cleared; the buffer cache levelled off at 386 MiB.

## What was changed

**A shader job that the GPU thread waits for now runs at normal thread priority** (commit
"video_core: run shader jobs a draw waits for at normal thread priority").

Shader workers run below normal priority so that background builds give way to the game. A job
was moved to the front of the queue when a draw started to wait for it, but a job that was already
running kept the low priority, and so shared whatever CPU the game left with every other shader
job in flight, while the GPU thread sat blocked on it. That is the path the 48.5 s in point 1 were
spent in. Now the worker running such a job gets normal priority until that stage ends, whether
the job was needed from the start or a draw began to wait for it halfway. Jobs that nothing has
waited for are never raised, so background and boot builds behave as before, and so do draws that
asynchronous shaders skip without waiting.

Windows only: elsewhere a thread cannot give itself back a priority it has lowered, and nothing
changes.

What is and is not known about it:

- The scheduling is covered by five test cases in `src/tests/video_core/shader_jobs.cpp` (raised
  on a promotion in either stage, not raised for Soon or Background, only the worker that runs a
  job promoted while queued, ended across a shutdown), and a sixth checks that Windows accepts
  the priority change for another thread. They, and the 25 older cases, pass under
  ThreadSanitizer on Linux, where the sixth only checks that nothing is changed.
- Whether it shortens the holds, and by how much, is **not measured**. Point 2 has three possible
  causes and this change addresses one of them. If the builds are slow because of the driver or
  the shaders themselves, it will do little.
- It cannot raise the gameplay frame rate in point 3.

## What was not changed, and what would decide it

| Finding | Why nothing was changed | Capture that would decide it |
| --- | --- | --- |
| 14 FPS in gameplay (3) | The time is on the GPU command and back threads; the log has no zones | `EDEN_NX2_GPU_PROFILE=1` and `EDEN_DKB_FRAME_TRACE=1` at the same place |
| Reads of GPU-written memory (4) | The wait is for the host GPU to reach the download; the path is already asynchronous | The profile above logs `Async read-back per second` |
| Slow driver builds (2) | Cause unknown | `EDEN_NX2_PIPELINE_BUILD_TRACE=<file>`, and the same run again with the cache now filled |
| Slow reads of small files (5) | Cause unknown; where the NUP file is stored is not in the log | `EDEN_NX2_FS_READ_TRACE=1` |
| Hitches while pipelines build (1) | Expected on a first run | `EDEN_NX2_STUTTER_TRACE=1` on a second run |

The debugging table in `AGENTS.md` describes each of these.

Two settings in this log are worth a second run on this machine, as observations rather than
recommendations: "Use the game's own DLSS" is off by default and no line in the log says whether
the game's network ran, and the texture cache limit was set to the card's whole memory. Neither
is shown by this log to cost anything.

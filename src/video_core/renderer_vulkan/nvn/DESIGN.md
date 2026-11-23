<!--
SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
SPDX-License-Identifier: GPL-3.0-or-later
-->

# NVN Vulkan replay (`Vulkan::Nvn`)

Executes the command lists the NVN recorder (`src/video_core/nvn`) produces, directly on Vulkan.
It implements `VideoCommon::Nvn::Sink` and is only constructed when the NVN HLE is chosen for the
game (`Settings::ResolveNvnHle`: `EDEN_NX2_NVN_HLE` (anything but `0` = on, `0` = off) when set, else
the `graphics_path` setting, default HLE, for a Switch 2 program); with the HLE off nothing in this
directory runs. The choice and its reason are logged (`[nvn] graphics path: ...`). A Vulkan device
without buffer device addresses or 64-bit shader integers, a failed context creation, or a renderer
that is not Vulkan leave the game on the emulated GPU (one log line each; the bridge's first NVN
call finds no sink and passes through to the game's own NVN).

## Scope

- This slice: memory pools, textures, sampler/program storage, and the ops 0-11 of the record
  contract (`End`, `CallList`, `ClearTexture`, `ClearBuffer`, `CopyBufferToBuffer`,
  `CopyBufferToTexture`, `CopyTextureToBuffer`, `CopyTextureToTexture`, `FenceSync`,
  `SignalEvent`, `WaitSync`, `WaitEvent`), `Present`, cross-queue sync, completion.
- Not here (Phase 3): graphics/compute pipelines, descriptors, draws, indirect draws, render
  conditions, sparse/virtual pools. Ops at 32 and above are counted (`ops_unsupported`) and logged
  once. Virtual pools (`flags & 0x200`) are logged and have no backing.
- Maxwell channel/engine state is never fabricated. The replay shares the `Device`, the queue and
  the present path with the legacy renderer and nothing else.

## Ownership and threads

`RendererVulkan` owns `std::optional<Nvn::Context>` (declared last, so it is destroyed first) and
exposes it through `NvnSink()`. The core never sees the Vulkan types.

| Thread | Role |
|---|---|
| caller (guest) | `Sink` calls only enqueue a command and, for `Submit`/`Present`, an `InFlight` entry (both under one mutex, so global order is preserved). |
| `NvnReplay` | Owns every host object (pools, textures, queue state, sync table). Executes commands in order, records `vk::CommandBuffer`s, submits to `device.GetGraphicsQueue()` holding `scheduler.submit_mutex` (as `PresentManager` does). |
| `NvnComplete` | Takes `InFlight` entries in seq order, waits for the timeline value, writes mirror ranges back, recycles resources, calls the completion callback in seq order and then publishes `completed_seq` for `Wait`. |

`Sink::Wait(seq)` waits for `completed_seq >= seq`, i.e. for the callback of `seq` to have returned.
Neither worker takes a guest lock.

### Lifecycle across boots (`core/hle/nvn/hle_runtime.*`, tests `src/tests/nvn/lifecycle.cpp`)

A GUI boots and stops games in one process, so nothing here may outlive its boot or assume it is the
first one. The Recorder lives in `Hle::Runtime`, the sink in the renderer, and they do not die
together: the renderer is destroyed in `System::ShutdownMainProcess` while a guest thread can still
be inside an NVN call, and the Recorder's destructor and every call it makes go to the sink.

- The Recorder is built over a `SinkGate` (`video_core/nvn/sink_gate.*`), which forwards to the real
  sink until `Detach()`. `Runtime::Shutdown` (first thing of the shutdown) stops the HLE, wakes the
  waiting guest threads and detaches the gate: it clears the completion/failure callbacks and the
  guest memory pointer (waiting for a callback that is running on the completion thread), after
  which no call reaches the sink. The renderer can then be destroyed safely.
- The Recorder and the guest memory stay alive for any thread still inside a call.
  `Runtime::Release` (after `cpu_manager.Shutdown()`, when no guest thread exists) destroys them;
  `ResetForNewProcess` (next boot's `RegisterAll`) does it too if `Release` did not run, and clears
  every per-boot piece of state: window table, debug domain ids, counters, the pool/texture tables
  of the function families (keyed by Recorder serials, which restart), `Common::BootCounter`s.
- Once-only logs and sampled traces kept in function-local statics use `Common::BootCounter`
  (`ResetAll()` at the start of each boot), state of a context lives in `Context::Impl`
  (capture schedule counters, watch log), state of a boot in `Runtime`. Nothing in a function-local
  static may be derived from a device or a Recorder.
- Device loss (`VK_ERROR_DEVICE_LOST` from a submit, a semaphore wait or any replay command): the
  context logs `<Critical>` once, tells its owner once through `Sink::SetFailureCallback` (the
  runtime logs and calls `System::Exit()`, which the Qt frontend turns into a stop; fukko-cli's exit
  callback is a no-op), and from then on completes every submission and present without running it
  (`dropped_after_device_loss` in the unsupported rows), skips the write-backs of a lost GPU and
  answers readbacks with failure, so no guest wait hangs while the frontend stops. `Device::ReportLoss`
  (a 15 s wait for the log and Nsight Aftermath) runs once on its own thread.
- Not NVN, found while measuring boots in one GUI process (private bytes after Stop grew ~185 MiB
  per boot, with and without the HLE): the `Common::Fiber`s of ~72 guest threads (a 2 MiB stack
  each) are still alive when the kernel shuts down, boot after boot. `KernelCore::Impl::Shutdown`
  frees the stacks of the fibers still alive (`Common::Fiber::ReleaseStacks`, valid because the
  cores are stopped) and logs a `<Warning>` with the count. Why those fibers' owners are never
  finalized is not fixed (the fiber objects and what else the leaked owners hold stay, ~35-55 MiB
  per boot including one 19 MiB heap block).

### Allocation

The device's VMA allocator is created `EXTERNALLY_SYNCHRONIZED` and used by the render thread, so
the replay thread cannot share it. `Device::CreateThreadSafeAllocator()` makes a second one
(identical parameters, no external-sync flag); `Context` wraps it in its own `MemoryAllocator`.
The `memory_allocator` constructor argument is accepted for the contract but not used. One
remaining exception: `PresentManager::RecreateFrame` (first present and window resize) allocates
the frame image from the renderer's allocator on the replay thread. With the HLE active no other
thread creates images, so this is a race only in theory.

## Queues and sync

- One timeline semaphore per NVN queue index, created lazily. All NVN queues submit to the single
  Vulkan graphics queue; barriers use `ALL_COMMANDS` scopes so Vulkan submission order gives the
  ordering NVN promises within a queue.
- A submission is one or more *segments* (command buffers), submitted together in one
  `vkQueueSubmit` of several `VkSubmitInfo`s. Every segment signals a fresh timeline value.
- `FenceSync` records `{timeline semaphore, signal value of the enclosing segment}` for the sync
  serial. `SignalEvent` records nothing on the GPU: the replay's event model takes the signal (with
  the enclosing segment's timeline value) and the recorder writes the guest word at completion.
- `WaitSync` on a sync fenced on the same queue is a no-op (same ordered stream). On another
  queue it ends the current segment and the next one waits on the producer's timeline value. Never
  a device-wide wait.
- Present is ordered with Submit: it is an `InFlight` entry with the same seq space.

### Streams (`StreamScheduler`, `video_core/nvn/queue_sync.h`)

A submission is not run whole in submission order. NVN queues are independent streams and the SDK
does not order submissions of different queues: DK submits `q1 [Wait e0==0; Signal e4=1; Wait
e0==2; Signal e4=3]` *before* `q0 [Signal e0=0; Wait e4==1; Signal e0=2; Wait e4==3; reset the
events]`, and the waits are only satisfied by signals that arrive later in the replay. So every
submission and present becomes a `StreamItem` (`SubmissionItem`, `PresentItem` in `nvn_exec.cpp`) in
its queue's FIFO of `Impl::streams`:

- `ProcessSubmit` / `ProcessPresent` first pump with release (a front item blocked longer than
  `ContextOptions::parked_timeout`, 5 s, goes on without what blocks it: held-back work must not
  outlive its timeout just because the guest keeps submitting), push the item and pump again.
  `ReplayMain` pumps with release every 250 ms while something is waiting and nothing arrives.
- The scheduler runs the runnable queue front with the lowest seq, restarts after every item that
  progressed (a signal can unblock streams of other queues), and never lets a later item of a queue
  pass an earlier one. A present is therefore behind everything its queue submitted before it.
- A `SubmissionItem` creates its `Recorder` (heap) at its first run. Before that it is held back
  while a `WaitSync` names a sync nobody has fenced yet (`DependenciesKnown`, `held_back_submissions`
  once per submission; released by the timeout: `parked_timeouts`, logged once per sync). Then
  `Recorder::RecordList(list, start, stop, release)` replays the records from a cursor (list, word).
- **Event model** (`Impl::events`, `video_core/nvn/queue_sync.h`, unit tested without Vulkan): the
  value of every event word as of the replay position, and for each the GPU signals (queue, timeline
  value of the segment holding the signal) it is built from. Event identity is the event's memory
  address. An event the model has not seen reads its word from guest memory (DK stores
  `0xFFFFFFFF` before the first frame). `SignalEvent` records update the model (`event_signals`); the
  guest word itself is written by the recorder's completion action. `Sink::NotifyEventCpu` (a
  command, ordered with `Submit`) applies `nvnEventSignal` and event re-initialisation, then pumps.
- A `WaitEvent` record the model satisfies (`event_waits`) needs the GPU to order it after the
  signalling segments: for every signal of another queue the recorder ends the segment (one break for
  all of them) and the next segment waits on that queue's timeline semaphore at the signal's value,
  like `WaitSync`. Signals of the same queue need nothing.
- A `WaitEvent` it does not satisfy **suspends** the stream: everything recorded so far is
  submitted (`Recorder::Suspend`: end the segment, `SubmitSegments`, begin a new one, the same
  sequence as an early submit), the `Recorder` and the cursor stay, and the other queues' streams
  run. The wait is evaluated again whenever something made progress. `event_waits_blocked` counts
  each wait once however often it is looked at.
- **Timeout.** A wait nothing satisfies is passed after the timeout (the scheduler calls `Advance`
  with `release`): `LOG_ERROR` once per (event address, mode, value) with the queue, seq, the wanted
  and the current value, counted in `waits_unsatisfied` (every release, also the ones not logged).
  Never skipped silently. The later work of other queues was never held up by it.

Invariants:

1. At every switch between streams the stream being left has everything it recorded already
   submitted: it finished (`Finish`) or suspended (`Suspend`). Storage layout tracking
   (`Storage::current_layout`), pool dirty ranges and timeline values assume recording order ==
   `vkQueueSubmit` order, and a recorder that is not running owns no unsubmitted work.
2. A wait only names a segment that has been submitted: the model only holds signals of records
   that were replayed, and a stream that replayed one has either finished or suspended.
3. Per-queue FIFO: one NVN queue is one ordered stream, so only one recorder per queue exists at a
   time and the queue's timeline values are allocated in recording order.
4. Presents stay behind the earlier work of their queue (FIFO), and are never suspended.
5. Completion callbacks are still in seq order. The completion thread waits at the front flight
   until it is submitted, so a suspended item delays the callbacks behind it, never reorders them.
   There is no cycle: the replay thread never waits for the completion thread on this path (a
   presentation pass waiting for a flight that is not submitted skips it, see `WaitFlight`).

Shutdown: `~Impl` joins the replay thread first, then `StreamScheduler::Clear()` destroys the
suspended items there, while every Vulkan object still exists and before the unsubmitted flights are
released to the completion thread. `Recorder::~Recorder` aborts: it ends the open command buffers and
waits for the pieces that were already submitted (the wait names only submitted segments, so it
ends).

### Pass boundary syncs (setting `nvn_sync_diet`)

Per render pass the replay used to put three full syncs (`ALL_COMMANDS`, every memory access) between
two consecutive passes: the render pass's external dependency out (`GetRenderPass`), the explicit
memory barrier of the next op (`EndPass` -> `HostMarkWritten` -> `FlushPending` / `PrepareImages` in
`BeginPass`, `Recorder::MemoryBarrier`), and the render pass's external dependency in. The first and
third add nothing to the middle one: a pipeline barrier recorded after `vkCmdEndRenderPass` has the
whole pass in its first scope and everything after it in its second, with the same stage and access
masks.

With `nvn_sync_diet` (the default; `PipelineCacheOptions::sync_diet`) `RenderPassDependencies` returns only the by-region self-dependency
(texture/attachment feedback inside a pass), so Vulkan's implicit external dependencies apply
(`TOP_OF_PIPE -> ALL_COMMANDS` at the start: waits for nothing; `ALL_COMMANDS -> BOTTOM_OF_PIPE` at
the end: blocks nothing), and the explicit barrier is the one sync at the boundary. It stays the
full barrier: NVN gives no order at a pass boundary (a render target change is not a barrier; only
the guest's `Barrier` op orders shader/ROP/texture-cache traffic, see the `Op::Barrier` comment in
`nvn_draw.cpp`), so the sync belongs to the replay's own splitting of one NVN stream into Vulkan
passes and copies and has to cover everything those ops do between them. The segment-start barrier
(`BeginSegment`) stays: command buffers of one queue have no implicit memory dependency, and it merges
with the first op's barrier. A typed barrier (source stages tracked from what the ops since the last
one did, destination from the consumer) is not done: it needs per-resource hazard tracking to stay
correct across barrier chains (a narrower source loses write-after-read and transitivity through
earlier compute work). Passes not begun by `BeginPass` (the masked clears of `nvn_draw_clear.cpp`)
flush the pending barrier themselves (`DrawHost::HostBarrierBeforePass`, a no-op with the setting
off). Checked with Vulkan synchronization validation (DK town on the HLE, Kirby newgame, MKW: no
hazard that the setting off does not have) and timed with `EDEN_NX2_NVN_GPU_PROFILE=frame` (DK city
and town, 2 interleaved reps: about 0.45 ms less GPU busy time per frame of 21-26 ms, fps and p99
unchanged at the 30 fps pace).

### Dispatch ordering (`EDEN_NX2_TRY_DISPATCH_ORDER=1`, candidate)

NVN does not order a dispatch after the shader writes of an earlier one: the game separates
dependent dispatches with `nvnCommandBufferBarrier` (OoT's compute queue in Kokiri: 131 dispatches,
51 barriers, e.g. `ORDER_INDIRECT_DATA | INVALIDATE_SHADER` before each indirect dispatch whose
arguments the previous one wrote). The replay puts a full memory barrier before every dispatch that
follows a dispatch (`HostMarkDispatchWritten`, flushed by `PrepareOutsidePass`). With
`ContextOptions::dispatch_order_by_barriers` (`EDEN_NX2_TRY_DISPATCH_ORDER=1`; off by default until
the A/B) `HostFlushBarrierBeforeDispatch` leaves that barrier out when nothing but earlier dispatches
wrote since the last barrier: a game `Barrier` (`HostMarkWritten`), a pass end, a copy, an upload,
a shadow or an image transition of the replay all still emit it, and every other consumer
(`FlushPending`: draws, passes, copies, shadow copies, readbacks) orders after the dispatches as
before. Indirect dispatches keep the full flush. Synchronization validation reports the unordered
dispatch pairs the game leaves (whole bound ranges, not the words they touch) as hazards; they are
the console's own ordering, not replay faults. TSV rows `dispatches after a dispatch with no
barrier between` (in both modes: the barriers the switch removes) and `memory barriers recorded`.

What a game barrier waits for (`nvn_barrier.h`, from the methods the NVN driver records per bit):
only `ORDER_PRIMITIVES` (WAIT_FOR_IDLE) and `ORDER_INDIRECT_DATA` (end of the GPFIFO segment) wait
for all earlier work, which in Vulkan is ALL_COMMANDS to ALL_COMMANDS; no narrower stage mask is
correct for them. `ORDER_FRAGMENTS` / `ORDER_FRAGMENTS_TILED` order pixel work only and the
`INVALIDATE_*` bits are `_NO_WFI` cache invalidations. With the setting on, such a barrier outside a
pass records nothing (TSV row `game barriers outside a pass without a wait for earlier work`; a
dispatch after it counts in `dispatches after a dispatch with no barrier between`, with the setting
off it does not): the last pass end is behind the pending barrier already, and dispatches before it
stay unordered for later dispatches as on the console. The replay's one lazily merged barrier
serves every later command, so its masks stay ALL_COMMANDS with MEMORY_READ | MEMORY_WRITE as
destination; its source access mask holds the writes only (a read there makes nothing available).

Rule for this family (coordinator decision): the replay orders dispatches exactly as the console's
driver does. The NVN driver's dispatch path records no WAIT_FOR_IDLE and no cache invalidation of
its own: `DispatchCompute` is a deferred op whose launch writes the QMD and calls MME macros
(methods 0xE74..0xEB0); NVK's `vkCmdDispatch` waits for nothing either. [INFERENCE: the macro
bodies are uploaded at run time and not decoded; a wait inside one would order every dispatch.]
So a write -> read pair of dispatches with no ordering game barrier between is the game's own
hazard and stays unordered; if the macros turn out to serialise dispatches, that exact ordering is
what the replay must reproduce. A hazard that changes the picture against the setting off is a
replay bug whatever its kind. DiagHub batch 8 (OoT VTX-HLE, oot-dark): the picture with the
setting on matches off (frames 2581-2941, animation drift only), and synchronization validation
found no dispatch hazard at all, only `vkCmdCopyBuffer` after `vkCmdCopyBuffer` WRITE_AFTER_WRITE
on the inline uniform updates (two updates of one range flushed after one pass: now ordered,
`FlushInlineUpdates`).

### Barrier merging (`EDEN_NX2_TRY_BARRIER_MERGE=1`, candidate)

OoT's frame records about 1,100 pipeline barriers (Nsight, river scene, 2026-11-06: 1,127 per frame,
about 3.0 ms of GPU time as the barrier calls themselves; HleSpeed6 / HleBarriers handoffs). Every one is
the replay's lazy ALL_COMMANDS barrier (`Recorder::MemoryBarrier`, `PrepareImagesSpan`): whoever writes
marks it pending, whoever needs the writes flushes it. The switch (value 1 = everything; 2, 3, 4 = one
part, `BarrierMergeMask`) removes the ones that order nothing, in three independent parts:

1. **Buffer commands with known ranges** (`BARRIER_MERGE_BUFFERS`, `nvn_barrier_track.h`). The replay's
   own copies and merges know exactly which bytes of which `VkBuffer` they read and write: the guest ->
   mirror copy-ins, the mirror -> snapshot copies, the storage-shadow copies, the write-back merges. Such a
   command goes through `DrawHost::HostBeginBufferOp(accesses)` / `HostEndBufferOp()` instead of
   `HostFlushBarrier()` / `HostMarkWritten()`. With the switch off those two are exactly the old calls. On,
   the recorder keeps the ranges since the last emitted barrier (`BarrierTracker`) and emits a barrier first
   only when (a) something it does not track wrote since (a game dispatch, a draw, a pass end, an image
   copy, an upload: the old pending flags), or (b) a range overlaps one noted and one of the two writes
   (read-after-write, write-after-read, write-after-write). Read-read and disjoint ranges need none. Any
   other consumer (`FlushPending`, `HostFlushBarrierBeforeDispatch`, a pass begin, an image op) sees the
   tracker non-empty and flushes as it always did, so nothing the tracker does not know about is ever
   skipped. The write-back merge's ranges are widened by 256 bytes on every side (line mode stores whole
   units of up to 32 words around a changed word); a null buffer, an inverted range or more than 256
   ranges since the last barrier saturate the tracker (every later buffer command gets its barrier).
   Nsight: 85.8 of the 221 dispatch -> dispatch barriers per frame (0.67 ms) sit between two write-back
   merge dispatches of one conflict flush (`FlushShadow` per entry, a `HostMarkWritten` after each), and
   21.8 (0.22 ms) between a merge and the copy that follows it.
2. **Inline updates that copy nothing** (`BARRIER_MERGE_INLINE`): `FlushInlineUpdates` forced a barrier in
   front of its copies and a mark after them even when every update was unresolved or rewritten whole by a
   later one; with nothing to copy it now returns first.
3. **Adjacent requests** (`BARRIER_MERGE_ADJACENT`): in the draw path's requests (`HostFlushBarrier`,
   `HostBarrierBeforePass`, `HostTransition(Many)`, `HostFlushBarrierBeforeDispatch`,
   `HostBeginBufferOp`) the barrier waits in `Recorder::deferred_barrier` (a memory barrier plus the layout
   transitions asked for, one entry per image: old layout of the first request, new layout of the last)
   until the next command is recorded. Every command is recorded through `Recorder::Cmd()`, which first
   records the waiting barrier; a request that finds one waiting (no command in between) joins it, whoever
   asks (the recorder's own `MemoryBarrier` / `PrepareImagesSpan` fold in too, but only the draw path
   starts a deferral). Two ALL -> ALL barriers with no command between them are one barrier, so this
   cannot change an order. Nsight: 395.8 barriers per frame sit directly in front of a pass's first draw
   or clear in 164.5 runs (231 more than one per run, 0.65 ms): the flush of the inline updates, the
   sampled-image transitions of `PrepareOutsidePass`, the attachments' transitions and flush of
   `BeginPass`. **Rule for code that keeps a `vk::CommandBuffer`:** ask `HostCmd()` again after any
   barrier request before recording with the old handle (the masked clears did not: `nvn_draw_clear.cpp`
   fetches it again); every other site that keeps one asks after its requests. `EndSegment` records what
   is left before the host barrier; `Abort` drops it (the layouts are reset).

What stays: the barriers between a game dispatch/draw and anything (their accesses are not known: raw
global memory reaches every pool), `Op::Barrier`, the image transition barriers themselves, the counter
engine's own. Untried: tracking the draw and dispatch accesses (bound ranges, images) so that a barrier
between a dispatch and an independent copy could go too (the remaining 134 dispatch -> dispatch, 170
dispatch -> copy, 134 copy -> dispatch, 123 copy -> draw per frame of the Nsight census).

TSV rows: `memory barriers recorded` (the requests, in both modes), `barrier merge: buffer commands ...`
(four rows: no barrier needed / left out / overlap / after untracked writes), `barrier merge: inline
update flushes that recorded no copy`, `barrier merge: barrier requests folded ...` and `... barrier
commands recorded for waiting requests ...` (vkCmdPipelineBarrier calls of the deferred requests; each
stands for one or more). Tests: `src/tests/nvn/barrier_track.cpp` (the hazard rules), `compute_vulkan.cpp`
(write-back chains, overlapping commands, inline updates: same pool bytes with and without), `draw.cpp`
(passes, masked clears, inline updates: same pictures). Measure with synchronization validation
(plan env `VK_ADD_LAYER_PATH=C:\eg\VulkanSDK\1.4.341.1\Bin`, `VK_LAYER_SETTINGS_PATH=F:\eg\results\sync-diet\vk_layer_settings.txt`,
setting `Renderer/debug=true`; `rundiag.py syncval`): no hazard key that the setting off does not have.

## Memory pools

GPU address == guest VA, so a buffer address resolves to `(pool, offset)` by range lookup in
`pools_by_address`.

| Pool kind | Backing |
|---|---|
| CPU-accessible (flag other than `CPU_NO_ACCESS`) and the guest storage is one contiguous host mapping with `minImportedHostPointerAlignment` alignment of pointer and size | **Imported**: `VK_EXT_external_memory_host`, zero copy; pool buffers are bound into large (up to 256 MiB, mapped-contiguous) imported chunks shared by neighbouring pools. Needs a host-coherent memory type. |
| CPU-accessible, import not possible or failed | **Mirror**: a device-local `VkBuffer`; see below. Counted by reason (`pools_mirrored: guest storage is several host runs` / `host import unavailable or failed`); the log line names the number of host runs and the first break. |
| `CPU_NO_ACCESS` | **DeviceLocal** buffer (one `vkCreateBuffer` + VMA, about a millisecond). Its guest storage's current contents are uploaded at the pool's first use (the game may still fill the storage after creating the pool): by `EnsureBacking`, or for all pools at once when the raw global table is built (`UploadWholePools`: one staging buffer and one submission for many pools; all-zero chunks are written with `vkCmdFillBuffer`, buffers do not start zeroed). |
| Virtual (`0x200`) | none: a table of mapped ranges (`MapVirtual`), see "Virtual pools" |

Why a pool is one host mapping or not: the guest's pages map to the host through the physical page
(`DeviceMemory` backing is one host mapping, guest physical = backing offset), so guest memory is
one host run exactly where its physical pages are consecutive. The kernel allocates a page group
(`KMemoryManager::AllocatePageGroupImpl`, heap growth, `MapPhysicalMemory`) as the biggest buddy
blocks first, 1 GiB at most (`KPageHeap::MemoryBlockPageShifts`), each at a random place: every
large allocation broke into pieces every 1 GiB (DK's 448 MiB pool over such a boundary, Kirby's
64 MiB + 4 KiB pools into 2-6 runs). Requests of 16 pages and more are now allocated as one
physically contiguous run, directly behind the previous one when that is free
(`KPageHeap::AllocateContiguous`), smaller ones fill the heap from the lowest address; placement
is no longer random (the guest cannot observe physical pages). The application pool is one
partition (`SetupPoolPartitionMemoryRegions`): Horizon's split at the middle of DRAM made a
second kind of break in every allocation across the middle (DK's 448 MiB city pool kept a 1 MiB
tail in a separate run after the contiguous allocation alone).

Backing (the host import of the chunk around the pool, or the device-local buffer) happens on the
`NvnBacking` thread right after `CreatePool` (a pool inside an already imported chunk needs
none), or at the first command that needs the pool, whichever comes first; `Pool::pending`
(acquire/release) tells the replay thread the backing is complete, `Impl::backing_mutex`
serializes the two threads (and the raw global table build).
Host import cost: a chunk import is ~1 ms of `vkGetMemoryHostPointerPropertiesEXT` and 3-5 ms of
`vkAllocateMemory` per 256 MiB once its pages are resident. Two things made it 30-470 ms in DK
(every one a 50-300 ms frame hitch: `vkAllocateMemory` holds a driver lock, the replay thread's
`vkQueueSubmit` waited for it, the guest's `nvnSyncWait` for the replay; DK's frame controller then
dropped to 30 fps for ~15 s):
- **One huge view.** The NVIDIA driver (621.01) queries the host pages of the whole view a
  pointer lies in, both in the properties call and in the import: its cost followed
  `VirtualQuery` of the pointer, ~15 ms per GiB of committed memory from the pointer to the end
  of the run (25-145 ms each in DK with 9.8 GiB committed in the 14 GiB backing view, even for a
  64 KiB import). `Common::HostMemory` maps the backing as 64 MiB views back to back
  (`HostMemory::BackingViewSize`, one contiguous range as before): 0.02-1.3 ms. Imports span
  views; a standalone benchmark checked that a GPU fill across a view boundary lands right.
- **Never-touched pages.** The import faults each one in (0.17 ms/MiB, 40-210 ms per window)
  while it holds the lock. `HostImporter::MakeResident` reads one byte per committed page first:
  the backing thread does it for the chunk window before taking `backing_mutex`, the import calls
  it again (cheap once resident) for its other callers. A read leaves the guest's bytes alone; on
  Windows only committed readable pages are read.
The earlier pre-touch experiment (DkSweep) touched the pages but kept the single view, whose
query cost hid the gain. Measurements and the benchmark: `F:/eg/results/hle-map-hitch`.
DK's first raw-global draw used to stall for ~1.6 s backing ~300 pools; the table now builds in
~80 ms. `ContextOptions::background_backing = false` backs on demand only.

`vkGetMemoryHostPointerPropertiesEXT` is loaded with `vkGetDeviceProcAddr`; the alignment is read
with `vkGetPhysicalDeviceProperties2`. `VK_EXT_external_memory_host` is in `FOR_EACH_VK_EXTENSION`
of the device, so it is enabled whenever the physical device offers it (the only change that is
not behind the HLE switch; it adds no feature structure and no calls).

### Mirror pools

The guest memory stays the CPU truth.
- **Before** an op that reads a range (copy source, later vertex/uniform/index/indirect binds),
  the replay compares the guest pages of that range (`Pool::MIRROR_PAGE`, 4 KiB; a CityHash64 per
  page in `Pool::mirror_pages`, each page looked at once per submission) with what the mirror last
  received, and `vkCmdCopyBuffer`s only the changed pages (runs of consecutive pages through a
  staging buffer, one barrier before and after) into the mirror. A draw that finds all its mirror
  ranges unchanged keeps its render pass (`HostMirrorNeedsUpload`, no commands recorded); only a
  changed page ends the pass for the copy (`render_pass_break: mirror upload`).
  `NotifyCpuWrite` (FlushMappedRange) makes the pages it names get compared again within the same
  submission. The first CPU upload copies the page. The upload clips out ranges owned by an
  unfinished submission (`Pool::gpu_dirty`). A page covered by one pending range also skips the
  hash without caching a comparison. Both decisions use the same per-lookup snapshot. Partial
  coverage still requires hashing so CPU changes outside the write can reach the mirror.
- **Page tracker gate** (default; `EDEN_NX2_NVN_MIRROR_HASH_ALWAYS=1` or no `GuestMemory::Track`:
  every page is hashed as above). At its first check a mirror pool is registered with the page
  tracker (`Pool::mirror_track`, every page armed). A page hashed after that, with no GPU write of
  it pending (no `gpu_dirty` range on it: its write-back is in the guest bytes), becomes `trusted`
  and is not hashed again (`pages_trusted` in the heartbeat) until something takes the trust away:
  a guest CPU write the tracker reports (`Recorder::MirrorTakeCpuWrites`: one `TakeWritten` of the
  whole pool when its `WriteEpoch` moved, into the per-page flags), a GPU write of the page
  (`NotePoolWrite`; the write-back and a report's timestamp land untracked), a host service's
  write or an untracked flush (`NotifyCpuWriteNow`), and a texel flush (`FlushTextureNow`). A
  guest flush with `flush_tracked` keeps the trust (the CPU's stores are in the tracker), as for
  the pool mirrors' audit below ("Guest flushes of mirrored pools"). OoT's Castle Town hashed
  ~30,500 unchanged pages (~125 MB, ~10 ms of the replay thread) per present without it.
- **After** an op that writes a range (copy destination, clear), the range is recorded as
  GPU-dirty and, at the end of the submission, copied into a download buffer.
- **At completion** the completion thread copies the download buffer into the guest bytes (host
  pointer or `GuestMemory::Write`) *before* the completion callback runs, then clears the dirty
  entries of that seq.

### Host writes (`Sink::NotifyHostWrite`)

A system service that writes guest memory the GPU reads (nx2:vdecsvc writing a decoded frame
into a decoder image, `Runtime::NotifyHostWrite` from `nx2_gsv.h`) acts like a DMA engine on the
console: the GPU's next read sees the bytes, whatever the pool's CPU access, and no NVN call comes
in between. In submit order the replay then: compares a mirror's pages again (as for
`NotifyCpuWrite`), copies the range into a GPU-only (CPU_NO_ACCESS) pool's buffer at once
(`UploadPoolRanges`, barriers against earlier and later submissions; `host_write_uploads`), and
uploads every texture that lies wholly inside the range again, even over GPU contents. A texture
only partly inside keeps its image (the rest may be GPU contents the guest bytes do not hold).

An NVN image (storage image) of an sRGB texture reads and writes the texel bits without sRGB
conversion. Vulkan stores no sRGB format, so such textures are created with STORAGE usage and
`EXTENDED_USAGE`, and their storage views use the UNORM format of the same bits
(`HostFormat::storage_view`); every other view of them leaves the storage usage out. OoT's title
movie is composited by a fragment shader that loads a 2560x1440 RGBA8 sRGB image: before, the draw
was dropped (`texture ... cannot be bound as a storage image of shader type 2`, one
`replay_ops_unsupported` per frame) and the screen stayed black.

The same holds for an NVN texture view of another format of the same texel size: the view can be
registered as an image even when the texture's own format cannot be stored. Every uncompressed
colour format that cannot be stored but whose texel size has a storable unsigned integer format
(`SameSizeUintFormat`: R8/R16/R32/R32G32/R32G32B32A32_UINT) is created with STORAGE usage and
`EXTENDED_USAGE` (`HostFormat::storage_alias`), as the emulated path's texture cache does for
images with a storage-capable view format. OoT's in-game compute passes (program 83e621f0eda967ea)
bind R32UI views of 854x480 and 1920x1080 RGB9E5 textures and store the shared-exponent words they
pack themselves (`SUST.P`); before, those dispatches were dropped (about two
`replay_ops_unsupported` per frame) and the 3D scene stayed nearly black.

A surface instruction (`SULD`/`SUST`/`SUATOM`) names the constant bank of its handle itself (bits
54-58); the lowering reports it as `required_surface_buffer`, and `NvnEnvironment::SurfaceBoundBuffer`
answers with it (the texture bank only when no surface named one). Before, the NVN environment
answered the texture bank (c2 when the program has no texture instruction), so a compute program
whose only handle use is a store read its image handle from c2 instead of its driver bank's image
table: the image binding was unsupported and the typeless store was declared float, whatever the
view's format.

## Textures

Per `TextureDesc`: a `Storage` (a `VkImage`, `MUTABLE_FORMAT`, cube-compatible when applicable) with
the format `NVNformat -> PixelFormat -> MaxwellToVK::SurfaceFormat`. Three-component formats and
`*_I2F/UI2F` have no host counterpart (texture fails, ops on it are dropped and logged once).
When the device cannot hold a compressed format (ASTC on a desktop GPU) `SurfaceFormat` returns an
uncompressed stand-in (`HostFormat::needs_transcode`): the replay decodes the guest's blocks on the
CPU when it uploads (`nvn_upload.cpp`, `Tegra::Texture::ASTC::Decompress` / `DecompressBCn`, in
parallel over levels and layers) and the image holds the decoded RGBA8 texels. For a 2D ASTC storage
in an imported pool, `nvn_astc_gpu.cpp` instead runs the same `astc_decoder.comp` directly from the
pool buffer into UNORM storage views of all levels and layers, avoiding the CPU decode and staging
copy. Its pipeline compiles on `NvnAstcInit` without sharing an externally-synchronized pipeline
cache; uploads before it is ready, other pool kinds, and unsupported layouts use the CPU path.
`EDEN_NX2_NVN_NO_ASTC_GPU=1` forces the CPU path, and `EDEN_NX2_NVN_ASTC_VERIFY=1` additionally does
the CPU decode and a readback comparison (slow, for validation).
**BC stand-ins.** With the `astc_recompression` setting at Bc1/Bc3 (or `EDEN_NX2_NVN_ASTC_STANDIN=
rgba8|bc1|bc3`, which overrides it) the 2D single-sample ASTC stand-ins are BC1_RGBA / BC3 images
(UNORM or SRGB as the ASTC format; `ToBcStandin` in `BuildTexture`, `HostFormat::bc_standin`), an
eighth / a quarter of the RGBA8 memory. Their upload decodes into a reused RGBA8 scratch image
(`nvn_bc_encode.h` `BcScratch`, up to 64 MiB of texels per fill, a level's layers in chunks), runs
the emulated path's `bc_encoder.comp` into the scratch's block buffer and copies the blocks into the
level (`Recorder::RecordBcUpload`): the GPU decoder fills the scratch from the pool, the CPU path's
decoded staging is copied into it, a partial CopyBufferToTexture likewise (4x4-aligned regions
only). Tile-backed sparse, 3D and multisampled ASTC storages stay RGBA8. A BC stand-in cannot be
cleared, bound as an image, viewed in a non-BC format, or copied to/from another stand-in in regions
that cut its 4x4 blocks (unsupported, logged); the storage census counts them per format as
`as BC3 ...`, `as BC7 ...`. `renderer_vulkan.cpp` still pins the setting's override to
Uncompressed so `MaxwellToVK` returns RGBA8; the replay reads the saved value itself.
`EDEN_NX2_NVN_ASTC_STANDIN=bc7` (the setting has no BC7) holds them as BC7 (same memory as BC3, far
less loss) with `bc7_encoder.comp`, DirectXTex's BC7 compute encoder (MIT) ported to GLSL: a pass
for modes 4/5/6, one per partitioned mode, then the encode, the best mode per block in the
scratch's `modes` buffer between them (`BcGpuEncoder::RecordBc7`). `EDEN_NX2_NVN_BC7_QUALITY`
picks the modes tried: 0 = 4/5/6, 1 (default) adds 1/3/7, 2 adds 0/2 (blocks modes 4-6 already
hold within ~45 dB skip the partitioned passes). The port orients each subset's box along its
covariance, checks the neighbours of the last pass's projected indices and refits the endpoints by
least squares, and keeps alpha 0/255 exact and texels on their side of alpha tests at 64/128/192
(an error penalty, endpoint rounding and the index choice). Its texel loops read no constant
arrays: glslang copies such an array into local memory at every dynamically indexed read, which
made the first port ~5x slower. Each upload's GPU time (decode or copy and encode) is
measured with a timestamp pair per submission (`BcUploadTimes`): uploads of 16 MiB stand-ins or
more get a log line, and every 60 presents a line sums the window's uploads.
**Smart BC stand-ins.** With the BC3 kind (the default) each stand-in gets the BC format its ASTC
blocks need (`SmartBc`, `SmartStandinFor`). `Tegra::Texture::ASTC::ClassifyChannels` (next to
`IsOpaque`, sharing its 11-bit mode table and endpoint-mode decoding) reads each block's colour
endpoint modes (CEMs) and void-extent colour: luminance CEMs (0, 1) only -> L, also
luminance+alpha (4, 5) -> LA, colour CEMs without alpha -> RGB, else RGBA; HDR CEMs, error blocks
and dual-plane blocks whose second plane is on R/G/B (for L/LA) count as colour or RGBA, all-zero
blocks (block-linear padding) are skipped, and it stops at the first block that makes it RGBA.
L -> BC4 (half of BC3), LA -> BC5 holding (L, A), RGB -> BC1 in four-colour mode (half), RGBA ->
BC3; sRGB has no BC4/BC5, so sRGB L/LA go to BC1/BC3. `EnsureImage` classifies the guest bytes
before the first image (`ChooseSmartStandin`; all zero: BC3 until a whole upload brings blocks).
The ASTC decode of an L/LA texture is (L, L, L, 1) / (L, L, L, A), so a BC4/BC5 view composes each
component of the guest's swizzle with (R, R, R, ONE) / (R, R, R, G) (`ComposeStandinSwizzle`, in
`ViewKeyOf`; the view format comes from the storage, `StandinViewFormat`). Each later whole upload
classifies again (`Recorder::RefitStandin`): bytes that no longer fit (alpha in BC1/BC4, colour in
BC4/BC5) recreate the stand-in as BC3 for good (old image and views retired after the submission,
logged, counted). A partial upload or a texture copy into or out of a smart stand-in first pins it
to BC3 (`PinStandin`: copies need one format on both sides, and a later format change must not
drop copied texels). An sRGB view rules BC4/BC5 out for its storage (one already BC4/BC5 is
recreated at an upload the view forces). Overlap transfers between stand-ins of two formats are
refused, as between RGBA8 and BC ones. With smart stand-ins the BC1/BC3 colour blocks are refined
(`bc_encoder.comp` REFINE): the better of the box fit and the extremes along the principal axis,
then least-squares refits and single 565-code steps, each kept when the error of the block's
texels inside the level drops; the emulated path's modes (0, 1, no REFINE) encode the same bits
as before. Switches:
`EDEN_NX2_NVN_SMART_BC=0` (BC3 everywhere, box fit: the old behaviour), `EDEN_NX2_NVN_SMART_BC1=0`
(opaque stays BC3), `EDEN_NX2_NVN_SMART_BC45=0` (L to BC1, LA to BC3), `EDEN_NX2_NVN_BC_REFINE=0`
(box fit); the startup line names them. The census adds `as BC4` /
`as BC5` and a `smart BC so far` line (scans per class, MiB scanned and scan time, re-classified,
pinned by copies); the 60-present window line counts the same per window.
Block-linear storages that need no decode (every texel size, BCn blocks the device holds, 2D, arrays,
3D, mip chains) of 64 KiB or more are untiled on the GPU (`nvn_untile_gpu.cpp`, idea from
switch2-recomp's "GPU untile"): the emulated path's `block_linear_unswizzle_3d_bcn.comp` (its
`BlockLinearUnswizzle3DPass`; 1/2/4-byte blocks now store only their own bytes) reads the tiled
bytes, as elements of the largest power of two up to 16 bytes that divides a row (what
`UnswizzleTexture` does), and writes the CPU path's linear rows into a device-local buffer that
the CPU path's regions copy into the image. The pool's bytes are read in place, at the command's
execution, where the GPU has them: an imported pool (storage shadows over it are flushed first) or
a GPU-only pool's device-local buffer (its initial upload done first; host writes are pushed into
it, GPU writes land there). Mirrors, virtual pools, and storages whose last GOBs lie past their
bytes copy the tiled bytes into staging (zero-padded, 1 MiB slices on the decode workers) first.
Before its pipeline is
built (`NvnUntileInit`), for decoded formats and for levels beyond the shader's 32-bit addressing
the CPU untiles (logged once per reason; `untile gpu/direct/fallback` in the `mem(...)` state line).
`EDEN_NX2_NVN_NO_UNTILE_GPU=1` forces the CPU path. Depth/stencil
D24S8/D32S8 are not stand-ins: copies, clears and attachments work on the host format;
`guest_layout_differs` only stops a CPU upload of guest bytes (`depth_upload_unsupported`).

- **Layout.** `nvn_texture_layout.*` is the nxnative layout (block height/depth from the packaged
  nibble or the automatic rule, per-level offsets, minimal layout, linear stride).
- **Storages are cached and lazy.** `Context::Impl::storage_cache` keeps every storage (key: pool,
  offset, size and everything the layout and the decode depend on) after its textures died, so the
  transient render targets DK re-creates over the same bytes every frame reuse one image
  (`storages_reused`; the 135k `alias_unsupported` of M1 were this churn). The VkImage is created
  at the first use (`EnsureImage`); unreferenced images are trimmed least-recently-used above 6 GiB;
  a finalized pool drops its storages.
- **Contract for code that records GPU work:** `Recorder::EnsureContents(s, will_overwrite)` before
  the GPU reads or writes a storage, `NoteGpuWrite(s)` after it wrote one (`PrepareImages` does it
  for the layouts that write).
- **Contents.** At the first use, `UploadStorage` reads the guest bytes (`GuestBytes`: host pointer,
  copy, or gathered through a virtual pool's mappings), unswizzles/decodes them
  (`PlanStorageUpload` + `FillStorageUpload`, Switch 2 GOB order via `native_tiling`; tiled storages
  without a decode on the GPU, above) and copies them
  in. Skipped when the caller overwrites the whole storage, for multisampled images and for GPU-only
  pools' storages after the first upload (their bytes never change again).
- **CPU writes (the video panel).** A storage uploaded from a CPU-accessible pool is
  `StorageOrigin::Cpu`: it remembers a sampled signature (256 lines of 64 B over the storage) and a
  full hash of the bytes it was built from. At most once per submission and present, when its check
  is due, the signature is compared (a difference proves a change and re-uploads; ~µs); the full
  hash every 90 presents. Checks happen every present at first, then back off up to every 30
  presents while the bytes stay equal. `FlushMappedRange` (pool and buffer), `nvnTextureInitialize`
  with packaged data and `WriteTexels` tell the replay ahead of the check (`NotifyCpuWrite`: look at
  once and compare the full hash; `NotifyTextureWritten`: upload again even over GPU contents).
  The check clock is the present count, plus one per 16 ms while nothing presents.
  A storage the GPU wrote is `StorageOrigin::Gpu`: the guest bytes are never looked at again
  (a flush of a nearby range must not undo GPU contents).
- **Overlapping textures.** Storages over overlapping bytes keep their own images. Every GPU write
  stamps the storage and its pool (`write_clock`); when a storage is read and an overlapping one was
  written more recently, `ResolveOverlaps` brings the overlapping bytes over. When both lay the
  bytes out alike (same start, tile height and depth, GOB columns, level offsets, texel size:
  same texels, other sizes or formats of the same texel size) it is a `vkCmdCopyImage`. Anything
  else goes through the pool's bytes (`TransferOverlapBytes`, `nvn_alias_gpu.*`,
  `host_shaders/nvn_alias_transfer.comp`), which is what the memory does: the block rows of both
  textures whose bytes reach the shared window (2D levels; 3D levels whole) are copied tightly into
  a scratch buffer (`vkCmdCopyImageToBuffer`, the pieces of `PlanStorageUpload` cut to those rows,
  so a small target over a big depth buffer costs the small target's rows), the reader's texels
  are scattered into a window of the pool's bytes at the
  offsets its layout gives them (Switch 2 GOBs, as `UnswizzleTexture` with native tiling, or
  linear rows), the writer's texels are scattered over them the same way, the window is gathered
  back into the reader's texels and copied into its image. Any non-compressed format of a
  single aspect (colour, D32F, D16) with 1 to 16 bytes per texel, mip chains, layers and 3D
  included; ASTC/BCn stand-ins, depth+stencil layouts and multisampled images are not moved and
  stay one `alias_unsupported` per pair, logged with the reason. Reading without a more recent
  overlapping write costs one integer compare.
- **ReadTexels.** `Sink::FlushTextureToGuest` reads a GPU-written texture back (one immediate
  submission) and swizzles it into its guest storage; after it the storage is CPU-sourced again.
- **View objects** (`view_of != 0`) reference the base storage; the view's base level/layer are
  added to copy/clear references. Image views and samplers for the draw path are created by the
  graphics slice; samplers and programs are only stored here.
- **Layouts** are tracked per image (whole image), changed with `ALL_COMMANDS` barriers recorded in
  submission order. If a submission fails while recording, the tracked layouts of the images it
  touched are reset to `UNDEFINED`.

## Virtual pools

A virtual pool (`PoolDesc` flag `0x200`) has no guest storage: the recorder gives it a GPU address
range of its own (`0x7D00...`, 2 MiB aligned) and `nvnMemoryPoolMapVirtual` reaches the sink as
`MapVirtual` (ordered with submissions). The replay keeps a table of runs per pool.
- A buffer range resolves (`ResolveBuffer`) to the physical pool and offset behind it when one
  physical range covers it; mappings adjacent in both pools count as one. A range over a hole or
  over scattered physical ranges fails and is counted (`virtual_unresolved`).
- A texture placed in a virtual pool reads its bytes by gathering the mapped parts from the physical
  pools (zeros for holes). A remap that changes the runs under a storage makes it upload again; one
  that maps what is already mapped changes nothing.
- A sparse texture (`NVN_TEXTURE_FLAGS_SPARSE`, tile shape in `SparseTileShape`) of 8 MiB or more
  in a virtual pool is **tile backed** (`nvn_sparse.h`, layout in `nvn_sparse_layout.h`; the
  tile rules are nxnative's, docs/reference/nvn-objects.md "Large sparse textures as Metal
  placement sparse textures"): its VkImage is a sparse residency image created without memory.
  A tile is 64 KiB: one block row of GOB columns of a level, the tiles of a layer follow each other
  (level 0 row by row, then the next tiled level ...) up to the mip tail, which is block-linear up
  to the end of the layer. A tile is an independent block-linear picture, so it unswizzles
  (and decodes ASTC/BCn stand-ins) on its own. `MapVirtualNow` marks the tiles of the remapped
  range (`SparseBacking::Remap`), a CPU write notification on a physical pool marks the tiles that
  map it (`ContentChanged`), and `StorageNeedsCommands` compares a few mapped tiles per present with
  their bytes (`VerifySparse`, the CPU may write without flushing). At the next use
  `Recorder::SyncSparse` binds 64 KiB pages (`SparsePagePool`, a budget of 40% of the device-local heaps,
  between 1 and 12 GiB) with `vkQueueBindSparse` for the newly mapped tiles, takes the pages of unmapped
  tiles back, and uploads only the mapped tiles that are dirty (staging per 64 MiB chunk). A tile
  nobody mapped, or one that found no page, reads as zero (`residencyNonResidentStrict`). Host
  sparse blocks that do not tile the guest's (ASTC with a 5-texel footprint decoded to RGBA8), a
  host mip tail that starts inside the guest's tiled levels, a shared mip tail, or a failing
  `vkCreateImage` make `EnsureImage` fall back to the ordinary whole-storage path. Smaller sparse
  textures stay ordinary images. Single-aspect depth (D16, D32F) is tiled like colour (the shadow
  arrays of Mario Kart World); depth+stencil is not. The log line `nx2:gpu map path` counts tiles
  bound/unbound/uploaded/verified. `EDEN_NX2_NVN_STORAGE_CENSUS=1` enables a replay-side inventory
  of the storages by kind and the largest ones every 900 presents. The `[nvn] replay stage`
  status line has `commit_MB` (process private bytes) next to the device heap usage.

### Storage cache budget

**MKW sparse binding batches (2026-11-14).** `SparseBacking::Prepare` compacts image binds
into horizontal host-block runs for MKW only, following the row-batching idea in
[`mkw-recomp` commit 30af22d](https://github.com/adriadam10/mkw-recomp/commit/30af22de9062367d3831d3787d136a6f43ebe275).
Only adjacent blocks of the same aspect, layer, mip, row, height, memory object and flags
merge; bound blocks also need consecutive physical offsets. Unbinds merge without a
physical-offset condition. Each record remains one host-block row, including clipped edge
blocks. Sorting and compacting do not change page ownership, residency, uploads, mip tails,
the memory budget or the existing wait before changing atlas residency. With the current
BC3 page allocation a full 512x512 guest tile uses two row binds instead of four individual
256x256 block binds, with the same 256 KiB backing. The neighboring-tile GPU test and the
per-block mapping comparison cover holes, remaps and physical-offset preservation. These
checks pass in the incremental build; no gameplay speedup is claimed.

**MKW coarse atlas levels.** The host blocks and opaque mip tail outside the guest's
tiled levels now retain backing for every MKW array layer even when no guest virtual
pages cover the tail. This adapts `mkw-recomp`'s `tail_alloc` rule: the game can populate
these foliage/lightmap levels with GPU copies rather than `MapVirtual`, and writes to
unbacked levels would be discarded. Tail residency is queued on attach and reserved before
detailed tiles, using the same sparse page pool/budget. Only tails with mapped guest bytes
are uploaded; a subsequent use of an unmapped tail must preserve its GPU-written texels.
Unmapping fine tiles still frees their pages. Other titles retain their prior tail rule.
The new BC6H regression uses a full 8192x8192x68 lightmap layout, distinct base/high-layer
tiles beyond 4 GiB, and copies/readbacks of the first coarse and final mip of layer 67
without tail mappings. These GPU readbacks and the BC3 pressure tests pass at simulated
8/12 GiB limits; gameplay appearance remains for manual testing.

**Sparse allocation reclamation and replay state (2026-11-14).** The page pool tracks live
pages per 4 MiB allocation. After a successful unbind fence and deferred page releases,
it frees wholly unused allocations beyond four cached blocks (16 MiB). Vacant metadata
slots are reused without renumbering live pages. Reserved-byte diagnostics count these
allocations separately from bound pages; VMA may retain additional allocator overhead.
The pressure test checks reclamation and refill, and neighboring-tile GPU readbacks cover
reuse after releasing whole allocations. BindUnit reuses scratch tile storage.

Repeated dynamic values compare their exact bits and preserve pending dirty groups;
the diagnostic no-state-elision switch still forces recording. An unknown predecessor
graphics pipeline restores all dynamic groups, including after masked clears. Identical
descriptor registrations preserve snapshot storage. Retirement compacts survivors in
place before invoking release callbacks, which may append more retirements. Staging-buffer
reuse remains deferred pending a separate lifetime and retention design.

**Tiny BC copies and coarse initialization (2026-11-14 live investigation).** A game copy
may end at a tiny ASTC mip edge while its BC stand-in destination does not end at a 4x4
block edge. The direct Vulkan copy cannot represent that region. Instead of rejecting
it, the replay samples the source and the destination block envelope into RGBA8 scratch,
then encodes and copies the affected BC blocks. Sampling uses raw UNORM views across
UNORM/sRGB copies. Texels outside the requested region are retained subject to normal
BC re-encoding error. Aligned copies retain the direct path; partial 3D BC copies remain
unsupported. The copy shader is built lazily on the first such copy.

Newly bound, unmapped MKW coarse color levels are zero-initialized once before use.
Later visits and remaps do not clear GPU-written coarse texels. The sparse prepare result
reports those newly resident tails separately from guest-data uploads. Regressions cover
tiny copies, surrounding texels, high array layers, untouched coarse levels and final
mips; these do not establish that every reported gameplay artifact is fixed.

Every `Storage` (one VkImage per pool, offset and shape) stays in `storage_cache` after its textures
died, so the game's transient render targets reuse their image. `TrimStorageCache` keeps the
cache under `ContextOptions::storage_cache_budget_bytes` (6 GiB): a pass runs when the cache is over
the budget and has grown by 1/32 of it since the last pass or 30 presents went by, and drops
storages only the cache holds, oldest first, until the cache is at 90% of the budget. First the
GPU-written ones that a later GPU write replaced (`StorageSuperseded`: the later-written storages
of the pool, which took the bytes over at their own use, cover every byte of its span; a storage
of the pool that is not superseded holds GPU texels no guest byte can rebuild and is never
dropped), then the ones that hold guest bytes. A game that re-allocates its render targets over
the same pool bytes in ever other shapes (a changing resolution, a frame graph) would otherwise add
one image per shape until the pool is finalized (`EvictPoolStorages`).

Two more rules (`EDEN_NX2_NVN_TRIM_LEGACY=1` turns them and the passes at presents off). A pass
also runs every 30 presents, cache over the budget or not: textures die and storages age without
a new image being made. **Age**: an idle render target (colour or depth attachment usage) unused
for `EDEN_NX2_NVN_TRIM_AGE_PRESENTS` presents (300, 0 off; `Storage::last_used_present`) is
dropped under the budget too when it costs nothing to rebuild: it holds guest bytes or nothing
(the next texture over them uploads them again), or it is GPU-written and superseded. **Device
budget**: when the device's usage plus the allocator's empty block space is over the
VK_EXT_memory_budget budget (the heartbeat's `device(usage/budget/slack)_MB`), the pass drops the
same kinds as over the cache budget (superseded, then guest bytes, oldest first) by the excess and
1/32 of the device budget, sparing storages used in the last 4 presents (the transient targets a
game re-creates every frame). GPU-written storages that are not superseded stay in every case:
the replay never writes GPU texels to guest memory, so nothing could rebuild them, and a texture
re-created over the same bytes in the same shape gets them back. `EDEN_NX2_NVN_STORAGE_BUDGET_MIB`
replaces the 6144 MiB budget.

`mem(... storage_cache(MB/superseded/aged/pressure_evicted)=` in the `[nvn] replay` status line
and the `[nvn] storage census` lines show it: the second has the idle GPU-written images and how
many of the first 1024 are superseded; the third who holds the cached images (live textures and
their render-target aliases, other holders: retired textures and views waiting for their last
submission, submissions in flight, bound targets; idle GPU-written; idle rebuildable), the bytes
per origin and the Resolution setting's twins (outside the cache's count); the fourth the render
targets by presents since their last use (all, held by live textures, idle GPU-written), the
evictions by rule and the device excess at the last pass; the fifth splits the ones live textures
hold by who keeps those textures (below).

### Memory line, texture holders, synthetic lists, strict device-local memory

**`[nvn] memory:`** (every heartbeat, 10 s; `EDEN_NX2_NVN_MEMORY_LOG=0` turns it off). Recorder
part: the lists at synthetic command memory (count, MiB, the byte cap and how many it and the
65,536 count cap dropped), the finalized textures that only lists keep (synthetic or guest command
memory) or that wait for their submissions, the re-initialised textures still alive and
`reinitialized_objects`. Replay part (`Sink::DescribeMemory`): process private bytes, private
working set and shared commit; every Vulkan heap (device or host) with the driver's budget and the
process's usage (VK_EXT_memory_budget) and the replay allocator's blocks and allocations in it (the
host heaps' usage is the Vulkan memory that sits in system RAM); the spills: allocations that asked
for device-local memory (images, DeviceLocal and Stream buffers) and got a memory type without
DEVICE_LOCAL, count and MiB per kind (storage images, pool buffers, shadows, mirrors, arenas,
staging, other; cumulative, `MemoryAllocator::TrackSpills`); the strict mode's counters; the
texture storages (and their scaled twins) by their strongest holder; the storage cache; the
imported chunks (kept until the context goes); pipelines and lowered programs (never evicted).

**Texture holders.** The heartbeat asks the recorder (`Recorder::PublishTextureHolders`) for every
texture the game no longer names and why it lives: *re-initialised* (the game initialized its
object again without a finalize: nothing will finalize the old one, so it is never destroyed;
this round only counts them), *synthetic lists* / *guest lists* (finalized, a recorded list
references it; a guest list wins), *retiring* (finalized, waiting for its submissions or an open
recording). View objects go with their texture. The report reaches the replay in order
(`Sink::NoteTextureHolders`); a storage several textures use is held by the strongest of them
(game, retiring, guest lists, synthetic lists, re-initialised). The memory line shows the previous
heartbeat's report; the census's fifth line splits the render targets held by live textures by it
per presents since use.

**Synthetic command lists** (`EDEN_NX2_NVN_SYNTH_LIST_MIB`, default 0: no byte cap). The game's
command memory callback is never invoked (`callbacks_skipped`): a recording that runs out continues
in synthetic memory (`0x7E00...`), and one that starts there gets a synthetic handle. A list in the
game's memory dies when the game records over its bytes or finalizes the pool; a synthetic one has
no such event, so only the 65,536 count cap (least recently used) ever drops it, and it keeps the
textures it references alive. With the byte cap, the least recently submitted or called synthetic
lists go while they are over it, their references released, but only ones no submission or call
used for 600 presents and that no other list calls (`List::callees`, counted per entry). A
submission copies the words it runs and retains its own references, so dropping a list never
affects work in flight. It is still not safe by construction, hence off by default: the game holds
the handle (an opaque value it may keep anywhere) and can submit or call it again at any time,
and nothing it does tells the recorder it will not; the real driver would have placed the list
in memory the callback supplied, whose lifetime only the game knows. A dropped list submitted
again is skipped (`unknown_call_handle`, logged). Use it as an A/B with the memory line and the
census showing what the synthetic lists hold.

**Strict device-local memory** (`EDEN_NX2_TRY_VMA_DEVICE_STRICT=1`, candidate, default off). VMA's
default for "prefer device, within budget" takes a host memory type once the device heap is at its
budget, and the allocation stays there. Strict: the replay's images and DeviceLocal buffers require
DEVICE_LOCAL; over the budget they are made in device-local memory anyway (the driver pages;
`over budget` counter, logged sparsely) and a storage trim pass runs at the next present; host
memory only when no device-local type can hold one (`host fallbacks`, every one logged). Stream
buffers keep their host fallback.

## Records

`Recorder` (in `nvn_exec.cpp`) walks `Submission::lists` (the recorder already expanded
`CallList`):

- `ClearTexture`: full level -> `vkCmdClearColorImage`; partial -> `vkCmdClearAttachments` inside a
  render pass with `LOAD`; integer clears use the uint union. Depth/stencil clears are not ClearTexture work
  (draw path). The channel mask is not in the record and is not applied.
- `ClearBuffer`: `vkCmdFillBuffer` (4-byte aligned).
- Buffer<->texture copies translate `row_stride`/`image_stride` to `bufferRowLength`/
  `bufferImageHeight` (per-row regions when they do not divide).
- `CopyTextureToTexture`: `vkCmdCopyImage`, `vkCmdResolveImage` for multisample -> single.
- A barrier (`ALL_COMMANDS`, memory write -> read|write) is emitted before an op that follows a
  write. Conservative on purpose; Phase 3 narrows it.
- After writes to imported pools a `HOST` barrier ends the segment.

## Present

`PresentTarget` (implemented over `PresentManager` + `EmuWindow`): `GetRenderFrame()`, recreate the
frame image when the window size differs (with a storage view when DLAA or FSR3 / Lossless Scaling
frame generation need one, `PresentFrameNeedsStorage`), draw the NVN texture into `frame->image`,
submit signalling `frame->render_ready` and the queue timeline, then
`PresentManager::PresentSubmitted(frame)` (a new method: `Present()` goes through the scheduler, which
has one writer) and `EmuWindow::OnFrameDisplayed()`.

The picture is the part of the texture the window's crop shows (`nvnWindowSetCrop`, kept by
`Recorder::WindowSetCrop` and carried in `PresentRequest::crop`; zero size = the whole texture;
`CropRectOf` clips it): Mario Kart World renders a dynamic-resolution part of its 3840x2160 present
texture and sets the crop to it. The blit, the passes (`PresentImage::crop`, normalised), the aspect fit
and the captures/screenshots all use that part. It goes into the window layout's screen rectangle
(`PresentTarget::ScreenLayout()`, i.e. the
Aspect Ratio setting, as on the emulated GPU; `PresentRectOf` falls back to fitting the picture's own
aspect when the layout is not the frame's) and the rest of the frame is the background colour. Two
ways to draw it:
- **Blit** (Window Adapting Filter Nearest or Bilinear, no anti-aliasing, no post-processing):
  `vkCmdBlitImage` of level 0 / layer 0 (`UNDEFINED -> TRANSFER_DST -> GENERAL`), nearest or linear.
- **Presentation passes** (every other filter, FXAA / SMAA, the ReShade chain; `PresentNeedsPasses`):
  the emulated GPU's own passes, not a copy of them. `PresentChain` (`nvn_present_chain.{h,cpp}`) owns a
  `BlitScreen` built over a `PassRecorder` (`present/pass_recorder.h`) in Replay mode, and
  `BlitScreen::DrawImageToFrame` runs `Layer` (anti-aliasing, post-processing, FSR / SGSR / FSR3
  scaling) and `WindowAdaptPass` on a host image instead of a guest framebuffer. The texture is
  copied raw (UNORM view of sRGB bytes) into the scratch image, which the passes sample in GENERAL.
  The frames' framebuffers use `CreateWrappedRenderPass(format, UNDEFINED)`, the window adapting
  pass's render pass, so they are compatible. Host images are not scaled by the resolution setting.

`PassRecorder` is the passes' "where do my commands go": the legacy scheduler (deferred, `Finish()`
submits and waits) or the replay's open command buffer (`Bind`/`Unbind` around every present; `Record`
runs at once; `CurrentTick()` is the flight's sequence number; `KeepAlive` puts objects into the
flight's keepalive). Resource rotation (`Layer`'s per-image resource ticks) waits through
`Context::Impl::WaitFlight`, which waits for that one flight's timeline value. A present is replayed
whole (it is a stream item that never suspends), so every present flight a pass recorded into was
submitted before the next present was recorded and the wait never waits for the replay thread; a
flight of a submission that is held back or suspended is not submitted and is skipped (logged).
`Finish()` (pass rebuilds only, never per frame) waits for every submitted flight but the open one.

**Resolution setting** (`nvn_rescale.{h,cpp}`, tests `src/tests/nvn/rescale.cpp`). The render targets the
game draws into are drawn at the scaled size on the host; nothing the game can observe changes (texture
sizes, `nvnTextureGet*`, copies to and from memory, `ReadTexels`, overlapping textures, uploads all stay on
the guest-sized image). `Settings::values.resolution_info` is fixed once per boot (`CreateGPU`), so the
factor is a constant of the run; at 1X nothing below exists (no twin, the push constants are the zeros and
1.0 they always were, `TranslationSwitches` carry no scale).

- **Twins.** A guest `Storage` that a scaled pass draws into owns a host-only twin (`Storage::scaled`, built by
  `MakeScaledTwin`): same format and layers, extent `ResolutionScalingInfo::ScaleUp(w, h)`, no guest bytes
  (`scale_owner`: it never uploads, is never checked against memory, takes part in no overlap). The texture
  object of a draw gets a twin texture (`Texture::scaled`, `Impl::ScaledTextureOf`: a host-only Texture over
  the twin storage whose `desc` is the scaled size) so views, framebuffers and layout tracking are the
  ordinary ones. `Storage::scale_state` (`Rescale::State`) says which image holds the newest pixels:
  `Native` (the twin is stale or unwritten), `Scaled` (the guest-sized image is stale), `Both` (one was made
  from the other). A GPU write to the twin (`Recorder::NoteGpuWrite` of a twin: passes, scaled copies,
  clears) makes the guest storage `Scaled` and stamps it as written for overlapping textures; a write to the
  guest-sized image (copies, uploads, overlap transfers, compute, unscaled passes) makes it `Native`.
- **Who reads which image.** Everything that touches guest bytes goes through `Recorder::EnsureContents`, which
  makes the guest-sized image current first (`MakeNativeCurrent`: a `vkCmdBlitImage` twin -> guest size, linear
  for colour, nearest for depth, stencil and integer formats, as the emulated GPU's `BlitScale`): buffer
  <-> texture copies, `ReadTexels` (`DownscaleNow`), overlap sources (`ResolveOverlaps`), unscaled passes,
  the game's own copies and clears. Reads at the scaled size: scaled passes (attachments; `PrepareScaledTarget`
  upscales GPU-written pixels the twin lacks, so the pass loads them), the sampled reads of a scaled pass and of
  compute dispatches (`DrawContext::ScaledTextureFor`: a twin that is current and needs no update from guest
  bytes), the present (`PresentSource`: blit or presentation passes read the twin, the crop rectangle in its
  pixels, the aspect ratio of the picture the guest's), and the game's DLSS evaluation (below).
- **Eligibility** mirrors the emulated texture cache (`ImageInfo::rescaleable`, `RescaleRenderTargets`): a 2D
  or 2D-array, block-linear, single-level, single-sample, non-sparse storage that is not a block-view alias
  or a decoded stand-in, whose format the device can blit (the emulated GPU would run a helper pass instead),
  and for colour taller than 288 rows (`RescaleHeightThreshold`); the scaled size must fit the device's
  framebuffer limits. A pass is scaled only when **every** attachment is eligible (else the pass and all its
  targets are guest-sized, `SetRenderTargets` decides, `DecideTargetScale`). Difference from the emulated GPU,
  which scales every eligible depth image: a depth-only pass (a pre-pass, a shadow map) scales its depth only
  when it is as large as a colour target of an earlier scaled pass (the scene depth); a 4096x4096 shadow map
  at 2X would otherwise be 8192x8192. The first frame of a scene size draws its pre-pass at the guest size.
- **Written storage images stay guest-sized** (the emulated 'blacklist': a compute invocation writes one
  texel, the dispatch is not multiplied), and so does everything a compute shader writes. Compute *reads*
  (texture fetches, `imageLoad` of a storage image the shader does not write) bind the twin like a scaled
  pass's samplers do.
- **Viewport, scissor, clear, copy.** The game's numbers stay guest pixels (`framebuffer_extent`, the render
  area push constant); `ApplyDynamicState` scales them (`Rescale::ScaleViewport` = the emulated
  `ScaleViewportValue`; scissors and clear rectangles by their edges, floor of the start, ceil of the end, so
  they tile and a non-empty rectangle stays non-empty). `ClearColor`/`ClearDepthStencil`/masked clears use the
  scaled rectangle. A `CopyTextureToTexture` between two eligible images whose source twin is current runs on the
  twins (`CopyTextureScaled`, regions through `Rescale::ScaleCopy`); every other copy reads the guest-sized
  source. A full `ClearTexture` of a texture that has a twin clears both (state `Both`).
- **Shaders.** The recompiler's `RescalingPass` runs in the HLE translation while the setting is active
  (`HostTranslateInfo::skip_rescaling_pass` = 1X; the pipeline-cache key carries the factors only while active,
  so 1X keys are unchanged); the replay used to push zero words. It now pushes, per draw, the bit of every texture / storage
  image descriptor (in stage order, `Bindings::texture_scaling_index`) that is bound to a twin
  (`DrawPlan::rescale_words`) and the down factor 1/scale in a scaled pass: `gl_FragCoord`, `texelFetch` /
  `imageLoad` coordinates and `textureSize` come out in guest pixels, point sizes scale up. The in-pass
  push constant elision (`PassBoundState::TakePushConstants`) compares the words and the down factor too.
  Normalized-coordinate sampling needs nothing. Not scaled (as in the emulated GPU): line widths.
- **The game's DLSS** (`Op::EvaluateDlss`). When the colour input, the depth and (for low-resolution motion
  vectors) the motion vectors are current twins and the scaled input rectangle is not larger than the output
  rectangle, DLSS (or the FSR1 / bilinear stand-in) is evaluated on the twins: rectangles, jitter and motion
  vector scale in scaled pixels, the output stays the guest-sized texture. Otherwise every image is read
  guest-sized (`EnsureContents` downscales). The status bar's `Render:` size is the size actually evaluated.
- **Failure modes.** A twin image that cannot be created makes the pass guest-sized (`RevertTargetScale`,
  logged); a failed downscale logs once and the reader sees the stale guest-sized image. Counters in the
  `rescale_*` TSV rows: scaled passes, guest-sized passes of an active setting, downscale / upscale blits,
  scaled copies, sampled bindings served by a twin.

**Screenshots** (`RendererBase::RequestScreenshot`): the replay never goes through
`RendererVulkan::Composite`, so `RecordPresent` serves the pending request itself
(`Context::SetScreenshotRequests(renderer_settings)`). Both paths ask `RendererSettings::ClaimScreenshot()`
first, so exactly one serves a request. Screenshots and the `EDEN_NX2_FRAME_CAPTURE` / `frame [N]`
captures are the guest's picture as the emulated path's `RendererVulkan::RenderScreenshot` renders it
(the guest layer through the scaling filter, stretched into `layout.screen`, black elsewhere), not the
frame: `RecordPictureReadback` clears an image of the requested size (captures: the nx2:vi2 path's
`Layout::DefaultFrameLayout(1280, 720)`) and the frame's format, blits the present blit's source into
`ScreenRect(layout.screen)` with the present's filter, and copies it to a host buffer. The frame is
not read: it has the window's size (a second resample), and DLAA, DLSS 5 neural rendering and the
final FSR change it, none of which the emulated path's screenshots contain. The request is claimed
once the image and host buffer exist (a request replaced meanwhile by one of another size is given
back); the completion thread hands the bytes to a worker that converts them (`ConvertPicture`: BGRA,
opaque, an HDR R16G16B16A16_SFLOAT picture tone mapped with `Common::Hdr::ToneMapToSdr`), calls the
callback (`callback(false)`, top row first) and `FinishScreenshot()`. A request made while no frame
is presented stays pending until the next present; a present that fails before reaching the GPU
gives the claim back. The layer stack of the request is not consulted: the replay only has the
guest's own picture. Pure parts (`nvn_screenshot.{h,cpp}`) are tested in `tests/nvn/screenshot.cpp`.

**Frame generation** (`nvn_frame_gen.{h,cpp}`, `PresentFrameGen`): created by the renderer's
`Context` constructor and owned by the replay thread. `RecordPresent` calls `PresentFrameGen::Process`
after the last stage that writes the frame (blit or passes, DLAA, DLSS 5, final FSR); the screenshot and
capture read-backs do not read the frame (they read the present's source: the passes sample it in GENERAL,
the blit reads it in TRANSFER_SRC). It runs the same `FrameGen::Process` DLSS branch as `RendererVulkan::Composite`
(Streamline constants, tags, options, Reflex, the `[dlss] rendered_fps=` line, pause/resume lines), so
that logic exists once. Only the motion-vector source differs: `FrameGen`'s `DlssInputProvider` hook
records `MotionVectorGenerator::Process(cmdbuf, ...)` of the presented frame into the present's own
command buffer (a barrier first makes the final writer's output visible to the flow pass). A resize
(`NeedsResize`) does not wait for the GPU as the legacy path does: the old generator is moved into the
flight's `keepalive` and a new one is created. `frame.dlss_frame_index` is set (UINT32_MAX when the
frame does not generate) before `PresentFrame` hands the frame over, so the Streamline-hooked
`vkQueuePresent` of `PresentManager::CopyToSwapchain` generates the extra frames. `ExecutePresent`
brackets `Recorder::Finish` (the queue submission) with the PCL render-submit markers; the simulation
markers and `slReflexSleep` are inside `FrameGen::Process`. `async_presentation` keeps DLSS-G off (one
log line, as in the legacy path); `[dlss] DLSS-G on the HLE present path` is logged when the first
frame generates. FSR3 and Lossless Scaling run the same `FrameGen` chains as the emulated GPU,
recording through the shared `PassRecorder`: `Process` records the chain and returns how many frames
to generate (the pacer plans with `PresentTarget::MaxExtraFrames()`), `GenerateFrames` acquires that
many more frames, `GenerateInto` writes each, the one submission signals all their `render_ready`
semaphores, and `ExecutePresent` presents the generated frames before the real one (counted in
`Core::g_generated_frames`). `Fsr3FrameGen` keeps one descriptor set per image view, because several
generated frames are recorded before any of them runs. Frame generation is only missing from builds
without HAS_LSFG (`FrameGenTypeSupported`, one warning per type). Pure parts (`ClassifyPresentFrameGen`,
`NotAppliedWarning`, `PresentFrameNeedsStorage`) are tested in `tests/video_core/dlss_frame_gen.cpp`;
`PresentNeedsPasses`, `SettingsNotApplied` and the replay `PassRecorder` in
`tests/nvn/present_passes.cpp`; `PresentRectOf` in `tests/nvn/screenshot.cpp`.

**`async_presentation`** already applies: `PresentManager` reads `Settings::async_presentation` when it
is built and `PresentSubmitted` branches on it (the legacy `Present` path too). On, the replay thread
only queues the submitted frame and the `VulkanPresent` thread runs `CopyToSwapchain` (swapchain
acquire, blit submit with the `render_ready` semaphore under `submit_mutex`, `vkQueuePresent`); the
frame goes back to the free queue afterwards, so `AcquireFrame` is where the replay waits for display
once it is ahead. Measured (HLE, warm cache, frame-stats causes `present_acquire` / `present_handover`
inside `present_blit`): with the setting off the replay thread spends 64% (town, 66 s of 103 s) to 74%
(newgame, 104 s of 143 s) of the run in `PresentFrame` waiting for the swapchain, `AcquireFrame` takes
~15 ms in total; with it on that wait moves to `AcquireFrame` (town 7.9 s, newgame 44.7 s; the present
thread absorbs the rest) and FPS does not improve (town 28.8 vs 29.9/30.8 FPS over all presents, newgame
41.1 vs 42.9/43.0; `ab_runner` >= 60 s mean 23.96 vs 24.4-24.96 and 42.4 vs 43.8-44.3). The replay
waits, it does not limit the frame rate, so no code change was made and the default stays off.

**Menu settings of the present and the pacing** (what each does on this path):
- *Frame Rate Cap* (`frame_pacing_mode`) and the speed limit: `PacePresent` (`functions_window.cpp`)
  on the guest's present, the rules in `core/hle/nvn/present_pacing.h` (`PlanPresentPacing`, the
  same as `Nx2Gsv::PaceToVblank`: the swap interval's period at the speed limit, never shorter than
  the cap's; Auto leaves it to `SpeedLimiter`). Replay frames skip the swapchain's renderer tick
  (`renderer_paced = false`), which is where the emulated GPU applies the cap. *Speed Limit %*
  (also its Turbo and Slow modes, `Settings::SpeedLimit()`): the single-core `SpeedLimiter` as
  before, and in every mode the display's slot period is scaled by 100 / limit
  (`ScaledDisplayPeriodNs`, used where `QueuePresentTexture` frees the previous texture), so the
  game gets its next texture, and with it its next frame, at the limited rate (as the VI conductor
  scales its vsync event). Limit off or 0: the 60 Hz slots as before.
- *Sync to framerate of video playback* (`use_video_framerate`): while the guest has NVDEC open
  (`System::GetNVDECActive`) the present is paced at the swap interval's period at 100%, whatever
  the speed limit (turbo, slow motion or off) says, as the VI conductor does for its vsync event.
- *Frame queue limit* (`EDEN_NX2_MAX_FRAMES_IN_FLIGHT=<n>`, default 1; no menu setting):
  `WaitPresentDepth` (`functions_window.cpp`, `PresentDepthLimit` in `present_pacing.h`) returns a
  present to the game only once the present `n` frames before it completed on the GPU. The default
  keeps only the current present potentially unfinished, preserving one frame of CPU/GPU overlap
  like the console.
  `self` returns a present only once that present itself completed (no frame left in flight: no
  CPU/GPU overlap, every GPU result the game reads next frame is complete).
  `0` or `off` removes this extra limit; then only the window's textures bound the queue (OoT: 3,
  so up to 2 frames unfinished), and results the game reads back without a fence (OoT's
  auto-exposure sums) can be that many frames old.
- *Force maximum clocks* (`renderer_force_max_clock`): `RendererVulkan` creates `TurboMode`; the
  scheduler's submit hook never fires for the replay's work, so the context's
  `ContextOptions::on_queue_submit` calls `TurboMode::QueueSubmitted` after every replay queue
  submission (TurboMode keeps its compute loop going for 100 ms after the last one).
- *Enable asynchronous presentation*: above; `WantsPresentThread` keeps it off while DLSS frame
  generation is active.
- *Window Adapting Filter*, sharpening, anti-aliasing, post-processing: the presentation passes
  above (the blit only for Nearest / Bilinear without the others); FSR / SGSR read the sharpening
  slider in their own passes.
- *Use disk shader cache*, *Use Vulkan pipeline cache*: see the pipeline cache section (off: no
  directory / no `nvn_pipelines.vkcache`).

**Acquire pacing of replay frames** (`Frame::renderer_paced`, `Swapchain::AcquireNextImage`,
`present/present_semaphore_ticks.h`). The wait that made up the `present_handover` / `present_blit`
time above (6-11.5 ms per present in town, perf HUD name `Vulkan::MasterSemaphore::Wait [gpu_fence:
host GPU: timeline semaphore tick] <- Scheduler::Wait <- Swapchain::AcquireNextImage`, once per present)
was the renderer scheduler's pacing tick of the swapchain image: it applies to frames the *renderer's*
scheduler rendered, but the replay's work never goes through it, so the image's tick is always "now" and
the wait flushed an empty submission and drained the whole GPU queue, the replay's own submission
included, before every present (the replay could not record frame N+1 while the GPU ran N). Frames
submitted through `PresentSubmitted` are `renderer_paced = false` and skip it; the frame rate is paced
by the guest present (`PacePresent`, `functions_window.cpp`) and the frame's own fences. What the tick
also protected, the acquire semaphore (it must have no wait pending when it is passed to the next
acquire), is guarded by a present timeline semaphore (`Swapchain::PresentTimeline`) that the blit
submission signals; `PresentSemaphoreTicks` keeps its value per acquire semaphore and the acquire waits
for it. Without timeline-semaphore support the old tick wait stays. Measured (town 110 s, user save,
70-108 s, `EDEN_NX2_FRAME_STATS`): `present_handover` 6.0-11.5 ms -> 0.2 ms, replay-thread wait 11.3 ->
0.2 ms per frame (perf HUD), the replay is no longer serial with the GPU (its own ceiling, estimated from
`replay_exec` ~9 ms plus the rest of its chain: ~110 instead of ~55-65 presents/s). The town frame itself
is limited by the host GPU (HUD `host_gpu` 18-21 ms busy of 23-24 ms
at full SM clock, 88-92% GPU utilisation), so FPS is unchanged there (42-44, run to run noise 1-3 FPS).

## Render conditions, counters and reports (`nvn_counters.{h,cpp}`, NvnIndirect)

Counter data and predicates live in one device-local scratch buffer (`CounterEngine`), so every
dependency is ordered on the GPU: a report written by an earlier submission is visible to a later
condition without the CPU seeing the completion.

- `NVNcounterData` is `{ u64 counter; u64 timestamp; }`, 16 bytes. The recorder does not write
  report data at completion; the replay does.
- `ReportCounter(TIMESTAMP | TIMESTAMP_TOP)`: counter field 0 (`vkCmdFillBuffer`); the time field
  (+8) depends on `EDEN_NX2_NVN_TIMESTAMPS` (next section). In `real` mode (the default) it is the GPU
  time of a `vkCmdWriteTimestamp` converted to the guest clock (the clock
  `nvnDeviceGetCurrentTimestampInNanoseconds` returns), stored by the completion thread; the smallest
  `guest now - GPU time` seen over the last 4 to 8 seconds (`GpuClockMap`) is the GPU-to-guest
  offset, so reports are never in the future and the mapping follows the GPU clock's drift against
  the guest clock (20 us/s measured; an all-time minimum, `EDEN_NX2_NVN_TS_OFFSET=global`, lets the
  reports fall behind by that much, hiding it from DK's frame-rate controller after minutes). In
  `legacy` mode it is the GPU emulation path's value.
- `ReportCounter(SAMPLES_PASSED)`: cumulative since the queue's last `ResetCounter`, whatever
  submission the draws were in. Draws are counted from a counter op on unless the rest of the
  submission shows the next `ResetCounter` before any SAMPLES_PASSED report (those samples count for
  nobody); when the submission ends first, they are counted (a later submission may report them).
  Draws only: clears are not counted, the masked clears drawn as triangles included
  (`HostPauseCounting`).
  - **In-pass queries** (the default, `nvn_in_pass_queries` setting, needs `hostQueryReset`; the
    first OoT HLE run with them lost the Vulkan device: OoT issues ~2200 resets in a row in one frame
    outside any pass, which overflowed the 256-event batch, fixed by the room check below. On by
    default since OoT's Kokiri Forest measured 7.7 -> 9.9 FPS, host GPU 130 -> 100 ms/frame, 2,238
    -> 1,172 passes/frame, same picture (`F:/eg/results/oothleperf/out2`); DK unchanged, HleAB):
    `HostBeforeDraw` begins a query inside the open pass before a counted draw; a report or reset
    ends it (a reset discards it), and so does the end of the pass (`HostPassEnding`). Neither
    report nor reset ends the pass: they go into the recorder's `CounterBatch` (query / reset /
    report events in guest order). When the pass ends (`HostPassEnded`) and the batch holds a
    report, `CounterEngine::ResolveBatch` copies the batch's query results into a scratch area,
    `nvn_counter.comp` walks the events from the queue's 64-bit accumulator (add, zero, store the
    running total per report; one dispatch per 256 events, 2 bits each in the push constants, each
    continuing from the accumulator the one before stored) and stores the accumulator back, and
    the report values are copied to their addresses. The areas come from a 2 MiB byte ring
    (`AreaRing`, sized to the batch, reused oldest first, waiting for an earlier submission that
    may still copy out of the reused bytes). A batch without a report (queries of a pass that
    ended for another reason) waits for the next pass, at most until the command buffer ends. What
    needs a report before its pass ends is a draw of the same pass that writes the report's bytes (a bound
    storage range, or any pool through raw global stores), and a draw that reads them (a bound
    buffer, vertex/index/indirect range, or any pool through raw global memory) once a game
    `Barrier` with an ordering bit (0xF) came after the report (`pass_reports_ordered`): it ends the
    pass first (`render_pass_break: draw reads a counter report of its pass`). A read before such
    a barrier is not ordered after the report on the console either (the report is a semaphore
    release behind the earlier draws that later work does not wait for; Mesa NVK waits for idle
    after its report writes, `nvk_query_pool.c`, and NVN's ordering point is the barrier), so it
    reads the bytes from before the report, as the emulated path does (its query cache writes the
    value when the fence completes). OoT's main view (a report after nearly every draw, raw
    global draws in between) ended ~80 passes per gameplay frame this way. Everything else that can
    read report memory runs outside a pass, after the pass end that wrote it: render conditions
    (their predicate is computed outside a pass), copies, dispatches, the mirror write-back and
    the CPU (after the submission). A batch is resolved early when it holds half its 4,096 events
    (a pass end) or cannot take another report or reset, inside a pass (ends the pass) or outside
    one; a reset right after a reset is no event. With 256 events OoT's main view (a query, report
    and reset around ~700 draws) ended its pass ~8 times for a full batch (`render_pass_break:
    ResetCounter`, ~15 per gameplay frame).
  - **Queries across passes** (`nvn_in_pass_queries` off, or no `hostQueryReset`): a query runs
    outside render passes (it spans whole passes) and starts lazily before the first draw after a
    counter op
    (`Recorder::BeginRenderCondition`); `EndDrawPass` + fold (copy the query result,
    `nvn_counter.comp` adds it to the queue's accumulator) happen at every report and at every
    command buffer end. The report copies the accumulator into the report address.
- Other counter types are reported as 0 (`counter_type_unsupported` in the unsupported TSV).
- `ReportValue`: 4 bytes by `vkCmdFillBuffer`, and (recorder) at completion.
- `SetRenderEnable(false)` skips draws and clears (`BeginRenderCondition` returns false).
  `SetRenderEnableConditional(mode, address)` compares the 64-bit counter fields of the reports at
  `address` and `address + 16` (RENDER_IF_EQUAL / RENDER_IF_NOT_EQUAL). With
  `VK_EXT_conditional_rendering` both counters are copied into a scratch slot, `nvn_counter.comp`
  stores the predicate, and every gated draw/clear is bracketed by
  `vkCmdBegin/EndConditionalRenderingEXT` (valid inside a render pass). Without the extension the
  work recorded so far is submitted, the two counters are read back on the CPU and the condition
  becomes enabled/disabled for the following ops (`render_conditions_cpu`).
- **Predicates computed ahead** (GPU conditions, `Recorder::PrefetchConditions`,
  `nvn_condition_prefetch.h`). Computing a predicate ends the render pass (copies and a dispatch
  outside it, with their barriers); OoT sets ~1,200 conditions per frame, mostly inside passes, on
  report pairs written thousands of ops earlier, so one pass end per condition cost more GPU time
  than its draws (976 of 1,160 pass breaks per gameplay frame). A condition that has to compute
  its predicate therefore also computes those of the conditions after it in the command list (up
  to 256, within 4,096 records and before the next `WaitSync`/`WaitEvent`/`End`), in one copy set
  and one dispatch, each into its own scratch slot; a later condition with the same address, mode
  and report pair uses that predicate without ending the pass (`render_conditions_prefetched`).
  Exactness: the predicate is the one the condition would compute at its own point only if no
  write reached either counter in between, so the lookahead skips pairs a `ReportCounter` /
  `ReportValue` ahead writes, and every write recorded later drops the predicates whose counters
  it overlaps (`NotePoolWrite`: reports, copies, fills, storage stores; stores into a shadowed
  range; any raw global store, a wait for another queue and the command buffer end drop them
  all). A pending `UpdateUniformBuffer` on the pair (it lands at the pass end), a scratch slot
  the ring handed out again, or a pair in a mirrored pool (CPU writes reach the mirror at the next
  upload) make the condition compute at its own point as before. `EDEN_NX2_NVN_CONDITION_PREFETCH=0`
  switches the batches off (every condition computes its own predicate).
- Only draws, `DrawTexture` and `ClearColor`/`ClearDepthStencil` are gated (as nxnative); copies,
  `ClearTexture` and `ClearBuffer` are not.

### Timestamp modes and DK's dynamic resolution

`EDEN_NX2_NVN_TIMESTAMPS=real|legacy` defaults to `real`. A report's time field is what the title's
GPU writes: 614.4 MHz ticks (divided by `fast_gpu_time`) for a Switch 1 title, whose
`nvnDeviceGetTimestampInNanoseconds` converts them with `GpuTicksToNanoseconds`, and nanoseconds for a
Switch 2 title, whose library returns the field unchanged (chip multiplier and divisor 1).
`real` reports hold the GPU time of the report's `vkCmdWriteTimestamp`, mapped onto the guest clock
(`CoreTiming::GetGlobalTimeNs`, also the answer of `nvnDeviceGetCurrentTimestampInNanoseconds`), so
the difference of two reports is the GPU time between them and a report is comparable with the
title's own tick counter. `legacy` writes `GPU::GetTicks()` when the flight completes, the GPU
emulation path's clock: every report of a submission is identical, so two reports of one frame
differ by 0. The mode affects only TIMESTAMP/TIMESTAMP_TOP, not occlusion counters or render
conditions.

DK consumes both uses (static decompile, 1.0.0 | 3.2.0): its frame-rate controller adds
(end-of-frame report - frame start) to the frame cost (`setGpuTimeStamp_` 0x1ea88c | 0x2045bc), and
its dynamic resolution reads the difference of two reports of the frame CB (agl::perf GPU-section
meter 0x793c0 | 0x79f0c: microseconds and percent of a 16.667 ms frame, smoothed over 3 and 5 frames)
in the DRS controller (0x72654 | 0x87c8c), which integrates the headroom into a render scale
(0x8e084). With `legacy` the load reads 0, so the scale stays at its maximum whatever the host GPU
does (the town's native 2880x1620 target); with `real` the scale follows the host GPU time as on
the console.

## Counters and logs

`ContextStats` (`pools_imported/mirrored/device_local/unsupported`, `host_import_failures`,
`textures_created/failed/transcode_unsupported`, `alias_unsupported`, `ops_unsupported/failed`,
`submissions`, `presents`, `held_back_submissions`, `parked_timeouts`, `event_signals`,
`event_waits`, `event_waits_blocked`, `waits_unsatisfied`). Everything unsupported is logged once
with a `[nvn]` prefix, never silently dropped. `waits_unsatisfied` is also a `CollectStats` row
(reason exactly `waits_unsatisfied`; the heartbeat and the unsupported table read it); the other
event counters are not problems, so they are only in `ContextStats` and `DescribeState`
(`events(signals/waits/blocked/released)`, `streams=<items waiting>`).

## GPU profiler (`nvn_gpu_profile.{h,cpp}`, tests `src/tests/nvn/gpu_profile.cpp`)

`EDEN_NX2_NVN_GPU_PROFILE=1` (`2`: also one interval per draw) measures GPU time with timestamp
queries. Unset, `Context::Impl` has no `Profiler`, no `Flight` and no query pool; the cost is one
null-pointer test per op (`InFlight::gpu_prof`, or `Impl::gpu_profiler` in the draw path). A device
without `hostQueryReset` or timestamps logs an error and stays off.

**Stamp semantics (GpuTruth, 2026-11-05).** A bare `vkCmdWriteTimestamp(BOTTOM_OF_PIPE)` does not
wait for compute shaders on NVIDIA (RTX 5090, driver 621.01; a standalone probe, host fence time as
truth): 8 dependent dispatches of 8.1 ms were stamped 0.99 ms, a 1-group
dispatch of 1.05 ms 1 us; `vkCmdWriteTimestamp2(ALL_COMMANDS)` and an execution-only barrier before
the stamp behave the same. The unfinished work then lands on whatever really waits next (the next
barrier, op or command buffer), which is how dispatches read <= 3 ms/frame and `Barrier` 17 ms on
OoT Kokiri while removing the dispatch workgroups (COSTSPLIT `nogroups`) removed ~54 ms of q1 busy.
So `Flight::StampSegment` records a memory barrier (ALL_COMMANDS -> ALL_COMMANDS, MEMORY_WRITE ->
MEMORY_READ|WRITE) before every timestamp outside a render pass: the stamp then follows the
completion of everything before it and an interval is the time of the work inside it. Only the
per-draw stamps of mode 2 and the `ops` stamps of records inside an open pass go without it (a
barrier is invalid there): those are issue-ordered, and the pass's time lands on the interval that
ends it. All of this exists only while `EDEN_NX2_NVN_GPU_PROFILE` (or the perf HUD's profiler) is
on. Test: `nvn gpu profile: a dispatch's stamped time scales with its work` (busy-loop compute shader,
1x and 4x iterations, 1 group and a dependent chain).

`EDEN_NX2_NVN_GPU_PROFILE=frame` (internally mode 3) is the mode that can judge synchronisation
changes: the other modes drain the GPU (barrier + stamp) at every pass/dispatch/copy boundary and
so remove exactly the overlap a barrier change creates. `frame` stamps only the first and last
command of each command buffer (`Flight::StampSegment`; `Flight::Stamp` is a no-op, so every
`Scope` and the pass/draw hooks record nothing) and logs, per 10 s window and for the run, `gpu ms
per frame over N frames (frame-only timer): busy (sum of command buffers) [avg p50 p95 p99 max];
span (first stamp to last, gaps included) [...]; idle between command buffers`. The two drains sit
at the command buffer's ends, so the work inside keeps its overlap. A frame is a present and the
command buffers since the previous one (`Window::frame_busy_ms`, `frame_span_ms`). Use `busy` to
compare GPU cost, `span` to see CPU starvation. The other modes print the same line from their
segment stamps (they include their own drains).

`EDEN_NX2_NVN_GPU_PROFILE=ops` (internally mode 4) names the `unattributed` time: mode 1 plus one
*op* interval per replayed record (`Flight::MarkOp` at the start of each record in
`Recorder::RecordList`, `Flight::EndOps` before the command buffer's last stamp), from the record's
start to the next record's start, so the pass end, barriers, counter work and uploads a record
causes are its own (`Tag::hash_a` = the `VideoCommon::Nvn::Op` number). Op intervals overlap the
other kinds: they are not summed into `in intervals` and are listed apart as `ops top N` (one row
per op number). It drains at every record outside passes (inside an open pass the stamps are
issue-ordered, see above): read relative numbers.

- **Recording (replay thread).** Every submission (and present) gets a `Flight`: a `VkQueryPool` of
  16384 timestamps (262144 in mode `ops`, one per replayed record of a ~10k-record OoT gameplay
  command buffer plus the scopes) taken from a free list (pools are reset by the completion thread with
  `vkResetQueryPool` and returned; pools of flights that never completed are parked, never reused).
  `Flight::Stamp` records a memory barrier and `vkCmdWriteTimestamp(BOTTOM_OF_PIPE)` (see "Stamp
  semantics"), so every boundary drains the pipeline.
  An interval is two query indices plus a `Tag`. Intervals: *segment* (one command buffer, first to
  last command), *pass* (just before `vkCmdBeginRenderPass` to just after `vkCmdEndRenderPass`),
  *dispatch*, *copy* (buffer/texture copies, feedback copies of sampled render targets, inline
  uniform updates), *clear* (`ClearTexture`, `ClearBuffer`, masked-clear passes), *present* (the
  present blit) and *upload* (guest bytes into images/mirrors outside a copy op). `GpuProf::Scope`
  wraps an op; nested scopes belong to the outermost one. Mode 2 adds *draw* intervals inside
  passes: a timestamp after every draw (taken when the next draw is prepared or the pass ends), so
  each draw's time is its own (the draws serialize, which costs time: read relative numbers).
- **Tag** (aggregation key): kind; size-0 flag (`DrawContext::prof_zero_size`: the descriptor build
  of any draw/dispatch inside bound a storage entry with an address and size 0 as the rest of its
  pool); passes: draws, vertex and fragment program hash of the *first* draw
  (`HashProgramStage`), colour format + size of the first colour target (depth format when none),
  `colours | depth<<4 | layers<<8 | log2(samples)<<24`; dispatches: compute program hash and
  groups; copies/clears/uploads: texture format and size or byte count, `hash_a` = what it was
  (feedback copy, inline update, mirror upload, image upload, storage shadow, vertex stage).
- **Completion thread (`NvnComplete`).** After the timeline value was reached `Profiler::Complete`
  reads the queries of the flight (value + availability, one call), converts with
  `timestampPeriod` and aggregates per tag; work that never ran (unavailable) is counted and
  skipped. The replay thread never waits for it.
- **Report.** Every 10 s (and once for the whole run at shutdown), `[nvn] gpuprof ...` lines in the
  log, all per present of the window:
  1. `gpu_busy=<ms>/present (<% of wall>; frame time <ms>)`: sum of the segment intervals;
  2. `in intervals`: sum per kind, `unattributed` (segment time outside every interval: barriers,
     layout transitions), size-0 intervals vs the others (passes, dispatches, copies, clears,
     uploads; draw intervals are not counted twice), `dropped` (queries over capacity),
     `unavailable`;
  3. `gpu idle between command buffers`: gaps of an empty queue (next command buffer's first
     timestamp minus the previous one's last), by size, and the part before the first command
     buffer after a present;
  4. `gpu busy % per second`; `depth-only passes`: passes with a depth target and no colour target
     (shadow maps): ms, passes, draws and vertices x instances per present, us per draw;
  5. `tags top 1..25`: ms per present, intervals per present, mean, tag; `draws top 1..25` (mode 2);
  6. `by program 1..12`: the same summed over shape and count per (kind, vs/cs, fs).
  The heartbeat (`hle_runtime.cpp`) adds `[nvn] gpuprof guest waits since last status:` (blocking
  waits per NVN function, count and ms) when the switch is set.
- **Overhead.** The drain at every boundary removes the overlap between consecutive passes and
  adds a few µs per interval: the profiled FPS is lower than the plain one (town: see below). Read
  the ms/present relatively, not as the unprofiled GPU time.
- `EDEN_NX2_NVN_NO_ZERO_SIZE=1` (experiment) binds the dummy buffer for size-0 storage entries, the
  behaviour before they bound the rest of their pool.
- `EDEN_NX2_NVN_NO_HOST_IMPORT=1` (experiment) gives CPU-accessible pools device-local mirrors.

### Town measurements (RTX 5090, `key_script_run_jump`, 110 s runs, `nvn/main` 17278b264c)

- **The frame is serial, not GPU-bound.** Per present: GPU busy 22.7 ms (61% of the 37 ms frame),
  GPU idle 14.6 ms, of which one gap of 13-14 ms before the big submission (117 passes, 8 dispatches,
  57 copies). Each frame has 6 submissions: 2 empty, 3 tiny size-0 dispatches, the big one, an empty
  fence one, the present. The guest blocks once per present in `nvnSyncWait` (26-29 ms) on the fence
  of the *previous* frame's big submission and only then records the next one (guest recording plus
  the replay's Vulkan recording, ~13 ms, GPU idle). Frame time = GPU 23 ms + CPU 13-14 ms.
- **Size-0 storage binding.** Intervals flagged size-0 cost 0.005 ms/present. With
  `EDEN_NX2_NVN_NO_ZERO_SIZE=1`: 36.3 vs 26.4 FPS (3 alternating runs each, profiler off), GPU busy
  17.1 vs 22.7 ms, the idle gap 9.3 vs 13.9 ms: the compute output is real and the game then draws
  more (shadow passes 2460 vs 1590 draws), i.e. more GPU and more CPU recording, not a binding cost.
- **Biggest GPU item: written storage in imported (host) pools.** pass vs=c470a15b/fs=3fc8f02c (R32F
  720x405 + depth, one draw) costs 4.5 ms per frame (20% of GPU), 22-25 ms at 1440x810 in the early
  game: 15-19 ns per fragment, linear in pixels. Its fragment shader does one `OpAtomicUMax` per
  fragment into a storage buffer that lives in an imported pool (backing 1, pool flags
  CPU_CACHED|GPU_CACHED): PCIe atomics. Binding a device-local scratch instead of written storage
  ranges <= 1 MiB (timing experiment, contents not kept): 31.9 vs 25.1 FPS, GPU busy 14.4 vs 24.1 ms.
  Mirroring every CPU pool (`NO_HOST_IMPORT`) is unusable: 1-2 FPS. The coherent fix is the storage
  shadows below (on by default).
- Profiler overhead: FPS 26.7 profiled vs 26.4 plain.

## Cost split (`nvn_costsplit.{h,cpp}`, tests `src/tests/nvn/costsplit.cpp`; measurement only)

`EDEN_NX2_NVN_COSTSPLIT=<class>:<start>[:<modes>][:<period=600>]` says WHY a class of work costs
what the GPU profiler measures, by taking one part of it away at a time. **The picture is wrong on
purpose while it runs.** Generalises the HleShadowDraw diagnostic (`diag/hle-shadow-draw`, which
found DK's depth passes paying ~6 of 7.4 ms for vertex fetch from imported pools).

- **Class**: `depth` (draws into a depth target, no colour target: shadow maps), `color` (>= 1 colour
  target), `compute` (dispatches), `prog:<hex hash>` (draws/dispatches whose programs include the hash
  the profiler prints), `all`.
- **Draw modes**: `base`; `noinst` (zero instances: our recording + the driver); `noraster` (empty
  scissors: vertex work and setup without fragments); `vram` (uniform / vertex / index / read-only
  storage ranges of imported pools copied into a 1 GiB host-visible, device-local-when-possible
  ring, ranges <= 4 MiB, not fenced: memory placement); `tri3` (<= 3 vertices/indices: fixed cost
  per draw); `nobind` (after a pass's first class draw, the same layout's descriptors and the bound
  vertex/index buffers are kept: binding cost); `vramu` / `vramv` / `vrams` (by name only, never in the default
  lists): `vram` for one kind of data, uniform / vertex+index / read-only storage (the placement split per kind).
  Indirect and mesh draws keep their counts
  (`unclamped` in the tally). **Compute modes**: `base`, `nogroups`, `onegroup` (direct and
  indirect), `vram`. `prog`/`all` run each mode on the kind it applies to, the other kind as base.
- **Schedule**: from present <start>, slots of <period> presents: base, mode 1, base, mode 2, ...
  (base between every mode, so scene drift shows as the spread of the base slots). The first 60
  presents of each slot (half of a shorter one) are warm-up and not summed. `[nvn] costsplit
  schedule ...` marks each slot.
- **Numbers**: the GPU profiler is forced to mode 1. Every submission carries the slot it was recorded
  in (`Flight::costsplit_slot`); passes and dispatches holding class work carry the count
  (`Interval::split`). The completion thread adds GPU busy and class time per slot; the replay thread
  adds class draws/dispatches, its time preparing them (`cpu_ms`, steady clock around PrepareDraw /
  the dispatch) and where each bound range lives (none/imported/mirror/device/virtual/arena).
- **Lines** (key=value): per finished slot (logged once a submission past the next slot's warm-up
  completed), each time followed by the totals per mode so far (`scope=run`, cumulative: the last
  ones count, so a killed run keeps them),
  `[nvn] costsplit class=.. mode=.. scope=window slot=K start=P|scope=run frames=.. busy_ms=..
  class_ms=.. passes=.. draws=.. dispatches=.. us_per_draw=..(compute: us_per_dispatch) cpu_ms=..`
  (all per frame) and `[nvn] costsplit tally ... ubo=a/b/c/d/e/f vtx=.. idx=.. ssbo=.. bytes=avg
  ubo/vtx/idx/ssbo copied=.. ring_mib=.. ring=device|host unclamped=..` (counts over the scope).
  `rundiag.py costsplit <run>` (MCP `cost_split`) tabulates them: overhead = noinst, raster =
  base - noraster, placement = base - vram, fixed per draw = tri3, binding = base - nobind.
- **DK city, depth class** (3.2.0 HLE, user settings, RTX 5090, `next` a5dfaa9c39 without vertex
  staging; F:/eg/results/costsplit/out-depth): base slots 6.99-7.48 ms of a 25.2 ms GPU frame (3840
  draws, 1.9 us/draw); `vram` 0.11 ms (placement ~7.1 ms: every uniform, vertex and index range of
  these draws is in imported host memory, 6.1 + 3.9 + 1.1 KB per draw; HleShadowDraw's vertex-only
  copy left 1.2 ms); `noinst` 0.05; `noraster` 7.16 (fragments ~0); `tri3` 1.15 (0.30 us/draw, itself
  mostly host fetches); `nobind` 12.2 (draws read the first draw's vertex buffers: other geometry, so
  read only its `cpu_ms`, 1.37 vs 1.64 ms).
- **OoT river, colour class** (hub-aa52838aeb, user settings, `color:1500:base,noinst,noraster,vram,tri3:300`, run
  `F:/eg/results/HlePasses/run-cs`): base 9.40 ms of a 25.6 ms GPU frame (1311 draws, 7.2 us/draw), `vram` 3.18 (placement
  6.2 ms: 5.7 uniform, 1 vertex, 0.94 index and 2.4 storage bindings per draw, all imported), `noinst` 3.11, `noraster`
  4.30 (fragments 5.1), `tri3` 5.41 (3.3 us/draw).
- **Off**: `Context::Impl::costsplit` is null; the draw path tests the cached pointer once per draw
  and dispatch, and `cs_mode` (always Count) at the hook sites.

## Storage shadows (`nvn_draw_shadow.cpp`, tests `compute_vulkan.cpp` "NVN storage shadow")

Why: an imported pool is the guest's own memory mapped into the GPU, so shader stores and atomics into
it cross PCIe (15-19 ns per atomic fragment: DK's shadow pass spent 4.5 ms per frame there). A
**written** storage range (`DescriptorBinding::is_written`) of at most 1 MiB
(`EDEN_NX2_NVN_SHADOW_MAX_KIB`) in an imported pool is bound through a device-local **shadow** instead.
`EDEN_NX2_NVN_SHADOW_STORAGE=0` switches it off (host binding as before),
`EDEN_NX2_NVN_SHADOW_LOG=1` logs the first shadowed ranges (pool address/size/flags, range, programs).

Life cycle (per `DrawContext`, i.e. per NVN queue; `Shadow{pool, begin, end, buffer, offset}`):

1. **Create** (`PrepareOutsidePass` -> `ApplyShadows`, outside a pass): a candidate range is copied
   host -> shadow (`vkCmdCopyBuffer`, after the pending barrier) and bound through the shadow. The shadow
   memory comes from `ShadowChunk`s (8 MiB device-local buffers, `InFlight::shadow`, returned to a free
   list of 4 by the completion thread; one chunk serves a whole command buffer, offsets keep the pool
   offset's alignment modulo 256 so uniform/storage offset limits hold).
2. **Live**: while it exists the shadow is the truth for `[begin, end)` and the host bytes are stale.
   Every use of the plan (`DrawPlan::ranges`: descriptor bindings, vertex/index buffers, and the indirect
   argument/count ranges `NoteExtraRead` adds) is classified (`ClassifyShadows`): a *descriptor binding
   inside the shadow* (storage or uniform) is served from it (no copies at all for the next pass or
   dispatch that binds the same range); any other use that overlaps it (partial overlap, vertex, index,
   indirect, a draw with raw global memory) ends the shadow first. Ops outside the draw path
   (`Recorder::ResolvePoolRange`: copies, clears, reports, conditions) and `DrawContext::ResolveBuffer`
   (inline updates, indirect arguments) do the same (`FlushShadowsOverlapping`; the open pass ends
   first, `BreakCause::ShadowAlias`).
3. **End** (`FlushShadow`): shadow -> host copy, `NotePoolWrite` (host barrier at segment end). On a
   guest CPU write while a shadow is live, `NotifyCpuWriteNow` ends the open pass and refreshes only
   the overlapping bytes in the shadow before its eventual copy-back. Mirror entries refresh both
   their mirror result and merge snapshot, preserving the CPU write while unrelated GPU changes
   remain eligible for write-back. Shadows still end no later than `Recorder::EndSegment`.

**Barriers** (`ShadowCopyBatch`): the copies of one loop over shadows (the creations of one plan, the
write-backs of one conflict, of one `FlushShadowsOverlapping` or of `FlushShadows`) share one memory
barrier before them and one `HostMarkWritten` after them. Live shadows cover disjoint pool ranges and
each has its own chunk slot, so no copy of a loop reads or writes bytes another copy of it writes; the
first copy's barrier orders them all after the dispatches and passes before, and the mark orders the
next use after them. A conflict write-back and the creation that follows it in the same plan are two
batches (the new shadow may cover bytes the write-back wrote), so a barrier stays between them.
OoT Kokiri (VTX-HLE, `oothleperf2/out4`): 52 shadows created and 52 written back per present
(44 at segment ends, 7 for raw-global draws), 1.25 MiB each way per present (~24 KiB per shadow).

Kept in host memory (counted in `ContextStats::shadow_skipped_*`, shown in the heartbeat): ranges over the
limit, size-0 "rest of the pool" entries, draws whose shaders use raw global memory, a candidate that
another use of the same draw overlaps, a candidate that first appears in an open pass (a pass break to
copy it in would cost more than it saves), no device memory.

Coherence: refresh copies are ordered after prior GPU use and before subsequent shader use or
write-back. They touch only the notified range (rounded to complete words for `vkCmdCopyBuffer`),
so unrelated GPU results in the shadow remain intact. CPU-side reads of guest bytes during recording
(texture uploads of overlapping storages, private uniform copies) still see the host bytes of the last
completed copy-back: the same staleness an unfinished earlier submission already caused (review L6).

## Write-back of GPU results (`nvn_write_back.{h,cpp}`, `host_shaders/nvn_wb_merge.comp`, tests `src/tests/nvn/write_back.cpp`)

Where the GPU's writes reach guest memory today (every site that copies GPU results back):

| Backing / object | GPU writes land | Copy-back | When the guest sees them |
|---|---|---|---|
| Imported pool | in guest memory itself (zero copy) | none | host barrier at the end of the segment (`NotePoolWrite`, raw global stores `HostNoteRawGlobalWrite`) |
| Storage shadow of an imported pool | device-local shadow | `FlushShadow`, at a conflicting use and at `EndSegment` | segment end |
| Mirror pool (import impossible) | device-local mirror | `NotePoolWrite` -> `readbacks` -> `WriteReadbacks` (download buffer) -> `CompleteMain` memcpy / `GuestMemory::Write` of the whole range | before the completion callback |
| CPU_NO_ACCESS pool | device-local buffer | none (the CPU cannot read it) | never |
| Texture written by the GPU | image | `FlushTextureToGuest` on demand (ReadTexels, `CmdFlushTexture`) | when the guest reads texels |
| Event / report words | counter scratch, recorder | report copies into the pool (`NotePoolWrite`); event words written by the recorder at completion | completion |

A whole-range copy also writes every word the GPU left alone with the value the copy started from:
a CPU write made to the range while the GPU ran would be lost. `NotifyCpuWriteNow` now refreshes
overlapping live shadow bytes before copy-back. For mirror entries it refreshes both result and
snapshot so the word-diff merge does not mistake the CPU write for a GPU change; GPU-changed words
outside the CPU-written range still merge. **Word-diff merge**: `MergeChangedWords(guest, snapshot,
result)` (CPU) and `WriteBackMerge::Record` (GPU, one dispatch per range and 64 MiB, push descriptors)
write only the 4-byte words (natural guest alignment) whose result differs from the snapshot the GPU
work started from; a partial first or last word changes only its changed bytes inside the range (CPU:
byte stores; GPU: `atomicAnd` / `atomicOr`). `MergeRange` offsets must be equal modulo 4 (snapshot
allocators keep the pool offset's phase); a range that is not is refused and logged. Idea and GPU form
after the TotK recompilation's write-back merge (vk_memory.c 1464-1640, MIT). Only changed words cross
PCIe. The pool mirror's completion readbacks also use `WriteBackMerge`.

## Pool mirrors (`nvn_pool_mirror.{h,cpp}`, `nvn_draw_shadow.cpp`, tests `pool_mirror.cpp`, `compute_vulkan.cpp` "NVN pool mirror")

`EDEN_NX2_NVN_POOL_MIRROR=0|1|2` (default 0; `ContextOptions::pool_mirror` overrides it): an imported pool
gets a persistent device-local copy of the whole pool (`Pool::vram_mirror`, same offsets) that storage
bindings, indirect arguments/counts and staged vertex/index ranges use instead of the guest memory over
PCIe. Mode 1 creates a mirror at the first **written** storage binding of the pool, mode 2 at any storage
binding. `EDEN_NX2_NVN_POOL_MIRROR_MAX_MIB` (4096) is the budget; a pool over it, a failed allocation, a
pool that is not page aligned or a guest memory that cannot track CPU writes keep the pool in guest memory
for good (`vram_mirror_refused`, never retried). Storage shadows (above) are the per-command-buffer form
of the same idea; with mirrors on, their entries become mirror entries.

`EDEN_NX2_TRY_STORAGE_VRAM=0` opts out of mode 1 for imported storage pools and raw-global table
aliases into the same mirror; it defaults on where buffer device addresses, 64-bit shader integers
and descriptor aliasing are supported. Since raw pointers can address any byte in such a pool,
raw-global dispatches copy in every stale page, and raw-writing dispatches snapshot every eligible
mirrored pool for the existing fence/segment write-back merge. Pools larger than
`EDEN_NX2_TRY_STORAGE_VRAM_MAX_MIB` (64 MiB) are not mirrored by this path; the ordinary global
mirror budget still applies.

A written range of a pool without a mirror stays in guest memory (`shadow_skipped_no_room`; with
`EDEN_NX2_NVN_SHADOW_LOG=1` the first ones are logged as `written storage range stays in host
memory`). OoT Kokiri has two such pools (305 and 88 MiB, imported) and ~62 of these bindings per present.
`EDEN_NX2_TRY_REFUSED_SHADOW` (on by default, `=0` opts out) binds those of at most
`EDEN_NX2_NVN_REFUSED_SHADOW_MAX_KIB` (default 4096) that are not in a raw-global plan through a copy shadow,
as with mirrors off (at most 32 live copy shadows). Measured on OoT Kokiri village (RTX 5080): the CMAA edge pass
`3816717874cb1f93` keeps its counter + edge list (1.76 MiB, over the mirrors-off 1 MiB `SHADOW_MAX_KIB`) in the
305 MiB pool; its per-warp atomics over PCIe cost ~10 ms a frame. In a copy shadow it costs ~1 ms: GPU 36 -> 24 ms,
23 -> 31-34 fps, picture unchanged.

**Validity** (`MirrorPages`: one bit per 4 KiB page). A page is valid when the mirror holds the guest
bytes: it was copied in and no other writer has changed the guest page since. Invalid at creation, and
made invalid by:
- guest CPU: the pool is registered with `GuestMemory::Track` (core `Memory::PageTracker`, the
  rasterizer's write trap) unarmed; a page copied in is armed (`Arm`), and the next use takes the pages
  the CPU wrote since (`TakeWritten`) and copies them again. One write fault per armed page per cycle;
- replay writes through the pool's own buffer (copies, fills, reports, storage stores that stayed in
  guest memory, inline updates): `Recorder::NotePoolWrite`.

Raw global stores use the raw table's actual backing: mirrored pools resolve to their VRAM mirror, so
those GPU writes keep the mirror current; non-mirrored pools have no mirror to invalidate.
`Recorder::Abort` invalidates all mirror pages because the commands that would have populated them did
not run.

**Use** (`DrawContext::ApplyMirrors`, outside a pass unless noted):
1. A written storage range becomes a mirror entry (`Shadow::mirror`, buffer = mirror, offset = pool
   offset): its stale pages are copied in (`RecordMirrorCopies`: one barrier, regions merged per pool),
   then a snapshot of the range goes to an `AllocShadow` chunk (offset modulo 256 kept, so the guest
   word phase holds for the merge). Inside an open pass a new written range stays in guest memory
   (`vram_bindings_host`) and `NotePoolWrite` makes its pages stale.
2. A read-only storage range is served from the mirror after the same copy-in; inside a pass only when
   the mirror is current there (`MirrorCurrent`: valid and not CPU-written) or a live entry holds it.
3. Indirect arguments and counts (`ResolveIndirectSource`): read from the mirror inside a live entry,
   or where it is current and no entry of this command buffer overlaps; else as before (shadows flushed,
   guest bytes).
4. Vertex/index staging of a pool with a mirror reads the mirror (copy-in + `TRANSFER -> VERTEX_INPUT`
   barrier) instead of a per-pass staging copy.
A copy-in never overwrites the bytes of a live entry (its GPU writes): those bytes are cut out of the
copy and the page stays stale, so the next use after the entry's merge copies it whole.

**End of an entry** (conflicting use, `MAX_LIVE_SHADOWS` = 128 reached, `EndSegment`): all mirror entries
go back together with `WriteBackMerge` (section above): only the 32-bit words that differ from the
snapshot reach the guest bytes, so CPU writes to other words of the range survive, and
`HostNoteMirrorWriteBack` notes the write without invalidating the mirror (guest == mirror there). A
device without the merge pipeline gets a plain copy (`writes plain` in the heartbeat). Entries never
outlive a command buffer, as shadows: a fence, a completion and `FlushAndWait` see the guest bytes.

**Write-back monitor** (`EDEN_NX2_NVN_WB_MONITOR=1`, diagnostic, off by default): per mirrored pool the
replay heartbeat shows `wbmon[addr+size entries=N(whole) entryMiB changedKiB dirtyKiB armed touched
copyinMiB]`, cumulative: the merged entries (whole-pool ones in parentheses), the KiB of words the
GPU merges changed (`nvn_wb_merge.comp` counts them per pool slot in `Impl::mirror_merge_stats`) and
the number of units (8 or 32 guest words) holding one, the pages armed with a CPU access trap after each partial-pool
merge (`GuestMemory::ArmAccessTrap`, entries up to 32 MiB) against the pages the guest then touched,
and the guest-written MiB copied in. It decides how much of the merge's PCIe traffic a lazy
write-back (results kept in the mirror until the guest touches the page) could save. A trapped
access costs microseconds: the monitor slows the guest.

**Merge line mode** (`EDEN_NX2_TRY_WB_LINES=8|32`, default 8, `=0` opts out; `WriteBackMerge::SetLineMode`,
`MirrorLineGuard`): OoT's merges
change ~2.5 MiB of words per frame in scattered runs and cost 4.2 ms of GPU time (Nsight, 182
dispatches): a store of an isolated word into host memory over PCIe costs 10-20 ns, a full 128-byte
line ~5 ns (`tests.exe "[.write-back-bench]"`: 0.8 MiB of random changed words 2.4 ms, 2 MiB in
half-filled groups 11.7 ms, 16 MiB contiguous 0.64 ms). In line mode (per merge range, `MergeRange::lines`)
the guest buffer is cut into
units of 8 words (32 bytes, a sector) or 32 words (a line), aligned as the host memory is, one per
8/32 invocations; a unit with one changed word stores it as before, a unit with several (all of it
inside the part, no partial word) is stored whole: the changed words from the result, the others as
the guest holds them an instant before. A CPU store to an unchanged word of such a unit that
lands between the shader's read of the unit and its store (about a microsecond) would be lost, so
line ranges are only recorded where no CPU store can come (`nvn_line_guard.h`):

- Every pool mirror has a `MirrorLineGuard`. When the replay thread records the merge of an entry
  (`WriteBackMirrorEntries`), the guard splits the range into runs. Pages the guest has not touched
  are *line* runs: an access trap (`GuestMemory::ArmAccessTrap`; armed once, it stays) sits on
  each, and the range counts as in flight until its token, kept in `InFlight::keepalive`, dies
  when the GPU is done with the submission. Touched ("hot") pages are *exact* runs (the word merge:
  only the words the GPU changed are stored, so a CPU store to any other word cannot be lost).
- The first guest access (a read too) to an armed page calls the guard back on the guest thread,
  before the access lands: it marks the pages hot, then waits (250 ms at most) until no line merge
  covering them is in flight. Merges recorded after the mark see the pages hot (one mutex), the
  ones before have stored everything: the guest's access lands outside any line store. The page
  stays hot (disarmed) for good.
- A pool whose guest touches more than 16 armed pages, or whose wait times out, merges exactly from
  then on and drops its traps. OoT touches one control page per pool every other frame: that page
  goes hot once and the rest of the pool keeps line mode.
- The replay thread never waits (it would wait for work it has not submitted); it never accesses
  guest memory through the tracked paths (the HLE uses the unsafe ones). Writes the HLE itself makes
  through host pointers and the GPU's own writes are ordered by the queue and not trapped, as for
  the exact merge.
- Counters in the replay heartbeat (`wbl[lineMiB exactMiB armed hot waits timeouts exactPools]`).
- Without access traps (`GuestMemory::SupportsAccessTraps`) no mirror has a guard: exact merge only.

**Guest flushes of mirrored pools** (on by default, `EDEN_NX2_TRY_FLUSH_TRACKED=0` opts out, `Impl::FlushedRuns`):
OoT calls `nvnMemoryPoolFlushMappedRange` on whole pools every other frame (the WB monitor's `notify(...)`
counters: 2.5 M pages notified in 1,550 calls on one 3 MiB pool, 17 of them written by the guest in all).
With the switch off every flush invalidates the range's mirror pages and the next use copies the whole range
in again (38 MiB per frame over PCIe, ~1.9 ms of GPU copy time). On, a guest flush (not a host service's
`NotifyHostWrite`, which still counts in full) acts only on the pages of the range the page tracker saw the
guest write since the mirror took them (`TakeWritten`, then invalidated) and on pages the mirror does not hold
(never copied in, so never armed in the tracker); pages valid in the mirror and clean were not written by the
guest CPU, whose stores all go through the tracker. Live-shadow refreshes (`OnCpuWrite`) use the same runs.
What that relies on: every writer of guest memory that is not a guest CPU store tells the tracker.
Audit (who writes pool memory, and how the mirror finds out):

| Writer | Seen by the tracker |
|---|---|
| Guest CPU stores (JIT, NCE write faults, exclusives, data cache ops) | yes, `Memory::Write*` / the write-protected page |
| Service buffers (`HLERequestContext::WriteBuffer*`, file reads, nvdrv ioctls, audio-out), `Memory::WriteBlock/CopyBlock/ZeroBlock` | yes, `Memory::WriteBlock` |
| `Memory::WriteBlockUnsafe` (audio renderer, audio device session) | yes now: it skips the device caches only, and marks tracked pages dirty |
| Kernel IPC copies through the heap pointers (`KPageTableBase::CopyMemoryFrom{Kernel,Heap}To...`, `WriteDebugMemory`) | yes now: `Memory::NotifyHostWrite` on the destination |
| HLE texel copies (`nvnTextureWriteTexels` storage, `nvnTextureReadTexels` caller buffer, packaged texture data) | yes now: `Recorder::NotifyHostWrite` (`Sink::NotifyHostWrite`, a full notification) |
| nx2 video decoder frames (`nx2_gsv.h`) | yes: `WriteBlock` and `NotifyHostWrite` |
| The replay's write-back of GPU results (`ProcessMemory::Write` = `Memory::WriteBlockUntracked`, counters, texel flush) | no, by design: the mirror that produced the bytes holds them; marking them would copy them back in |
| Kernel-owned memory the services fill through `KSharedMemory::GetPointer` (hid, irs, time, fonts), the VI shared buffer, zero fills of freshly allocated pages | not pool memory (the game cannot pass them to NVN); new mappings reset the tracker (`OnRemap`) |
| Legacy GPU (device memory manager downloads, Maxwell DMA, query cache) and nvdrv vic/nvdec | no; the legacy GPU does not run a game's GPFIFO under VTX-HLE |

Known gaps: a second virtual mapping of the same physical pages (shared or transfer memory mapped twice) written
through the other address is not seen by the tracker, which follows virtual pages; the HLE's DLSS network storage
zero fill is not announced (the replay never reads it).

**Invalidate** (`nvnMemoryPoolInvalidateMappedRange`, `nvnBufferInvalidateMappedRange`; candidate
`EDEN_NX2_TRY_INVALIDATE_WRITEBACK=<ms>`, off, 1 = 250 ms; `nvn_gpu_writes.h`, `Sink::GpuWriteMarker` /
`PollGpuWrites`, `functions_pool.cpp InvalidateMappedRange`): on the console an invalidate only drops stale
CPU cache lines; the GPU's writes are in memory as far as the GPU has run, which on the console is as the
game submits. OoT's auto-exposure controller reads D1's four light sums right after an invalidate with no
fence and then zeroes them (the next frame reads the new sums). Here the GPU's writes reach guest memory only
when the replay (which lags the guest by a frame or more) has recorded them and the GPU ran the write-back
(imported pools: the merge at the end of the command buffer; mirror pools: `CompleteMain`'s readback), so
the CPU reads its own zero. With the switch on, `Recorder::NotePoolWrite` notes every pool write with its
flight's seq in a `GpuWriteLog` (guest addresses), and the invalidate (guest thread): reads `LastSeq`, takes
the marker (commands enqueued so far), polls (1 ms slices of `Call::WaitUntil`) until the replay dispatched
them, takes the newest unfinished submission <= `LastSeq` that wrote the range, and waits until
`CompletedSeq` reaches it, bounded by the limit (one warning, the read goes on). Submissions held back or
suspended at an event wait have no noted writes (the GPU would wait too); writes of submissions made after the
call are never waited for. It adds no new synchronization to the game's own: a game that fences before the read
is unaffected. `EDEN_NX2_DIAG_INVALIDATE=1` logs every flush/invalidate of up to 4 KiB with the raw words at
the range start (`[nvn] mapped-range ...`). Not covered: raw global stores to non-mirrored imported pools
(no range is known).

**Raw-global draws and live entries** (on by default, `EDEN_NX2_TRY_RAW_KEEP_ENTRIES=0` opts out): without it a
draw with raw global memory ends every live entry (`ClassifyShadows`: merge back, then the raw-writing draw snapshots
every storage-mirrored pool whole again) - OoT: 55 raw draws, 156 entries created and ~100 merges per present, 264 MiB
of snapshot copies (heartbeat `shadows(...)`, tags `copy storage-shadow`). The raw table resolves every storage-mirrored
pool to its mirror, which is also the entries' buffer, so raw loads and stores already meet the entries' bytes, and the
word-diff merge counts a raw store as a GPU change like any other. By default the raw draw ends only entries that
are not mirror entries; a raw-writing draw adds a whole-pool entry only where the pool has none. Entries end at a
conflicting use, at `MAX_LIVE_SHADOWS` (a partial entry goes back first) and at `EndSegment`. Test: `compute_vulkan.cpp`
"raw-global draws keep the mirror entries live".
**Limits (known, not handled):**
- *Cross-queue copy-in.* Entries are per `DrawContext` (per NVN queue), the mirror is per pool. A copy-in
  recorded for queue B only cuts out B's live entries. If queue A has a live entry in a page and B, not
  synchronised with A, copies in a stale page that A's entry partly covers, B's copy can overwrite A's GPU
  results in the mirror before A merges them back (the words A wrote are then lost). NVN lets two queues
  touch different bytes of one page without a sync, so this is a real race at page granularity; it needs
  queue-ordered copy-ins (or entries visible across queues) to close. Fenced use is safe: A's entries end
  at A's `EndSegment`, before A's signal.
- *Timestamp reports.* A timestamp report into a pool (`nvn_exec.cpp` `TimestampReport`) is written by
  the CPU at completion through a raw host pointer (`counters.cpp`), which the page tracker does not see.
  `NotePoolWrite` invalidates the page at record time, but a copy-in recorded before that completion
  re-validates it with the old bytes, and the completion write is never seen. Only matters for a report
  inside a range bound as storage / read as indirect arguments (none seen so far).
- Raw global stores invalidate every *read-only* mirror (below); storage mirrors are in the raw table and take
  the stores themselves. A title with such a draw every frame re-copies the pages its later draws use
  (watch `ro_mirrors(... raw invalidations ...)` per present).

Heartbeat: `pool_mirrors(n/refused/MiB/copy_in runs,MiB/served,host/writes n,MiB,plain/invalidate_all)`.
Copied-in pages after warm-up are an upper bound for guest write faults (a trapped write costs ~0.3 us,
~1.9 us when it arrives as a fastmem fault; HleCpuTraps' `[page_tracker][bench]`); pages made stale by
`NotePoolWrite` or `invalidate_all` are copied without a fault.

**Measured** (OoT on VTX-HLE, RTX 5090, user renderer settings, save `oot`, 2 reps per arm interleaved,
`EDEN_NX2_NVN_GPU_PROFILE=1`, run dir `F:/eg/results/HlePoolMirror2/ab`; mode 0 vs 1, same picture):
- title screen and its intro film (csv 20-50 s): 25.9 -> 46.6 FPS, guest `nvnSyncWait` 35 -> 14 ms a
  frame; per present 31.7 MiB copied in (2 runs), 5 entries / 55 MiB merged back, no `invalidate_all`.
- Kokiri Forest gameplay (log 100-300 s): 11.99 -> 12.07 FPS (rep spread 0.73), GPU busy 81.8 -> 81.0 ms,
  dispatch 1.83 -> 0.56 ms, upload (mirror copy-ins) 0.00 -> 0.50 ms a present. 20 mirrors (400 MiB);
  per present 1,400 bindings served from mirrors, 225 left in guest memory (first written in an open
  pass), 83 entries, 7.7 MiB copied in, 3 `invalidate_all` (raw global stores). The frame is not GPU-bound
  there: 81 ms, of which the guest waits 65 ms in `nvnSyncWait` while the GPU's profiled work is ~20 ms.
Default stays 0: no gameplay gain where the save plays.

### Read-only mirrors: vertex, index and uniform ranges (`EDEN_NX2_TRY_VTX_MIRROR`, `EDEN_NX2_TRY_UBO_MIRROR`)

CostSplit (`F:/eg/results/HlePasses/run-b5`, OoT river, colour passes 9.9 ms): putting vertex+index ranges in VRAM
saves 2.8 ms, uniforms 2.0 ms, storage 0. All draws read imported host memory over PCIe. A mirror for those
reads is a `PoolMirror` with `storage == false`:

- It copies in **page by page what is used** and nothing else. Vertex and index ranges come from the scan that
  `StageVertexRanges` already makes of the pass ahead (the bound stream ranges and each indexed draw's index
  bytes); `MirrorTakeStale` copies the stale pages of exactly those ranges, at most `VTX_MIRROR_COPY_BYTES` (4 MiB)
  per pass (the range that crosses the bound still goes in whole; the next ranges stay in guest memory and
  come with later passes). Passes of fewer than 32 draws use it too (no copy of their own is worth it, a mirror
  copies nothing when current). Uniform ranges mark their pages in `PoolMirror::wanted`; `PrefetchMirrors` copies
  the stale pages of that working set at every pass begin (earlier uniform use, not the whole pool).
- It is **not** in the raw global table and not in the raw-writing draws' snapshots (a storage mirror is; every
  mirror added to those loops costs a tracker call per raw draw and a snapshot + merge per raw-writing draw:
  148 pools made the A/B of UBO_MIRROR slower, HlePasses run-ab4). Raw pointers therefore reach the pool's guest
  bytes, so a draw with raw global stores (`plan.raw_global_writes`) makes every page of every read-only mirror
  stale (`MirrorInvalidateReadOnly`, a bitmap fill; the pages come back when used) and is never served from one.
- A written storage range on the pool promotes it to a storage mirror (`VramMirror(..., storage)`): its valid
  pages stay valid, it joins the raw table (`pool_generation`) and gets the write-back monitor and merge guard.
- Currency check without the tracker: `GuestMemory::WriteEpoch(id)` is a lock-free counter that grows with each
  page of the registration that turns dirty. `MirrorCollectCpuWrites` collects the whole pool's dirty pages when it
  moved and remembers the value read before (`epoch_clean`); while it is unchanged `MirrorCurrent` and
  `MirrorTakeStale` skip the tracker (before: one mutex + owner search per uniform per draw).
- Everything else is the pool mirror's rules: copy-ins never overwrite a live entry's GPU writes (cut out by
  `RecordMirrorCopies`), replay writes through the pool's buffer invalidate (`NotePoolWrite`), guest CPU writes
  are seen by the page tracker, an aborted submission invalidates all.
- `EDEN_NX2_NVN_DIAG_RO_RAW_KEEP=1` (measurement only; the picture may be wrong): a raw-writing draw leaves the
  read-only mirrors valid. Its gap to the real switch is the price of the invalidation (the recopies it causes);
  identical pictures mean no raw store of the run reached a read-only mirror.
- `EDEN_NX2_TRY_RO_MIRROR_EXACT` and `EDEN_NX2_TRY_STORAGE_RAW_WRITE_THROUGH` (both on by default, `=0`
  opts out) enable raw store write-through to read-only and storage mirrors respectively. OoT Castle Town
  market at night: whole-pool snapshots 262 -> 92 per present, 345 -> 62 MiB copied each way, +2.5 fps. `Impl::RawWriteThroughTable`
  is a second table in the raw table's format, with one entry per eligible pool: its guest base,
  alias target device address and size. Read-only mirrors are included only with
  `TRY_RO_MIRROR_EXACT`; imported storage mirrors are included only with
  `TRY_STORAGE_RAW_WRITE_THROUGH`. Overlapping pools are excluded so every guest address has at
  most one alias. The table is rebuilt when its mirror set or either switch changes; the driver
  bank carries its count, address and guest bounds (`Profile::RAW_GLOBAL_ALIAS_*`, 0xF18..0xF37).
  The shader (`Info::raw_global_store_alias`) checks the bounds, binary-searches the table and
  repeats each mapped raw store into the alias: plain 32/64/128-bit stores and 8/16-bit field
  inserts. Raw atomics repeat the same operation on the alias word; the instruction result remains
  the result from the raw-table target.
- With `TRY_RO_MIRROR_EXACT`, a raw-writing draw keeps included read-only mirrors current instead
  of invalidating them. With `TRY_STORAGE_RAW_WRITE_THROUGH`, raw stores also reach the imported
  guest backing for included storage mirrors; `PoolMirror::written_through` lets `ApplyMirrors`
  skip those pools' whole-pool snapshot/merge, while explicit storage descriptor writes still
  snapshot their written ranges. Mirrors made after the draw's table or left out of it retain the
  existing invalidation/snapshot behavior. The draw keeps the table buffer and mirrored pools
  alive (`NoteUse`). Two invocations racing on the same word in one draw may leave different
  winners in the raw-table target and alias (a data race the console leaves unordered too).
  Counters: heartbeat `ro_write_through(draws)/alias_tables`, TSV row "pool mirrors: raw
  write-through alias tables built".
- Counters: heartbeat `ro_mirrors(n/promoted/raw invalidations/vertex+index ranges served/copy-bound skips)`,
  plus the pool mirror copy-in KiB; TSV rows "pool mirrors: read-only ...".
- Tests: `tests/nvn/compute_vulkan.cpp` covers raw stores/atomics through read-only mirrors and
  storage mirrors, CPU writes between draws, promotion, and the first copy of used pages;
  `pool_mirror.cpp` covers `MirrorPages::Set` / `TakeInvalidWithin`; `core/page_tracker.cpp`
  covers write epochs.

### Vertex and index coverage (`nvn_vtxcov.{h,cpp}`, `EDEN_NX2_NVN_VTXCOV_DIAG`, `EDEN_NX2_TRY_VTXCOV_*`)

OoT Kokiri (RTX 5080, build 44, standing still): ~42 ms of GPU per frame, latency-bound; the cost split puts ~5 ms
on vertex+index placement, but with `TRY_VTX_MIRROR` + `TRY_RO_MIRROR_EXACT` only ~800 of ~4,000 vertex/index
ranges per frame came from VRAM and the GPU time did not move. **Diagnostic** (`EDEN_NX2_NVN_VTXCOV_DIAG=1`, on by
itself with `TRY_VTX_MIRROR` or any `TRY_VTXCOV` switch): every vertex stream and index buffer a draw binds
(`PrepareDraw`, after `PrepareOutsidePass`/`BeginPass`) is counted by `VtxCov`, with its bound bytes, separately for
vertex and index. Served (device-local): `staged` (a staging copy), `mirror` (a pool mirror through the pass's
scan), `mirror_bind` (a pool mirror at the bind, switches below), `device` (the pool itself is device-local).
Not served, read from imported memory:

| Reason | Meaning |
|---|---|
| `stage_off` | `EDEN_NX2_NVN_VERTEX_STAGE=0` |
| `shader_writes` | a draw of the pass wrote through shaders (storage, image or raw stores): staged copies are not used |
| `too_large` | over 1 MiB (`max_range_bytes`), the scan leaves it out |
| `total_cap` | the scan reached 64 MiB first |
| `not_scanned_rt` / `_call` / `_end` / `not_scanned` | bound past the op that ended the pass's scan while the pass went on: a SetRenderTargets that kept the targets, a CallList (the called list's records), the end of the list; other |
| `index_indirect` | an indirect draw's index buffer (count on the GPU, never staged) |
| `pending`, `shadowed` | the pool was still being backed / a live storage shadow covered the range at the scan |
| `few_draws` | a pass under 32 draws without pool mirrors |
| `no_mirror_size` / `_budget` / `_align` / `_refused` | a pass under 32 draws and `VramMirror` makes no read-only mirror: pool over `TRY_STORAGE_VRAM_MAX_MIB`, read-only half of the budget used, pool not page aligned, refused for good earlier (budget, tracker, allocation, or the storage size limit when a storage use came first) |
| `no_device_memory` | no shadow chunk for its staging copy |
| `copy_bound` | the pass's mirror copy-ins reached their bound first |
| `mirror_stale`, `raw_writes` | bind-time serving (below): the mirror is not current there / the draw stores through raw pointers into a read-only mirror's pool |
| `dropped`, `size_short` | staged by the scan but the copy was dropped since (a pool write) / staged with fewer bytes |

The scan records what it did with each range (`DrawContext::stage_outcomes`) and the op that ended it
(`stage_unscanned`); a bind that finds no copy looks its address up there. Every 10 s of presents the replay thread
logs `[nvn] vtxcov: presents=N per present, vertex/index: served=v/i{reason=v/i,..} not_served=v/i{..} bytes
served_KiB=v/i not_served_KiB=v/i{..} copy_in_KiB=.. stage_copy_KiB=.. switches=..` (per present, reasons with no
bind left out; `copy_in` the pool mirror copy-ins, `stage_copy` the staging gathers). Totals: `ContextStats::vtxcov_binds`.

**Switches** (on by default together with `TRY_VTX_MIRROR` and `TRY_RO_MIRROR_EXACT`; they act with pool mirrors;
`EDEN_NX2_TRY_VTXCOV=0` turns all four off, `EDEN_NX2_TRY_VTXCOV_<NAME>=0` one; the startup line
`[nvn] vtxcov: switches ...` names them; the coverage counters need `EDEN_NX2_NVN_VTXCOV_DIAG=1`). OoT Kokiri
village: every vertex/index range served from VRAM, GPU 39-40 -> 36-37 ms, stable across area transitions:

- `EDEN_NX2_TRY_VTXCOV_INPASS=1`: a vertex/index range of an imported pool that no staged copy serves (not scanned,
  indirect index, past the copy bound, a pass whose copy was dropped) is served at its bind from the pool's mirror
  (made read-only at the bind if needed) when `MirrorCurrent` holds for it; otherwise its pages go into
  `PoolMirror::wanted` and `PrefetchMirrors` copies the stale ones in at every pass begin (it now runs for these
  pools without `TRY_UBO_MIRROR`). OoT draws the same geometry every frame, so the second frame finds it current.
  Ranges over 16 MiB (`VTXCOV_BIND_MAX_BYTES`: an indirect draw's index range runs to the end of its pool) are left
  alone. Why it is exact: a page is valid only after a copy-in, and copy-ins are recorded outside passes and ordered
  before every later vertex fetch by the all-stages barrier of the next pass begin (`HostFlushBarrier`, whose first
  scope also takes earlier command buffers of the queue) or by the staging scan's own `TRANSFER -> VERTEX_INPUT`
  barrier; every writer since is seen at record time: replay writes through the pool's buffer (`NotePoolWrite`),
  raw-writing draws (`MirrorInvalidateReadOnly`, or the write-through alias with `TRY_RO_MIRROR_EXACT`), the guest
  CPU (page tracker / write epoch). A use overlapping a live storage shadow or mirror entry ends it before the bind
  (`ClassifyShadows`). The same rule already serves read-only storage ranges inside a pass. A draw with raw stores
  into a read-only mirror's pool is not served (`raw_writes`). Not a cost: a prefetch copies only stale wanted pages.
- `EDEN_NX2_TRY_VTXCOV_SHADERWRITES=1`: the same bind-time serving after a draw of the pass wrote through shaders
  (staged copies stay unused there). Why it is exact: a shader store reaches pool bytes only through a descriptor
  write (noted per draw at record time, `NotePoolWrite`, or redirected into a mirror entry that a vertex use ends
  first) or a raw store (invalidates read-only mirrors at record time, or writes the mirror through with
  `TRY_RO_MIRROR_EXACT`; storage mirrors are the raw table's target and hold the newest bytes), so a current page
  holds what the guest bytes hold. A store and a later vertex fetch in the same pass are unordered on the console
  too (NVN orders them only through a Barrier, which ends the pass).
- `EDEN_NX2_TRY_VTXCOV_BOUND=1`: the stale pages a pass copies into mirrors for its scan stop at 32 MiB
  (`VTXCOV_COPY_BYTES`) instead of 4 MiB. Cost only: copies are exact at any size; a pool's first use may copy more
  in one pass.
- `EDEN_NX2_TRY_VTXCOV_BIGPOOL=1`: read-only mirrors also for pools over `TRY_STORAGE_VRAM_MAX_MIB` (64 MiB; the
  read-only half of `POOL_MIRROR_MAX_MIB` still bounds them). The limit is there for storage mirrors (raw-global
  dispatches copy in every stale page of each one and raw-writing draws snapshot it whole); read-only mirrors are
  not in the raw table and copy in only used pages. Such a mirror is never promoted and the pool gets no storage
  mirror (`VramMirror(..., storage)` returns null, not refused for good): its storage ranges stay in guest memory,
  their writes make the mirror's pages stale (`NotePoolWrite`), as for any read-only mirror.

Tests: `compute_vulkan.cpp` "NVN vertex coverage" (a stream bound after a SetRenderTargets that kept the pass is
`not_scanned_rt`, and `mirror_bind` from the next pass on with INPASS, guest writes seen through the prefetch;
BIGPOOL mirrors a pool over the limit and does not promote it), `pool_mirror.cpp` (the line), `pass_state.cpp`
(`VertexStageScan::capped`).

### Uniform snapshot at submit (`EDEN_NX2_NVN_UBO_SNAPSHOT`, `uniform_snapshot.*`, `nvn_uniform_snapshot.*`)

The replay runs a frame or more behind the game, and uniform bindings name guest memory directly. A game
that rewrites per-object uniform blocks right after it submits (OoT: transforms, skinning, light constants)
then has its draws read the next frame's constants: stretched polygons across the screen, lights on the
wrong objects. After MacNativeGames' TotK layer (MIT; `CREDITS.md`):

- **Capture** (`Recorder::Impl::CaptureUniformsLocked`, game thread, under `sink_mutex`, after the deferred
  arrays were resolved): the `BindUniformBuffers` ranges of every list (at most 64 KiB each), radix-sorted
  by address, kept only inside CPU-accessible physical pools (`snapshot_pools`: not `CPU_NO_ACCESS`, not
  virtual) and cut at the pool end. **Only bytes the guest CPU owns are copied**: a binding stays live
  unless every page it touches was found CPU-written (page tracker: each pool is registered with
  `GuestMemory::Track` at its first capture, a page is armed when first bound, `TakeWritten` at each
  capture) in this frame or the previous one (`UniformSnapshotCpuPages`, by present count), and none
  of its pages has GPU write history (`UniformSnapshotShared`, below). Survivors merge as before:
  overlapping ranges of one pool merge (so no binding straddles two), ranges up to 256 bytes apart
  merge while the result stays within 64 KiB. The bytes go into `Submission::uniform_snapshot`
  (recycled buffers) with the `completed_seq` read before the copy. Why: MacNativeGames' host lands
  every GPU write in guest memory; here a raw-writing draw's stores sit in the VRAM mirror until the
  whole-pool merge at the end of the command buffer (noted only then, after the binds of that command
  buffer) or reach unmirrored pools without any note. Round 1 copied such ranges and bound stale
  GPU results (OoT build 6: the whole scene blown out white).
- **GPU write history** (`UniformSnapshotShared`, process-wide, page intervals, sticky until the pool is
  finalized): every write the replay notes (`NoteWrite`; not inline updates, not mirror write-backs,
  which repeat stores noted at their draw) and, in the strict mode, every guest range of the raw table
  at a raw-writing draw (`NoteRawWrite`, once per table generation). Captures never copy those pages.
- **Binding** (`buffer_info`, `SnapshotUniform`): a uniform binding inside a live range is bound to an arena
  copy of the whole range (one upload per range and submission, phase kept modulo 256). Bindings under a
  pending inline update and misaligned ones keep their private copy, built from the snapshot bytes.
  `ReadUniformWord` (handle words, storage pointers) and `FreezeOracle` read the same bytes.
- **Drop rule** (`UniformSnapshotTracker`, replay thread): every pool write the replay records
  (`Recorder::NotePoolWrite`, `HostNoteShadowedWrite`) is logged with its seq until no snapshot can need it.
  A range is dropped (its bindings read memory again) when a write of a submission later than the capture's
  `completed_seq` overlaps it: earlier in the same submission, an earlier submission still unfinished in the
  submit call, or another queue's work replayed meanwhile (checked when the replay of the submission starts,
  then live for the rest of it). Inline updates carry their bytes (`NoteInlineWrite`): one of the submission
  itself or of an earlier one is layered into the copy in replay order; one of a later submission drops.
  Strict mode: a raw-writing draw also drops every range overlapping the raw table's guest ranges (in
  the active views, and at activation while such a draw is newer than the capture); mode 2 leaves raw
  stores to the CPU-written test. A `WaitEvent` passed after the CPU changed that event
  later than the submit (`NotifyEventCpu`, which is ordered with the submissions) drops the whole copy for
  the rest of the submission: the game may have written uniforms before it signalled.
- **Switch**: `Recorder::SetUniformSnapshot(UniformSnapshotMode)`, decided in `hle_runtime.cpp`: game patch
  `oot-ubo-snapshot` (OoT, strict), `EDEN_NX2_NVN_UBO_SNAPSHOT=0|1|2` overrides it for any title (0 off,
  1 strict, 2 raw-reachable ranges left to the CPU-written test). Off: submissions carry no snapshot,
  nothing is tracked, and the replay paths are the ones before (one null check per uniform binding and
  per pool write).
- Counters: Debug line per second on both sides (`[nvn] uniform snapshot: ...` bindings, the ones left live
  because not CPU-written lately or GPU-written, ranges, KiB, ms on the submit threads, GPU-written runs
  known; `[nvn] uniform snapshot replay: ...` drops by GPU writes / CPU-signalled waits / raw stores,
  layers, writes tracked); TSV rows "uniform snapshot: ...". Cost of the CPU-written test: one write
  fault per bound uniform page per frame the CPU writes it (as a pool mirror's tracked pages). Test:
  `tests/nvn/uniform_snapshot.cpp` (capture and merge, CPU-written and GPU-history filters).

## Report output directory and debug switch cost

Every NVN file the core writes (`census-<title>.tsv`, `unimplemented-<title>.tsv`,
`unsupported-<title>.tsv`) goes to `Core::NvnHle::OutputDirectory()` (`core/hle/nvn/config.h`, the
only definition): `$EDEN_NX2_NVN_OUTPUT_DIR` when set and non-empty, else `nvn` inside the emulator's
log directory (`Common::FS::EdenPath::LogDir`). The bridge creates it when it activates and logs
`[nvn] output directory <path>` once. (`EDEN_NX2_NVN_CENSUS_DIR` no longer exists.)

Per-call, per-draw, per-submit and per-present cost with every debug switch unset:

- The HLE dispatcher keeps no per-call counters: the per-slot call counters run only with
  `EDEN_NX2_NVN_CENSUS` or `EDEN_NX2_NVN_THREADS` (`Bridge::count_calls`, fixed at construction).
  `Call` reads and writes guest memory through `GuestMemory` directly (no `std::function`).
- The replay reads its switches once, during static initialization, into `Debug::g_config`
  (`Debug::Get()` is an inline load of that global, no guard). Per draw the unset cost is one
  `tracing` byte test; the experiments (`_FACE`, `_NO_DEPTH`, `_DEPTH_PROBE`, `_RELAX_DEPTH_EQUAL`,
  `_DEPTH_MODE`, `_FLIP_VIEWPORT`) are read only when a pipeline is built or the viewport changes.
- `EDEN_NX2_NVN_PROFILE`, `_SWEEP` and `_SKIP_DRAWS` were removed: the first was never read, the other
  two were one-off bisection tools (`_ONLY_DRAWS` / `_ONLY_VS` cover the same need).

## Draw path: storages, 3D targets, sampled targets (`nvn_draw*.cpp`)

- **Lazy images.** `Storage::image` does not exist before the first use. Per draw, `PrepareOutsidePass`
  asks `StorageNeedsCommands` for every storage the draw samples (this creates the image), ends the
  pass and calls `EnsureContents` only for the ones that answer true; attachments are ensured with
  `will_overwrite` in `BeginPass` (their stale guest bytes are never uploaded). Image views are made
  after that: descriptors are built in two phases (`BuildDescriptors` resolves textures, view keys and
  the plan; `FinishImages`, called by `PushPlan`, creates the views and patches the placeholder
  `VkDescriptorImageInfo`s). Views are keyed by texture serial, hold the storage, and are retired with
  the texture (`DestroyNow`) after its last use.
- **3D render targets.** A 3D storage is created with `2D_ARRAY_COMPATIBLE`; an attachment is a `2D`
  (one slice) or `2D_ARRAY` (a slice range) view of it, `baseArrayLayer` = slice (`RenderTargetRef`
  layer / view min_layer + nlayers; no view = every slice from the base layer), framebuffer layers =
  slice count, the tracked layout stays whole-image. Masked clears draw one pass per slice.
  Sampling follows the type the shader was translated for (`QueryTexture` reports the texture's real
  target): a 3D texture is a `3D` view.
- **Block-view render targets** (GPU block encoders: OoT renders BC4 through an `R16G16B16A16_UINT`
  view, one texel per block). Vulkan cannot attach a compressed image, so `TargetAlias` draws into
  an alias texture of the view format over the level's bytes (`BlockViewAliasOf`: the level's size in
  blocks, its GOB columns, block height and block depth, at the layer's offset). A 2D level's alias
  is one layer; a 3D level's alias is 3D with all its slices (block-linear slices interleave in
  blocks of 2^tile_d), and the target keeps its slice range on it. The blocks reach the compressed
  texture through the overlap transfers (`TransferOverlapBytes`) when it is read. A view of several
  layers of a 2D array is reported unsupported.
- **A texture sampled while it is a render target of the draw** (`feedback_copy`): the pass is ended,
  the sampled levels/layers of the target are copied into a scratch image of the same shape
  (`DrawResources::GetScratch`, one per draw slot, created lazily and kept), the draw samples the copy
  and the pass restarts. `VK_EXT_attachment_feedback_loop_layout` is not used: copy-on-read is exact
  for shaders that read what the draw does not write and needs no pipeline/render-pass variants.
  Multisampled targets and storage images that are also targets are counted
  (`feedback_unsupported`).
- **Base instance (c0+0x644).** NVN2 vertex programs read gl_BaseInstance from the driver bank. The
  SDK's instanced draws with a base instance (or, for elements, a base vertex) go through draw macros
  that write it there (elements: {base vertex, base instance} at c0+0x640/0x644; arrays: the base
  instance at 0x644) and write 0 back after the draw; every other draw macro leaves the words 0
  (OoT's macro RAM replayed with `gpu_capture_mme.tu_replay`, capture
  `F:/eg/results/hle-mesh-rec/.../04-sample-f2271`). The replay passes the base instance as the host
  draw's firstInstance (indirect draws: the record's own word), so the translation reads the vertex
  stage's c0+0x644 as Vulkan's BaseInstance (`NvnEnvironment::GetReplaceConstBuffer`). OoT's text
  draws two instanced draws per string from one glyph record buffer, the second from its base
  instance; with the word left 0 it read the first draw's records and drew nothing visible
  (dialogue, title, load screen strings). c0+0x640 (base vertex of indexed draws) is not
  provided: Vulkan's BaseVertex is firstVertex for non-indexed draws, where NVN keeps 0, and no OoT
  program reads the word.

## Pipeline cache (`nvn_pipeline_cache.h`, tests `src/tests/nvn/pipeline_cache.cpp`)

`HashProgramStage` includes the register count and the compute driver-bank layout (control
`0xa9`, `0xb1..0xb3`, `0xb6`, `0xba..0xbb`), in addition to the instruction/code bytes and
the control prefix. Identical compute code with different descriptor table offsets must not
share a lowered program or pipeline. Truncated compute controls have a distinct identity.

The opt-in `nvn_shader_refresh` setting starts without disk caches or asynchronous draw skips.
At present boundaries, after 60 seconds, `ClearShaders` drains preload, asynchronous, preparation
and optimization workers and the disk writer, then clears graphics/compute pipelines, lowered
programs, failed entries, translated shader sets, GPL parts and the driver compilation cache.
It keeps render passes (framebuffers reference them), textures and GPU-written results. Existing
flights retain pipeline shared pointers until completion. `ShaderGeneration` invalidates each
queue's local pipeline and memo before its next draw, including its binding-state tracking.
The diagnostic memory summary and log expose the refresh count. `ContextOptions` can shorten
the interval for headless tests; the UI mode uses 60 seconds. Disk files remain untouched in
this mode. The game-list Vulkan cache removal includes the emulated and NVN cache file names.

Files: `nvn_pipeline_cache.cpp` (lookups, translation: lower -> `NvnEnvironment` + `TranslateNvnStage`
-> `ConvertLegacyToGeneric` + `EmitSPIRV`), `nvn_pipeline_vk.cpp` (render passes, descriptor and
pipeline layouts, `VkPipeline` assembly), `nvn_pipeline_state.*` (NVN enum -> Vulkan, profile,
graphics pipeline create info), `nvn_pipeline_disk.*` (file format), `nvn_pipeline_cache_disk.cpp`
(writer queue, driver blob, preload), `nvn_pipeline_impl.h` (private `Impl`).

- Key: stage content hashes + `GraphicsFixedState` bytes + `RenderPassKey`; candidates of a key are
  variants, valid when every recorded texture-type query is answered identically by the oracle.
- Vertex input mode (`VertexInputMode`, chosen once per cache): **Dynamic** when
  `device.IsExtVertexInputDynamicStateSupported()` (and `PipelineCacheOptions::allow_dynamic_vertex_input`),
  else **Static**. The NVN replay does not look at the legacy `vertex_input_dynamic_state` user
  setting: `vulkan_device.cpp` keeps the extension/feature when `EDEN_NX2_NVN_HLE` is set (the
  workarounds that disable it on drivers known to be broken still apply, which gives Static).
  - Static (fallback): the key holds the vertex layout words (`GraphicsFixedState::attributes`,
    `stream_stride`, `stream_divisor`), the pipeline bakes bindings/attributes/divisors
    (`BuildVertexInput` -> `VkPipelineVertexInputStateCreateInfo`), `vertex_types` is zero.
  - Dynamic: the key holds **only** `vertex_types[k]`, the `Shader::AttributeType` (float / sint /
    uint / scaled / disabled) of location k that the vertex shader is translated for (`MakeKeyState`
    fills it from the NVNformat with `ResolveVertexFormat`); formats, offsets, stream indices,
    strides and divisors are zeroed (`Disk::CanonicalizeState(state, mode)`), so draws that differ
    only in layout share one `VkPipeline`. The pipeline is created with
    `VK_DYNAMIC_STATE_VERTEX_INPUT_EXT` and zero bindings. The shader translation reads the same
    type words (`VertexAttributeType`), so `RuntimeInfo::generic_input_types` and the SPIR-V
    content id (`HashRuntime`) depend on the type class only in both modes.
  - Per draw (`DrawContext::PrepareDraw`): `BuildVertexInput(fixed, pipeline->attributes_read)` fills
    a fixed-size `VertexInput` (attributes of the vertex shader's reads that have a vertex format;
    one binding per used stream with stride, `inputRate` = INSTANCE when the NVN divisor is non-zero
    and `divisor` = that divisor, else 1; no `vkCmdBindVertexBuffers2`: strides come from the binding
    descriptions). It is rebuilt only when `BindVertexAttribs` / `BindVertexStreams` changed the
    words or the bound pipeline reads other locations, and `vkCmdSetVertexInputEXT` is re-issued only
    when it was rebuilt or the command buffer's vertex input state is unknown (command buffer /
    segment start, or a pipeline bound after a pass begin/end or by the clear path). Problems (a
    non-vertex NVNformat, a scaled format without a Vulkan counterpart, stream beyond the device
    limit) are `CollectRows` rows via `PipelineCache::NoteIssue` instead of at pipeline creation.
  - The mode is part of the disk identity (the header's profile hash is seeded with it) and of the
    `GraphicsFixedState` layout (format version 4), so a file written in one mode is rejected by the
    other and a Dynamic key is never loaded as a Static one.
- Render pass: attachments are compact (used color slots ascending, then depth/stencil), LOAD/STORE,
  `COLOR_ATTACHMENT_OPTIMAL` / `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`; the color reference array is
  indexed by slot (unused = `VK_ATTACHMENT_UNUSED`).
- Dynamic state: viewport, scissor, line width, depth bias, blend constants, stencil compare/write
  mask and reference. Front face is the NVN value, inverted for an upper-left window origin (the
  replay uses a negative-height viewport for a lower-left origin).
- Failures: a stage that cannot be lowered/translated/emitted fails the request (null), is
  remembered by hash and logged once (`[nvn] shader rejected: ...`); every unsupported state is a
  `CollectRows` row and logged once.
- Key stability (a second identical run must compile <= 5% of the first run's pipelines): the key
  holds nothing that changes between launches - stage content hashes (`HashProgramStage`), the
  `RenderPassKey` (VkFormat ints, no padding, asserted) and the `GraphicsFixedState` bytes in
  *canonical* form (`MakeKeyState` -> `Disk::CanonicalizeState`, applied by `GetGraphics` before
  keying and by the loader to every stored record, with the cache's vertex input mode). The NVN
  state words are copied out of guest state objects, and the bits outside their fields (polygon
  bits 8..31, depth-stencil bits 39..63, blend target index and bits 43+, multisample bytes the
  pipeline ignores, unused attribute words and streams, a blend word of a target that does not
  blend, `restart_index`, `alpha_ref_bits` under ALWAYS, ...) hold stale guest memory: raw, one DK
  run produced 22656 distinct "pipelines" for 533 real ones and none of them matched in the next
  launch; with the layout in the key (Static) DK still compiled 557 pipelines cold and 30 (5.4%)
  warm. In Dynamic mode the layout is gone from the key. Texture-type queries are recorded in handle order
  (`NvnEnvironment::HandleRecords`, deterministic), hashes use fixed seeds, no serial or pointer is
  hashed. Remaining key noise in Dynamic mode: the type class of locations the vertex shader does
  not read still distinguishes keys (the read set is known only after translation).
- Disk: `<directory>/nvn_pipelines.k<16 hex digits>.bin`, one file per **translation key**, not per
  git revision (a new build used to start cold: DK town compiled 483 pipelines and skipped 11,995
  draws in 65 s cold, 98 skips warm). The key (`Disk::MakeTranslationKey`, stored whole in the
  header, the file name is its CityHash) is `FORMAT_VERSION`, the **translation epoch**, a hash of
  the translation switches, device/vendor/driver/pipeline-cache UUID and the profile hash (shader
  profile, host info, vertex input mode). The epoch (`Disk::TranslationEpoch()`) is the SHA-256
  that `CMakeModules/GenerateTranslationEpoch.cmake` computes at every build over
  `shader_recompiler/**`, `video_core/nvn/shader/**`, the pipeline translation sources
  (`nvn_pipeline_{cache,cache_disk,disk,impl,state,vk}`), `state_layouts.h`,
  `cooperative_matrix.h`, `.patch/sirit/**` and the sirit pin of `cpmfile.json`, so a forgotten
  `FORMAT_VERSION` bump cannot serve stale SPIR-V and a build that did not touch translation
  (new git revision, a change in another module) reuses the file. The list in the script is the
  contract: a file that influences translated SPIR-V belongs there (the GLSL/GLASM text backends
  under `shader_recompiler/backend/` are excluded: nothing in the SPIR-V path includes them).
  Settings and environment switches the recompiler reads (`TranslationSwitches`: loop safety
  setting, `EDEN_NX2_FP32_DENORM_PRESERVE`, `EDEN_NAN_OUTPUTS`, `EDEN_NX2_UNDEF_ZERO`,
  `EDEN_NX2_TENSOR`, `EDEN_NX2_RAW_COVERAGE` when on) are hashed at open, so another value gets
  its own file. **Not inputs:** HDR output, upscaler, anti-aliasing, scaling filter, frame
  generation, and the resolution setting. The last one reached the SPIR-V through the rescaling
  pass (a `resolution_setup` change started an empty cache: Res1X/Res2X/Res3X were three files on
  the same build), but the HLE never rescales (`PushPlan` pushes zero scaling words), so
  `MakeHostInfo` sets `HostTranslateInfo::skip_rescaling_pass` and the SPIR-V is the same at every
  resolution.
  The status bar's "Building: N shader(s)" counts the compiles queued to the async workers
  (`PipelineCacheOptions::shader_notify`, marked in `AsyncQueued` / `AsyncDone`).
  The loading screen's "Loading Shaders x / y": the cache opens when the guest names its title
  (after the rasterizer's `LoadDiskResources`, which the frontend drives), so its preload reports
  through `PipelineCacheOptions::load_progress` (`GPU().LoadProgress()`, set by the frontend's
  emu thread): Prepare, Build done/total records (rate limited, the last always), Complete.
  `EDEN_NX2_TRY_DIRECT_UNUSED_ATOMICS=1` (default off) emits unused-result storage atomics directly as
  hardware atomics, without subgroup pre-combine/waterfall. This matters for guest RED operations that
  are already warp-aggregated; the flag is included in the profile hash.
  `EDEN_NX2_TRY_NATIVE_F32_ATOMICS=0` opts out of VK_EXT_shader_atomic_float's buffer float32
  atomic-add feature for guest F32 add atomics; by default it is used when the Vulkan feature exists.
  Vulkan's feature applies to storage-buffer operations for both `VK_EXT_external_memory_host` imports
  and device-local allocations. NVIDIA's native float atomic add has FTZ and round-to-nearest semantics,
  matching the guest's `ATOM.ADD.F32` semantics; the switch and feature are part of the profile hash.
  `EDEN_NX2_TRY_ATOMIC_RESERVE=0` opts out of used-result compute IAdd subgroup reservation; by default
  it is used when subgroup atomic combine is supported. The switch and feature are part of the profile hash.
  The revision stays in the header for diagnostics only. Files of format 14 and older (one per git
  revision, `nvn_pipelines.<12 characters>.bin`) are never read (they cannot be proven to match any
  epoch) and age out: at most 6 cache files are kept per directory, oldest write time first, the
  file in use is refreshed at start. Header with format
  version, revision, translation key,
  device, driver, pipeline-cache UUID and profile hashes (all repeated outside the key and checked
  again on load; a renamed or colliding file is discarded); CRC-checked append-only SHADER /
  GRAPHICS / COMPUTE records; a bad record - frame, checksum or payload - ends parsing and the
  valid prefix is kept, the tail is cut before appending, a bad header discards the file, and
  `nvn_pipelines.vkcache` (driver blob shared by all builds, validated by the driver, rewritten
  atomically at most every 60 s and at shutdown). Setting `use_disk_shader_cache` off gives the cache
  no directory (nothing is read, written or created, as the emulated GPU without
  `LoadDiskResources`); `use_vulkan_driver_pipeline_cache` off (`PipelineCacheOptions::
  driver_cache_file`) keeps the pipeline file but never reads, writes or creates the `.vkcache`
  (the driver's cache still serves the run in memory). Unlike the emulated GPU
  (`DriverCachesPipelines`: no `.vkcache` on NVIDIA / AMD proprietary drivers, where it copied the
  whole blob at each save and stalled frames), this path keeps writing it by default on every
  driver: no stall was seen here (Main's decision, 2026-11-06).
  Decoder bounds of enum fields must cover the whole enum (the reader once stopped at
  `R16G16B16A16_SNORM` and rejected every file with a `B10G11R11_FLOAT` texture query; the tests
  encode every enumerator). A background loader rebuilds the pipelines on worker threads;
  `GetGraphics` / `GetCompute` that miss while the loader is still registering the cached keys wait
  for it (`WaitLoaderQueued`), and a key in flight is waited for, so a cached pipeline is never
  compiled a second time at run time. Logs `[nvn] pipeline cache: N loaded, M compiled at run time`
  once the preload finished, from `CollectRows` (the 10 s census and shutdown) whenever a count
  changed, and at destruction; `[nvn] pipeline compile: <ms> ms (translate <ms>, create <ms>)
  <kind> <key hash>` for every run-time compile (a `slow pipeline compile` warning over 50 ms),
  `[nvn] pipeline compile time: ...` totals and `[nvn] pipeline cache async workers: ...` (thread
  count, peak jobs queued or running, first job queued to last done) with the counts. The async
  worker count is `clamp(hardware threads / 4, 2, 6)`; `EDEN_NX2_NVN_ASYNC_WORKERS=<n>` forces it.
- **Asynchronous compiles** (`PipelineCacheOptions::async_compile`, set from
  `Settings::values.use_asynchronous_shaders` when the cache opens; off = every lookup is
  synchronous exactly as described above). `GetGraphics` / `GetCompute` take an optional
  `bool* not_ready`; only a caller that can skip passes it (the draw path; dispatches never skip).
  - **Which draws may skip** (`MaySkipUnbuiltDraw`, the emulated path's rule from
    vk_pipeline_cache.cpp `BuiltPipeline` / `CanSkipDraw`): `ResolvePipeline` passes `not_ready`
    only when the draw may skip; any other draw looks up synchronously and waits for a compile in
    flight (`WaitPending`). A draw whose programs store to memory (`StoresGlobalMemory`: the SPH
    DoesGlobalStore bit, word 0 bit 16, of any present stage, known before translation) always
    waits: skipping it loses state, not a frame of geometry. Otherwise a draw with a depth target
    may skip, and a depth-less draw only when it has more than 6 elements; a depth-less draw of
    <= 6 vertices/indices (a fullscreen pass) and a depth-less indirect or mesh draw (counted 0)
    wait, as they often build a texture the game never draws again (OoT's first gameplay frame
    renders 3- and 6-index depth-less passes into the slices of its sky tables).
  - A miss claims the key in `pending_*` under the same lock as the search (so a finished worker
    or a second lookup cannot slip in between), deep-copies the program bytes and asks the oracle for
    a frozen copy (`TextureOracle::Freeze(stage_mask)`), queues the compile on the `NvnPipelineAsync`
    workers (2-6 threads at normal priority; the preload workers are a low-priority background
    queue and stay separate) and answers "not ready". The worker runs the ordinary
    `CompileGraphics` / `CompileCompute` (same translation, same disk records, same stats), inserts
    the pipeline into the key's variant list and only then releases the claim. A later draw that asks
    again finds the pipeline. A lookup that passed `not_ready` never waits for it; a draw that may
    not skip waits for the claim to be released.
  - **Frozen oracle** (`DrawContext::FreezeOracle`, nvn_draw_descriptors.cpp): a worker must not read
    replay state, and translation asks the oracle for handles it only discovers while translating.
    The copy holds the per-stage binding arrays (combined / separate texture / image words), the bytes
    of every bound uniform buffer (<= 64 KiB each, guest memory plus this submission's inline
    updates; handle words read from a uniform buffer are rebased with the bound pools as live) and, for
    every word in them that names a live texture id, that texture's type / format / integer flag.
    `Query` then answers exactly what `QueryTexture` answered at the request. The recorded answers
    are the pipeline's variant key, so the draw that asked and every later draw revalidate against the
    live oracle as before (`QueriesMatch`): a pipeline built for other answers is simply another
    variant. An oracle that cannot be frozen (default `Freeze` = nullptr) compiles synchronously.
  - **Results:** a compile that produced no pipeline (rejected stage, assembly failure, exception in
    the worker; none escapes the worker) puts the key in `failed_*`: later lookups answer null
    *without* `not_ready` (counted as `pipeline: draw skipped, the pipeline was rejected earlier`;
    rejected stage hashes keep their own `shader:` rows). `stat_failures` / `async_failed` count it.
  - **Not-ready lookups** are counted (`async_not_ready`), the first 16 are logged
    (`[nvn] draw skipped awaiting async pipeline (#n): stages <hashes>`). The draw path counts them
    in their own row, `draw skipped awaiting async pipeline` (not in `draws_skipped`, which stays
    real failures), plus `frames with a draw skipped awaiting an async pipeline`; one `[nvn] async
    pipelines: present N: x draws skipped` line per such frame (<= 6000). The pipeline stays dirty in
    the draw state, so every following draw asks again.
  - **Dispatches never skip** (nvn_compute.cpp): `GetCompute` is called without `not_ready`, so a
    dispatch waits for a compile in flight, as the emulated path's `ComputePipeline::Configure`
    waits for its build. A skipped dispatch leaves what it writes stale for every later draw
    (one-time setup, GPU-driven culling). The cache keeps answering `not_ready` for compute to a
    caller that passes it (the pipeline-cache tests use compute requests for the asynchronous
    machinery).
  - **Preload:** with the option on, a lookup that misses before the loader finished registering the
    cached keys (`loader_queued`), or that finds its key pending (preload or a worker), answers
    "not ready" instead of waiting; it never claims a key the preload may still deliver. Synchronous
    lookups keep waiting (`WaitLoaderQueued` / `WaitPending`) and add the wait to `Stats().preload_waits`
    / `preload_wait_ns` (`[nvn] pipeline cache: N lookups waited ...`).
  - `Flush()` also waits for the asynchronous jobs. `[nvn] pipeline cache async: N compiles queued, M
    failed, K lookups answered not ready` follows the counts line.
  - Debug: `EDEN_NX2_NVN_CACHE_DIR=<dir>` keeps the cache files in `<dir>` instead of the profile's
    shader directory (builds with the same translation key share the files there; an A/B run that
    must start cold needs its own directory).
  - Debug: `EDEN_NX2_FRAME_CAPTURE_EVERY=<n>`, `_RANGE=<first>-<last>` and `_SKIPS=<n>` (the first n
    presents whose frame had skipped draws, each with the frame 8 presents later) change which presents
    `EDEN_NX2_FRAME_CAPTURE` saves; the frame-stats causes `present_acquire` / `present_handover` split
    `present_blit`.
  - **Measured** (RTX 5090, `EDEN_NX2_FRAME_STATS`, own cache directory per run, town 110 s / newgame
    150 s; the `.csv` of every run is under `results/ab/nvn-dk/async/`): the cold run compiles 584 / 849
    pipelines on the replay thread; the stalls are two bursts when an area loads (town p1650: 906 ms
    frame with 315 ms of `pipeline`; newgame p1319: 1048 ms with 391 ms). With async on the same
    bursts become frames of 230-280 ms (max 279.7 / 272.7 ms against 905.7 / 1048.5), frames over
    100 ms 16 -> 14 (town) and 19 -> 16 (newgame), `pipeline` stall 1.36 s -> 2 ms; mean FPS is
    unchanged (the run is bound elsewhere). What stays above 100 ms with a warm cache is texture
    upload (`tex_upload` 276-294 ms in the 590-610 ms frames), not compiles. The preload is not on
    the critical path: it starts ~1.7 s after boot, reads 520-710 shader and 566-841 graphics
    records and finishes within 22-300 ms; in the synchronous warm runs no lookup ever waited for it
    (`preload_waits` 0) and the `pipeline` stall of a whole warm run was 3-14 ms (the 5-18 run-time
    compiles).
- **Pipeline libraries** (`nvn_pipeline_library` setting, `PipelineCacheOptions::graphics_pipeline_library`,
  `nvn_pipeline_gpl.cpp`; off by default; tests "TranslationState ...", "Dynamic fixed state ...",
  "GPL: ..." in `pipeline_cache.cpp`). The device enables VK_EXT_graphics_pipeline_library only
  under `EDEN_NX2_GPL=1` (legacy renderer, which keeps its own switch) or `nvn_pipeline_library`
  while the program runs on the NVN HLE, and
  only with fast linking and independent interpolation decorations (the SPIR-V backend decorates
  fragment inputs only). A graphics miss is then built on the replay thread and never answered "not
  ready" (the asynchronous setting keeps applying to compute):
  - **Translation reuse:** `MakeRuntimeInfo` reads the draw's state only through
    `TranslationState` (input topology class, point size, alpha test and reference, depth mode,
    dual-source blend, color output type per target, vertex type class per location). Pipelines with
    the same stage hashes and `TranslationState` share one `ShaderSet` (translated stages, their
    `VkShaderModule`s, the layout the shader parts are compiled with) as long as the oracle still
    gives its texture answers; others translate (and are a new set).
  - **Parts** (`GraphicsPipelineLibraryCache`, shared with the legacy renderer; the owner pointer
    of a set stands for its layout): vertex input and fragment output parts are keyed by their state,
    pre-rasterization and fragment shader parts by set + render pass + their state. The vertex
    input part always has vertex input and input assembly state: a mesh pipeline has none, and a
    part cannot know it is one (VUID-VkGraphicsPipelineCreateInfo-pStages-02097), so it gets an
    empty vertex input and a triangle list (`MakeLibraryPartCreateInfo`); the NVIDIA driver 621.01
    crashed on the null input assembly state of that part. With
    VK_EXT_extended_dynamic_state(2) the GPL pipelines declare cull mode, front face, depth bias
    enable, depth test/write/compare and stencil test/ops dynamic (`DynamicFixedState`, the same
    function that fills the static create info of the default path), so the shader parts do not
    depend on them; the replay sets them after binding such a pipeline (`fixed_state_set`, unknown
    after a static pipeline or an unknown predecessor). Missing parts are compiled side by side
    (`part_workers`, 3 threads plus the calling thread), then fast-linked (no link-time optimization).
    A shader part is a full shader compile (100 ms to seconds on a cold driver cache), so a "fast
    link" of a new shader set costs that, not the link: with `use_asynchronous_shaders` a draw that
    may be skipped (`MaySkipUnbuiltDraw`) takes the default path's route: the translation, the parts
    and the link run on `async_workers` and the draw is skipped until the pipeline is published; a
    draw that must not be skipped (and every draw with the setting off) builds on the replay thread.
    One translation per `TranslationKey` runs at a time (`TranslationHold`): workers asking for the
    same key wait for the first and reuse its set. The parts are compiled with the driver
    pipeline cache (saved to `nvn_pipelines.vkcache`); the parts themselves are in memory only.
    Pipelines the parts cannot express (no fragment program, static rasterizer discard) are compiled
    whole (`gpl_whole`).
  - **Swap:** each fast-linked pipeline is compiled whole on `optimize_workers` (background,
    `clamp(hardware threads / 8, 1, 4)`) with its own layout, render pass and dynamic states and
    published in `GraphicsPipeline::optimized_handle`; the replay binds `Handle()` (rebinding when it
    changes), the fast-linked one stays alive with the object.
  - **Program creation:** `CmdCreateProgram` lowers the program's stages on a background thread
    (`PrepareProgram`, `programs_prepared`): its first draw starts at translation. The translation
    itself needs the draw (textures behind the handles, targets, vertex formats), so it cannot move
    to creation.
  - Disk cache unchanged: shader and pipeline records as on the default path, the boot preload
    compiles them whole, the optimized compiles fill `nvn_pipelines.vkcache`. After the preload,
    `QueueWarmShaderParts` compiles the library parts of the newest cached shader sets (one
    record per set, at most `MAX_WARMED_SHADER_SETS`, 4 background jobs on the preload workers;
    the loading screen does not wait): the sets are made from the cached stages (`FindShaderSetOf`
    dedupes against run-time sets by shader record ids), so the first run-time pipeline of a known
    set finds it (`gpl_translations_reused`) and links without compiling shaders. A failed key is
    remembered (`failed_graphics`) and not built again per draw. Counts:
    `[nvn] pipeline libraries: N fast links (M from parts compiled already), ... optimized compiles
    swapped in (... over 50 ms, ... background)`; per pipeline `[nvn] pipeline compile: ... graphics
    fast-linked|graphics whole (gpl)` (the "slow pipeline compile" warning is a replay-thread stall
    unless the compile ran on an async worker).
- Provoking vertex: NVN's is the last vertex (the SDK's initial GPU state sets `SET_PROVOKING_VERTEX`
  0x1684 to 1; `nvnCommandBufferSetProvokingVertexMode` writes the mode into it, OoT sdk+0x686300),
  Vulkan's the first. `GraphicsFixedState::provoking_first` (0 = last, so old disk records keep their
  meaning) follows `Op::SetProvokingVertex`; pipelines chain
  `VkPipelineRasterizationProvokingVertexStateCreateInfoEXT{LAST}` when the device has
  `provokingVertexLast`, else a pipeline issue is logged and flat outputs come from the first vertex.
- Known gaps: tessellation primitive/spacing/winding are not part of the NVN pipeline state this
  module receives (recompiler / Vulkan defaults); advanced blend modes are
  ignored (basic funcs used); vertex-divisor feature bit is not queried (the extension is
  mandatory on the device). Real-shader translation is covered by the shader coverage tool and the
  game, not by `pipeline_cache.cpp`.
- **Shader log** (`Settings::values.gpu_log_shaders` / `EDEN_NX2_SHADER_LOG=1`, core in
  `video_core/shader_log.*`, HLE hooks in `nvn_pipeline_log.cpp` and `nvn_draw_log.cpp`; off = one
  relaxed load in `PipelineCache::GetGraphics/GetCompute`). When a lookup returns, the first time
  per pipeline object (`PipelineBase::logged`; a rejected key or a fallback program + reason has no
  object and is remembered in a per-cache set instead), the hook submits: each new stage hash
  (`HashProgramStage`) once: the BNSH code area and the control section byte for byte (`<hash>.bin`
  / `.ctl`; graphics code starts with the 0x80-byte SPH), rejected ones with the reason and the
  failed instruction (`LoweredNvnProgram::failed_instruction`); and one pipeline record (key = the
  `GraphicsKey` hash, variant >= 1 mixes in the hash of the texture queries) with outcome
  Translated (compile ms measured in `CompileGraphics/CompileCompute`), DiskCache (rebuilt by the
  preload), Rejected (the reason `NoteRow` saw last on the compiling thread, or the stage's
  `Reject()` reason) or Fallback (the NDSR dispatch the replay replaces by its upscale stage,
  `PipelineCache::LogComputeFallback`), the NVN state as JSON, `<hash>.ir.txt` (IR emitted from,
  dumped before `EmitSPIRV`) and `<hash>.spv` of the first pipeline that used the hash, and
  `p<key>.inputs.json` from the replay (`InputsProvider`, `DrawContext::DescribeInputs`): the
  constant buffer bytes the shader may read (`constant_buffer_used_sizes`; the bound size for a
  pipeline from the disk cache), handle words with the ids and the NVN texture/sampler
  descriptors behind them (the HLE has no guest TIC/TSC: `tic`/`tsc` are null), push constants and
  storage ranges. An async lookup that answers "not ready" calls `NoteAsyncSkip(base key)`.
  `ShaderLog::NoteFrame()` runs once per present in `Context::Impl::ExecutePresent`.

## Compute driver bank (c0) and the tensor upscaler (`nvn_compute.cpp`, `nvn_compute_bank.*`)

- c0 of a compute dispatch: `+0x00` the declared bank size, `+0x04/+0x08/+0x0c` the group counts
  (direct dispatches; 0 for an indirect one, the GPU reads those), then the push words at `+0x10`,
  the uniform and storage entries and the handle tables of the control section. Measured on DK's
  legacy cbuf 0 (`EDEN_NX2_NVN_HLE` off): word 0 = 0x20/0x30/0x70/0xc0/0xf0 for programs whose
  tables end there, word 1 = the dispatch's x group count (18 and 10 seen; DK's legacy dispatches
  all have y = z = 1). Two DK shaders (`agl_common` 16x16x1) read c0+4; the HLE left the header 0.
- DK's tensor upscaler (agl `NDSR`, 24 compute shaders with HMMA, 8 push words, `EDEN_NX2_TENSOR` unset)
  gets its push words from `nvnCommandBufferPushComputeConstants`: `{0, 0, src_w-1, src_h-1, 0, 0,
  dst_w-1, dst_h-1}` (source and destination rectangles, e.g. 1439x809 -> 2879x1619). All 8 words are
  read by the shaders (c0+0x10..0x2c, found by scanning the translated IR). The old census (legacy
  GPU path) shows no calls because the legacy path never dispatches NDSR in DK's town run (no 128x1x1
  group in 76,000 dispatches); with real GPU timestamps HLE DK picks the 1440x810 tier and dispatches
  one NDSR program per presented frame (the default legacy timestamp mode keeps native 2880x1620).
  Zeroing the push words, or skipping the dispatch, leaves the scene as the clear colour (blue); with
  the real words the frame is correct.
- **The replacement is structural** (`DrawContext::UpscaleNetwork`, `Upscaler::IsUpscaleNetworkDispatch`;
  no title, build or program hash): the NDSR dispatch shape (128x1x1 threads, 1 sampled texture, 2 image
  slots, >= 8 push words), HMMA in the program's code (opcode 0x23c, `CodeUsesTensorMath`, cached per
  program hash) and rectangle words whose output is larger than the input on both axes. The output is
  the bound image that accepts those rectangles, not necessarily the first image slot (OoT binds a
  480x270 auxiliary image before its 2560x1440 output). DK's 24 programs
  and Ocarina of Time's gameplay network (HLE `3756d1d2..`/`484437fc..`, 188/176 HMMA; same push-word
  layout at c0+0x10, 29 push words) classify; OoT's opening-video chain (32 threads, 2 textures) does not.
  Replaced when `ReplacesGameNetwork`: the Upscaler setting alone decides (DLSS, DLAA, FSR,
  Bilinear, or Auto/Game without tensor cores), or Auto with the Resolution setting above 1X (the network
  reads the scaled scene only at its native sample positions; an explicit Game keeps it). DLAA is not an
  Anti-Aliasing choice: a successful DLSS/DLAA upscale suppresses a duplicate final DLAA pass, while
  FXAA/SMAA stays a separate final-frame setting for non-NVIDIA upscalers. Otherwise the
  game's network runs and nothing below exists (no chain, no pass depths).
- **Scene colour.** OoT's network does not sample the scene: its single texture is a packed R32_UINT (the
  colour sqrt-encoded as RGB9E5) written by the dispatch before it, itself fed by two earlier ones (an
  analysis at 1/4 size, a class map from a 192x128 LUT). While a replacement is possible the replay
  remembers the textures the last 16 compute dispatches sample and write (`Impl::compute_chain`);
  view bindings use their base texture's serial, because OoT's packing dispatch writes an R32 view
  of the base texture the network samples. `FindNetworkColour` takes the network input itself when no
  dispatch wrote it (DK) and otherwise the first sampled texture of its newest writer that is a float colour
  and that no compute dispatch of the chain wrote (the scene's R11G11B10F; the class map is excluded). No colour: the game's program runs.
- **Depth, motion, twins.** The scene depth is the depth of the last render pass drawn into that colour
  (`Impl::pass_depth_of_color`, same size, standard Z); OoT has no velocity target and no jitter, so the
  motion vectors are the host's optical flow (`UpscaleJob::host_inputs`) with the real depth instead of the
  far plane (`EvaluateDlss`), pre-exposure 1 (LDR). With the Resolution setting the current twins of the
  colour and the depth are used (`UpscaleTwinOf`; the game's DLSS keeps its stricter size rule), the
  rectangle scaled to their pixels; the output stays guest-sized. The scaled input must be no larger than the output on
  either axis, or at least as large on both by at most `DlaaCropSlack` (2 px: OoT's 854x480 at 3X is
  2562x1440 for a 2560x1440 output); DLAA then crops the right/bottom excess (`RunUpscale` turns a
  host-input DLSS request at ratio 1 into DLAA, `PromoteNearNativeToDlaa`). A larger twin (953x536 or
  1920x1080 at 3X) is not evaluated; the guest-sized image is. One `[nvn] NDSR upscale replaced by <stage>
  (input WxH -> OxP twin|native)` line per input size and stage; `native: <reason>` names why the twin was
  not used (does not fit the output / never drawn scaled / stale / lacks guest bytes / image not created)
  and `twin, far-plane depth (<reason>)` when only the depth twin is missing. The prepass dispatches still
  run (cheap, unused).

## The game's own DLSS (`Op::EvaluateDlss`, `nvn_upscale.cpp`, `core/hle/nvn/functions_dlss.cpp`)

- Star Fox and OoT call NVN2's DLSS API (24 `nvnDlssNetwork*`, 33 `nvnDlssParameters*`, and
  `nvnCommandBufferEvaluateDlss`). The HLE keeps `NVNdlssNetworkBuilder`, `NVNdlssNetwork` and
  `NVNdlssParameters` in place in guest memory with the layouts of the game's own SDK. Their sizes
  are read from that SDK's code at boot (`ReadSdkDlssSizes`, `sdk_dlss_layout.*`: the functions of
  its GetProcAddress implementation table, the bytes the two SetDefaults store and the memset size
  of `nvnDlssNetworkInitialize`; log line `[nvn] sdk DLSS objects: ...`), the network's field
  offsets come from the table of known layouts by its size: 0xc0 / 0x258 / 0xc0 for API
  231.39/231.40 (OoT sdk+0x6599d0..0x65b010, Star Fox sdk+0x6611d0..0x6627b0, identical semantics),
  0xb8 / 0x220 / 0xc0 for 231.30 (SDK 20.5, the network 0x38 bytes shorter), 0xc0 / 0x218 / 0xc8 for
  231.16 (no Resize). A network size of no known layout refuses every Initialize (nothing written).
  Parameters (all SDKs) `+0x00` input colour, `+0x10` output colour, `+0x20` motion vectors, `+0x30`
  depth, `+0x40` bias mask (each texture + view pointer), `+0x50` jitter, `+0x58` motion vector scale,
  `+0x60` pre-exposure, `+0x68` exposure texture, `+0x78` history reset byte, `+0x7c` evaluation
  index, `+0x80` input size, `+0x88..+0xaf` the five rectangle origins, `+0xb0..+0xbf` the four
  overlay switches. `GetStorageSize` is the SDK's formula (checked against the OoT and Star Fox
  SDKs' code; the 231.30/231.16 SDKs compute it with other code, not checked), with the history
  textures sized by the HLE's own texture builder as the SDK does. The network's `+0xa0` (the SDK's
  feature handle, at that offset in every layout) is a per-Initialize id. The SDK's backend (optimal
  render sizes, feature create/evaluate) is a system library the SDK loads at run time ("dlss",
  "Handshake"), not part of the game: `GetOptimalSettings` keeps NGX's published ratios.
- `EvaluateDlss` reads both objects at record time and emits one `Op::EvaluateDlss` record
  (`EvaluateDlssPayload`: six `RenderTargetRef`s, the rectangles, jitter, scale, reset, the
  network's output size/flags/method/preset and its id) with the SDK's rules: an input size of 0x0
  is the input texture's size, the output origin counts only with flag bit 28, the exposure texture
  and pre-exposure only for HDR colour without automatic exposure (`(flags & 0x41) == 0`). The
  evaluation index is not passed on (the SDK only puts it in its devtools trace).
- The replay runs it through the same routine as the NDSR replacement (`DrawContext::RunUpscale`):
  DLSS when the setting is DLSS or DLAA (or Auto/Game for the game's own DLSS) and Streamline can
  evaluate, else FSR1 (bilinear if FSR cannot, or when chosen), so the output rectangle is always
  written; rectangles that reach outside their image are clipped first. Flags map as the SDK
  forwards them to NGX: bit 0 set = LDR (clear selects DLSS's HDR path), `0x2` motion vectors at the
  render size (else the output size), `0x4` jittered motion vectors, `0x8` inverted depth, `0x40`
  automatic exposure. The game's motion vector scale (NGX, vectors to pixels of the motion vector
  image) divided by that image's rectangle size is Streamline's `mvecScale`. With DLSS Quality Auto
  the network's method and preset choose Streamline's mode and preset (`Dlss::Decide`); each
  network evaluates on its own Streamline viewport (its id), so networks keep separate histories.

## Draw diagnostics and world-geometry findings (`nvn_debug.*`, `nvn_draw_debug.cpp`)

Env-gated, free when unset: `EDEN_NX2_NVN_DRAWLOG=n` (first n draws of a selected frame: stage
hashes, state words, viewport, targets, vertex/uniform/texture bindings, descriptor fallbacks, and a
per-target draw table plus pass-break reasons; `rc=cond` marks a draw gated by a GPU render-condition
predicate, and a mesh-task indirect draw gets a `watch draw-args` line with the {count, first} the
GPU read, since its `IND=` words are guest memory at record time, before the culling ran),
`EDEN_NX2_NVN_DUMP_TARGETS=<dir>` (+ `_FRAME`, `_MAX`, `_WIDTH`: every pass's color/depth targets as
TGA with mean/nonzero/range statistics; unscaled depth and float/wide colour targets also as `.raw`),
`EDEN_NX2_NVN_DUMP_SPIRV=<dir>`, `EDEN_NX2_NVN_CACHE_LOG=1` (why a pipeline compiled at run time).
Experiments: `_FACE=1|2`, `_NO_DEPTH`, `_FLIP_VIEWPORT`, `_DEPTH_MODE`.
`EDEN_NX2_NVN_LOG_BACKING=<program hash hex>[:<max dispatches>]` (default 8) logs each storage
binding of the selected compute program's first dispatches: guest address, size, effective backing
(imported host pool, pool mirror, shadow copy, device-local pool, raw-global table, or other), and
whether the translated shader info marks the slot as an atomic target.

`descriptor fallback (graphics|compute): <reason>` rows in the unsupported TSV count every dummy
texture/buffer a draw bound, by reason.

- **Empty and unsized buffer bindings** (`DescReasonIsFallback`; counted in the same rows but not
  warned about). A uniform buffer bound with address 0 or with size 0 reads zeros (the dummy
  buffer; constant-buffer reads are bounds checked and nxnative reads its all-zero bank); DK has
  11.5k draws a run that reference such a slot. A storage buffer bound with an address and size 0
  is the rest of its pool (loads and stores go through the address; the c0 entry's size is never
  read by a shader: `ULDC.64` of the address only), which the compute output of DK's GPU
  particles needs (`EDEN_NX2_NVN_NO_ZERO_SIZE=1` loses it). A storage slot nothing was bound to
  (DK's particle emitters, e.g. program 59d4a71494e471ef: after a depth-buffer test the shader
  appends through slot 14 behind a warp-aggregated atomic counter in slot 15; the game binds
  storage slots 0..10 only) is only reached by code the game does not run with it:
  the hardware would fault on a store through address 0. The dummy is bound for reads; a written
  binding gets its own 4 KiB slice of the sink buffer per (program, stage, slot) and
  `DrawResources::CheckSink` (every 120 presents) logs a `[nvn]` warning with the program hash
  when a nonzero word shows up in a slice, so a game that does store through one is reported
  instead of silently polluting the dummy that unbound uniform buffers read zeros from.

- **Depth EQUAL after a pre-pass.** DK draws a depth pre-pass (color mask 0, LESS) and then its
  G-buffer draws with depth func EQUAL, no depth write, with *different* vertex shaders. The replay
  tests EQUAL exactly (`EDEN_NX2_NVN_RELAX_DEPTH_EQUAL=1` tests it as LESS_OR_EQUAL). The NVN profile
  decorates the vertex position output `Invariant` (`Profile::position_invariant`) and every
  floating point add/mul/fma of the position stages `NoContraction` (`Profile::position_precise`;
  the guest's FADD/FMUL/FFMA already are). Measured on a town frame: the canonical position
  expression of a pre-pass and a G-buffer vertex shader of the same mesh is identical, with strict
  EQUAL the terrain of the first area and the platform render the same as with LESS_OR_EQUAL; a
  LEQUAL+depth-write probe (`EDEN_NX2_NVN_DEPTH_PROBE=1`) only moves depth for fur shells and a few
  alpha-tested props (1.7% of the pixels, not ulp-level).
- **Window origin.** NVN clip space is OpenGL's and row 0 of a render target is its top, so viewports
  are always negative-height and the NVN winding is the Vulkan one (`nvn_viewport.h`); the origin only
  places the rectangle. With the old rule (positive height for UPPER_LEFT) DK's G-buffer was stored
  upside down and render-to-texture text (world banner) was sampled flipped.
- **Handles of the wrong dimensionality.** DK binds 1x1 / 4x4 2D placeholder textures to slots a
  shader samples as 3D / cube / cube array; they stay a counted dummy fallback (3 draws per frame in
  the cave, DK's head light). Handle words read from a uniform buffer are rebased with the bound
  pool base (`RebaseHandleWord`); DK's pool bases are all 0.
- More experiments: `_DEPTH_PROBE=1|2`, `_ONLY_DRAWS=a,b,..`; depth dumps also write `.raw` floats.
- Validation findings fixed: sampled depth images without DEPTH_STENCIL_ATTACHMENT usage use
  SHADER_READ_ONLY layout; storage image views use the identity swizzle and do not claim
  STORAGE usage for formats without storage support. Not ours: StencilExportEXT capability
  without the extension, shader_quad_control without maximal reconvergence (device setup), and a
  geometry shader whose inputs exceed maxGeometryInputComponents.

## Black terrain root cause (cave, DK gsys terrain) and more diagnostics

- The terrain is GPU-driven: a point draw (rasterizer discard, `mode=0`, ~1300 vertices) runs a vertex
  shader that reads a table of 64-bit addresses and writes `{count 0, instances 1, 0, 0, 0}` into every
  chunk's indirect record; the depth pre-pass draws then run a geometry shader that compacts the
  surviving triangles (atomic add on `args[0]`, u16 index stores); the G-buffer pass draws each chunk
  with `DrawElementsIndirect`. The reset stores go to raw global memory (addresses read from memory).
  The recompiler's global-memory pass ties them to the table's storage buffer (its BFS walks through the
  load that produced the address), so they land out of range and are dropped: `instanceCount` stays 0
  (the indirect draws draw nothing: ALWAYS+write probe leaves the depth identical) and `args[0]` grows by
  ~6 per frame, measured with `EDEN_NX2_NVN_WATCH=ind` (0x630, 0x900, 0xbd0, ... every 120 frames).
  nxnative documents the same behaviour (nvn_mtl.m, "Raw global memory", DMSH_KIND_GMEM_MAP).
- Not the cause: polygon offset (units/factor/clamp map like nxnative, the enable bits decode right and the
  terrain draws have none), depth range, depth clip control, attachment load/store.
- Fixed with raw global memory (`HostTranslateInfo::nvn_raw_global_memory`, `Profile::support_raw_global_memory`,
  `Info::uses_raw_global_memory/stores_raw_global_memory`; kill switch `EDEN_NX2_NVN_NO_RAW_GLOBAL=1`):
  the storage buffer pass stops its address search at memory loads and does not use the unbiased
  fallback; remaining global accesses are emitted through `PhysicalStorageBuffer64` pointers. The
  shader translates the guest address with a table in device memory: the driver bank (c0) holds the
  entry count (u32 at 0xf00) and the table's device address (u64 at 0xf08), past every NVN2 word of
  that bank (the mesh stage's storage table ends at 0x870), the table is a sorted,
  non-overlapping array of {guest, device, size} u64 triples that the shader binary-searches (the last
  entry whose base is <= the address; any number of pools fit). `RawGlobalMemoryTable` builds it
  with `FlattenRawGlobalRanges`: one source per imported chunk (a buffer over the whole chunk,
  `ImportedChunk::address_buffer`, priority 0), one per pool with its own buffer (CPU_NO_ACCESS,
  priority 1, they cut the chunk ranges around them) and per virtual run, then uploads it to a
  device-local buffer (`RawTableBuffer`, a new one per rebuild; a draw keeps the one it used alive
  through `InFlight::keepalive`, once per submission). Mario Kart World uses a mapped device-local
  allocation when available, writes and flushes it before submission, and falls back to a staged
  copy. This also applies to the raw store alias table. DK has ~300 small GPU-only pools inside
  chunk windows: 615 entries, no drops (`raw_global_ranges_dropped` is 0; it counts the entries of
  a table that could not be uploaded). Accesses outside every range are dropped (loads return 0).
  Pool buffers have SHADER_DEVICE_ADDRESS usage and imports allocate with
  DEVICE_ADDRESS when the device has the feature (`EDEN_NX2_NVN_HLE` keeps the BDA feature on in
  `vulkan_device.cpp`; Vulkan 1.2+ never set the extension flag). Pipelines with raw accesses are
  treated as writers: the pass ends before an indirect draw reads arguments (ALL_COMMANDS barrier).
  Result: the chunk records are reset to {0,1,0,0,0} every frame (`EDEN_NX2_NVN_WATCH=ind`).
  Unit tests: `tests/nvn/raw_global.cpp`; `EDEN_NVN_RAW_GLOBAL=1` runs the coverage tool with the path.
- Diagnostics added: `EDEN_NX2_NVN_ONLY_VS=h,..`, `_DEPTH_PROBE=3` (ALWAYS+write), `_OPLOG=1` (buffer/pass ops
  and dispatch bindings of the selected frame), `_WATCH=gs|ind|<hex addr>[:<hex bytes>]` (a buffer range
  copied back at every dispatch/barrier/render target change and logged), indirect arguments, index
  address, polygon offset state and storage buffers in `_DRAWLOG`.

## Tests (`src/tests/nvn/vulkan.cpp`, tag `[nvn]`)

Layout and format tables (no GPU); clear + texture->buffer copy of a 64x64 RGBA8 texture (default
pool path and forced mirror); buffer copy with a second queue waiting on the first queue's fence
(producer first, consumer first = held back, mirror); host-pointer import probe on
`Common::HostMemory` (logs `[nvn] host import: supported|unsupported (<VkResult>)`) and a GPU write
through a pool over that memory; DK's two-queue event ping-pong with data dependencies through pool
memory (3 frames, both submit orders, import and mirror), a wait nothing signals (timeout, the other
queue keeps running) and a CPU signal that releases a waiting stream (`RunEventPingPong`,
`RunEventNeverSignalled`, `RunEventCpuSignal`). GPU tests `SKIP("no Vulkan device")` without a device.

## Open items

- `[UNVERIFIED]` the 16-bit packed format bit orders (RGBA4, RGB565, BGR565).
- `[UNVERIFIED]` layer handling of the clear/copy regions against the recorder's encoding
  (`layer_count`, `region.z/depth`).
- Sparse tiled mip padding in the unswizzle (`TextureLevelGobs` is computed but unswizzling assumes
  an unpadded row).
- ASTC/BC transcode on devices without native support.
- The per-op barriers are the simple version.

## Render pass breaks (`BreakCause`, rows `render_pass_break: <cause>` in the unsupported TSV)

Every `EndPass` of an open pass counts a cause. Town measurements (per present, 3,500 frames):
begins 108.0 -> 90.5. What was removed: a game `Barrier` ends a pass only when a draw of the pass
wrote through a shader (storage buffer, storage image, raw global memory; NVN barrier bits 0xF);
an unchanged `SetRenderTargets` keeps the pass; `SetRenderCondition` ends it only when a predicate
is computed (kind 0), not for SetRenderEnable(true/false); storages that flip between attachment
and sampled use (learned from layout breaks, `FlipTargetList`, 256 entries; a full list replaces the
entry used longest ago, it no longer stops learning) are moved to their sampled layout in
the barrier of the next pass begin, so a draw that samples one does not end the pass. Sampled
image transitions are batched into one barrier call (`HostTransitionMany`). Remaining causes are
inherent to Vulkan: a target change, compute, a feedback copy of a sampled target, samples-passed
reports, copies, masked clears. Replay-thread time did not move (~8.0 ms/frame submit): the
cost is per draw (2,260 draws/frame), not per pass. `EDEN_NX2_NVN_OPLOG=1` now logs draws,
reports, counters, conditions and the layout-break culprit.

**Sampled layouts of storage-written images.** A colour image in GENERAL (a shader stored to it as a
storage image) is sampled in GENERAL, which allows every device access (Vulkan spec, "Image
Layouts"); other sampled images use SHADER_READ_ONLY (depth: DEPTH_STENCIL_READ_ONLY). Before, a
write-then-sample chain (OoT's compute blur/mip passes: 26 storage-writing and 56 sampling
dispatches per frame) flipped GENERAL <-> SHADER_READ_ONLY at every use: a whole-image transition in
a full barrier per flip, which also ordered dispatches the game left unordered. NVN has no layouts;
the game's barrier orders the chain (`Op::Barrier` -> `HostMarkWritten`, flushed before the next
dispatch or pass), and without one the console does not order it either. Rules
(`BuildDescriptors`, `PrepareOutsidePass`, `FinishImages`): a sampled-only use records nothing for
an image already in its layout and never moves an image into GENERAL (that would note a GPU write;
when loading the contents moved the image out of GENERAL it is sampled read-only); uses of one
image in two layouts merge to GENERAL (`DrawPlan::UseImage`) and every descriptor of the image
names the plan's layout (`DrawPlan::LayoutOf`; a sampled descriptor of an image also bound for
storage said SHADER_READ_ONLY while the image was in GENERAL). Rows: `image layout transitions`,
`sampled bindings left in GENERAL`.

## Newgame "cracked tunnel" (M2 blocker) was a key-script timing artifact

The grey, cracked, high-contrast frames at presents ~4,201-6,841 of the newgame run are DK's
intro cinematic (a rock wall with a punched crater, "Y Punch" prompt, HUD hidden), which waits for
the Y press (SDL scancode 27 in `key_script_newgame.txt`). That script is wall-clock based: the Y
block (72-98 s) was written for the legacy path, which reaches the cinematic at present ~3,841
(77 s); the HLE reaches it at present 4,201 (~99 s), after the last Y press, so it never advances.
With Y pressed from 102 s the run goes crater -> flying rocks -> DK close-ups -> lit tunnel with
HUD and the animated hint-video panel (presents ~5,600+). `C:\eg\tools\key_script_newgame_hle.txt`
presses Y every 2 s from 72 s to 126 s, so it works on both timelines.

Diagnostics added for this (env-gated): `EDEN_NX2_NVN_THREADS=1` (heartbeat dumps every guest
thread: state, pc/lr with the thunk slot name, frame-pointer backtrace, and the NVN call counts
since the last line; it also turns the per-slot call counters on), `EDEN_NX2_NVN_DUMP_COUNT/
_DUMP_PASS/_DUMP_DRAW=<n>` (frame ranges, per-pass dumps, CPU view of a draw's buffers).

## Replay recording cost (`nvn_exec.cpp`, `nvn_draw*.cpp`)

ETW of the `NvnReplay` thread in the DK town (about 2,260 draws and 9,500 NVN calls per frame)
showed the cost in per-draw lookups, not in Vulkan. What was removed, with the invariant that keeps
each cache exact:

- **Early submission.** `Recorder::MaybeSubmitEarly` (checked every 32 ops, between ops) ends the
  command buffer, submits every segment recorded so far and starts a new one once the piece has
  taken `EDEN_NX2_NVN_EARLY_FIRST_US` (300 µs, the first piece) or `EDEN_NX2_NVN_EARLY_SUBMIT_US`
  (1,500 µs, later ones; 0 turns it off) of recording, at a point where no render pass is open
  (`EDEN_NX2_NVN_EARLY_PASS_US`, default 4,000: a pass that is still open that long is ended). The
  GPU then runs the first part of the frame while the rest is recorded; before, it idled for the
  whole recording of the submission. It is the `WaitSync`/`FlushAndWait` segment machinery: every
  segment signals its own timeline value in submission order, `MarkSubmitted` still publishes the
  last one and the binary present signal is only on the last segment. `SubmitSegments` flushes only
  the arena/staging chunks it has not flushed yet. A recorder that fails after an early piece
  waits for the pieces before it lets the flight go (`Recorder::Abort`). Counted in the log line
  `[nvn] early submits: N ...` at shutdown.
- **Buffer range lookups** (`Impl::ResolveBuffer`): 16 recent results, valid while
  `pool_generation` is unchanged, each limited to the part of its pool in which no other pool
  starts (the address map's answer is the same for every address in it). `EnsureBacking` returns
  before reading the clock once a pool is backed.
- **Texture of descriptor id** (`DrawContext::TextureOfId`): direct-mapped slots keyed by (id,
  serial) holding the `textures` node; `Impl::texture_epoch` is bumped when a texture is erased.
  The draw plan holds raw `Texture*` for its image bindings (the textures outlive the draw).
- **Views** (`DrawResources::GetView`): the texture remembers its `texture_views` node
  (`views_epoch`, bumped by `TakeViews`).
- **`NoteUse`**: 256 recent (serial, seq) pairs in front of the `last_use` map.
- **`StorageNeedsCommands`**: the storage remembers its `storage_index` node
  (`storage_index_epoch`, bumped when a pool's index is evicted).
- **Pipeline lookup** (`ResolvePipeline`): 8 recent (program serials, raw fixed state, pass formats)
  -> pipeline entries; a draw whose state words were set again to the values of an earlier draw
  skips `MakeKeyState`, the key hash and the cache map (texture-type queries are still checked).
- **Constant banks** (`BuildDescriptors`): the graphics driver (c0) and handle (c2) banks are built
  once per submission and per change of what they hold (storage buffer table of the stage, push
  constants, handle tables, raw global table), tracked by `bank_clock` generations that
  `BindBuffers` / `BindHandles` / `PushConstantsOp` bump; the arena copy is reused by the draws in
  between (it was a 4 KiB + 2.4 KiB memset and refill per stage and draw). Compute banks are still
  built per dispatch.
- **Guest side** (`recorder.cpp`): `Submission::{texture,sampler}_pool_snapshot` are `IdSnapshot`s
  sharing the recorder's immutable table (it was copied at every submit), deferred bind arrays are
  read through the host pointer, and the expansion vector is reserved. Texture-pool registrations
  follow the object at their guest address across `TextureInitialize`: reinitialization replaces the
  base and view serials in future snapshots, while snapshots already submitted stay unchanged.

## Pass state tracking (`nvn_pass_state.h`, tests `src/tests/nvn/pass_state.cpp`)

Inside an open render pass only draws record commands, so `DrawContext::bound_state` remembers what the
pass has recorded: the pipeline layout of the last bound pipeline, the push constants, the push
descriptor set (exact comparison of the write shapes, buffer infos and image infos, never a hash), the
vertex buffer of every stream and the index buffer. A draw skips what is already there (counters
`draw state elided: ...`). Rules: the state is `Reset` at every pass begin, pass end and command buffer
start (compute, copies, masked clears and blits record outside a pass and may disturb anything);
descriptors are skipped only while the bound pipeline layout is the same object (a different layout
object disturbs a push set); push constants survive a pipeline bind whose layout has the same push
constant range (every graphics layout of the replay has the same single range) and are dropped by one
without it; generated indices bind their own index buffer and forget the tracked one.
`EDEN_NX2_NVN_NO_STATE_ELIDE=1` records everything again (A/B). City measurement (DK Ingot Isle, 3.2.0,
user settings): on the Batch C tree the whole run was 4.4 fps with and without the tracking (the
regression of that tree dominated), so the speed effect is not demonstrated; the draw cost per call
(1.3-1.6 us in the shadow passes) is still open, see the gpuprof numbers above.

## Vertex staging (`nvn_vertex_stage.h`, `nvn_vertex_stage_gpu.h`, `DrawContext::StageVertexRanges`, tests `pass_state.cpp`)

Guest geometry lives in imported host memory (`Pool::Backing::Imported`), so every vertex fetch crosses
PCIe. Measured at DK's Ingot Isle city spot (3.2.0, RTX 5090, `EDEN_NX2_NVN_GPU_PROFILE=1`, run dirs
`F:/eg/results/hle-shadow-draw/out-diag1|2`, measurement-only switches on branch
`diag/hle-shadow-draw`): the depth-only (shadow map) passes take 7.3-7.5 ms of a 25.7 ms GPU frame for
3,840 draws and 2.04 M vertices (1.9-2.0 us per draw). Split by diagnostic modes, per frame:
- the replay's own commands (pipeline/descriptor/vertex/index binds, push constants, dynamic state),
  every draw recorded with zero instances: 0.05 ms;
- rasterisation and the fragment shader (empty scissor): no change (7.49 ms);
- uniform buffers, or uniform and index buffers, read from device-local copies: no change (7.62 / 7.54);
- vertex buffers read from device-local copies: **1.20 ms** (frame 26.1 -> 20.5 ms);
- three indices per draw (the fixed cost of a draw): 1.40 ms.

So ~6.2 ms of the 7.4 ms is the vertex attribute fetch from host memory, not DK's vertex work and not
the replay's per-draw commands. DK binds a vertex range of ~3.9 KiB per draw (one per mesh). With the
vertex buffers in VRAM the index fetch is the next host read: CostSplit (`vram` mode: uniform, vertex
and index data in VRAM) takes the depth passes from 7.24 to 0.1 ms where vertex data alone leaves 1.2.

**What is staged.** When a pass begins (`BeginPass`, after the barrier that orders everything recorded
before), the replay scans the records ahead in the command list (`DrawHost::HostRecordsAhead`) up to
the first op that ends a pass (`OpKeepsPass`), a SetRenderTargets (a later pass stages its own ranges)
or the list's End (`ScanPassVertexRanges`), collecting the vertex buffer ranges bound when the pass
begins and by every BindVertexBuffers on the way, and (index staging) the index range (address,
count x index size) of every direct DrawIndexed that binds its guest indices
(`VertexStageBindsIndices`): not line loops, quads, quad strips, or 8-bit indices on a device without
`VK_EXT_index_type_uint8`, whose indices are converted into generated ones, and not indirect draws,
whose count is on the GPU. With at least 32 draws ahead, the ranges of imported pools (at most 1 MiB
each, 64 MiB per pass, not under a live storage shadow) are staged. Vertex and index ranges are bytes
alike: one table keyed by guest address (`staged_vertex`) serves both, and a draw binds the copy of a
stream, or of its index buffer, whose start address was staged with at least the draw's size (a
converted draw never binds the guest indices, so it never reads an index copy).

**How it is copied.** Ranges of a pool up to 4 KiB apart share one copy region (`CoalesceVertexStageSources`;
the gap bytes copied stay within the bytes of the ranges themselves). The regions are carved from the
storage-shadow chunks of the submission (`AllocShadow`, the chunk offset keeps the source offset modulo
256, so attribute and index alignment hold; 32 free chunks are kept, a busy frame stages 20-40 MiB;
the chunks carry VERTEX_BUFFER and INDEX_BUFFER usage). The copy is a compute gather
(`VertexStageGpu`, `host_shaders/nvn_vertex_stage.comp`): one workgroup of 256 invocations per task of
at most 4096 16-byte units, one dispatch per run of regions with the same source and destination
buffer, the task table in the arena; one COMPUTE (or TRANSFER) -> VERTEX_INPUT barrier (attribute and
index reads) follows.
`vkCmdCopyBuffer` is used only for a source window the shader cannot bind (or a device without push
descriptors). Why a shader and not a transfer: copies
from host memory cost per byte (~0.15 ms/MiB + 0.7 us/region on an idle GPU); while another process
shared the GPU and its PCIe link `vkCmdCopyBuffer` slowed 3-5x (0.4-0.75 ms/MiB, 13-17 ms per present
for DK, a net loss of 4-6 fps), shader loads of the same memory ~1.4x (many more reads in flight).

**Reuse across passes.** A copy serves later passes of the same command buffer (DK draws the same meshes
into several shadow cascades) as long as nothing recorded since can have written its bytes: copies are
dropped at every pass end that is not a pure render target / image change, after a pass that wrote
through shaders or has counter reports to write (`StagedVertexSurvives`), at every op recorded outside
a pass and at every command buffer start; a pool write the replay records drops the copies it overlaps
(`Recorder::NotePoolWrite` -> `DrawContext::OnPoolWrite`).

**Why it is exact.** The copies read the same bytes in the same stream position as the draws (nothing
runs between the copy and the pass); an index copy holds exactly the bytes the draw's index fetch
would read (its start address and at least count x index size), so the draw reads the same indices.
Inside the pass only the pass's draws can write memory; once one of
them writes through shaders (`pass_shader_writes`) the remaining draws read the pools directly again,
vertex and index buffers alike
(NVN orders a shader write before a later vertex fetch only through a Barrier, which ends such a pass).
Counter reports landing in a range a draw reads end the pass once a game barrier ordered them
(`HostReportPending`, `pass_reports_ordered`; before that the copy holds the bytes the pool holds too),
and a range under a live storage shadow is not staged (a draw reading it ends the pass and copies the
shadow back first).
Guest CPU writes a later pass must see are ordered before it through a wait the replay passes before
recording that pass, and a command buffer is submitted only after it is recorded, so reused copies read
those writes as well. Inline uniform updates are applied at pass end both with and without staging.

Counters (unsupported TSV): `vertex staging: passes / ranges copied / copy regions / KiB copied /
ranges and KiB reused from an earlier pass / vertex buffer binds and index buffer binds from a staged
copy`, `shadow chunks: created / destroyed`; GPU profiler tag `copy vertex-stage`.

Vertex staging (the numbers below were measured without index staging) is on by default;
`EDEN_NX2_NVN_VERTEX_STAGE=0` turns it off, index staging with it. DK city, user settings, interleaved 2 reps
each (`F:/eg/results/vertex-stage3/out-excl`), exclusive, off / on: GPU busy 24.9 / 17.1 ms per
present, depth-only passes 7.14 / 2.62 ms, all passes 19.4 / 10.1 ms, copy class 0.54 / 2.2 ms (22.3
MiB staged per present, 2.6 k ranges in 1.1 k regions, 1.4 MiB reused from earlier passes), fps 30.0 /
31.8, p99 33.8 / 33.8. Sharing the GPU with other emulators the result depends on the load: with
incidental load (`F:/eg/results/vertex-stage2/out-loaded-gpu`) on was never slower (fps 30.0-30.8 vs
22.7-30.3, GPU busy 17.6-23.4 vs 22.0-34.5 ms); under a heavy load (one rep each,
`F:/eg/results/vertex-stage3/out-loaded`) the copy took ~3.9 ms for 2.65 ms of depth time saved: fps
19.7 vs 20.3, p99 77 vs 68 ms, GPU busy 44.9 vs 42.6 ms. That is the known trade-off: the game alone on
the GPU (the normal case) gains ~7.8 ms of GPU time per present; a second emulator on the same GPU can
make the copy cost more than it saves (~3% fps in that one rep). Pictures at the same frames match
within run-to-run noise (`imgcompare.py ab`, `F:/eg/results/vertex-stage3/imgab`, `imgab2`).

**Span window (`EDEN_NX2_TRY_STAGE_SPAN=1`, candidate, off by default).** The 32-draw minimum is per scan window,
and a window ends at the pass's first SetRenderTargets: OoT (VTX-HLE) runs ~150 passes per present, ~140 of them
under 32 draws (GPU profile, river: passes of >= 32 draws 2.4 ns/vertex and 3.8 us/draw, passes of 8-32 draws
5.6 ns/vertex and 10.8 us/draw, shadow passes of 5-30 draws up to 24 ns/vertex), so almost none staged (7.7 passes
per present) and the rest read vertices and indices over PCIe. With the switch the scan goes on past render target
changes (`SpanWindow`: copies survive those breaks, `StagedVertexSurvives`), stops behind the first one at a
Barrier, a render condition or a CallList, and counts the draws of the whole window; a pass beginning inside an
already scanned window (`stage_covered_begin/end`, reset whenever copies are dropped or overwritten) does not scan
again. A mesh indirect argument conversion break (reads pool memory, writes only its arena) no longer drops the
copies. Everything else (limits, exactness argument, copies reused by address) is as above.

Index staging is on by default too (with vertex staging); `EDEN_NX2_NVN_INDEX_STAGE=0` turns only it
off. Same plan, vertex staging on, index staging off / on, interleaved 2 reps each
(`F:/eg/results/index-stage/out-ab`), exclusive: GPU busy 16.95 / 13.63 ms per present, depth-only
passes 2.56 / 0.89 ms, all passes 10.0 / 6.96 ms, copy class 2.17 / 2.25 ms (22.3 / 24.3 MiB staged
per present, ranges 2.7 k / 4.2 k, ~4.0 k index binds per present from a copy), fps 32.6 / 32.6 (DK's
own pacing, rep spread 4.4), p99 34.3 / 33.8 ms. The index ranges are small (~2 MiB per present) and
ride in the same gather, so the copy class barely grows. Pictures at the same frames differ between the
arms no more than between two reps of one arm (`F:/eg/results/index-stage/imgab`: mean |luma| diff
3.7-7.9 across arms, 5.4-8.9 within an arm; DK's pose and the animated fog).

## Feedback copies per target epoch (`PruneFreshFeedback`)

City measurement (DK Ingot Isle, main 6c5f2bfec8, whole run): 14.2 M draws, 1.25 M passes, of which 972 k
broke for a feedback copy (993 k colour feedback copies, 947 k of them "already copied since
SetRenderTargets"); only ~223 k passes were DK's own target changes. DK renders mip k of a texture while
sampling a view of ALL its mips (blur and probe chains: 720x408 5 levels, 128x128 cube arrays), so every
draw sampled "a render target of the draw" and copied it again. A copy is now skipped when the scratch
still holds the same source with the levels and layers the draw samples, in the same
`render_target_epoch`: the epoch is bumped by every target change, every Barrier op and every pass end
other than a feedback break (compute, copies, clears, uploads, reports...), so anything the guest can
order against the sampled texture invalidates the scratch; a source whose guest bytes changed
(`HostNeedsCommands`) is copied again. What the skip changes is the value a draw reads from a subresource
that EARLIER DRAWS OF THE SAME EPOCH wrote: the hardware gives no order for that without a barrier (the
texture cache is not coherent with ROP writes), and a game that needs it issues a Barrier, which bumps
the epoch. Counters: `feedback_copy: skipped (...)`, `...overlaps an attached subresource` / `disjoint`
(EDEN_NX2_NVN_FEEDBACK_LOG=1 logs the first 40 with the sampled view and the attached levels).

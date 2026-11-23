// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "core/hle/service/nx2_native_image_directory.h"

// NVN2 native images (nvnNativeImage*, nvnTextureInitializeFromNativeImage): a pool and a texture
// made out of an image another server created (nx2:gsv 429; Mario Kart World's GameChat camera
// pool is one, and the game aborts at start-up when it cannot be made).
//
// The SDK builds them on driver extensions the HLE does not provide: it asks nx2:gsv about the
// image (QueryNativeImage), makes an NvRm memory handle for it in guest code and hands that to
// nvnMemoryPoolBuilderSetNativeHandleNVX, a driver function that only the real driver can resolve
// back into memory. These bodies stand for those SDK functions themselves and do what they do
// with the facts nx2:gsv holds (Nx2NativeImageDirectory): the pool is the image's pages in guest
// memory and the texture is the 2D texture the SDK builds over it, both made through the HLE's own
// NVN bodies.
namespace Core::NvnHle::Hle {
namespace {

// NVNnativeImageBuilder, as the SDK's own setters (kept in guest code) store it.
constexpr u64 BuilderDevice = 0x00;
constexpr u64 BuilderHandle = 0x08;     ///< nn::gs::NativeImageHandle
constexpr u64 BuilderPermission = 0x0C; ///< NVNnativeImageMemoryPermission
constexpr u64 BuilderFormat = 0x10;

// NVNnativeImage: an NVNmemoryPool, then what nvnTextureInitializeFromNativeImage needs (the
// SDK's layout).
constexpr u64 ImageDevice = 0x100;
constexpr u64 ImageFormat = 0x108;
constexpr u64 ImageOffset = 0x110;     ///< storage offset in the pool (the first page's remainder)
constexpr u64 ImageDescriptor = 0x118; ///< the 0x80-byte GpuOunceUserImageDescriptor
constexpr u64 ImageFlags = 0x198;
constexpr u64 ImageEnd = 0x1A0;

// GpuOunceUserImageDescriptor words (gsv 428 reply).
constexpr size_t DescriptorValid = 0;   ///< byte 0x00 == 1
constexpr size_t DescriptorExtent = 2;  ///< (width - 1) | (height - 1) << 16
constexpr size_t DescriptorLayoutIs1 = 8; ///< byte 0x20 == 1
constexpr size_t DescriptorKind = 9;    ///< 1: RGBA formats, 2: the 0x99-0x9c formats
constexpr size_t DescriptorOffset = 10; ///< of the image in the memory the address names
constexpr size_t DescriptorSize = 11;   ///< bytes

constexpr u64 PageSize = 0x1000;
/// NVNmemoryPoolFlags CPU_CACHED | GPU_CACHED: the image's bytes are guest memory other servers
/// and the game both touch.
constexpr s32 PoolFlags = 0x4 | 0x20;
/// NVNtextureFlags the SDK passes, NVNtextureTarget 2D, NVNtextureSwizzle G, A.
constexpr u64 TextureFlags = 5;
constexpr u64 Target2D = 1;
constexpr u64 SwizzleG = 3;
constexpr u64 SwizzleA = 5;
constexpr u64 SwizzleR = 2;
constexpr u64 SwizzleB = 4;

/// Scratch builders live below the caller's stack pointer: nothing else uses those bytes while the
/// body runs (the SDK builds them on its stack the same way).
constexpr u64 BuilderScratch = 0x200;

using Descriptor = std::array<u32, 32>;

void nvnNativeImageInitialize(Call& c) {
    const u64 image = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const u64 device = c.Load<u64>(builder + BuilderDevice);
    const u32 handle = c.Load<u32>(builder + BuilderHandle);
    const u32 permission = c.Load<u32>(builder + BuilderPermission);
    const u32 format = c.Load<u32>(builder + BuilderFormat);

    const auto info = Service::Nx2NativeImageDirectory::Get().Resolve(handle);
    if (!info) {
        LOG_WARNING(Core, "[nvn] nvnNativeImageInitialize: native image handle {} is unknown to "
                          "nx2:gsv ({} sessions asked), refused",
                    handle, Service::Nx2NativeImageDirectory::Get().SessionCount());
        c.ret.SetBool(false);
        return;
    }
    // The SDK refuses a descriptor that does not describe a user image with a size.
    const Descriptor& desc = info->descriptor;
    if ((desc[DescriptorValid] & 0xFF) != 1 || (desc[DescriptorLayoutIs1] & 0xFF) != 1 ||
        desc[DescriptorKind] == 0 || desc[DescriptorSize] == 0) {
        LOG_WARNING(Core, "[nvn] nvnNativeImageInitialize: handle {} has an unusable descriptor, "
                          "refused",
                    handle);
        c.ret.SetBool(false);
        return;
    }

    // The pool is the pages that hold the image; the image starts `in_page` bytes into the first.
    const u64 address = info->address + desc[DescriptorOffset];
    const u64 storage = address & ~(PageSize - 1);
    const u64 in_page = address - storage;
    const u64 pool_size = ((address + desc[DescriptorSize] - 1) | (PageSize - 1)) - storage + 1;

    const u64 pool_builder = c.ctx.sp - BuilderScratch;
    c.CallNvn("nvnMemoryPoolBuilderSetDefaults", {pool_builder});
    c.CallNvn("nvnMemoryPoolBuilderSetDevice", {pool_builder, device});
    c.CallNvn("nvnMemoryPoolBuilderSetStorage", {pool_builder, storage, pool_size});
    c.CallNvn("nvnMemoryPoolBuilderSetFlags", {pool_builder, static_cast<u64>(PoolFlags)});
    if ((c.CallNvn("nvnMemoryPoolInitialize", {image, pool_builder}) & 0xFF) == 0) {
        LOG_WARNING(Core, "[nvn] nvnNativeImageInitialize: the pool at {:#x}+{:#x} was refused",
                    storage, pool_size);
        c.ret.SetBool(false);
        return;
    }

    c.Zero(image + ImageDevice, ImageEnd - ImageDevice);
    c.Store<u64>(image + ImageDevice, device);
    c.Store<u32>(image + ImageFormat, format);
    c.Store<u64>(image + ImageOffset, in_page);
    c.Store<Descriptor>(image + ImageDescriptor, desc);
    c.Store<u32>(image + ImageFlags, info->flags);
    LOG_INFO(Core, "[nvn] native image {} -> pool {:#x}+{:#x} (image at +{:#x}, {}x{}, format {:#x}, "
                   "permission {})",
             handle, storage, pool_size, in_page, (desc[DescriptorExtent] & 0xFFFF) + 1,
             (desc[DescriptorExtent] >> 16) + 1, format, permission);
    c.ret.SetBool(true);
}

void nvnNativeImageFinalize(Call& c) {
    const u64 image = c.args.NextPointer();
    c.CallNvn("nvnMemoryPoolFinalize", {image});
    c.Zero(image + ImageDevice, ImageEnd - ImageDevice);
}

/// The 2D texture over the image the SDK builds: the formats its layout kind allows, the
/// extent from the descriptor, the swizzle that makes the stored order read as RGBA.
void nvnTextureInitializeFromNativeImage(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 image = c.args.NextPointer();
    const u64 device = c.Load<u64>(image + ImageDevice);
    const u32 format = c.Load<u32>(image + ImageFormat);
    const u64 offset = c.Load<u64>(image + ImageOffset);
    const Descriptor desc = c.Load<Descriptor>(image + ImageDescriptor);

    if ((desc[DescriptorValid] & 0xFF) != 1 || (desc[DescriptorLayoutIs1] & 0xFF) != 1) {
        c.ret.SetBool(false);
        return;
    }
    u64 red = 0;
    u64 blue = 0;
    if (desc[DescriptorKind] == 1) {
        // 0x25, 0x37, 0x38 and 0x6d
        const u32 index = format - 0x37;
        if (format != 0x25 && !(index <= 0x36 && ((u64{1} << index) & 0x0040000000000003ull) != 0)) {
            c.ret.SetBool(false);
            return;
        }
        red = SwizzleR;
        blue = SwizzleB;
    } else if (desc[DescriptorKind] == 2) {
        if (format - 0x99 > 3) {
            c.ret.SetBool(false);
            return;
        }
        red = SwizzleB;
        blue = SwizzleR;
    } else {
        c.ret.SetBool(false);
        return;
    }

    const u64 builder = c.ctx.sp - BuilderScratch;
    c.CallNvn("nvnTextureBuilderSetDefaults", {builder});
    c.CallNvn("nvnTextureBuilderSetDevice", {builder, device});
    c.CallNvn("nvnTextureBuilderSetFlags", {builder, TextureFlags});
    c.CallNvn("nvnTextureBuilderSetTarget", {builder, Target2D});
    c.CallNvn("nvnTextureBuilderSetSize2D",
              {builder, (desc[DescriptorExtent] & 0xFFFF) + u64{1}, (desc[DescriptorExtent] >> 16) + u64{1}});
    c.CallNvn("nvnTextureBuilderSetFormat", {builder, format});
    c.CallNvn("nvnTextureBuilderSetSwizzle", {builder, red, SwizzleG, blue, SwizzleA});
    c.CallNvn("nvnTextureBuilderSetStorage", {builder, image, offset});
    c.ret.SetBool((c.CallNvn("nvnTextureInitialize", {texture, builder}) & 0xFF) != 0);
}

} // namespace

void RegisterNativeImageFunctions(Registry& registry) {
    registry.AddNv<&nvnNativeImageInitialize>(
        "_Z24nvnNativeImageInitializeP14NVNnativeImagePK21NVNnativeImageBuilder");
    registry.AddNv<&nvnNativeImageFinalize>("_Z22nvnNativeImageFinalizeP14NVNnativeImage");
    registry.AddNv<&nvnTextureInitializeFromNativeImage>(
        "_Z35nvnTextureInitializeFromNativeImageP10NVNtextureP14NVNnativeImage");
}

} // namespace Core::NvnHle::Hle

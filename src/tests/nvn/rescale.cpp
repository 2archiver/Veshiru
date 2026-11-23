// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The Resolution setting on the NVN replay (nvn_rescale.h): eligibility, the pass rule, the
// scaling of viewports / scissors / clears / copies, the state of a guest storage and its twin, the
// rescaling words, the guest-visible sizes, and what the recompiler's RescalingPass does to the
// instructions the replay's push constants steer. Pure logic, no device.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_plan.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_pass_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_rescale.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/upscaler.h"

namespace {

using namespace Vulkan::Nvn;
using Rescale::Factor;
using Rescale::Reason;
using Rescale::Shape;
using Rescale::State;

constexpr Factor X1{1, 0};
constexpr Factor X2{2, 0};
constexpr Factor X3_2{3, 1};
constexpr Factor X1_2{1, 1};
constexpr Factor X3_4{3, 2};

/// A 864x480 colour render target: what OoT's scene is at its dynamic resolution floor.
Shape SceneColor() {
    Shape shape;
    shape.two_d = true;
    shape.width = 864;
    shape.height = 480;
    return shape;
}

Shape SceneDepth() {
    Shape shape = SceneColor();
    shape.color = false;
    return shape;
}

Reason ReasonOf(const Factor& factor, auto edit) {
    Shape shape = SceneColor();
    edit(shape);
    return Rescale::Classify(factor, shape);
}

} // namespace

TEST_CASE("Rescale: scaled sizes are the emulated GPU's, for every Resolution setting",
          "[nvn][rescale]") {
    using Settings::ResolutionSetup;
    for (const auto setup :
         {ResolutionSetup::Res1_4X, ResolutionSetup::Res1_2X, ResolutionSetup::Res3_4X,
          ResolutionSetup::Res1X, ResolutionSetup::Res5_4X, ResolutionSetup::Res3_2X,
          ResolutionSetup::Res2X, ResolutionSetup::Res3X, ResolutionSetup::Res4X,
          ResolutionSetup::Res5X, ResolutionSetup::Res6X, ResolutionSetup::Res7X,
          ResolutionSetup::Res8X}) {
        Settings::ResolutionScalingInfo info;
        Settings::TranslateResolutionInfo(setup, info);
        const Factor factor{info.up_scale, info.down_shift};
        CHECK(factor.Active() == info.active);
        for (const u32 size : {0u, 1u, 3u, 480u, 853u, 854u, 864u, 1080u, 2560u}) {
            CHECK(factor.Scale(size) == info.ScaleUp(size));
        }
        CHECK(factor.UpFactor() == info.up_factor);
        CHECK(factor.DownFactor() == info.down_factor);
    }
    CHECK(X2.Scale(864) == 1728);
    CHECK(X2.Scale(480) == 960);
    CHECK(X1_2.Scale(1) == 1); // never 0 for a non-zero size
    CHECK(X3_4.Scale(480) == 360);
    CHECK_FALSE(X1.Active());
    // The one the game is running at, as the replay reads it.
    const auto saved = Settings::values.resolution_info;
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res3_2X,
                                      Settings::values.resolution_info);
    CHECK(Rescale::CurrentFactor() == X3_2);
    Settings::values.resolution_info = saved;
}

TEST_CASE("Rescale: eligibility mirrors the emulated texture cache", "[nvn][rescale]") {
    CHECK(Rescale::Classify(X2, SceneColor()) == Reason::Eligible);
    CHECK(Rescale::Classify(X2, SceneDepth()) == Reason::Eligible);
    // 1X: nothing is a scaled image.
    CHECK(Rescale::Classify(X1, SceneColor()) == Reason::Inactive);
    CHECK(ReasonOf(X2, [](Shape& s) { s.two_d = false; }) == Reason::NotTwoD);
    CHECK(ReasonOf(X2, [](Shape& s) { s.linear = true; }) == Reason::Linear);
    CHECK(ReasonOf(X2, [](Shape& s) { s.alias = true; }) == Reason::Alias);
    CHECK(ReasonOf(X2, [](Shape& s) { s.sparse = true; }) == Reason::Sparse);
    CHECK(ReasonOf(X2, [](Shape& s) { s.transcoded = true; }) == Reason::Transcoded);
    CHECK(ReasonOf(X2, [](Shape& s) { s.levels = 2; }) == Reason::Levels);
    CHECK(ReasonOf(X2, [](Shape& s) { s.samples = 4; }) == Reason::Multisampled);
    CHECK(ReasonOf(X2, [](Shape& s) { s.blit_ok = false; }) == Reason::NoBlit);
    // RescaleHeightThreshold: a colour image of at most 288 rows is not a picture.
    CHECK(ReasonOf(X2, [](Shape& s) { s.height = 288; }) == Reason::TooSmall);
    CHECK(ReasonOf(X2, [](Shape& s) { s.height = 289; }) == Reason::Eligible);
    // A depth image has no such floor (the emulated GPU's depth rule); the pass rule decides.
    CHECK(ReasonOf(X2, [](Shape& s) {
              s.color = false;
              s.height = 64;
          }) == Reason::Eligible);
    // The scaled size must fit the device's framebuffer.
    CHECK(ReasonOf(X2, [](Shape& s) { s.max_dimension = 1727; }) == Reason::TooLarge);
    CHECK(ReasonOf(X2, [](Shape& s) { s.max_dimension = 1728; }) == Reason::Eligible);
    CHECK(ReasonOf(X2, [](Shape& s) { s.max_dimension = 959; }) == Reason::TooLarge);
    // Downscales are eligible too (0.5X renders 432x240 from 864x480).
    CHECK(Rescale::Classify(X1_2, SceneColor()) == Reason::Eligible);
    CHECK(std::string_view{Rescale::ReasonName(Reason::NoBlit)}.size() > 0);
}

TEST_CASE("Rescale: a pass is scaled only when every attachment is eligible", "[nvn][rescale]") {
    Rescale::SceneSizes scene;
    const std::array<Shape, 1> color{SceneColor()};
    const Shape depth = SceneDepth();
    const auto scaled = [&](const Factor& factor, std::span<const Shape> colors, const Shape* z) {
        return Rescale::PassScaled(factor, colors, z, scene);
    };
    // The scene's colour (and depth).
    CHECK(scaled(X2, color, nullptr));
    CHECK(scaled(X2, color, &depth));
    CHECK_FALSE(scaled(X1, color, &depth));
    // One attachment that cannot scale makes the whole pass guest-sized.
    Shape small = SceneColor();
    small.height = 200;
    const std::array<Shape, 2> mixed{SceneColor(), small};
    CHECK_FALSE(scaled(X2, mixed, nullptr));
    Shape no_blit_depth = SceneDepth();
    no_blit_depth.blit_ok = false;
    CHECK_FALSE(scaled(X2, color, &no_blit_depth));
    // Nothing attached: a pass without attachments is guest-sized.
    CHECK_FALSE(scaled(X2, {}, nullptr));
    // A depth-only pass (a pre-pass, a shadow map) scales only a depth of the scene's size.
    CHECK_FALSE(scaled(X2, {}, &depth));
    scene.Add(864, 480);
    CHECK(scaled(X2, {}, &depth));
    Shape shadow_map = SceneDepth();
    shadow_map.width = 4096;
    shadow_map.height = 4096;
    CHECK_FALSE(scaled(X2, {}, &shadow_map));
    // With a colour target of the same pass the depth is the scene's whatever its size.
    CHECK(scaled(X2, color, &shadow_map));
    CHECK(scene.Contains(864, 480));
    CHECK_FALSE(scene.Contains(480, 864));
    CHECK(scene.Size() == 1);
}

TEST_CASE("Rescale: viewports, scissors and clears scale with the pass", "[nvn][rescale]") {
    // A viewport: every number scales; a downscale rounds to whole pixels like the emulated GPU's.
    const VkViewport viewport{10.0f, 20.0f, 100.0f, -50.0f, 0.0f, 1.0f};
    const VkViewport doubled = Rescale::ScaleViewport(X2, viewport);
    CHECK(doubled.x == 20.0f);
    CHECK(doubled.y == 40.0f);
    CHECK(doubled.width == 200.0f);
    CHECK(doubled.height == -100.0f);
    CHECK(doubled.minDepth == 0.0f);
    CHECK(doubled.maxDepth == 1.0f);
    const VkViewport halved = Rescale::ScaleViewport(X1_2, VkViewport{3.0f, 5.0f, 101.0f, -51.0f, 0.2f, 0.9f});
    CHECK(halved.x == 2.0f);
    CHECK(halved.y == 3.0f);
    CHECK(halved.width == 51.0f);
    CHECK(halved.height == -26.0f);
    CHECK(halved.minDepth == 0.2f);
    CHECK(halved.maxDepth == 0.9f);

    // A scissor / clear rectangle: the scaled pixels the guest rectangle covers.
    CHECK(Rescale::ScaleRect(X2, {16, 32, 100, 50}) == Rescale::Rect{32, 64, 200, 100});
    CHECK(Rescale::ScaleRect(X2, {0, 0, 864, 480}) == Rescale::Rect{0, 0, 1728, 960});
    // Neighbouring rectangles still tile at a fractional scale (the shared edge may overlap by a
    // pixel, never leave a gap), and a one pixel rectangle never vanishes at a downscale.
    const Rescale::Rect left = Rescale::ScaleRect(X3_2, {0, 0, 333, 100});
    const Rescale::Rect right = Rescale::ScaleRect(X3_2, {333, 0, 531, 100});
    CHECK(left.x == 0);
    CHECK(static_cast<s32>(left.width) + left.x >= right.x);
    CHECK(right.x + static_cast<s32>(right.width) == 1296);
    for (s32 x = 0; x < 8; ++x) {
        CHECK(Rescale::ScaleRect(X1_2, {x, x, 1, 1}).width == 1);
        CHECK(Rescale::ScaleRect(X1_2, {x, x, 1, 1}).height == 1);
    }
    // An empty rectangle stays empty.
    CHECK(Rescale::ScaleRect(X2, {5, 5, 0, 0}).width == 0);
}

TEST_CASE("Rescale: a copy between two scaled images", "[nvn][rescale]") {
    // The whole image covers the whole scaled image.
    const Rescale::CopyRegion whole =
        Rescale::ScaleCopy(X2, {0, 0, 0, 0, 864, 480}, 1728, 960, 1728, 960);
    CHECK(whole == Rescale::CopyRegion{0, 0, 0, 0, 1728, 960});
    // Offsets scale like viewport origins, the extent like an image size.
    const Rescale::CopyRegion part =
        Rescale::ScaleCopy(X2, {16, 32, 48, 64, 100, 50}, 1728, 960, 1728, 960);
    CHECK(part == Rescale::CopyRegion{32, 64, 96, 128, 200, 100});
    // Clamped to what both scaled images hold.
    const Rescale::CopyRegion clamped =
        Rescale::ScaleCopy(X2, {16, 32, 48, 64, 100, 50}, 1728, 960, 150, 960);
    CHECK(clamped.width == 54);
    CHECK(clamped.height == 100);
    CHECK(Rescale::ScaleCopy(X2, {0, 0, 900, 0, 10, 10}, 1728, 960, 1728, 960).width == 0);
}

TEST_CASE("Rescale: which image of a storage holds the newest pixels", "[nvn][rescale]") {
    CHECK_FALSE(Rescale::NativeIsStale(State::Native));
    CHECK(Rescale::NativeIsStale(State::Scaled));
    CHECK_FALSE(Rescale::NativeIsStale(State::Both));
    CHECK_FALSE(Rescale::TwinIsCurrent(State::Native));
    CHECK(Rescale::TwinIsCurrent(State::Scaled));
    CHECK(Rescale::TwinIsCurrent(State::Both));
    // A write to one image makes the other stale; a blit makes them equal.
    CHECK(Rescale::AfterNativeWrite() == State::Native);
    CHECK(Rescale::AfterScaledWrite() == State::Scaled);
    CHECK(Rescale::AfterBlit() == State::Both);
    // So after a scaled pass a guest read needs the downscale, after the downscale it does not,
    // and a write of the guest-sized image brings the twin to need an upscale (it is not current).
    State state = State::Native;
    state = Rescale::AfterScaledWrite();
    CHECK(Rescale::NativeIsStale(state));
    state = Rescale::AfterBlit();
    CHECK_FALSE(Rescale::NativeIsStale(state));
    CHECK(Rescale::TwinIsCurrent(state));
    state = Rescale::AfterNativeWrite();
    CHECK_FALSE(Rescale::TwinIsCurrent(state));
}

TEST_CASE("Rescale: the words that tell a shader which bindings are twins", "[nvn][rescale]") {
    Rescale::Words words;
    CHECK_FALSE(words.Any());
    words.SetTexture(0);
    words.SetTexture(37);
    words.SetImage(3);
    words.SetImage(33);
    const Rescale::Words::Data data = words.Get();
    CHECK(data.size() == 6);
    CHECK(data[0] == 1u);
    CHECK(data[1] == 1u << 5);
    CHECK(data[2] == 0u);
    CHECK(data[3] == 0u);
    CHECK(data[4] == 1u << 3);
    CHECK(data[5] == 1u << 1);
    CHECK(words.TextureIsSet(37));
    CHECK_FALSE(words.TextureIsSet(36));
    CHECK(words.ImageIsSet(33));
    // Beyond the words the shader can read nothing is set.
    words.SetTexture(Rescale::TEXTURE_BITS);
    words.SetImage(Rescale::IMAGE_BITS);
    CHECK(words.Get() == data);
    CHECK_FALSE(words.TextureIsSet(Rescale::TEXTURE_BITS));
    words.Clear();
    CHECK_FALSE(words.Any());

    // The same layout as the emulated GPU's RescalingPushConstant (what the shader reads).
    Rescale::Words mine;
    Vulkan::RescalingPushConstant reference;
    for (u32 index = 0; index < 100; ++index) {
        const bool texture = index % 3 == 0;
        if (texture) {
            mine.SetTexture(index);
        }
        reference.PushTexture(texture);
    }
    // (the reference advances image bits in its own counter)
    for (u32 index = 0; index < 50; ++index) {
        const bool image = index % 5 == 1;
        if (image) {
            mine.SetImage(index);
        }
        reference.PushImage(image);
    }
    CHECK(mine.Get() == reference.Data());

    // A draw's plan starts with nothing scaled again.
    DrawPlan plan;
    plan.rescale_words.SetTexture(2);
    plan.Clear();
    CHECK_FALSE(plan.rescale_words.Any());
}

TEST_CASE("Rescale: push constants are only skipped when the words are the same", "[nvn][rescale]") {
    PassBoundState state;
    PassBoundState::PushWords scaled;
    scaled.words[0] = 0b101;
    scaled.down_factor = 0.5f;
    // The first draw of a pass pushes; a draw with the same words skips.
    CHECK_FALSE(state.TakePushConstants(false, scaled));
    CHECK(state.TakePushConstants(false, scaled));
    // Another texture of the draw is a twin: pushed again, then skipped.
    PassBoundState::PushWords other = scaled;
    other.words[0] = 0b111;
    CHECK_FALSE(state.TakePushConstants(false, other));
    CHECK(state.TakePushConstants(false, other));
    // The factor changes with the pass.
    other.down_factor = 1.0f;
    CHECK_FALSE(state.TakePushConstants(false, other));
    // A guest-sized pass pushes zeros and 1.0: what the replay pushed before scaling existed.
    state.Reset();
    CHECK_FALSE(state.TakePushConstants(false));
    CHECK(state.TakePushConstants(false));
    CHECK(state.TakePushConstants(false, PassBoundState::PushWords{}));
}

TEST_CASE("Rescale: the twin is host-only; guest-visible sizes stay native", "[nvn][rescale]") {
    Storage native;
    native.extent = {864, 480, 1};
    native.layers = 1;
    native.levels = 1;
    native.desc.target = static_cast<s32>(TexTarget::Dim2D);
    native.desc.width = 864;
    native.desc.height = 480;
    native.desc.levels = 1;
    native.layout.size = 864 * 480 * 4;
    native.vram_bytes = 864 * 480 * 4;
    native.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    native.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    native.create_flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

    const std::shared_ptr<Storage> twin = MakeScaledTwin(native, X2);
    // What the game can observe of its storage is untouched.
    CHECK(native.extent.width == 864);
    CHECK(native.extent.height == 480);
    CHECK(native.desc.width == 864);
    CHECK(native.desc.height == 480);
    CHECK(native.layout.size == 864u * 480u * 4u);
    CHECK(native.scale_state == State::Native);
    CHECK(native.scaled == nullptr); // attaching the twin is Impl::ScaledTwinOf's job
    // The twin: the scaled image, an owner, nothing to upload or compare with guest bytes.
    CHECK(twin->extent.width == 1728);
    CHECK(twin->extent.height == 960);
    CHECK(twin->extent.depth == 1);
    CHECK(twin->levels == 1);
    CHECK(twin->layers == 1);
    CHECK(twin->samples == VK_SAMPLE_COUNT_1_BIT);
    CHECK(twin->aspect == native.aspect);
    CHECK(twin->usage == native.usage);
    CHECK(twin->create_flags == native.create_flags);
    CHECK(twin->scale_owner == &native);
    CHECK(twin->contents_valid);
    CHECK(twin->origin == StorageOrigin::Gpu);
    CHECK_FALSE(twin->cached);
    CHECK(twin->scaled == nullptr);
    CHECK(twin->vram_bytes == 1728u * 960u * 4u);
    // The twin's description keeps the guest's formats and block info, whatever the size.
    CHECK(twin->desc.target == native.desc.target);
    CHECK(twin->desc.format == native.desc.format);

    // A 0.5X twin is smaller than the guest image.
    const std::shared_ptr<Storage> half = MakeScaledTwin(native, X1_2);
    CHECK(half->extent.width == 432);
    CHECK(half->extent.height == 240);
}

namespace {

/// A tiny shader IR: the instructions the rescaling pass patches, built by hand.
struct IrFixture {
    explicit IrFixture(Shader::Stage stage) {
        program.stage = stage;
        program.post_order_blocks.push_back(&block);
    }

    Shader::IR::Inst* Add(Shader::IR::Opcode op, std::initializer_list<Shader::IR::Value> args,
                          Shader::IR::TextureInstInfo info) {
        return &*block.PrependNewInst(block.end(), op, args, info.raw);
    }
    Shader::IR::Inst* Fetch(Shader::TextureType type, u32 index) {
        Shader::IR::IREmitter ir{block};
        coords = type == Shader::TextureType::ColorArray2D
                     ? ir.CompositeConstruct(ir.Imm32(3u), ir.Imm32(4u), ir.Imm32(1u))
                     : ir.CompositeConstruct(ir.Imm32(3u), ir.Imm32(4u));
        Shader::IR::TextureInstInfo info{};
        info.descriptor_index.Assign(index);
        info.type.Assign(type);
        return Add(Shader::IR::Opcode::ImageFetch,
                   {Shader::IR::Value{}, coords, Shader::IR::Value{}, ir.Imm32(0u),
                    Shader::IR::Value{}},
                   info);
    }
    Shader::IR::Inst* Dimensions(Shader::TextureType type, u32 index) {
        Shader::IR::IREmitter ir{block};
        Shader::IR::TextureInstInfo info{};
        info.descriptor_index.Assign(index);
        info.type.Assign(type);
        return Add(Shader::IR::Opcode::ImageQueryDimensions,
                   {Shader::IR::Value{}, ir.Imm32(0u), ir.Imm1(false)}, info);
    }
    [[nodiscard]] size_t Count(Shader::IR::Opcode op) {
        size_t count = 0;
        for (const Shader::IR::Inst& inst : block.Instructions()) {
            count += inst.GetOpcode() == op ? 1 : 0;
        }
        return count;
    }

    Shader::ObjectPool<Shader::IR::Inst> pool;
    Shader::IR::Block block{pool};
    Shader::IR::Program program;
    Shader::IR::Value coords;
};

/// Runs the rescaling pass the way a translation does with the setting at 2X.
struct ScaledSetting {
    ScaledSetting() : saved{Settings::values.resolution_info} {
        Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res2X,
                                          Settings::values.resolution_info);
    }
    ~ScaledSetting() {
        Settings::values.resolution_info = saved;
    }
    Settings::ResolutionScalingInfo saved;
};

} // namespace

TEST_CASE("Rescale: the recompiler's pass keeps a shader in guest pixels (fragment)", "[nvn][rescale]") {
    using Shader::IR::Opcode;
    const ScaledSetting setting;
    IrFixture f{Shader::Stage::Fragment};
    Shader::IR::IREmitter ir{f.block};
    const Shader::IR::F32 frag_x = ir.GetAttribute(Shader::IR::Attribute::PositionX);
    Shader::IR::Inst* const frag_x_inst = frag_x.Inst();
    const Shader::IR::F32 used = ir.FPMul(frag_x, frag_x); // so that the position has users
    (void)used;
    Shader::IR::Inst* const fetch = f.Fetch(Shader::TextureType::Color2D, 2);
    const Shader::IR::Value original_coords = f.coords;
    Shader::IR::Inst* const fetch_cube = f.Fetch(Shader::TextureType::ColorCube, 3);
    const Shader::IR::Value cube_coords = f.coords;
    Shader::IR::Inst* const dims = f.Dimensions(Shader::TextureType::Color2D, 2);
    (void)dims;

    Shader::Optimization::RescalingPass(f.program);

    // gl_FragCoord is multiplied by the down factor the replay pushes (1/scale in a scaled pass).
    CHECK(f.Count(Opcode::ResolutionDownFactor) >= 1);
    CHECK(f.Count(Opcode::FPMul32) >= 1);
    // The position instruction became an alias of the patched value (ReplaceUsesWith), so every
    // user reads position * down factor.
    CHECK(frag_x_inst->GetOpcode() == Opcode::Identity);
    CHECK(frag_x_inst->Arg(0).Inst()->GetOpcode() == Opcode::FPMul32);
    // A texel fetch of a 2D texture takes its coordinate from the descriptor's bit (a twin's
    // coordinate is the guest's scaled up, keeping the fragment's place inside its pixel).
    CHECK(f.Count(Opcode::IsTextureScaled) == 2); // the fetch and the size query
    CHECK(fetch->Arg(1).InstRecursive() != original_coords.InstRecursive());
    CHECK(f.Count(Opcode::FPFloor32) >= 2);
    // textureSize comes out in guest pixels again for a scaled texture.
    CHECK((f.Count(Opcode::SDiv32) + f.Count(Opcode::UDiv32)) >= 2);
    // A cube texture is not scaled: its fetch is untouched and owns no bit.
    CHECK(fetch_cube->Arg(1).InstRecursive() == cube_coords.InstRecursive());
}

TEST_CASE("Rescale: the recompiler's pass keeps a compute shader's reads in guest pixels",
          "[nvn][rescale]") {
    using Shader::IR::Opcode;
    const ScaledSetting setting;
    IrFixture f{Shader::Stage::Compute};
    Shader::IR::Inst* const fetch = f.Fetch(Shader::TextureType::Color2D, 0);
    const Shader::IR::Value original_coords = f.coords;
    Shader::IR::Inst* const array_fetch = f.Fetch(Shader::TextureType::ColorArray2D, 1);
    const Shader::IR::Value array_coords = f.coords;
    Shader::IR::Inst* const buffer_fetch = f.Fetch(Shader::TextureType::Color1D, 2);
    const Shader::IR::Value line_coords = f.coords;

    Shader::Optimization::RescalingPass(f.program);

    // The dispatch is not multiplied: a coordinate read becomes the scaled texel the invocation's
    // guest pixel starts, chosen by the descriptor's bit; there is no fragment position to use.
    CHECK(fetch->Arg(1).InstRecursive() != original_coords.InstRecursive());
    CHECK(array_fetch->Arg(1).InstRecursive() != array_coords.InstRecursive());
    CHECK(f.Count(Opcode::IsTextureScaled) == 2);
    CHECK(f.Count(Opcode::IMul32) >= 2);
    CHECK(f.Count(Opcode::SelectU32) >= 4);
    CHECK(f.Count(Opcode::GetAttribute) == 0);
    CHECK(f.Count(Opcode::FPFloor32) == 0);
    CHECK(f.Count(Opcode::ResolutionDownFactor) == 0); // a compute shader has no down factor
    // A 1D texture is not a picture: untouched.
    CHECK(buffer_fetch->Arg(1).InstRecursive() == line_coords.InstRecursive());
}

TEST_CASE("Rescale: OoT's scene twin is taken by the upscaler at 2X and 3X", "[nvn][rescale][dlaa]") {
    using VideoCore::Upscaler::Backend;
    using VideoCore::Upscaler::PromoteNearNativeToDlaa;
    using VideoCore::Upscaler::UpscaleInputFitsOutput;
    constexpr u32 out_w = 2560;
    constexpr u32 out_h = 1440;
    const auto twin_fits = [&](const Factor& factor, u32 width, u32 height) {
        const Rescale::Rect scaled = Rescale::ScaleRect(factor, {0, 0, width, height});
        return UpscaleInputFitsOutput(scaled.width, scaled.height, out_w, out_h);
    };
    // 2X: 1708x960 is DLSS Quality's input.
    CHECK(Rescale::ScaleRect(X2, {0, 0, 854, 480}).width == 1708);
    CHECK(twin_fits(X2, 854, 480));
    CHECK(twin_fits(X2, 953, 536)); // 1906x1072
    // 3X: 854x480 is 2562x1440, two pixels wider than the output. Its twin is evaluated (cropped
    // by DLAA), the user-reported case that fed DLSS the guest-sized 854x480 instead.
    const Rescale::Rect scaled_3x = Rescale::ScaleRect(Factor{3, 0}, {0, 0, 854, 480});
    CHECK(scaled_3x.width == 2562);
    CHECK(scaled_3x.height == 1440);
    CHECK(twin_fits(Factor{3, 0}, 854, 480));
    // A twin much larger than the output (953x536 and 1920x1080 at 3X) is still not evaluated;
    // so is one that is only wider (mixed overhang) or more than two pixels over.
    CHECK_FALSE(twin_fits(Factor{3, 0}, 953, 536));
    CHECK_FALSE(twin_fits(Factor{3, 0}, 1920, 1080));
    CHECK_FALSE(UpscaleInputFitsOutput(2562, 1000, out_w, out_h));
    CHECK_FALSE(UpscaleInputFitsOutput(2563, 1440, out_w, out_h));
    CHECK(UpscaleInputFitsOutput(2560, 1440, out_w, out_h));
    // A DLSS (or Auto) request whose input is the output's size or the two-pixel overhang is DLAA;
    // a real ratio keeps DLSS, and no other backend changes.
    CHECK(PromoteNearNativeToDlaa(Backend::Dlss, 2562, 1440, out_w, out_h) == Backend::Dlaa);
    CHECK(PromoteNearNativeToDlaa(Backend::Dlss, 2560, 1440, out_w, out_h) == Backend::Dlaa);
    CHECK(PromoteNearNativeToDlaa(Backend::Dlss, 1708, 960, out_w, out_h) == Backend::Dlss);
    CHECK(PromoteNearNativeToDlaa(Backend::Dlss, 2859, 1608, out_w, out_h) == Backend::Dlss);
    CHECK(PromoteNearNativeToDlaa(Backend::Fsr, 2562, 1440, out_w, out_h) == Backend::Fsr);
    CHECK(PromoteNearNativeToDlaa(Backend::Dlaa, 2562, 1440, out_w, out_h) == Backend::Dlaa);
}

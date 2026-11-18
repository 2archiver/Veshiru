// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace Common::Log {

/// Write UTF-8 to the inherited handle. No ANSI escapes are added, even for a console.
void WriteStandardStream(std::string_view text, bool to_stderr = false) noexcept;

/// Best-effort live output. A slow or unread redirect must not stall the file logger or teardown.
class StandardStreamSink {
public:
    StandardStreamSink();
    ~StandardStreamSink();
    void Enqueue(std::string text, bool to_stderr, bool important = false);
    void Flush(std::chrono::milliseconds timeout);
    void Close();

private:
    struct State;
    std::shared_ptr<State> state;
    std::thread writer;
};

} // namespace Common::Log

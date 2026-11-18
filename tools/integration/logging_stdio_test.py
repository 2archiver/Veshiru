#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def test(probe, root):
    completed = subprocess.run([probe, "normal"], cwd=root, capture_output=True, timeout=8,
                               check=True)
    assert "Booting café" in completed.stdout.decode("utf-8")
    assert b"<Error>" not in completed.stdout and b"early error" not in completed.stdout
    assert b"[Error]" in completed.stderr and b"early error" in completed.stderr
    assert b"\x1b" not in completed.stdout + completed.stderr
    assert b"orderly shutdown" in completed.stdout

    completed = subprocess.run([probe, "normal"], cwd=root, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=8, check=True)
    lines = completed.stdout.decode("utf-8").splitlines()
    times = [float(re.match(r"\[([\d.]+)s\]", line)[1]) for line in lines]
    assert len(times) == 4 and times == sorted(times)

    # Keep the read end open without draining: the console pipe fills but warns/errors still
    # reach the mirror and the logger returns. A separate check closes the read end entirely.
    for broken in (False, True):
        mirror = root / "mirror.log"
        mirror.unlink()
        with subprocess.Popen([probe, "flood"], cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as proc:
            try:
                if broken:
                    proc.stdout.close()
                assert proc.wait(timeout=8) == 0
            finally:
                if proc.poll() is None:
                    proc.kill()
        logged = mirror.read_text(encoding="utf-8")
        assert "<Warning>" in logged and "warning persisted before shutdown" in logged
        assert "<Error>" in logged and "early error" in logged
        assert "orderly shutdown" in logged
        assert logged.count("\n") <= 24_006

    if hasattr(os, "mkfifo"):
        os.mkfifo(root / "log-fifo")
        subprocess.run([probe, "fifo"], cwd=root, timeout=8, check=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="veshiru-stdio-") as tmp:
        test(Path(sys.argv[1]).resolve(), Path(tmp))
    print("Logging: UTF-8 redirects, flushing, unread/broken pipes, and shutdown passed")

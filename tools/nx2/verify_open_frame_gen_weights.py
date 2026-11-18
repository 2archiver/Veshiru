#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reproduce the compiled Nano v17 FP16 table from the pinned upstream checkpoint."""

import argparse
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from math import prod, sqrt
from pathlib import Path
import pickle
import re
import struct
from zipfile import ZipFile

CHECKPOINT_SHA256 = "6d006aa1abeda7038b3dfac476962968d58b9c052f17327c7d41d5b7fb721f46"
CHECKPOINT_SIZE = 403315
PACKED_SHA256 = "306d7d38f46a9cfdc2f27b65c145cff268d8fde7802b4fdd5c576054e5201feb"
CPP_TABLE = (Path(__file__).resolve().parents[2] /
             "src/video_core/renderer_vulkan/present/open_frame_gen_weights.cpp")
ZIP_ROOT = "nano_v17_best/"


def rebuild_contiguous(storage, offset, shape, stride, *_):
    fmt, data, count = storage
    if offset != 0 or count != prod(shape) or stride != tuple(
            prod(shape[i + 1:]) for i in range(len(shape))):
        raise ValueError("Unexpected checkpoint tensor layout")
    return shape, struct.unpack(f"<{count}{fmt}", data)


class CheckpointReader(pickle.Unpickler):
    """Only the four globals used in the hash-checked Nano v17 state dictionary."""

    def __init__(self, archive):
        self.archive = archive
        super().__init__(BytesIO(archive.read(ZIP_ROOT + "data.pkl")))

    def find_class(self, module, name):
        return {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_contiguous,
            ("torch", "FloatStorage"): "f",
            ("torch", "LongStorage"): "q",
        }[(module, name)]

    def persistent_load(self, value):
        kind, fmt, key, _, count = value
        if kind != "storage" or fmt not in ("f", "q") or not key.isdecimal():
            raise ValueError("Unexpected checkpoint storage")
        return fmt, self.archive.read(ZIP_ROOT + "data/" + key), count


def load_checkpoint(path):
    with path.open("rb") as source:
        data = source.read(CHECKPOINT_SIZE + 1)
    if len(data) != CHECKPOINT_SIZE or sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match the pinned upstream Nano v17 blob")
    with ZipFile(BytesIO(data)) as archive:
        return CheckpointReader(archive).load()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack_checkpoint(state):
    def tensor(key):
        return state[key][1]

    values = []
    for layer in ("e1", "e2", "e3", "btn", "d1", "d2", "d3"):
        for convolution, norm in ((0, 1), (3, 4)):
            conv = f"{layer}.{convolution}"
            bn = f"{layer}.{norm}"
            shape, weights = state[conv + ".weight"]
            bias = (tensor(conv + ".bias") if conv + ".bias" in state else (0.0,) * shape[0])
            alpha = [g / sqrt(v + 1e-5) for g, v in zip(tensor(bn + ".weight"),
                                                      tensor(bn + ".running_var"))]
            plane_size = prod(shape[1:])
            values.extend(f32(w * alpha[i // plane_size]) for i, w in enumerate(weights))
            values.extend(f32(f32(f32(b - mean) * a) + beta)
                          for b, mean, a, beta in zip(bias, tensor(bn + ".running_mean"),
                                                        alpha, tensor(bn + ".bias")))
    for layer in ("se.fc.0", "se.fc.2", "head.0", "head.2"):
        values.extend(tensor(layer + ".weight"))
        values.extend(tensor(layer + ".bias"))
    return struct.pack(f"<{len(values)}e", *values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--emit", action="store_true", help="Print the verified C++ initializer body")
    args = parser.parse_args()
    try:
        packed = pack_checkpoint(load_checkpoint(args.checkpoint))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    digest = sha256(packed).hexdigest()
    if digest != PACKED_SHA256:
        parser.error(f"Unexpected packed weight hash: {digest}")
    if args.emit:
        bits = [value for (value,) in struct.iter_unpack("<H", packed)]
        for i in range(0, len(bits), 16):
            print("    " + ", ".join(f"0x{value:04X}u" for value in bits[i:i + 16]) + ",")
        return
    match = re.search(r"kPackedWeights\s*\{(.*?)\};", CPP_TABLE.read_text(encoding="utf-8"), re.S)
    if match is None:
        parser.error("Could not find the compiled initializer")
    compiled = b"".join(struct.pack("<H", int(word, 16))
                        for word in re.findall(r"0x([0-9A-Fa-f]{4})u", match.group(1)))
    if compiled != packed:
        parser.error("The packed table does not match the compiled initializer")
    print(f"Verified {len(packed) // 2} FP16 values byte-for-byte; SHA-256 {digest}")


if __name__ == "__main__":
    main()

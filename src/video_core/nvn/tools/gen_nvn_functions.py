#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2020 h1k421
# SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
# SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
# SPDX-License-Identifier: GPL-3.0-or-later AND MIT
"""Generate the NVN tables committed next to nvn_types.h.

Inputs (downloaded from https://github.com/seedds/nxnative, never committed):
    dm/gfx/nvn/names.txt        one NVN procedure name per line, line index = function ID
    dm/gfx/nvn/nvn_protos.h     generated NVNT_<name> prototypes (booleans replaced by uint32_t)
    dm/gfx/nvn/include/nvn.h    enums and PFNNVN<NAME>PROC prototypes (booleans kept as NVNboolean)

Outputs (in --out, default: the directory above this script):
    nvn_functions.inc   NVN_FUNCTION(id, name, has_signature)            one line per name
    nvn_signatures.inc  NVN_SIGNATURE(id, name, "ret", "params")         one line per name with a prototype
    nvn_enums.inc       enum class NVNxxx : s32 { ... }                  from nvn.h, sentinels removed

Usage: gen_nvn_functions.py [--src C:\\eg\\results\\nvn\\src] [--out <dir>]
"""

import argparse
import os
import re
import sys

# nxnative header bugs, corrected here. Key: function name. Value: (return type, params) after the fix.
# Each override is checked against what the headers contain, so an upstream fix makes the script complain.
OVERRIDES = {
    # returned void in nvn_protos.h
    "nvnMemoryPoolBuilderGetMemory": ("void*", "const NVNmemoryPoolBuilder*"),
    # nvn_protos.h has uint32_t for the last parameter, nvn.h has NVNboolean (the PFN one is kept anyway)
    # nvn.h and nvn_protos.h return NVNmemoryPool by value (NVN_NOPROTO in nxnative); the getter returns a pointer
    "nvnBufferBuilderGetMemoryPool": ("NVNmemoryPool*", "const NVNbufferBuilder*"),
    "nvnTextureBuilderGetMemoryPool": ("NVNmemoryPool*", "const NVNtextureBuilder*"),
    # non-const this in the header
    "nvnMultisampleStateGetRasterSamples": ("int", "const NVNmultisampleState*"),
    # returns the callback's void* user data, not void
    "nvnCommandBufferGetMemoryCallbackData": ("void*", "const NVNcommandBuffer*"),
}


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def norm_params(params):
    params = re.sub(r"//[^\n]*", "", params)
    params = re.sub(r"\s+", " ", params).strip()
    return params.replace(" *", "*").replace("* ", "*").replace(",", ", ").replace(",  ", ", ")


def norm_ret(ret):
    return re.sub(r"\s+", " ", ret).strip().replace(" *", "*")


def parse_protos(text):
    """NVNT_<name> typedefs -> {name: (ret, params)}."""
    result = {}
    for m in re.finditer(r"^typedef\s+(.+?)\s*\(\*NVNT_(\w+)\)\s*\((.*?)\);\s*$", text, re.M):
        result[m.group(2)] = (norm_ret(m.group(1)), norm_params(m.group(3)))
    return result


def parse_pfn(text, upper_to_name):
    """PFNNVN<NAME>PROC typedefs -> {name: (ret, params)}; names the table does not know are returned separately."""
    result, unknown = {}, []
    for m in re.finditer(r"^typedef\s+(.+?)\s*\(\*PFNNVN(\w+?)PROC\)\s*\((.*?)\);", text, re.M | re.S):
        name = upper_to_name.get("NVN" + m.group(2))
        if name is None:
            unknown.append("PFNNVN" + m.group(2) + "PROC")
            continue
        result[name] = (norm_ret(m.group(1)), norm_params(m.group(3)))
    return result, unknown


def common_word_prefix(names):
    split = [n.split("_") for n in names]
    prefix = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return "_".join(prefix) + "_" if prefix else ""


def parse_enums(text):
    """Returns [(enum_name, [(enumerator, value_text)])] in file order, `*_LARGE` sentinels removed."""
    enums = []
    pattern = re.compile(r"(?:typedef\s+enum|enum\s+(NVN\w+))\s*\{(.*?)\}\s*(NVN\w+)?\s*;", re.S)
    for m in pattern.finditer(text):
        name = m.group(1) or m.group(3)
        members = []
        for line in m.group(2).split("\n"):
            line = re.sub(r"//.*", "", line).strip().rstrip(",")
            if not line:
                continue
            mm = re.fullmatch(r"(NVN_\w+)\s*=\s*(0[xX][0-9A-Fa-f]+|\d+)", line)
            if not mm:
                sys.exit(f"unparsed enumerator in {name}: {line!r}")
            members.append((mm.group(1), mm.group(2)))
        sentinel = [k for k, v in members if v.lower() == "0x7fffffff"]
        if len(sentinel) != 1 or not sentinel[0].endswith("_LARGE"):
            sys.exit(f"{name}: expected exactly one *_LARGE sentinel, got {sentinel}")
        members = [(k, v) for k, v in members if k != sentinel[0]]
        enums.append((name, sentinel[0][: -len("LARGE")], members))
    return enums


def emit_enum(name, sentinel_prefix, members):
    keys = [k for k, _ in members]
    prefix = sentinel_prefix
    if not all(k.startswith(prefix) for k in keys):
        prefix = common_word_prefix(keys)
    # An enumerator must start with a letter once the prefix is stripped.
    stripped = []
    for k in keys:
        s = k[len(prefix):]
        if not s or not (s[0].isalpha() or s[0] == "_"):
            head = prefix.rstrip("_").split("_")[-1]
            s = head + "_" + s
        stripped.append(s)
    if len(set(stripped)) != len(stripped):
        sys.exit(f"{name}: enumerator collision after stripping {prefix!r}")
    out = [f"enum class {name} : s32 {{"]
    for s, (_, v) in zip(stripped, members):
        out.append(f"    {s} = {v},")
    out.append("};")
    return "\n".join(out), len(members)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\eg\results\nvn\src")
    ap.add_argument("--out", default=os.path.dirname(here))
    args = ap.parse_args()

    names = [n.strip() for n in read(os.path.join(args.src, "names.txt")).splitlines()]
    if any(not n for n in names):
        sys.exit("names.txt has an empty line")
    if len(set(names)) != len(names):
        sys.exit("names.txt has duplicate names")
    upper_to_name = {n.upper(): n for n in names}

    protos = parse_protos(read(os.path.join(args.src, "nvn_protos.h")))
    pfn, pfn_unknown = parse_pfn(read(os.path.join(args.src, "nvn.h")), upper_to_name)

    sigs = {}
    notes = []
    for name in names:
        p, f = protos.get(name), pfn.get(name)
        if p is None and f is None:
            continue
        if f is not None:
            chosen = f  # keeps NVNboolean
            if p is not None:
                strip = lambda s: (norm_ret(s[0]).replace("NVNboolean", "uint32_t"), norm_params(s[1]).replace("NVNboolean", "uint32_t"))
                if strip(p) != strip(f):
                    notes.append(f"prototype mismatch {name}: protos={p} pfn={f}")
        else:
            chosen = p
        if name in OVERRIDES:
            if OVERRIDES[name] == chosen:
                notes.append(f"override for {name} is no longer needed")
            chosen = OVERRIDES[name]
        sigs[name] = chosen
    for name in OVERRIDES:
        if name not in sigs:
            sys.exit(f"override for unknown function {name}")
    for name in protos:
        if name not in upper_to_name.values():
            notes.append(f"prototype without a name in names.txt: {name}")

    os.makedirs(args.out, exist_ok=True)
    header = (
        "// SPDX-FileCopyrightText: Copyright (c) 2020 h1k421\n"
        "// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)\n"
        "// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)\n"
        "// SPDX-License-Identifier: GPL-3.0-or-later AND MIT\n\n"
        "// Generated by tools/gen_nvn_functions.py from nxnative dm/gfx/nvn/{names.txt,nvn_protos.h,include/nvn.h}.\n"
        "// Do not edit.\n\n"
    )

    with open(os.path.join(args.out, "nvn_functions.inc"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write("// NVN_FUNCTION(id, name, has_signature): id is the 0-based line index in names.txt.\n")
        for i, n in enumerate(names):
            f.write(f"NVN_FUNCTION({i}, {n}, {1 if n in sigs else 0})\n")

    with open(os.path.join(args.out, "nvn_signatures.inc"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write("// NVN_SIGNATURE(id, name, \"return type\", \"parameters\"): C spelling, header bugs corrected.\n")
        for i, n in enumerate(names):
            if n in sigs:
                ret, params = sigs[n]
                f.write(f'NVN_SIGNATURE({i}, {n}, "{ret}", "{params}")\n')

    enums = parse_enums(read(os.path.join(args.src, "nvn.h")))
    nmembers = 0
    with open(os.path.join(args.out, "nvn_enums.inc"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write("// Enumerators drop the NVN_<ENUM>_ prefix and the *_LARGE sentinel; values are the SDK's.\n\n")
        for name, sp, members in enums:
            text, n = emit_enum(name, sp, members)
            nmembers += n
            f.write(text + "\n" + ("\n" if name != enums[-1][0] else ""))

    print(f"names={len(names)} signatures={len(sigs)} no_signature={len(names) - len(sigs)} "
          f"enums={len(enums)} enumerators={nmembers}")
    if pfn_unknown:
        print("PFN typedefs that are not table functions:", ", ".join(pfn_unknown))
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()

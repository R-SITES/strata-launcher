#!/usr/bin/env python3
"""Build the launcher's flags catalogue.

Source of truth is the engine's own --help, where each entry is:
    --flag PLACEHOLDER    description...
i.e. the placeholder ends at the first run of 2+ spaces. That gap is the only reliable
separator - collapsing whitespace makes `--kv q4_0  4-bit K/V ...` look like prose.

Some flags the launcher actually passes are NOT in the help (`--spec`, `--spec-min-p`,
`--no-prefill-borrow`, `--serve`): the engine accepts them and they demonstrably change
behaviour, so they are added explicitly and marked undocumented rather than omitted.
"""
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELP = os.environ.get("STRATA_HELP", str(HERE / "engine-help.txt"))
lines = open(HELP, encoding="utf-8", errors="replace").read().splitlines()

flag_re = re.compile(r"^\s{2,}(--[a-z0-9-]+)(?:\s+(\S+))?\s{2,}(.*)$")
bare_re = re.compile(r"^\s{2,}(--[a-z0-9-]+)\s*$")
cont_re = re.compile(r"^\s{22,}(\S.*)$")

cat: "OrderedDict[str, dict]" = OrderedDict()
cur = None
for ln in lines:
    m = flag_re.match(ln)
    b = bare_re.match(ln)
    if m:
        name = m.group(1)
        ph = m.group(2) or ""
        desc = m.group(3).strip()
        if name in cat:
            # the engine lists some flags more than once as alternative values
            # (--kv fp16|int8, then --kv q4_0, then --kv k8v4) - merge them
            prev = cat[name]
            if ph and ph != prev.get("placeholder"):
                opts = prev.setdefault("alt", [])
                if prev.get("placeholder"):
                    opts.append(prev["placeholder"])
                opts.append(ph)
            cat[name]["help"] = (cat[name]["help"] + " " + desc).strip()[:320] if desc else cat[name]["help"]
        else:
            cat[name] = {"flag": name, "placeholder": ph, "help": desc}
        cur = name
    elif b:
        cur = b.group(1)
        cat.setdefault(cur, {"flag": cur, "placeholder": "", "help": ""})
    elif cur and cont_re.match(ln):
        cat[cur]["help"] = (cat[cur]["help"] + " " + ln.strip()).strip()[:320]

# typing
NUMERIC = ("N", "M", "MiB", "ms", "GB", "GiB", "us", "U", "SEED", "STEPS", "TOKENS", "F", "T")
for f in cat.values():
    ph = f["placeholder"]
    alts = f.pop("alt", None)
    if alts:
        # several spellings of the same flag: one select covering all of them
        base = [o for o in ph.split("|")] if "|" in ph else ([ph] if ph else [])
        f["type"] = "select"
        seen, opts = set(), []
        flat = []
        for o in base + alts:
            if not o:
                continue
            flat += [p.strip() for p in o.split("|") if p.strip()]
        for o in flat:
            if o not in seen:
                seen.add(o)
                opts.append(o)
        f["options"] = opts
        f["placeholder"] = "|".join(opts)
    elif "|" in ph:
        f["type"] = "select"
        f["options"] = [o.strip() for o in ph.split("|") if o.strip()]
    elif ph:
        f["type"] = "number" if ph in NUMERIC else "string"
    else:
        f["type"] = "bool"
    low = f["help"].lower()
    if "experimental" in low or "not with" in low or "opt-in" in low:
        f["caution"] = "experimental / limited"

# flags the launcher passes that the help does not list
UNDOCUMENTED = {
    "--serve": ("bool", "", "serve mode (what the launcher runs); not in the engine's help"),
    "--spec": ("number", "T", "MTP draft depth (default 4). NOT in the help but accepted; "
                            "4 measured 64 tok/s, 8 collapses to ~5 - see the skill"),
    "--spec-min-p": ("number", "P", "MTP minimum draft probability (default 0.5)"),
    "--no-prefill-borrow": ("bool", "", "prompt path must not lend expert slots (long prompts hang without it)"),
    "--stream-experts": ("bool", "", "stream experts from the GGUF into VRAM; no host arena"),
    "--prefill": ("number", "CHUNK", "prompt chunk size (4096 used; 8192 measured ~2x prompt speed)"),
    "--vram-reserve-mib": ("number", "MIB", "VRAM kept free; the engine sizes its expert cache off this"),
}
for k, (typ, ph, why) in UNDOCUMENTED.items():
    if k in cat:
        cat[k].setdefault("type", typ)
        continue
    cat[k] = {"flag": k, "placeholder": ph, "type": typ, "help": why, "undocumented": True}

out = list(cat.values())
json.dump({"flags": out}, open(os.environ.get("STRATA_FLAGS_DB", str(HERE / "flags_db.json")), "w"), indent=1)

from collections import Counter
print(f"{len(out)} flags in the catalogue")
print("types:", dict(Counter(f["type"] for f in out)))
print(f"selects: {sum(1 for f in out if f['type']=='select')}  "
      f"bool: {sum(1 for f in out if f['type']=='bool')}  "
      f"undocumented: {sum(1 for f in out if f.get('undocumented'))}")
print("\nthe ones we care about:")
for f in out:
    if f["flag"] in ("--spec", "--spec-min-p", "--expert-cache", "--expert-cache-per-layer", "--kv",
                     "--kv-resident", "--prefill", "--vram-reserve-mib", "--draft-vocab", "--mtp"):
        print(f"  {f['flag']:24s} {f['type']:7s} {f.get('placeholder',''):6s} "
              f"{','.join(f.get('options',[]))[:30]:32s} {f['help'][:50]}")

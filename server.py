#!/usr/bin/env python3
"""Strata Launcher - a small local web UI for the Strata engine on Intel Arc.

Serves one page and a handful of endpoints. It does NOT run in a container: the whole
point of this launcher is that it launches Strata's SYCL engine natively, carrying the
things a bare invocation gets wrong (oneAPI env, the two VERIFY vars, the required
--prefill, the draft vocabulary, CORS origins).

Standard library only, loopback by default.

    python3 server.py            # http://localhost:9877
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATA = Path(os.environ.get("STRATA_ROOT", "~")).expanduser() / "strata"
MODELS_DIR = Path("~/models").expanduser()
PORT = int(os.environ.get("STRATA_LAUNCHER_PORT", "9877"))
LOG = ROOT / "strata-launcher.log"

# oneAPI env: the engine links oneMKL's SYCL libs, so this must be present or the engine
# dies instantly with "libmkl_sycl_blas.so.6: cannot open shared object file".
ENV_SOURCING = (
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; "
    "source /opt/intel/oneapi/umf/1.1/env/vars.sh >/dev/null 2>&1; unset MKLROOT; "
    "export SYCL_CACHE_PERSISTENT=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 ZES_ENABLE_SYSMAN=1 "
    "STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1"
)

_proc: subprocess.Popen | None = None
_lock = threading.Lock()

# The three engines a model install can use, in the order we prefer them.
KNOWN_MODELS = {
    "coder-iq1-m": {
        "label": "Coder IQ1_M  (ISTA-DASLab, half the experts, code-focused)",
        # What chat clients DISPLAY. The serve layer sends this over /v1/models and the web chat shows
        # it in the header + the token bar, so the artifact name (GSQ-RCO, IQ1_M, Coder) belongs here:
        # "IQ2_XS" or "strata-dream-launch" tells the user nothing about what they are talking to.
        "served_name": "Qwen3.8-Flash-Next-Coder-IQ1_M",
        "dir": "strata-coder-iq1-m",
        "profile": "expert-profile-coder.bin",
        "mtp": "strata-mtp",
        "shard_bytes": [29608446496, 28800138432],
    },
    "iq2-xs": {
        "label": "IQ2_XS  (full 512-expert model)",
        "served_name": "Qwen3.8-Flash-Next-IQ2_XS",
        "dir": "strata-iq2-xs",
        "profile": "expert-profile.bin",
        "mtp": "strata-mtp",
        "shard_bytes": [39225954592, 28800138432],
    },
}


def sh(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def find_model(key: str) -> dict | None:
    meta = KNOWN_MODELS.get(key)
    if not meta:
        return None
    d = MODELS_DIR / meta["dir"]
    if not d.is_dir():
        return None
    shards = sorted(d.glob("*-00001-of-*.gguf"))
    if not shards:
        return None
    # the image encoder, if this install has one: the mmproj GGUF setup (or a hand build) left beside
    # the shards. Its presence is the honest test for "Launch with vision" - nothing else is needed.
    mmproj = sorted(d.glob("mmproj-*.gguf"))
    shard1 = shards[0]
    n = shard1.name.rsplit("-", 1)[-1].replace(".gguf", "")
    parts = sorted(d.glob(f"*{n}*.gguf"))
    ple = next((p for p in parts if "-00002-of-" in p.name), None)
    pack = d / "pack"
    rt = MODELS_DIR / meta["mtp"] / "rt"

    # download progress against the known shard sizes, so a partial install is VISIBLE
    # rather than absent (a missing shard used to hide the whole model)
    want = meta.get("shard_bytes") or []
    present_total = sum(p.stat().st_size for p in parts if p.stat().st_size)
    expected_total = sum(want)
    complete = bool(want) and len(parts) >= len(want) and \
        all(p.stat().st_size == b for p, b in zip(parts, want))
    shard_state = ", ".join(
        f"shard {p.name.rsplit('-of-', 1)[0].rsplit('-', 1)[-1].lstrip('0')}: "
        + ("ok" if p.stat().st_size == b else f"{p.stat().st_size / 1e9:.1f}/{b / 1e9:.1f} GB")
        for p, b in zip(parts, want))

    return {
        "key": key,
        "label": meta["label"],
        "dir": str(d),
        "shard1": str(shard1),
        "ple": str(ple) if ple else "",
        "pack": str(pack),
        "pack_ready": (pack / "index.txt").is_file() and (pack / "tokenizer" / "vocab.json").is_file(),
        "profile": str(STRATA / "data" / meta["profile"]),
        "mtp": str(rt),
        "mmproj": str(mmproj[0]) if mmproj else "",
        "mtp_ready": (rt / "experts.bin").is_file() and (rt / "draft_vocab.bin").is_file(),
        "engine": str(STRATA / "build-sycl-aot" / "strata"),
        # the name chat clients show (header pill, token bar, /v1/models)
        "served_name": meta.get("served_name") or key,
        # download state, so a partial install is VISIBLE rather than absent
        "complete": complete,
        "percent": round(100 * present_total / expected_total, 1) if expected_total else 0.0,
        "present_gb": round(present_total / 2**30, 1),
        "expected_gb": round(expected_total / 2**30, 1),
        "shard_state": shard_state,
        "ready": bool(complete and ple and (pack / "index.txt").is_file()
                      and (rt / "experts.bin").is_file()),
    }


# ---------------------------------------------------------------- launcher-applied flags
# These are NOT engine flags. The engine's CLI would reject them as unknown options and exit
# at start, which the server reports only as "the engine exited before it was ready" - so they
# are taken OUT of the arg merge and the launcher performs the action itself instead.
DRAFT_VOCAB_FILES = {          # setup's --draft-vocab choices -> the data/ file each one means
    "cjk": "draft_vocab.bin",
    "en": "draft_vocab_en.bin",
    "cyrillic": "draft_vocab_cyrillic.bin",
    "fr": "draft_vocab_fr.bin",
}
LAUNCHER_FLAGS = ("--draft-vocab",)


def shared_rt_dir() -> Path:
    """Every model shares ONE MTP runtime dir, so the draft vocabulary is an rt-wide choice."""
    for meta in KNOWN_MODELS.values():
        return MODELS_DIR / meta["mtp"] / "rt"
    return MODELS_DIR / "strata-mtp" / "rt"


def draft_vocab_current(rt: Path) -> str:
    """Which subset rt/draft_vocab.bin currently holds, matched by size. '?' = none of them."""
    try:
        n = (rt / "draft_vocab.bin").stat().st_size
    except OSError:
        return ""
    for name, fn in DRAFT_VOCAB_FILES.items():
        try:
            if (STRATA / "data" / fn).stat().st_size == n:
                return name
        except OSError:
            continue
    return "?"


def apply_draft_vocab(rt: Path, choice: str) -> dict:
    """Copy data/draft_vocab_<choice>.bin over the shared rt/draft_vocab.bin.

    That file IS the mechanism (4-byte token ids; the draft head is built over exactly those
    ids) - there is no engine flag for it. It changes the draft vocabulary for whichever model
    starts next, since rt/ is shared.
    """
    fn = DRAFT_VOCAB_FILES.get(choice)
    if not fn:
        return {"error": f"unknown draft vocab {choice!r}"}
    src = STRATA / "data" / fn
    dst = Path(rt) / "draft_vocab.bin"
    if not src.is_file():
        return {"error": f"{src} is missing"}
    size = src.stat().st_size
    if dst.is_file() and dst.stat().st_size == size:
        return {"choice": choice, "copied": False, "already": True, "bytes": size,
                "target": str(dst)}
    shutil.copy2(src, dst)
    return {"choice": choice, "copied": True, "bytes": size, "source": str(src),
            "target": str(dst)}


# Every numeric setting the panel can send, bounded. A box reading "05" - what an Android keyboard
# produces when the field has no decimal key - reached the config as `--spec-min-p 5`, the engine could
# never clear a draft at min-p 5, and the launch ran at 33 tok/s with "drafts accepted 0 of 0"
# (measured 2026-10-06). Out of range is REFUSED, and a valid value is canonicalised (05 -> 5,
# 0.50 -> 0.5) so what the box shows is what the engine receives.
SETTING_BOUNDS = {
    "context": (1024, 1048576, "context", True),
    "port": (1, 65535, "port", True),
    "reasoning_budget": (0, 32768, "thinking budget", True),
    "spec": (0, 32, "spec n-max", True),          # a count of drafted tokens, like the panel's field
    "spec_min_p": (0, 1, "spec min-p", False),    # a probability: decimals are the point
    "prefill": (0, 32768, "prefill chunk", True),
}


def canonical_opts(opts: dict) -> dict:
    """Launch options with the numeric boxes validated and canonicalised.

    Raises ValueError naming the box and the number it read, so the API answers 400 rather than starting
    an engine whose flags differ from the panel that requested it. Whole-number fields are checked here
    too, so the API cannot accept what the panel refuses.
    """
    out = dict(opts)
    for key, (lo, hi, label, whole) in SETTING_BOUNDS.items():
        raw = out.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            n = float(str(raw).strip())
        except ValueError:
            raise ValueError(f"{label} must be a number (it reads {raw!r})") from None
        if not (lo <= n <= hi):
            raise ValueError(f"{label} must be between {lo:g} and {hi:g} (it reads {n:g})")
        if whole and abs(n - round(n)) > 1e-9:
            raise ValueError(f"{label} must be a whole number (it reads {n:g})")
        out[key] = f"{n:g}"
    return out


def build_config(m: dict, opts: dict) -> dict:
    opts = canonical_opts(opts)
    ctx = int(opts.get("context") or 262144)
    reserve = 1024 if ctx <= 32768 else 2048
    args = [
        "--pack", m["pack"],
        "--native", m["shard1"],
        "--ple-gguf", m["ple"],
        "--expert-profile", m["profile"],
        "--mtp", m["mtp"],
        "--stream-experts",
        "--expert-cache", "auto",
        "--no-prefill-borrow",     # long prompts hang without it on this platform
        "--prefill", str(opts.get("prefill") or 4096),   # REQUIRED in serve mode
        "--vram-reserve-mib", str(reserve),
        "--max-context", str(ctx),
        "--spec", str(opts.get("spec") or 4),
        "--spec-min-p", str(opts.get("spec_min_p") or 0.5),
    ]
    if ctx >= 65536:
        args += ["--kv-resident", "32768"]

    # UI flag overrides: {flag: value}. A bool arrives as true (flag appended bare); anything
    # else is flag + value. An override REPLACES any existing occurrence, so the panel can
    # change a flag the template already set (e.g. --expert-cache auto -> 11800) instead of
    # emitting a duplicate the engine would see twice.
    # Flags the MODEL owns, never the panel. These four ARE the model identity: letting a stale
    # panel entry override them means a launch silently loads different weights - the real
    # symptom was the IQ2_XS command carrying the Coder's --pack after a preset load. The model
    # selection is the only authority for them.
    MODEL_OWNED = ("--pack", "--native", "--ple-gguf", "--expert-profile")
    overrides = dict(opts.get("flags") or {})
    draft_choice = str(overrides.get("--draft-vocab") or "").strip().lower()
    # Launcher-applied flags come OUT of the merge entirely: the engine has no such option and
    # hands the unknown flag back as a start failure, so the launcher does the work instead.
    for lf in LAUNCHER_FLAGS:
        overrides.pop(lf, None)
    for fname, fval in overrides.items():
        if not str(fname).startswith("--"):
            continue
        if str(fname) in MODEL_OWNED:
            continue
        while fname in args:
            i = args.index(fname)
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt is not None and not str(nxt).startswith("--"):
                del args[i:i + 2]
            else:
                del args[i]
        if fval is True or fval == "" or fval is None:
            args.append(str(fname))
        else:
            args += [str(fname), str(fval)]

    cfg = {
        "exe": m["engine"],
        "cwd": str(STRATA),
        "backend": "sycl",
        # what the chat clients DISPLAY: the real artifact name, not the launcher's internals
        "model_name": m.get("served_name") or f"strata-{m['key']}",
        "port": int(opts.get("port") or 8085),
        "host": opts.get("host") or "0.0.0.0",
        "api_key": opts.get("api_key") or "",
        "log": str(ROOT / "engine.log"),
        # 0 means UNLIMITED thinking (which is why the model sometimes never answers)
        "reasoning_budget_tokens": int(opts.get("reasoning_budget") or 64),
        # a browser client on another origin gets blocked without this
        "cors_origins": opts.get("cors_origins")
        or ["http://localhost:3001", "http://127.0.0.1:3001",
            "http://localhost:3000", "http://127.0.0.1:3000"],
        "args": args,
    }
    # setup.py records the draft vocab in the config and refreshes rt/ from it. Only write the
    # key when a subset was actually chosen, so a launch that never picked one leaves the file
    # exactly as setup wrote it.
    if draft_choice in DRAFT_VOCAB_FILES:
        cfg["draft_vocab"] = draft_choice
    # The image encoder, when the caller asked for vision: serve/server.py spawns `strata-vision`
    # (llama.cpp mtmd + an mmproj) as a child of the engine server, so the projector never takes
    # engine VRAM. All three paths must exist or the launch dies at startup, so refuse first.
    #
    # BOTH HALVES ARE NEEDED: the config block spawns the encoder, and `--vision` is what makes the
    # ENGINE accept the embeddings it sends. Without the flag the encoder loads, runs on the image,
    # and the engine answers "this engine was started without --vision" - measured 2026-10-06, after
    # a launch that carried the block and not the flag (fans spun up, then "no response from the
    # server" and the whole server shut itself down).
    if opts.get("vision") in (True, "true", "1", 1):
        vision_exe = STRATA / "build-vision" / "bin" / "strata-vision"
        if not vision_exe.is_file():
            vision_exe = STRATA / "build-sycl-aot" / "strata-vision"     # setup.py's copy
        if not m.get("mmproj"):
            raise ValueError("no mmproj-*.gguf beside this model's shards, so it has no vision encoder")
        if not vision_exe.is_file():
            raise ValueError(f"the image encoder is not built ({vision_exe}) - run "
                             f"scripts/build-strata-vision.sh")
        cfg["vision"] = {"exe": str(vision_exe), "mmproj": m["mmproj"], "model": m["shard1"],
                         "gpu": False, "threads": 8}
        if "--vision" not in cfg["args"]:
            cfg["args"].append("--vision")
    return cfg


def write_run_script(m: dict, cfg: dict) -> Path:
    cfg_path = STRATA / f"strata-{m['key']}-launcher.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    script = ROOT / f"run-{m['key']}.sh"
    script.write_text(
        "#!/bin/bash\n"
        "# generated by the Strata Launcher - do not edit by hand\n"
        f"cd {STRATA} || exit 1\n"
        f"{ENV_SOURCING}\n"
        f"exec python3 sycl/serve/server_intel.py --engine strata \\\n"
        f"  --config {cfg_path} \\\n"
        f"  --tokenizer {m['pack']}/tokenizer \\\n"
        f"  --port {cfg['port']}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def engine_command(m: dict, cfg: dict) -> str:
    """The exact command line the launcher runs, as text (shown in the UI, prefills Quick Launch)."""
    text = (
        f"cd {STRATA} && python3 sycl/serve/server_intel.py --engine strata \\\n"
        f"  --config {STRATA}/strata-{m['key']}-launcher.json \\\n"
        f"  --tokenizer {m['pack']}/tokenizer \\\n"
        f"  --port {cfg['port']}"
    )
    # The draft vocab is a FILE copy, not an engine flag, so the launcher performs it - say so
    # in the command text, or the command shown would not be the whole truth of what the launch
    # changes. A shell comment is inert wherever this text gets pasted.
    if cfg.get("draft_vocab"):
        text += (f"\n# launcher, not an engine flag: draft vocab {cfg['draft_vocab']} -> "
                 f"cp {STRATA}/data/{DRAFT_VOCAB_FILES[cfg['draft_vocab']]} {m['mtp']}/draft_vocab.bin")
    return text


def _normalize_command(cmd: str) -> str:
    """Join a pasted multi-line command back into ONE line.

    Two ways a wrapped command dies, both seen in the wild. A trailing space AFTER the
    continuation backslash: bash escapes the space and ends the command there. Or the
    backslash is dropped in transport, leaving lines like `--config ...` that bash runs as
    their own commands (the symptom is "--config: command not found").

    Guessing wrong in the other direction is worse than the bug: merging two commands the
    user DID want separate. So join only on positive evidence of a wrap - the previous line
    ends with a shell operator, the next line starts with a flag, or the next line is bare
    KEY=VALUE assignments (never a command on its own). Anything else stays on its own line.
    """
    lines = [ln.strip() for ln in cmd.splitlines() if ln.strip()]
    if not lines:
        return cmd
    out: list[str] = []
    for ln in lines:
        prev = out[-1].rstrip() if out else ""
        operand = prev.endswith(("\\", "&&", "||", "|", ";", "="))
        is_flag = ln.startswith("-")
        is_assign = all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w)
            for w in ln.split() if not w.rstrip(";").endswith("&&")
        ) and bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", ln))
        if out and (operand or is_flag or is_assign):
            if prev.endswith("\\"):
                prev = prev[:-1].rstrip()
            out[-1] = prev + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def run_command(cmd: str, port: int | None) -> dict:
    """Run a pasted command with the oneAPI env prepended. This is the point of Quick Launch:
    a Strata command run in a plain shell dies on libmkl_sycl_blas.so.6, and that is the single
    most common way a hand-typed invocation fails."""
    global _proc
    with _lock:
        r = running()
        if r["running"]:
            return {"error": "a Strata server is already running - kill it first "
                             "(two engines would fight for the 32 GB card)"}
        cmd = _normalize_command(cmd)
        full = f"{ENV_SOURCING}\n{cmd}\n"
        script = ROOT / "run-quicklaunch.sh"
        script.write_text("#!/bin/bash\n# generated by the Strata Launcher Quick Launch\n" + full,
                          encoding="utf-8")
        script.chmod(0o755)
        logf = open(LOG, "wb")
        logf.write(b"# quick launch\n")
        logf.write(full.encode())
        logf.flush()
        _proc = subprocess.Popen([str(script)], stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    return {"ok": True, "pid": _proc.pid, "port": port, "script": str(script)}


def _port_open(port: int) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(0.6)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        s.close()


_ACESTEP_ARGV0_HINTS = ("acestep", "ace-step")


def _acestep_pids() -> list[int]:
    """Every live ACE-Step process. Matches the PROGRAM (argv[0]) so a shell or agent
    command that merely MENTIONS acestep is never matched."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                parts = [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
        except Exception:  # noqa: BLE001
            continue
        if not parts:
            continue
        if any(h in parts[0].lower() for h in _ACESTEP_ARGV0_HINTS) \
                or "acestep.api_server" in " ".join(parts).lower():
            pids.append(int(entry))
    return sorted(pids)


def _vllm_pids() -> list[int]:
    """A native vLLM server, anchored on the venv binary path (a bare 'vllm serve'
    pattern would match shells whose command text merely contains it)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                parts = [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
        except Exception:  # noqa: BLE001
            continue
        if len(parts) < 2:
            continue
        if parts[0].endswith("/vllm") and parts[1] == "serve":
            pids.append(int(entry))
    return sorted(pids)


def free_vram() -> dict:
    """Sweep the OTHER known VRAM holders so a Strata launch can size a real expert cache.

    Strata sizes that cache off FREE VRAM at start, so this is the difference between
    60+ tok/s (9766 slots) and ~35 (1034 slots, everything else mirrored in host RAM).

      1. ComfyUI :8188/free - unload models, service STAYS up (gentle).
      2. ACE-Step :8001 - stop (no unload API; cold boot ~25-60s next use).
      3. A native vLLM server - stop (SIGTERM then SIGKILL).
    The Strata engine is NEVER touched here - that is what Kill server is for.
    Kokoro :5090, SadTalker :5091 and the r1p helper :8010 are never touched.
    """
    actions, errors = [], []

    if _port_open(8188):
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8188/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                code = r.status
            actions.append("ComfyUI models unloaded" if code == 200 else f"ComfyUI /free returned {code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ComfyUI free failed: {exc}")
    else:
        actions.append("ComfyUI not running")

    if _acestep_pids() or _port_open(8001):
        try:
            req = urllib.request.Request("http://127.0.0.1:8010/stop-acestep", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=20)
        except Exception:  # noqa: BLE001
            pass
        pids = _acestep_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        for _ in range(8):
            if not _acestep_pids() and not _port_open(8001):
                break
            time.sleep(1)
        for pid in _acestep_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
        if _acestep_pids() or _port_open(8001):
            errors.append("ACE-Step still running after stop attempt")
        else:
            actions.append(f"ACE-Step stopped ({len(pids)} process{'es' if len(pids) != 1 else ''})")

    vpids = _vllm_pids()
    if vpids:
        for pid in vpids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        for _ in range(10):
            if not _vllm_pids():
                break
            time.sleep(1)
        for pid in _vllm_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
        if _vllm_pids():
            errors.append("vLLM still running after stop attempt")
        else:
            actions.append(f"vLLM stopped ({len(vpids)} process{'es' if len(vpids) != 1 else ''})")

    vram_free = vram_total = ram_free = ram_total = None
    py = Path("~/vllm-xpu/venv/bin/python").expanduser()
    if py.is_file():
        c, o = sh([str(py), "-c",
                   "import torch;f,t=torch.xpu.mem_get_info();print(f'{f/2**30:.1f} {t/2**30:.1f}')"])
        if c == 0 and o.strip():
            try:
                vram_free, vram_total = (float(x) for x in o.split())
            except Exception:  # noqa: BLE001
                pass
    c, o = sh(["bash", "-lc", "free -g | awk 'NR==2{print $7\" \"$2}'"])
    if c == 0 and o.strip():
        try:
            ram_free, ram_total = (float(x) for x in o.split())
        except Exception:  # noqa: BLE001
            pass

    msg = "; ".join(actions)
    if errors:
        msg = (msg + " | " if msg else "") + "ERRORS: " + "; ".join(errors)
    return {"success": not errors, "message": msg or "Nothing to free",
            "vram_free_gb": vram_free, "vram_total_gb": vram_total,
            "ram_free_gb": ram_free, "ram_total_gb": ram_total,
            "errors": errors}


def running() -> dict:
    """Is a Strata server up, and on which port?

    Checks BOTH the python server and the engine binary: the engine is the server's child,
    and if the child outlives the parent it still holds ~30 GiB of VRAM while a naive
    'is the server running' check reports down and lets a second engine start.
    """
    code, out = sh(["bash", "-lc",
                    "pgrep -af 'server_intel.py' | head -3; "
                    "pgrep -af 'build-sycl-aot/strata' | head -3"])
    procs = [l for l in out.splitlines() if "pgrep" not in l and l.strip()]
    port = None
    for line in procs:
        if "--port" in line:
            try:
                port = int(line.split("--port")[1].split()[0])
            except Exception:  # noqa: BLE001
                pass
    return {"running": bool(procs), "procs": procs, "port": port}


def probe(port: int) -> dict:
    """Ask the running server for its own health."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return {}


class Handler(BaseHTTPRequestHandler):
    server_version = "StrataLauncher/1.2.0"

    def log_message(self, *a):  # keep the console quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._file(ROOT / "index.html", "text/html; charset=utf-8")
        elif path in ("/strata-sq.jpg", "/strata-256.png", "/strata-icon.png"):
            ctype = "image/jpeg" if path.endswith(".jpg") else "image/png"
            self._file(ROOT / path.lstrip("/"), ctype)
        elif path == "/api/models":
            found = {k: find_model(k) for k in KNOWN_MODELS}
            self._json({"models": [v for v in found.values() if v]})
        elif path == "/api/state":
            r = running()
            health = probe(r["port"]) if r.get("port") else {}
            tail = ""
            if LOG.is_file():
                tail = LOG.read_text(errors="replace")[-4000:]
                tail = tail[-1500:]
            self._json({**r, "health": health, "log_tail": tail,
                        "draft_vocab_current": draft_vocab_current(shared_rt_dir()),
                        "engine_log": (ROOT / "engine.log").read_text(errors="replace")[-800:]
                        if (ROOT / "engine.log").is_file() else ""})
        elif path == "/api/preview":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            key = (q.get("model") or ["coder-iq1-m"])[0]
            m = find_model(key)
            if not m:
                self._json({"error": f"model {key!r} not installed"}, 400)
                return
            opts = {k: v[0] for k, v in q.items()}
            try:
                cfg = build_config(m, opts)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"command": engine_command(m, cfg), "config": cfg,
                        # the flags live in the config file, so surface them: otherwise the
                        # launch is unreadable and nobody can tell what it will run
                        "engine_args": cfg["args"],
                        "engine_argv": cfg["exe"] + " --serve " + " ".join(cfg["args"]),
                        "config_path": str(STRATA / f"strata-{m['key']}-launcher.json")})
        elif path == "/api/readconfig":
            # Load a config the way the launcher wrote it, so a saved command preset can
            # restore the flags and settings it points at. The flags are IN the config, not
            # on the command line, which is why a command-only preset restored nothing.
            p = ""
            try:
                from urllib.parse import parse_qs, urlparse
                p = (parse_qs(urlparse(self.path).query).get("path") or [""])[0].strip()
            except Exception:  # noqa: BLE001
                p = ""
            if not p:
                self._json({"error": "no path given"}, 400)
                return
            cp = Path(os.path.expanduser(p))
            if not cp.is_file():
                self._json({"error": f"config not found: {p}"}, 404)
                return
            try:
                c = json.loads(cp.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"bad config json: {exc}"}, 400)
                return
            args = c.get("args") or []
            flags, i = {}, 0
            while i < len(args):
                a = str(args[i])
                if a.startswith("--"):
                    if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                        flags[a] = str(args[i + 1])
                        i += 2
                    else:
                        flags[a] = True
                        i += 1
                else:
                    i += 1
            settings = {
                "context": flags.get("--max-context"),
                "port": c.get("port"),
                "reasoning_budget": c.get("reasoning_budget_tokens"),
                "spec": flags.get("--spec"),
                "spec_min_p": flags.get("--spec-min-p"),
                "prefill": flags.get("--prefill"),
                "host": c.get("host"),
                "cors_origins": ",".join(c.get("cors_origins") or []),
            }
            # a config written by a launch that chose a draft vocab must restore that choice too,
            # or a command-only preset would silently keep whatever rt/ happens to hold
            if str(c.get("draft_vocab") or "").lower() in DRAFT_VOCAB_FILES:
                flags["--draft-vocab"] = str(c["draft_vocab"]).lower()
            self._json({"flags": flags, "settings": settings,
                        "model_name": c.get("model_name"), "path": str(cp)})
        elif path == "/api/flags":
            db = ROOT / "flags_db.json"
            if db.is_file():
                data = json.loads(db.read_text(encoding="utf-8"))
                cur = draft_vocab_current(shared_rt_dir())
                for f in data.get("flags", []):
                    if f.get("launcher"):
                        f["current"] = cur      # what rt/ holds RIGHT NOW, so the panel can say
                self._json(data)
            else:
                self._json({"flags": [], "error": "flags_db.json missing - run parse_flags.py"})
        elif path == "/api/stats":
            # The engine log is opened "wb" by the next launch, so the numbers that explain a launch
            # survive only in the stats file. This endpoint is the manual trigger for the same pass
            # /api/launch runs before it truncates the log.
            c, o = sh([sys.executable, str(ROOT / "stats_log.py")], timeout=120)
            total = 0
            if (ROOT / "stats.csv").is_file():
                total = max(0, len((ROOT / "stats.csv").read_text(errors="replace").splitlines()) - 1)
            self._json({"ok": c == 0, "rows_total": total, "csv": str(ROOT / "stats.csv"),
                        "md": str(ROOT / "stats.md"), "output": o[-2500:]})
        elif path == "/api/system":
            now = time.time()
            if getattr(self.server, "_sys_cache", None) and now - self.server._sys_cache["ts"] < 4:
                self._json(self.server._sys_cache["data"])
                return
            core_c, core_o = sh(["bash", "-lc", "nproc"])
            mem_c, mem_o = sh(["bash", "-lc", "free -g | awk 'NR==2{print $7\" \"$2}'"])
            ram_free = ram_total = None
            if mem_c == 0 and mem_o.strip():
                try:
                    ram_free, ram_total = (float(x) for x in mem_o.split())
                except Exception:  # noqa: BLE001
                    pass
            vram_free = vram_total = None
            py = Path("~/vllm-xpu/venv/bin/python").expanduser()
            if py.is_file():
                c, o = sh([str(py), "-c",
                           "import torch;f,t=torch.xpu.mem_get_info();print(f'{f/2**30:.1f} {t/2**30:.1f}')"],
                          timeout=30)
                if c == 0 and o.strip():
                    try:
                        vram_free, vram_total = (float(x) for x in o.split())
                    except Exception:  # noqa: BLE001
                        pass
            data = {"ram_free_gb": ram_free, "ram_total_gb": ram_total,
                    "ram_used_gb": (round(ram_total - ram_free, 1)
                                    if ram_free is not None and ram_total is not None else None),
                    "vram_free_gb": vram_free, "vram_total_gb": vram_total,
                    "vram_used_gb": (round(vram_total - vram_free, 1)
                                     if vram_free is not None and vram_total is not None else None),
                    "cores": core_o.strip() if core_c == 0 else None,
                    # what the engine cares about: it sizes its expert cache off free VRAM
                    "vram_ok": (vram_free is not None and vram_free >= 24)}
            self.server._sys_cache = {"ts": now, "data": data}
            self._json(data)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        global _proc
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        path = self.path.split("?")[0]

        if path == "/api/preview" or path == "/api/flags-preview":
            # POST form so the UI can send its flag overrides (a JSON object) with the preview
            key = body.get("model") or "coder-iq1-m"
            m = find_model(key)
            if not m:
                self._json({"error": f"model {key!r} not installed"}, 400)
                return
            try:
                cfg = build_config(m, body)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"command": engine_command(m, cfg), "engine_args": cfg["args"],
                        "config_path": str(STRATA / f"strata-{m['key']}-launcher.json")})
        elif path == "/api/free-vram":
            res = free_vram()
            self.server._sys_cache = None      # force a fresh probe next poll
            self._json(res, 200 if res["success"] else 500)
        elif path == "/api/launch":
            key = body.get("model")
            m = find_model(key)
            if not m:
                self._json({"error": f"model {key!r} not installed"}, 400)
                return
            if not m["pack_ready"]:
                self._json({"error": "pack missing - run tools/iq_pack.py first"}, 400)
                return
            if not m.get("complete"):
                self._json({"error": f"shards incomplete - {m.get('percent')}% "
                                     f"({m.get('shard_state')})"}, 400)
                return
            if not m["mtp_ready"]:
                self._json({"error": "MTP draft layer missing (rt/experts.bin + draft_vocab.bin)"}, 400)
                return
            if not Path(m["engine"]).is_file():
                self._json({"error": f"engine not built at {m['engine']}"}, 400)
                return
            with _lock:
                if running()["running"]:
                    self._json({"error": "a Strata server is already running - kill it first"}, 409)
                    return
                try:
                    cfg = build_config(m, body)
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
                    return
                dv = {}
                if cfg.get("draft_vocab"):
                    # launcher-applied, and only here: it must happen with the card free, i.e.
                    # while nothing is running (checked above), and before the engine starts
                    dv = apply_draft_vocab(m["mtp"], cfg["draft_vocab"])
                    if dv.get("error"):
                        self._json({"error": f"draft vocab: {dv['error']}"}, 400)
                        return
                script = write_run_script(m, cfg)
                # engine.log is opened "wb" on the next line, so whatever the PREVIOUS launch printed
                # is about to go. Snapshot it into stats.csv first - that is the only record that
                # survives, and it is what the Stats button reads.
                sh([sys.executable, str(ROOT / "stats_log.py")], timeout=120)
                logf = open(LOG, "wb")
                logf.write(f"# launching {m['key']} on port {cfg['port']}\n".encode())
                logf.flush()
                _proc = subprocess.Popen([str(script)], stdout=logf, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self._json({"ok": True, "pid": _proc.pid, "port": cfg["port"],
                        "draft_vocab": dv,
                        "config": str(STRATA / f"strata-{m['key']}-launcher.json"),
                        "script": str(script)})
        elif path == "/api/quicklaunch":
            cmd = (body.get("cmd") or "").strip()
            if not cmd:
                self._json({"error": "empty command"}, 400)
                return
            try:
                port = int(str(body.get("port") or "8085"))
            except Exception:  # noqa: BLE001
                port = 8085
            res = run_command(cmd, port)
            self._json(res, 400 if res.get("error") else 200)
        elif path == "/api/kill":
            r = running()
            if not r["running"]:
                self._json({"ok": True, "message": "nothing running"})
                return
            # Kill the whole session/process group first: the launched script is a session
            # leader, so this takes the python server AND the engine child with it. Killing
            # only the python server leaves the engine holding ~30 GiB of VRAM.
            killed = []
            if _proc is not None:
                try:
                    os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
                    killed.append(_proc.pid)
                except Exception:  # noqa: BLE001
                    pass
            for line in r["procs"]:
                try:
                    pid = int(line.split()[0])
                    if pid not in killed:
                        os.kill(pid, signal.SIGTERM)
                        killed.append(pid)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(3)
            left = running()
            # anything that ignored SIGTERM: the engine can be mid-kernel and slow to exit
            if left["running"]:
                for line in left["procs"]:
                    try:
                        os.kill(int(line.split()[0]), signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(2)
                left = running()
            self._json({"ok": True, "message": "stopped", "killed": killed,
                        "still_running": left["running"]})
        else:
            self.send_error(404)


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[strata-launcher] http://127.0.0.1:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

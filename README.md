# Strata Launcher

![Strata Launcher](strata-icon.png)

A web UI that helps you launch and manage [Strata](https://github.com/Niko1221/Strata) inference servers on
Intel Arc GPUs. Browse your local Strata model installs, configure engine flags through a visual panel, save
launch presets, and start a server with one click.

Strata is a purpose-built inference engine for the Qwen3.8-Flash-Next mixture-of-experts model family; its
Intel path is a SYCL port. This UI wraps the SYCL server so you do not have to memorize the flag set or
hand-write a config JSON every launch.

## Layout

```
strata-launcher/
├── server.py       ← Python backend (serves the UI, builds configs, starts/kills the engine)
├── index.html      ← The single-file web UI
├── flags_db.json   ← Catalogue of engine flags (built from the engine's own --help)
├── parse_flags.py  ← Regenerates flags_db.json from `strata --help` output
├── launch.sh       ← Port-probing starter (opens the browser)
├── strata-*.png/svg ← Icon assets
└── LICENSE, README.md, .gitignore
```

`python3 server.py` starts a web server on **http://localhost:9877** (`STRATA_LAUNCHER_PORT` to change it).
The browser loads `index.html`, which talks to the backend over a small JSON API.

## Requirements

- Python 3 (standard library only — no pip install)
- A working Strata install: the `strata` SYCL engine binary plus at least one model install
- Linux (the oneAPI environment is sourced automatically before every engine launch)

## Setup

The launcher expects two roots:

- **`STRATA_ROOT`** (env, default `~`) — the directory holding the `strata/` repo checkout.
- **`MODELS_DIR`** (env) — where your model installs live (`strata-coder-iq1-m/`, `strata-iq2-xs/`,
  `strata-mtp/` for the shared draft head).

Run:

```bash
export STRATA_ROOT="$HOME"          # so the repo is found at $HOME/strata
python3 server.py                   # http://localhost:9877
```

Or just `./launch.sh`, which probes the port first and opens the browser.

## What the UI does

- **Models** — scans the models directory for Strata installs (shard GGUFs + `pack/` + `rt/`) and reports
  which are ready to load.
- **Flags panel** — every engine flag with a checkbox and a value field/select, searchable. Changes re-render
  the exact command that will run, so the flags are never hidden in a config file.
- **Command panel** — the precise `server_intel.py …` invocation the launch will use.
- **Launch** — with a hover dropdown: plain launch, or launch with the image encoder (vision) when the
  selected model has a projector (`mmproj-*.gguf`) beside its shards.
- **Quick Launch** — paste or type a full command; the launcher prepends the oneAPI environment so a
  hand-typed Strata command works.
- **Presets** — save a full launch state (settings + every ticked flag) as a pill; click to restore,
  ✕ to delete. Saved presets live in the browser profile, so they are per-browser; the built-in mechanism
  (a preset defined in the code, so it exists in every profile) is present and currently ships empty — add
  your own to `BUILTIN_PRESETS` in `index.html` if you want one that survives a fresh profile.
- **Stats** — a per-launch and per-request record read back from the engine log.
- **Free VRAM** — unloads other GPU tenants (ComfyUI, ACE-Step, vLLM) so the expert cache gets the whole
  card; on this engine that is the difference between a fast and a starved launch.
- **Hardware panel** — live VRAM/RAM bars.

## A note on tuning

The tuning values that measured best on a 32 GB Arc-class card (whole expert set resident, native context
window, speculative decode with the English draft head) are worth starting from, but treat every number as a
starting point: size `--expert-cache` to **your** model's expert count and A/B the rest on your own hardware.
The flag panel's help text and the engine's own `--help` are the authority on what each flag does.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/models` | Installed models + readiness |
| GET | `/api/state` | Running? port, `/health` passthrough, engine log tail |
| GET | `/api/system` | VRAM/RAM/cores (cached probe) |
| GET | `/api/flags` | The flag catalogue |
| GET | `/api/preview` | The exact command a launch would run |
| POST | `/api/launch` | Build the config, start the engine |
| POST | `/api/quicklaunch` | Run a pasted command (with the oneAPI env prepended) |
| POST | `/api/kill` | Kill the launched session's process group |
| POST | `/api/free-vram` | Unload other GPU tenants |

## License

MIT — see `LICENSE`.

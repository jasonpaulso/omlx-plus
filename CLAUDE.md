# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

oMLX is an LLM inference server for Apple Silicon built on MLX: OpenAI/Anthropic-compatible HTTP API with continuous batching, paged SSD KV caching, and LRU model management, plus a SwiftUI menu-bar app (`apps/omlx-mac/`) that embeds the Python server. This repo is a fork (`origin` = jasonpaulso/omlx); upstream is jundot/omlx.

The `discovery/` directory is gitignored external repos — exclude it from analysis and searches.

## Commands

```bash
pip install -e ".[dev]"                    # dev install (Python 3.11–3.13)
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e . # with native kernels (needs full Xcode, CMake ≥3.27)

omlx serve --model-dir ~/models            # run server in foreground
omlx start / omlx stop                     # managed service (Homebrew/app)

pytest -m "not slow and not integration"   # unit tests (what CI runs)
pytest tests/test_file.py::test_name       # single test
pytest -m slow                             # needs Apple Silicon + local model files
pytest -m integration                      # needs a running server
pytest -m turboquant                       # TurboQuant KV cache suite

black --check .                            # format check (line length 88)
ruff check .                               # lint (vendored files under omlx/patches/*/vendor excluded)
mypy omlx                                  # type check
```

Test conventions and dev setup are in `docs/CONTRIBUTING.md`. Tests use one-test-file-per-source-module naming; `tests/conftest.py` provides MockTokenizer and a torch stub so unit tests don't need real models.

Packaging: `python packaging/build.py --venvstacks-only` builds the embedded-Python layers, then `apps/omlx-mac/Scripts/build.sh release` produces the .app (requires venvstacks 0.7.0+, full Xcode). Homebrew formula in `Formula/omlx.rb`.

Dependency pins matter: `mlx==0.32.0`, mlx-lm/mlx-vlm/mlx-embeddings pinned to git commits, `transformers>=5.12.1,<5.13`. Don't bump casually.

## Architecture

Request lifecycle: HTTP route (`omlx/server.py`, all endpoints live here) → `get_engine()` leases a model from `EnginePool` → request enqueued into `Scheduler.waiting` → `EngineCore` (`engine_core.py`) drives `scheduler.step()` on a dedicated MLX executor thread → decode runs through mlx-lm's `BatchGenerator` → `RequestOutputCollector` buffers tokens → SSE streaming response, engine lease released after stream.

- **`omlx/scheduler.py`** — continuous batching. `_schedule_waiting` admits requests into the running batch with homogeneity checks (cache status, VLM vs text, SpecPrefill isolation) and backpressure gates; waiting queue is capped and returns 503 when full. Aborts are deferred: `abort_request()` only records the ID; the abort applies at the next `step()`.
- **`omlx/engine_pool.py`** — LRU model management. `EngineEntry` tracks size estimates, lease counts, `is_pinned`; loading a model evicts LRU entries until it fits under the memory ceiling. `ProcessMemoryEnforcer` (in server.py) polls memory and can trigger eviction. `model_registry.py` enforces single-scheduler ownership of a model (prevents BatchKVCache corruption).
- **`omlx/cache/`** — paged KV cache with SSD offload (not GPU-tiered): `prefix_cache.py` does hash-based prefix lookup over `paged_cache.py` (block metadata) + `paged_ssd_cache.py` (disk I/O, LRU eviction). Only active when `paged_ssd_cache_dir` is set (`cache/factory.py`).
- **`omlx/api/`**, **`omlx/admin/`** — Pydantic request/response models and the admin dashboard sub-app. Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/messages` (Anthropic), `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/audio/*`, model load/unload, `/admin/*`.
- **`omlx/patches/`** — per-model-family compatibility fixes applied post-load (contains vendored upstream files). **`omlx/custom_kernels/`** — optional native kernels. **`omlx/speculative/`**, **`omlx/engine/dflash.py`** — speculative decoding paths.
- **Config**: `config.py` + `settings.py`, precedence CLI > env > settings file > defaults; the macOS app syncs via a bootstrap file.

The Mac app (`apps/omlx-mac/`, Xcode project) manages the embedded server process (`Sources/Server/`), talks to it over HTTP (`Sources/Net/OMLXClient`), and drives the menu bar UI. Commit messages follow conventional-commit style with scopes like `omlx-mac`, `tokenizer`, `dflash`.

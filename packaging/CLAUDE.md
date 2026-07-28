# Packaging

`packaging/build.py` builds the embedded Python layers (venvstacks) consumed by the Swift `.app` — it strips unused packages (torch/pandas/cv2, ~780 MB) and the `[bundle]` extra in `pyproject.toml` is its single source of truth for the mlx-base layer. Changes to the `omlx/` package need no `build.py` change (the bundle reinstalls the package); only touch `build.py` for the macOS packaging/codesign pipeline itself.

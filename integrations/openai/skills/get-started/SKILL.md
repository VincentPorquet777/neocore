---
name: get-started
description: Set up NeoCore memory on this computer, or check that it is working. Use after installing the NeoCore plugin, or when the user asks to set up, repair or check NeoCore.
---

# Set up NeoCore

NeoCore is the open-source `neocore-memory` Python package (Apache-2.0,
https://github.com/VincentPorquet777/neocore). It runs on this computer; nothing is sent to the
NeoCore team unless the user later opts in to "Help improve NeoSift".

1. **Check first.** Run `neocore doctor`. If it reports a working install, say so and stop.
2. **Explain before installing.** Tell the user, in a few lines, that setup will:
   - install the `neocore-memory` package with `pipx` (Python 3.10 or newer);
   - download a small embedding model (~35 MB, checked against a pinned fingerprint);
   - on a computer with a graphics card, also install NeoSift, the local relevance filter
     (a larger download). The user can skip it with `--no-sift`;
   - add NeoCore's memory server and two hooks to `~/.codex/config.toml` (and to the other AI
     tools it finds, such as Claude Code), so memory is recalled before each prompt and each turn
     is saved. Codex asks the user to trust new hooks the first time; NeoCore never trusts them
     for the user.
   Ask for a go. Do nothing until the user agrees.
3. **Install**, once they agree:
   - If `pipx` is missing, install it the usual way for this system (`python -m pip install --user pipx`
     then `python -m pipx ensurepath`), or ask the user.
   - `pipx install "neocore-memory[embed]"`
   - `neocore setup --dry-run` and show the user the plan (the files it will change).
   - `neocore setup` (add `--no-sift` if the user chose to skip NeoSift). Run from here, it leaves
     "Help improve NeoSift" off; the user can turn it on later with `neocore share on`.
4. **Verify.** Run `neocore doctor` and report the result plainly. Tell the user to start a new
   Codex conversation so the memory server and hooks load.
5. **Undo, if asked:** `neocore uninstall` removes the hooks and keeps the memory;
   `pipx uninstall neocore-memory` removes the package.

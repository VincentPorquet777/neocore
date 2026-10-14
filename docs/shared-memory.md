# Shared memory: one memory for every AI and every machine

Tell one AI something once; every other AI you use (Claude Code and Codex, on any of your
machines) knows it from its first message.

NeoCore does this with two things on each machine and one memory somewhere:

- **Hooks** feed each new prompt the memories that matter for it, and hand each finished turn back
  to the memory. Nothing to remember to do.
- **Memory tools** (MCP) let the AI look things up with intent: `neocore_search`,
  `neocore_open`, `neocore_timeline`, `neocore_facts`, `neocore_remember`.

## Pick a rung

Same program on every rung; only where the memory lives changes. Move between rungs at any time;
the memory itself is never deleted by setup.

| Rung | Where the memory lives | Good for | Needs |
|---|---|---|---|
| Solo | this machine | one computer | nothing |
| Home hub | an always-on computer at home | a few machines at home and on the move | Tailscale or a Cloudflare Tunnel |
| Own VPS | your server | many machines, always on | a small Linux VPS |
| Hosted | someone runs the server for you | no server to look after | a URL and a key |

## Install (one command)

Linux and macOS:

```bash
sh scripts/install/install.sh setup
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install/install.ps1 setup
```

The script makes a private Python in `~/.neocore/venv` (with `uv` when present, otherwise
Python 3.10+), installs NeoCore into it and runs `neocore` with your arguments. Run it again to
upgrade; nothing is added twice. `NEOCORE_PACKAGE` installs something else (a wheel file, say);
`NEOCORE_PYTHON` picks the Python.

`setup` is the Solo rung. The two other entry points are `server init` and `connect`, below.

## On the machine that keeps the memory (hub or VPS)

```bash
sh scripts/install/install.sh server init --person alex --device desktop --device laptop
```

This creates the server folder (`~/.neocore/server`, mode 700), issues **one key per device**
(files only you can read, under `device-keys/`), and on Linux installs and starts a systemd user
service (`neocore-memserver`, listening on `127.0.0.1:8793` and on a unix socket in the server
folder). It prints the next step for each device.

`--facts` lets the server form facts and sleep (retire facts a newer one replaced) with the model
sign-in on that machine (see `neocore.llm`). Without it, recall works on your own words only and
no model is ever called. The per-prompt relevance filter is off on a server.

Reaching the server from other machines:

- **Tailscale**: the server's tailnet address, e.g. `http://hub:8793`.
- **Cloudflare Tunnel**: route a hostname (e.g. `memory.example.com`) to `http://127.0.0.1:8793`.
- **SSH only**: no open port at all; each device connects with `--ssh HOST` (below).

## On every other device

Copy that device's key file over (a password manager or `scp`; never paste it into a chat), then:

```bash
sh scripts/install/install.sh connect --url https://memory.example.com --key-file laptop.key
```

Over SSH instead of HTTPS (the device keeps a local forward to the server's socket):

```bash
neocore connect --url http://127.0.0.1:18890 --key-file laptop.key \
  --ssh myvps --socket /home/alex/.neocore/server/http.sock
```

`connect` copies the key to `~/.neocore/memory.key` (mode 600), writes `remote.json`, adds the
hooks to Claude Code (`~/.claude/settings.json`) and Codex (`~/.codex/config.toml`), registers the
memory tools in both, and checks the server. Start a new chat: its first message has the memory.

`--tools-only` gives a device the memory tools without sending its own turns (a shared or guest
machine). `--dry-run` lists every step and writes nothing. `neocore disconnect` goes back to a
memory on this machine (the old settings are kept beside, switched off); `neocore status` shows
the server's health as this device sees it.

Every file setup changes is backed up beside itself first (`*.before-neocore`). Your own hooks
are left alone.

## When the server is away

Nothing is lost: each device uploads from its own Claude Code and Codex transcripts and moves its
place only after the server confirmed a batch, so while the server is away the transcript is the
queue. Turns arrive when it is back, in order and without duplicates; turns are redacted before
they leave the device. A prompt whose recall cannot reach the server in time just goes ahead
without memory; the AI is never held up.

## Keys and scopes

A key names one device of one person. It carries the scopes it may **read** and the one scope it
**writes**; the server stamps every captured turn with the key's write scope and recalls only from
its read scopes, whatever the device sends. Keys are stored hashed; the plain key only ever goes
to a file.

```bash
neocore server keys issue --device phone --person alex --out phone.key
neocore server keys list
neocore server keys revoke KEY_ID      # at once
```

Separate scopes keep separate memories on one server: `--write-scope team --read-scopes team` for
a team machine that must never see your personal scope, `--read-scopes personal,team` for you.
Facts never cross scopes either: a fact is shown only to a key that may read every turn it came
from, and sleep only compares facts within one scope.

## Running it

All on the server, all read-only unless said otherwise:

| Command | Shows |
|---|---|
| `neocore server scopes` | every scope: turns, conversations, facts, first and last activity, which keys read and write it |
| `neocore server audit --summary` | per device and endpoint: calls, failures, p50/p95 time (never any text) |
| `neocore server audit --failed --since 2026-10-14` | only errors and late recalls |
| `neocore server export --scope personal --out file.jsonl` | a scope's (or one `--session`'s) turns and facts, to a new file (mode 600) |
| `neocore server forget --session codex:ID --reason "asked"` | how much would be withdrawn; add `--yes` to withdraw, `--erase` to delete the text for good |

`forget` withdraws a scope, a conversation or a single turn. Withdrawn turns and every fact formed
from them stop appearing in recall, tools and exports at once; the withdrawal itself is recorded
with who did it and why. The audit log records who called what and how long it took, never
prompts, replies or memories.

## Privacy

- The memory is yours: one SQLite file on the machine you chose. Solo and self-hosted rungs send
  nothing anywhere else.
- A server calls no model unless you turn facts on (`--facts`), and then only the model sign-in
  on the server machine.
- Turns are redacted on the device before upload and again on the server, the fact former is told
  never to keep keys, tokens or passwords, and `neocore_remember` tells the AI the same. Still: do
  not paste secrets into chats.
- The embedder download (`setup`) is pinned by SHA-256; a file that does not match is refused.

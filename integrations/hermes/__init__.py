"""NeoCore memory provider for Hermes Agent.

The owner's own conversations (the CLI, the desktop app, and direct messages from the ids in
``owners``) join the machine's NeoCore memory: the same one Claude Code and Codex use after
``neocore setup``, or the memory server this machine is connected to (``neocore connect``). What
the owner tells any of them, Hermes knows, and the other way round. Every other conversation (a
group, someone else's direct messages) keeps a memory of its own and never sees the owner's.

Every finished turn is kept verbatim in one SQLite file. Before each turn
NeoCore recalls the dated excerpts that turn needs, with zero model calls (BM25, plus a small local
embedding model once it is downloaded). On a machine with a graphics card (NVIDIA, or Apple
silicon) NeoSift, NeoCore's own relevance model, then reads the turn and the recalled memories
locally and keeps only the ones that help. Permission is decided before ranking: a messaging chat
recalls only its own memory unless the ``scope`` setting says otherwise.

Settings live in ``$HERMES_HOME/neocore/config.json`` (``hermes memory setup`` writes them).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from agent.memory_provider import (
    MemoryProvider,
    RecallStatus,
    is_trivial_prompt,
    spawn_context_thread,
)

logger = logging.getLogger(__name__)

NAME = "neocore"
LABEL = "NeoCore"
DATA_DIR = "neocore"  # inside the profile's HERMES_HOME
EMBEDDER = "bge-small-en-v1.5-onnx-Q"
SCOPES = ("auto", "profile", "chat", "user")
SIFT_MODES = ("auto", "off")
SHARE_MODES = ("auto", "off")
DEFAULTS: dict[str, Any] = {"scope": "auto", "budget_chars": 8000, "dense_recall": True,
                            "neosift": "auto", "shared": "auto", "owners": ""}
HOST = "hermes"  # how the shared memory names Hermes's turns, next to claude-code and codex
SPAWN_SYNC = True  # tests turn the detached background sync off
# NeoSift scores a wide pool and keeps the best: recall this many excerpts for it to read.
SIFT_POOL: dict[str, Any] = {"top_spans": 48, "top_facts": 0, "budget_bytes": 40_000,
                             "focus_turns": 2}
SIFT_RETRY_DAYS = 7  # after a failed NeoSift install, try again this much later
PROFILE_SCOPE = "hermes"
SYSTEM_NOTE = (
    "NeoCore long-term memory is on. Before a turn, dated excerpts from earlier conversations "
    "may appear in a memory block. To look further back, call neocore_search. Call neocore_forget "
    "only when the person asks you to forget something."
)
SHARED_NOTE = (
    "NeoCore memory is on: one memory shared by every AI harness this person uses (Claude Code, "
    "Codex, Hermes, ...), on all their machines. Before a turn, dated items from it may appear "
    "in a memory block, each saying which AI and project it came from. To look further, call "
    "neocore_search; neocore_open reads one item in full, neocore_timeline shows what they "
    "worked on when, neocore_facts lists current facts. Call neocore_remember only when the "
    "person asks you to remember something."
)
# Recalled context Hermes fenced into a message is not something the person said: never store it.
_FENCED = re.compile(r"<memory-context>.*?</memory-context>", re.S | re.I)
_SEARCH_LIMIT = 48_000

SEARCH_TOOL = {
    "name": "neocore_search",
    "description": (
        "Search long-term memory (earlier conversations, verbatim and dated) for anything the "
        "current context does not hold. Returns excerpts with the id of each source message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "max_chars": {"type": "integer", "description": "Most text to return (default 8000)."},
        },
        "required": ["query"],
    },
}
FORGET_TOOL = {
    "name": "neocore_forget",
    "description": (
        "Delete one remembered message for good (only when the person asks you to forget "
        "something). Take the id from neocore_search. An audit record of the deletion stays, "
        "without the text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Source message id from neocore_search."},
            "reason": {"type": "string", "description": "Why, in the person's words."},
        },
        "required": ["id", "reason"],
    },
}

# The embedding model is read-only and the same for every profile, so one copy per folder serves
# every provider instance in the process (a gateway opens one per chat).
_EMBEDDERS: dict[str, Any] = {}
_EMBEDDERS_LOCK = threading.Lock()
_DOWNLOADS: set[str] = set()


def _tag(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def load_config(folder: Path) -> dict[str, Any]:
    config = dict(DEFAULTS)
    try:
        stored = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    if isinstance(stored, dict):
        config.update({k: v for k, v in stored.items() if k in DEFAULTS})
    if config["scope"] not in SCOPES:
        config["scope"] = DEFAULTS["scope"]
    try:
        config["budget_chars"] = max(1000, min(int(config["budget_chars"]), _SEARCH_LIMIT))
    except (TypeError, ValueError):
        config["budget_chars"] = DEFAULTS["budget_chars"]
    dense = config["dense_recall"]
    config["dense_recall"] = dense is True or str(dense).lower() in ("true", "1", "yes", "on")
    config["neosift"] = str(config["neosift"]).lower()
    if config["neosift"] not in SIFT_MODES:
        config["neosift"] = DEFAULTS["neosift"]
    config["shared"] = str(config["shared"]).lower()
    if config["shared"] not in SHARE_MODES:
        config["shared"] = DEFAULTS["shared"]
    config["owners"] = str(config["owners"] or "")
    return config


def owns(config: dict[str, Any], context: dict[str, Any]) -> bool:
    """Whether this conversation is the owner's own, and so joins the shared memory: the CLI,
    the desktop app or an API call (no messaging chat), everything with ``scope: profile`` (one
    person), and direct messages from an id listed in ``owners`` (``123``, or ``telegram:123``).
    A group or channel never does, whoever writes in it."""
    if config["shared"] != "auto":
        return False
    if config["scope"] == "profile":
        return True
    if not (context.get("gateway_session_key") or context.get("chat_id")):
        return True
    if str(context.get("chat_type") or "dm") != "dm":
        return False
    owners = {o.strip() for o in config["owners"].split(",") if o.strip()}
    platform = str(context.get("platform") or "")
    ids = {str(context.get(k) or "") for k in ("user_id", "user_id_alt")} - {""}
    return any(i in owners or f"{platform}:{i}" in owners for i in ids)


def sift_home() -> Path:
    """NeoSift's install is shared with NeoCore's other hosts (Claude Code, Codex) on the machine."""
    return Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore")


_SIFT_LOCK = threading.Lock()
_SIFT_BUSY: set[str] = set()


class SharedMemory:
    """The machine's NeoCore memory (``home``, normally ``~/.neocore``), or the memory server
    it is connected to. One per process: a gateway opens a provider per chat."""

    def __init__(self, home: Path) -> None:
        from neocore.shared import client

        self.home = home
        self.remote = client.load(home)
        self.lock = threading.RLock()
        self._memory: Any = None
        self._tools: Any = None
        self._synced = 0.0

    @property
    def local(self) -> bool:
        return self.remote is None

    def memory(self) -> Any:
        with self.lock:
            if self._memory is None:
                from neocore.assistant.memory import DevMemory

                self._memory = DevMemory(self.home)
            return self._memory

    def tools(self) -> Any:
        with self.lock:
            if self._tools is None:
                from neocore.shared.tools import Tools

                self._tools = Tools(self.memory(), write_lock=self.lock)
            return self._tools

    def reload(self) -> None:
        """Read the settings again next time (NeoSift or the embedder were just installed)."""
        with self.lock:
            if self._memory is not None:
                self._memory.store.close()
            self._memory = self._tools = None

    def sync_soon(self) -> None:
        """What a Claude Code or Codex stop hook does: a detached ``sync`` (index, facts, sleep,
        and this machine's Claude Code and Codex transcripts), at most once per ``SYNC_EVERY``."""
        from neocore.assistant import memory as dev

        if not SPAWN_SYNC or time.time() - self._synced < dev.SYNC_EVERY:
            return
        if dev._sync_due(self.home):
            self._synced = time.time()
            dev._detached("sync")


_SHARED: dict[str, SharedMemory] = {}


def shared_memory(home: Path) -> SharedMemory:
    key = str(home.resolve())
    with _EMBEDDERS_LOCK:
        if key not in _SHARED:
            home.mkdir(parents=True, exist_ok=True)
            from neocore.assistant import memory as dev
            from neocore.shared import client

            if client.load(home) is None and not (home / "config.json").exists():
                # Hermes is the first NeoCore on this machine: the settings `neocore setup`
                # writes, so Claude Code or Codex set up later join this memory as it is.
                dev._write_json(home / "config.json",
                                {"budget_bytes": 8000, "facts_enabled": True, "relevance": "off"})
            _SHARED[key] = SharedMemory(home)
        return _SHARED[key]


def install_sift(home: Path, marker: Path) -> None:
    """Install NeoSift once when this machine has a graphics card (its own Python environment and
    the pinned model, ~2-3 GB; it is switched on only if it passes its speed check). A failed try
    is not repeated for ``SIFT_RETRY_DAYS``. Recall works as before meanwhile."""
    key = str(home.resolve())
    with _SIFT_LOCK:
        if key in _SIFT_BUSY:
            return
        _SIFT_BUSY.add(key)
    try:
        from neocore.assistant import neosift

        if neosift.installed(home):
            return
        try:
            last = json.loads(marker.read_text(encoding="utf-8"))
            if time.time() - float(last.get("at", 0)) < SIFT_RETRY_DAYS * 86400:
                return
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        device = neosift.gpu()
        if not device:
            return
        logger.info("NeoCore: graphics card found (%s); installing NeoSift in the background",
                    device)
        try:
            settings = neosift.install(home)
            marker.unlink(missing_ok=True)
            config_path = home / "config.json"
            if config_path.exists():  # the shared memory: filter with it, as `neocore setup` does
                from neocore.assistant import memory as dev

                config = dev._read_json(config_path)
                if config.get("relevance", "off") != "filter":
                    dev._write_json(config_path, {**config, **settings})
                    shared = _SHARED.get(str(home.resolve()))
                    if shared is not None:
                        shared.reload()
            neosift.ensure(home)
            logger.info("NeoCore: NeoSift is ready; recall is filtered from now on")
        except Exception as error:  # noqa: BLE001 - recall keeps working unfiltered
            marker.write_text(json.dumps({"at": time.time(), "error": str(error)[:300]}),
                              encoding="utf-8")
            logger.warning("NeoCore: NeoSift not installed (%s); recall stays unfiltered", error)
    finally:
        with _SIFT_LOCK:
            _SIFT_BUSY.discard(key)


def scope_for(setting: str, context: dict[str, Any]) -> str:
    """The one memory this conversation reads and writes."""
    chat = str(context.get("gateway_session_key") or context.get("chat_id") or "")
    user = str(context.get("user_id") or "")
    if setting == "profile":
        return PROFILE_SCOPE
    if setting == "user" and user:
        return "user:" + _tag(f"{context.get('platform') or ''}:{user}")
    if setting in ("auto", "chat", "user") and chat:
        return "chat:" + _tag(chat)
    return PROFILE_SCOPE


def clean(text: Any) -> str:
    return _FENCED.sub("", str(text or "")).strip()


def model_folder(folder: Path) -> Path:
    return folder / "models" / EMBEDDER


def _embedder_ready(model: Path) -> bool:
    if not (model / "model_optimized.onnx").exists() or not (model / "tokenizer.json").exists():
        return False
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def shared_embedder(model: Path) -> Any:
    key = str(model.resolve())
    with _EMBEDDERS_LOCK:
        if key not in _EMBEDDERS:
            from neocore.embedder import LocalEmbedder

            _EMBEDDERS[key] = LocalEmbedder(model, threads=2)
        return _EMBEDDERS[key]


def download_embedder(model: Path) -> None:
    """Fetch the pinned model files once (each is refused unless its SHA-256 matches)."""
    key = str(model.resolve())
    with _EMBEDDERS_LOCK:
        if key in _DOWNLOADS:
            return
        _DOWNLOADS.add(key)
    try:
        from neocore.cli import _download_embedder

        _download_embedder(model)
        logger.info("NeoCore: embedding model ready; recall also matches meaning from next start")
    except Exception as error:  # noqa: BLE001 - recall keeps working by words; retried next start
        logger.warning("NeoCore: embedding model download failed (%s); word matching only", error)
    finally:
        with _EMBEDDERS_LOCK:
            _DOWNLOADS.discard(key)


class NeoCoreMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._folder: Path | None = None
        self._config: dict[str, Any] = dict(DEFAULTS)
        self._store: Any = None
        self._engine: Any = None
        self._engine_lock = threading.Lock()
        self._scope = PROFILE_SCOPE
        self._session_id = ""
        self._writes = True
        self._shared_chat = False
        # Per session: ids of messages the chat still holds in its context (not recalled back).
        self._held: dict[str, set[str]] = {}
        self._held_lock = threading.Lock()
        self._jobs: queue.Queue[Any] | None = None
        self._worker: threading.Thread | None = None
        self._status: RecallStatus | None = None
        # NeoSift: the last finished turn per session (it reads the end of the reply before),
        # and a scorer tests can put in its place.
        self._last: dict[str, tuple[str, str]] = {}
        self._sift_scorer: Any = None
        # The owner's conversations: the shared memory, and per session the time of its last
        # compression (turns before it may be recalled back) and its turn count.
        self._shared: SharedMemory | None = None
        self._since: dict[str, str] = {}
        self._turns: dict[str, int] = {}
        self._project = HOST

    # -- identity ----------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return NAME

    def is_available(self) -> bool:
        try:
            import neocore  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        return ("NeoCore's Python package (neocore-memory) is not installed; "
                "`hermes plugins enable neocore` installs it.")

    # -- lifecycle ---------------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        from neocore.recall import GovernedRecall
        from neocore.store import Store

        self._folder = Path(kwargs["hermes_home"]) / DATA_DIR
        self._folder.mkdir(parents=True, exist_ok=True)
        self._config = load_config(self._folder)
        self._session_id = session_id
        self._scope = scope_for(self._config["scope"], kwargs)
        self._shared_chat = self._scope != PROFILE_SCOPE
        # Subagents, cron runs and flushes read memory but never write it.
        self._writes = str(kwargs.get("agent_context") or "primary") == "primary"
        self._project = str(kwargs.get("agent_identity") or HOST)
        if owns(self._config, kwargs):
            self._start_shared(session_id, kwargs)
            return
        embedder = None
        model = model_folder(self._folder)
        if self._config["dense_recall"]:
            if _embedder_ready(model):
                embedder = shared_embedder(model)
            else:
                spawn_context_thread(download_embedder, name="neocore-model", args=(model,)).start()
        self._store = Store(self._folder / "memory.db")
        self._engine = GovernedRecall(
            self._store, embedder=embedder,
            embedding_model=embedder.model_id if embedder else "")
        if kwargs.get("resumed"):
            # The resumed transcript is back in context: do not recall it into itself.
            rows = self._store.rows("SELECT evidence_id FROM evidence WHERE thread_id=?",
                                    (self._thread(session_id),))
            self._hold(session_id, {str(r["evidence_id"]) for r in rows})
        self._jobs = queue.Queue()
        self._worker = spawn_context_thread(self._work, name="neocore-writer")
        self._worker.start()
        self._jobs.put(("index",))
        if self._config["neosift"] == "auto" and self._sift_scorer is None:
            from neocore.assistant import neosift

            home = sift_home()
            if neosift.installed(home):
                # Start the local server now, so the first real turn is already filtered.
                spawn_context_thread(neosift.ensure, name="neocore-sift", args=(home,)).start()
            elif self._writes:
                spawn_context_thread(install_sift, name="neocore-sift-install",
                                     args=(home, self._folder / "neosift-install.json")).start()

    def _start_shared(self, session_id: str, kwargs: dict[str, Any]) -> None:
        home = sift_home()
        self._shared = shared = shared_memory(home)
        self._jobs = queue.Queue()
        self._worker = spawn_context_thread(self._work, name="neocore-writer")
        self._worker.start()
        if not shared.local:
            return  # the server indexes, filters and forms facts
        from neocore.assistant import memory as dev

        config = dev._read_json(home / "config.json")
        model = home / "models" / EMBEDDER
        if self._config["dense_recall"] and not config.get("embedder_dir"):
            if _embedder_ready(model):
                dev._write_json(home / "config.json", {**config, "embedder_dir": model.as_posix()})
            else:
                spawn_context_thread(self._fetch_shared_embedder, name="neocore-model",
                                     args=(model,)).start()
        if self._config["neosift"] == "auto" and self._sift_scorer is None:
            from neocore.assistant import neosift

            if neosift.installed(home):
                spawn_context_thread(neosift.ensure, name="neocore-sift", args=(home,)).start()
            elif self._writes:
                spawn_context_thread(install_sift, name="neocore-sift-install",
                                     args=(home, home / "neosift-install.json")).start()
        # Open the store and load the embedder now, so the first real turn is not the slow one.
        spawn_context_thread(shared.memory, name="neocore-warm").start()

    def _fetch_shared_embedder(self, model: Path) -> None:
        download_embedder(model)
        if self._shared is not None and _embedder_ready(model):
            from neocore.assistant import memory as dev

            path = self._shared.home / "config.json"
            config = dev._read_json(path)
            if not config.get("embedder_dir"):
                dev._write_json(path, {**config, "embedder_dir": model.as_posix()})
            self._shared.reload()

    def shutdown(self) -> None:
        if self._jobs is not None and self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=10)
        self._jobs = self._worker = None
        if self._store is not None:
            with self._engine_lock:
                self._store.close()
            self._store = self._engine = None

    def system_prompt_block(self) -> str:
        return SHARED_NOTE if self._shared is not None else SYSTEM_NOTE

    # -- recall ------------------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._status = None
        if is_trivial_prompt(query):
            return ""
        session = session_id or self._session_id
        if self._shared is not None:
            try:
                rendered, count = self._shared_recall(query, session)
            except Exception as error:  # noqa: BLE001 - a recall failure must never fail the turn
                logger.warning("NeoCore shared recall failed: %s", error)
                return ""
            if count:
                self._status = RecallStatus(provider_label=LABEL, count=count)
            return rendered if count else ""
        if self._engine is None:
            return ""
        with self._held_lock:
            held = sorted(self._held.get(session, ()))
        budget = self._config["budget_chars"]
        if self._sift_ready():
            try:
                sifted = self._sift(query, held, session, budget)
            except Exception as error:  # noqa: BLE001 - unfiltered recall below instead
                logger.warning("NeoCore: NeoSift step failed (%s); recall unfiltered", error)
                sifted = None
            if sifted is not None:
                rendered, count = sifted
                if not count:
                    return ""  # NeoSift read them all: nothing recalled helps with this turn
                self._status = RecallStatus(provider_label=LABEL, count=count)
                return rendered
        try:
            result = self._recall(query, held, budget)
        except Exception as error:  # noqa: BLE001 - a recall failure must never fail the turn
            logger.warning("NeoCore recall failed: %s", error)
            return ""
        count = len(result.receipt.get("delivered", ())) + len(result.fact_ids)
        if not count or not result.rendered.strip():
            return ""
        self._status = RecallStatus(provider_label=LABEL, count=count)
        return result.rendered

    def _sift_ready(self) -> bool:
        if self._config["neosift"] != "auto":
            return False
        if self._sift_scorer is not None:
            return True
        from neocore.assistant import neosift

        return neosift.installed(sift_home())

    def _sift(self, query: str, held: list[str], session: str,
              budget: int) -> tuple[str, int] | None:
        """Recall wide, let NeoSift score every excerpt against this turn, keep the ones that
        help (best first, within the budget). ``None`` when NeoSift could not answer: the
        caller then recalls unfiltered, as on a machine without NeoSift."""
        from neocore.assistant import neosift
        from neocore.assistant.filter import Gate
        from neocore.recall import _render

        with self._engine_lock:
            result = self._engine.recall(query, allowed_scopes={self._scope},
                                         exclude_evidence_ids=held, **SIFT_POOL)
        excerpts: dict[str, list[Any]] = {}
        for span in result.spans:
            excerpts.setdefault(span.evidence_id, []).append(span)
        if not excerpts:
            return "", 0
        items = [{"kind": "excerpt", "id": evidence_id, "spans": spans,
                  "text": "\n".join(s.text for s in spans)}
                 for evidence_id, spans in excerpts.items()]
        context: list[dict[str, str]] = []
        if session in self._last:
            user, assistant = self._last[session]
            context = [{"speaker": "User", "text": user[:20_000]},
                       {"speaker": "Assistant", "text": assistant[:20_000]}]
        gate = Gate(sift_home(), {"relevance": "filter", "relevance_scorer": "neosift",
                                  "learning": "off", **neosift.POLICY}, scorer=self._sift_scorer)
        keep, _record = gate.select(query, context, items, "hermes")
        if keep is None:
            return None
        rendered, delivered, _facts = _render([s for item in keep for s in item["spans"]], budget)
        count = len({s.evidence_id for s in delivered})
        return (rendered, count) if count and rendered.strip() else ("", 0)

    def _last_turn(self, session: str) -> Any:
        from neocore.assistant.memory import Turn

        if session not in self._last:
            return None
        user, assistant = self._last[session]
        return Turn(HOST, session, "", self._project, "", user, assistant)

    def _shared_recall(self, query: str, session: str) -> tuple[str, int]:
        """The shared memory's recall, exactly as Claude Code and Codex get it (NeoSift or the
        hosted filter when the memory has one), minus what this chat already holds."""
        shared = self._shared
        assert shared is not None
        since = self._since.get(session, "")
        last = self._last_turn(session)
        if shared.local:
            delivered: set[str] = set()
            with shared.lock:
                memory = shared.memory()
                text = memory.recall(query, exclude_session=session, host=HOST, since=since,
                                     last=last, delivered=delivered)
                memory.hold(HOST, session, since, delivered)
            return (text, len(delivered)) if delivered and text.strip() else ("", 0)
        from neocore.shared import client

        answer = client.ask(shared.remote, shared.home, query, session, HOST,
                            seconds=6.0, since=since, last=last)
        if answer.get("error"):
            logger.info("NeoCore memory server: no memory this turn (%s)", answer["error"])
        text = str(answer.get("text") or "")
        count = sum(1 for line in text.splitlines() if line.startswith("["))
        return (text, max(count, 1)) if text.strip() else ("", 0)

    def recall_status(self) -> RecallStatus | None:
        return self._status

    def _recall(self, query: str, exclude: list[str], budget: int) -> Any:
        with self._engine_lock:
            return self._engine.recall(query, allowed_scopes={self._scope},
                                       exclude_evidence_ids=exclude, budget_bytes=budget)

    # -- capture -----------------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: list[dict[str, Any]] | None = None,
                  turn_author: dict[str, Any] | None = None) -> None:
        if not self._writes or self._jobs is None:
            return
        user, assistant = clean(user_content), clean(assistant_content)
        if not user and not assistant:
            return
        author = turn_author or {}
        if self._shared_chat and author.get("name") and user:
            user = f"{author['name']}: {user}"  # several people may share one chat
        metadata = {"source": "hermes"}
        if author.get("id"):
            metadata["author_id"] = str(author["id"])
        session = session_id or self._session_id
        self._last[session] = (user, assistant)
        if self._shared is not None:
            from neocore.assistant.memory import Turn

            count = self._turns[session] = self._turns.get(session, 0) + 1
            now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            now += f".{int(time.time() * 1000) % 1000:03d}Z"
            turn_id = _tag(f"{session}:{count}:{now}:{user[:200]}")
            self._jobs.put(("shared", Turn(HOST, session, turn_id, self._project, now, user,
                                           assistant)))
            return
        self._jobs.put(("turn", session, user, assistant, metadata))

    def _work(self) -> None:
        jobs = self._jobs
        while jobs is not None:
            job = jobs.get()
            if job is None:
                return
            try:
                if job[0] == "shared":
                    self._store_shared(job[1])
                    continue
                if job[0] == "turn":
                    _, session, user, assistant, metadata = job
                    from neocore.assistant.memory import redact

                    ids = self._store.capture_turn(
                        thread_id=self._thread(session), user=redact(user),
                        assistant=redact(assistant), scope=self._scope, metadata=metadata)
                    self._hold(session, set(ids))
                # Index in small steps so a recall never waits long behind a big first index.
                while True:
                    with self._engine_lock:
                        counts = self._engine.backfill(64)
                    if counts.get("vectors_added", 0) + counts.get("fact_vectors_added", 0) == 0:
                        break
            except Exception as error:  # noqa: BLE001 - keep the writer alive for later turns
                logger.warning("NeoCore capture failed: %s", error)
            finally:
                jobs.task_done()

    def _store_shared(self, turn: Any) -> None:
        shared = self._shared
        assert shared is not None
        if shared.local:
            with shared.lock:
                memory = shared.memory()
                memory.capture(turn)
                memory.recall_engine.backfill(64)  # the words now; meaning with the next sync
            shared.sync_soon()
            return
        from neocore.shared import client

        sent = client.send_turns(shared.remote, shared.home, [turn])
        if sent.get("error") and sent["error"] != "tools-only":
            logger.warning("NeoCore: turn not sent to the memory server (%s)", sent["error"])

    def flush(self) -> None:
        """Wait until every queued turn is stored and indexed (tests, shutdown)."""
        if self._jobs is not None:
            self._jobs.join()

    @staticmethod
    def _thread(session_id: str) -> str:
        return f"hermes:{session_id}"

    def _hold(self, session: str, ids: set[str]) -> None:
        with self._held_lock:
            self._held.setdefault(session, set()).update(ids)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        # Compressed turns leave the context: recall may bring them back from now on.
        with self._held_lock:
            self._held.pop(self._session_id, None)
        self._since[self._session_id] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return ""

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs: Any) -> None:
        with self._held_lock:
            if rewound:
                self._held.pop(new_session_id, None)
            if not reset and parent_session_id and parent_session_id in self._held:
                # /branch, /resume after compression: the history the new id carries is in context.
                self._held.setdefault(new_session_id, set()).update(self._held[parent_session_id])
        if not reset and parent_session_id in self._since:
            self._since.setdefault(new_session_id, self._since[parent_session_id])
        self._session_id = new_session_id

    # -- tools -------------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._shared is not None:
            from neocore.shared.tools import TOOLS

            return [{"name": t["name"], "description": t["description"],
                     "parameters": t["inputSchema"]} for t in TOOLS]
        return [SEARCH_TOOL, FORGET_TOOL]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self._shared is not None:
            return self._shared_tool(tool_name, args)
        if self._engine is None:
            return json.dumps({"error": "NeoCore memory is not initialized"})
        try:
            if tool_name == "neocore_search":
                return json.dumps(self._search(args))
            if tool_name == "neocore_forget":
                return json.dumps(self._forget(args))
        except Exception as error:  # noqa: BLE001 - tools report errors as JSON, never raise
            return json.dumps({"error": str(error)})
        return json.dumps({"error": f"unknown tool {tool_name}"})

    def _shared_tool(self, name: str, args: dict[str, Any]) -> str:
        """The shared memory's own tools (the ones Claude Code and Codex get), as plain text."""
        shared = self._shared
        assert shared is not None
        if name == "neocore_remember" and not self._writes:
            return "Not stored: this run only reads memory."
        try:
            if shared.local:
                from neocore.assistant.memory import SCOPE

                answer = shared.tools().call(name, args, scopes=frozenset({SCOPE}),
                                             write_scope=SCOPE, device=HOST)
                if name == "neocore_remember":
                    with shared.lock:  # found by every harness from the next prompt on
                        shared.memory().recall_engine.backfill(64)
                return answer
            from neocore.shared import client

            reply = client.mcp(shared.remote, shared.home, {
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": name, "arguments": args}})
            if not isinstance(reply, dict):
                return "memory unavailable: bad reply"
            if reply.get("error"):
                return str((reply["error"] or {}).get("message") or "memory unavailable")
            content = (reply.get("result") or {}).get("content") or []
            return "\n".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
        except Exception as error:  # noqa: BLE001 - tools report errors, never raise
            return f"error: {error}"

    def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "query is empty"}
        try:
            budget = max(500, min(int(args.get("max_chars") or 8000), _SEARCH_LIMIT))
        except (TypeError, ValueError):
            budget = 8000
        result = self._recall(query, [], budget)
        sources: dict[str, dict[str, str]] = {}
        for span in result.spans:
            sources.setdefault(span.evidence_id, {
                "id": span.evidence_id, "when": span.occurred_at, "speaker": span.speaker,
                "excerpt": span.text[:240]})
        return {"found": len(sources), "memory": result.rendered, "sources": list(sources.values())}

    def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(args.get("id") or "").strip()
        reason = str(args.get("reason") or "").strip() or "asked to forget"
        rows = self._store.rows("SELECT sensitivity FROM evidence WHERE evidence_id=?",
                                (evidence_id,))
        if not rows or str(rows[0]["sensitivity"]) != self._scope:  # the store calls scope that
            return {"error": "no message with that id in this conversation's memory"}
        with self._engine_lock:
            self._store.forget(evidence_id, reason=reason, by="agent")
            self._engine.refresh()
        return {"forgotten": evidence_id}

    # -- setup -------------------------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "scope", "default": DEFAULTS["scope"], "choices": list(SCOPES),
             "description": "Which memory a conversation reads and writes. auto: each messaging "
                            "chat has its own, CLI and desktop share the profile's. profile: one "
                            "memory for everything (one person). chat: always per chat. user: "
                            "follows a person across chats."},
            {"key": "budget_chars", "type": "integer", "default": DEFAULTS["budget_chars"],
             "minimum": 1000, "maximum": _SEARCH_LIMIT, "step": 1000,
             "description": "Most recalled text added before a turn, in bytes."},
            {"key": "dense_recall", "type": "boolean", "default": DEFAULTS["dense_recall"],
             "description": "Also match meaning, not only words: downloads a 65 MB embedding "
                            "model (bge-small, pinned by SHA-256) from Hugging Face once. Off: "
                            "word matching only, no download."},
            {"key": "neosift", "default": DEFAULTS["neosift"], "choices": list(SIFT_MODES),
             "description": "NeoSift, NeoCore's own relevance model: with a graphics card "
                            "(NVIDIA or Apple silicon) it reads each turn and the recalled "
                            "memories locally and keeps only the ones that help. auto: installs "
                            "it once in the background when a graphics card is found (~2-3 GB, "
                            "shared with NeoCore for Claude Code and Codex). off: never."},
            {"key": "shared", "default": DEFAULTS["shared"], "choices": list(SHARE_MODES),
             "description": "auto: your own conversations (CLI, desktop, and direct messages "
                            "from the ids in owners) use this machine's NeoCore memory, the one "
                            "Claude Code and Codex share (or the memory server it is connected "
                            "to), so every harness remembers the same things. Groups and other "
                            "people keep their own. off: Hermes keeps a memory of its own."},
            {"key": "owners", "default": DEFAULTS["owners"],
             "description": "Your own user ids on messaging platforms, comma separated (e.g. "
                            "telegram:123456789). Direct messages from them join the shared "
                            "memory. Empty: only the CLI and desktop do."},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        folder = Path(hermes_home) / DATA_DIR
        folder.mkdir(parents=True, exist_ok=True)
        current = load_config(folder)
        current.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})
        (folder / "config.json").write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")


def register(ctx: Any) -> None:
    ctx.register_memory_provider(NeoCoreMemoryProvider())

"""NeoCore memory provider for Hermes Agent.

Every finished turn is kept verbatim in one SQLite file inside the profile's home. Before each turn
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
DEFAULTS: dict[str, Any] = {"scope": "auto", "budget_chars": 8000, "dense_recall": True,
                            "neosift": "auto"}
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
    return config


def sift_home() -> Path:
    """NeoSift's install is shared with NeoCore's other hosts (Claude Code, Codex) on the machine."""
    return Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore")


_SIFT_LOCK = threading.Lock()
_SIFT_BUSY: set[str] = set()


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
            neosift.install(home)
            marker.unlink(missing_ok=True)
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
        return SYSTEM_NOTE

    # -- recall ------------------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._status = None
        if self._engine is None or is_trivial_prompt(query):
            return ""
        session = session_id or self._session_id
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
        self._last[session_id or self._session_id] = (user, assistant)
        self._jobs.put(("turn", session_id or self._session_id, user, assistant, metadata))

    def _work(self) -> None:
        jobs = self._jobs
        while jobs is not None:
            job = jobs.get()
            if job is None:
                return
            try:
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
        return ""

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs: Any) -> None:
        with self._held_lock:
            if rewound:
                self._held.pop(new_session_id, None)
            if not reset and parent_session_id and parent_session_id in self._held:
                # /branch, /resume after compression: the history the new id carries is in context.
                self._held.setdefault(new_session_id, set()).update(self._held[parent_session_id])
        self._session_id = new_session_id

    # -- tools -------------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_TOOL, FORGET_TOOL]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
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
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        folder = Path(hermes_home) / DATA_DIR
        folder.mkdir(parents=True, exist_ok=True)
        current = load_config(folder)
        current.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})
        (folder / "config.json").write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")


def register(ctx: Any) -> None:
    ctx.register_memory_provider(NeoCoreMemoryProvider())

"""NeoCore: governed long-term memory that recalls only what the moment needs.

For Claude Code and Codex, run ``neocore setup``. To give any other assistant a memory::

    from neocore import Memory

    memory = Memory("memory.db")
    memory.add(conversation="chat-1", user="I moved to Lisbon in May.", assistant="Noted!")
    print(memory.recall("Where do I live now?").rendered)

Recall makes no model calls. Facts (``form_facts``) are optional and use any model you pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

from neocore.recall import GovernedRecall, RecallResult  # noqa: E402
from neocore.store import Store  # noqa: E402

__all__ = ["GovernedRecall", "Memory", "RecallResult", "Store", "__version__"]


class Memory:
    """A store plus recall, for any assistant. Everything lives in one SQLite file."""

    def __init__(self, path: str | Path = ":memory:", *, embedder_dir: str | Path | None = None,
                 threads: int = 2, **recall_options: Any) -> None:
        self.store = Store(path)
        embedder = None
        if embedder_dir:
            from neocore.embedder import LocalEmbedder

            embedder = LocalEmbedder(Path(embedder_dir).expanduser(), threads=threads)
        self.engine = GovernedRecall(
            self.store, embedder=embedder,
            embedding_model=embedder.model_id if embedder else "", **recall_options)

    def add(self, *, conversation: str, user: str, assistant: str = "",
            at: str | None = None, turn: str = "", revision: str = "1", branch: str = "",
            scope: str = "internal", metadata: dict[str, Any] | None = None) -> list[str]:
        """Remember one exchange. ``at`` is an ISO time (default now); ``scope`` is who may
        recall it (recall names the scopes it may read)."""
        return self.store.capture_turn(
            thread_id=conversation, user=user, assistant=assistant, occurred_at=at,
            turn_id=turn, revision=revision, branch_id=branch, scope=scope, metadata=metadata)

    def recall(self, query: str, *, scopes: Sequence[str] = ("internal",),
               exclude_conversation: str = "", **options: Any) -> RecallResult:
        """What memory holds for ``query``: a dated, budgeted block (``.rendered``) and a
        receipt naming every source. Zero model calls."""
        excluded = [str(r["evidence_id"]) for r in self.store.rows(
            "SELECT evidence_id FROM evidence WHERE thread_id=?", (exclude_conversation,))
        ] if exclude_conversation else []
        return self.engine.recall(query, allowed_scopes=set(scopes),
                                  exclude_evidence_ids=excluded, **options)

    def form_facts(self, ask: Callable[[str, dict[str, Any]], dict[str, Any]], *,
                   max_batches: int = 4, roles: Sequence[str] = ("user",)) -> dict[str, int]:
        """Turn new messages into short dated facts with ``ask(prompt, schema) -> dict`` (see
        ``neocore.llm`` for Codex, Claude and OpenAI-compatible backends)."""
        from neocore.assistant.memory import _fact_extractor

        return self.engine.form_facts(_fact_extractor(ask), former="memory",
                                      max_batches=max_batches, roles=roles)

    def revoke(self, evidence_id: str, *, reason: str) -> None:
        self.store.revoke(evidence_id, reason=reason)

    def forget(self, evidence_id: str, *, reason: str) -> None:
        self.store.forget(evidence_id, reason=reason)
        self.engine.refresh()

    def abandon_branch(self, branch_id: str, *, reason: str) -> None:
        self.store.abandon_branch(branch_id, reason=reason)

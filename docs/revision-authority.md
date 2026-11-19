# Conversation revision authority

Governed recall identifies a stored message by its thread, branch, turn and role. User and
assistant messages in one turn have independent revision histories. When a legacy row lacks a
turn ID, its Evidence ID identifies the turn.

For each current branch, recall chooses the winning stored revision across all scopes. The
authority check projects the Evidence ID, scope and revision identity fields. Recall admits the
winner only when the caller has its scope, its access disposition is active and the inquiry
has not excluded it. A restricted, revoked or excluded winner suppresses older revisions.
The same eligible Evidence IDs govern excerpts and every cited source of a formed fact.

If every revision label in a family contains only ASCII digits, recall compares decimal
magnitudes. Leading zeroes are insignificant and counters may exceed a machine integer.
If the family includes any other label, all its labels use the existing lexical ordering.
For equal winning labels or equal decimal magnitudes, the first stored row wins. Capture time
can reflect a replay, so it does not override an explicit revision counter.

## Physical erasure

When a conversation message is forgotten, the store commits the audit entry, a revision marker
and the Evidence deletion in one transaction. The marker retains a SHA-256 digest of the
thread, branch, turn and role together, plus the supplied revision label needed for comparison.
A one-byte hash prefix of the thread divides markers into 256 buckets. Recall can skip threads
outside those buckets and always checks the full family digest before suppressing a revision.
It contains no excerpt, scope, raw thread or turn identifier, or arbitrary message metadata.
The existing audit entry retains the Evidence ID, source hash, reason, authorizer and time.
An exact capture replay cannot recreate the deleted row.

An erased revision participates in the same ordering and wins any tie at its rank. An older
surviving revision therefore stays out of automatic recall when the winner is erased. A strictly
higher revision can become current. An authorized explicit history lookup can still open a
surviving sibling by its own Evidence ID. A refresh purges derived excerpts and facts for the
deleted Evidence.

Stores opened for writing create the marker table automatically. Earlier physical deletions
left only the audit tombstone. When the deleted row is already gone, its family and revision
cannot be reconstructed from that tombstone. The new rule applies to erasures with a recorded
marker. A legacy row with malformed metadata can still be erased, but cannot supply a revision
marker. Read-only pre-migration snapshots retain their historical behavior.

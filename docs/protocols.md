# Protocol And State Invariants

## Object Identity

Reusable KV identity includes tenant, model, tokenizer, adapter, prefix hash,
block hash, layer, block index, token count, and physical layout digest.
Content hash equality alone is insufficient.

## Object Publication

```text
active tail -> sealed buffer -> pool publish -> object generation
            -> Ready event -> catalog record -> page-table lease
```

Publication becomes visible only after the pool reports the object as ready.
Retries must return the same committed generation or a newer generation that
invalidates the prior reference.

## Read Lease

A lease binds a pool id, lease id, expiration, and exact object generations.
The runtime validates it before `begin_step`. The pool guarantees those objects
will not be deleted or rewritten until expiry or explicit release.

## Step Transaction

`begin_step` records sequence generation, page-table generation, leases, and
per-layer plans. `commit_step` appends the active tail only when all generations
still match. `abort_step` releases the single-writer slot without publishing KV.

## Recovery

Persistent catalog records are hints. On restart, the controller resolves every
object with its pool, checks generation and layout, and only then installs a hot
directory entry. Missing objects are removed from the catalog.

## Tensor Handles

Tensor handles are ephemeral capabilities containing owner, device, address,
length, registration key, and generation. They may cross a trusted data-plane
protocol but are never written to Holt or exposed through the public gateway.

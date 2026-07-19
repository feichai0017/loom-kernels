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

## Attention Operation

The fast-path request contains Q, an optional atomic K_new/V_new append, an
output tensor, layout, mask, scale, deadline, and `KvView`. Device tensors cross
the data plane as registered handles. Historical KV crosses the protocol only
as ordered block identities. A sealed-prefix shard receives Q without an
append; exactly one mutable-tail owner may receive the append.

Every non-empty `KvView` must carry a non-zero page-table generation and at
least one lease id. Each block must belong to the requested layer and be bound
to one of those leases in the node runtime. The current K_new/V_new append and
attention execution form one ordered operation.

## Recovery

Persistent catalog records are hints. On restart, the controller resolves every
object with its pool, checks generation and layout, and only then installs a hot
directory entry. Missing objects are removed from the catalog.

## Tensor Handles

Tensor handles are ephemeral capabilities containing owner, device, address,
length, registration key, and generation. They may cross a trusted data-plane
protocol but are never written to Holt or exposed through public control APIs.

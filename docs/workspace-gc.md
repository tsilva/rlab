# Proof-gated successful workspace cleanup

Workspace cleanup is additive and dormant by default. It deletes only the five exact paths in
the immutable layout-v1 manifest for a successful queue-backed training launch. Failed and
canceled non-secret workspaces remain available for forensic inspection. Attempt credential
files and every container object that copied those credentials are removed on every terminal
outcome independently of workspace deletion.

## Safety boundary

A cleanup row is not claimable until all of the following are durable in PostgreSQL:

- authoritative train/evaluation/publication success and a closed artifact ledger;
- a full-byte SHA-256 readback receipt for every model, metadata, and recipe object;
- an exact object version under an attested non-expiring write-once policy;
- all expected telemetry obligations, including authenticated zero-batch closures;
- absence of retained telemetry batches, attempt env files, credential-bearing container
  objects, and active host-operation leases;
- an immutable cleanup proof for the current lifecycle generation.

Deletion is performed only by the host helper after a short-lived batch has been signed by the
isolated root signer. The helper validates the manifest, ownership identities, container mounts,
host boot, continuous-clock deadline, policy key, and monotonic batch epoch. A durable host
journal bridges helper mutation and database acknowledgement. Ambiguous delivery enters
`rollback_review`; it is never retried as a fresh destructive action.

## Services and policy

Install the root LaunchDaemon only after creating a root-owned mode-0600
`/Library/Application Support/rlab/workspace-signer/database.env` containing only
`WORKSPACE_SIGNER_DATABASE_URL`:

```sh
sudo /absolute/path/to/rlab/.venv/bin/python -m rlab.workspace_signer_service install \
  --key-revision key-v1 --replace
```

The private key stays under the root-only signer directory. Install only its public key on each
enabled host:

```sh
rlab fleet workspace-gc install-key \
  --key-revision key-v1 \
  --public-key '/Library/Application Support/rlab/workspace-signer-public.pem'
```

The workspace controller requires `RLAB_ARTIFACT_DURABILITY_POLICY_FILE` outside dormant mode.
The root-protected JSON policy must bind protocol version 1, endpoint, bucket, non-root prefix,
policy scope, verifier identity, and immutable preflight receipt SHA-256. These booleans must all
be true: `non_expiring_write_once`, `runtime_delete_denied`, `runtime_overwrite_denied`,
`runtime_policy_admin_denied`, `content_addressed_keys`, and `version_identity_required`.
Runtime artifact uploads use `.../sha256/<digest>/<filename>` keys. The controller streams every
stored object after upload and requires an object version identity; HEAD metadata and ETags are
not accepted as byte proof.

## Rollout

Use `rlab fleet workspace-gc status` and `doctor` first. A safe rollout is deliberately staged:

1. `qualify` enters paused qualification mode at an expected revision.
2. `create-schedule` records the exact host set and global concurrency cap.
3. Run the counterbalanced benchmark and canaries described by the schedule.
4. `record-qualification` accepts only evidence satisfying all throughput, tail, mixed-backlog,
   boundary, service-rate, and quiescence limits.
5. `complete-schedule` fails until every scheduled host has a receipt.
6. `enable-machine` binds each machine revision to its exact passed receipt SHA-256.
7. `promote` enters paused `promotion_verifying`; it does not resume ordinary admission.
8. Run one ordinary evidence-bound verification launch, then use `record-promotion` after its
   cleanup and journal cleanup complete and the quiescence gate passes.
9. `activate` performs the final revision-checked gate. It requires all admitted machines to be
   qualified, a healthy signer lease, no active host-operation lease or rollback-review row, and
   the ordinary promotion receipt before it can unpause work.

`disable` leaves ordinary work active but disables cleanup. `rollback` pauses new work and
disables deletion. Neither command erases receipts or review evidence.

## Drain and recovery

`rlab fleet drain --machine NAME` now requests drain and blocks new root claims immediately. The
machine controller continues only bounded active-lineage reconciliation, terminal credential
cleanup, and journal cleanup. It asks the helper for a nonce/revision-bound zero-residue receipt
and commits final `drained=true` only if the database zero predicate is still true in the same
ordered machine-control transaction. Direct `drained=true` writes are rejected by a database
trigger. There is no force-drain path while credentials, tracked containers, journals, deletion
states, or operation leases remain.

Completed deletion is terminal for host-dependent recovery. Reopen paths reject `deleting`,
`host_deleted`, `completed`, and `rollback_review`; operators should use durable object/W&B
evidence or an explicit review resolution instead of reconstructing host paths.

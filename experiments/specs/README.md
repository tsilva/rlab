# Structured Experiment Specs

Experiment specs are checked-in JSON documents that describe the hypothesis,
training delta, seed set, queue profile, and selection gate for a campaign
candidate.

Use them when the queue should own the experiment payload instead of relying on
ad hoc shell history.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-campaign add-spec-file \
  experiments/specs/mario-level1/b55-lowkl-lrdecay-post21-revalidate.json

UV_CACHE_DIR=.uv-cache uv run rlab-campaign enqueue-train-from-spec \
  experiments/specs/mario-level1/b55-lowkl-lrdecay-post21-revalidate.json \
  --runtime-image-ref-file rlab-train-image.json
```

The spec file is allowed to contain `run_name_template` and
`run_description_template` values with `{seed}` and `{utc}` placeholders.
`train_config` should omit secrets and should not include seed-specific fields
unless the spec intentionally runs a single seed.

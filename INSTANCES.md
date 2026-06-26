# GPU Instances

Last updated: 2026-06-25

Use this file as the repo-local source of truth for known GPU instances, launch targets, benchmark-backed concurrency, and operational gotchas. Re-check live availability before launching, but do not rediscover these basics from scratch unless the facts here fail.

## Quick Choice

| Use case | Target | Default shape |
| --- | --- | --- |
| Highest-throughput Mario PPO screening | `k8s/beast-3` | 5 concurrent children, `env_threads=4` |
| Lower-contention RTX4090 confirmation batch | `k8s/beast-3` | 3-4 concurrent children, `env_threads=4` |
| Small-GPU batch screening | `kubernetes` on beast-2 | 4 concurrent children, `env_threads=2` |
| Faster individual turnaround on RTX2060 | `kubernetes` on beast-2 | 2 concurrent children, `env_threads=4` |
| Modal baseline GPU launch | `modal-t4` | 1 child, `n_envs=32`, `env_threads=0` |

Refresh these defaults when changing `n_envs`, `n_steps`, model size, deterministic CUDA flags, runtime package version, W&B/artifact behavior, or target node CPU shape.

## Unified Compute Targets

Default control plane: the Mac-local SkyPilot API server at
`http://127.0.0.1:46580`. Beast-3 and RunPod are compute targets, not required
control-plane hosts. The old beast-3 SkyPilot API server
`http://192.168.0.151:46580` is a legacy/remote endpoint and should not be the
default if RunPod should work while beast-3 is off.

Local Mac control-plane setup as of 2026-06-24:

- `~/.sky/config.yaml` no longer pins `api_server.endpoint`; SkyPilot uses the
  default local endpoint.
- `sky api start --host 127.0.0.1` starts the local API/dashboard server.
- `sky check runpod` succeeds from the local API server using local
  `~/.runpod/config.toml`.
- Local kubeconfig context `beast-3` points at beast-3 Kubernetes via
  `https://beast-3:6443`; `sky check kubernetes` enables context `beast-3`.
  The k3s server certificate is valid for short host `beast-3` and LAN IP
  `192.168.0.151`, but not the full Tailscale FQDN
  `beast-3.tail50040f.ts.net`.
- beast-2 is managed from its host-side SkyPilot server against local k3s.
  Use `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26` and run SkyPilot
  with `KUBECONFIG="$HOME/.kube/config"`.

Machine-readable defaults live in `experiments/instances.json`. SkyPilot
manifests and runner profiles may set `"target": "<name>"`, and the
`rlab-compute` and `rlab-skypilot` CLIs accept `--target <name>` to override
that without editing JSON. Prefer `rlab-compute` for provider-neutral direct
launch manifests; keep using `rlab-skypilot` for SkyPilot-only runner profiles
and low-level SkyPilot utilities.

Current target names:

| Target | Alias examples | Infra | Default shape |
| --- | --- | --- | --- |
| `rtx4090` | `beast-3` | `k8s/beast-3` | 5 children, `env_threads=4` |
| `rtx2060` | `beast-2` | `kubernetes` | 4 children, `env_threads=2` |
| `runpod-rtx4090` | `runpod4090` | `runpod` | 1 child, `env_threads=2` |
| `runpod-l4` | `l4` | `runpod` | 1 child, `env_threads=2` |
| `runpod-t4` | `t4` | `runpod` | unavailable in current SkyPilot RunPod catalog |
| `modal-t4` | `modal`, `t4-modal` | Modal | 1 child, `n_envs=32`, `env_threads=0` |
| `local-macbook` | `macbook`, `local` | local CLI only | 1 child, no SkyPilot launch |

Examples:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute targets

UV_CACHE_DIR=.uv-cache uv run rlab-compute launch \
  experiments/launches/rlab_rtx4090.example.json \
  --target modal-t4

UV_CACHE_DIR=.uv-cache uv run rlab-skypilot render-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --target runpod-l4 \
  --output sky_train_runner_runpod_l4.yaml

UV_CACHE_DIR=.uv-cache uv run rlab-skypilot launch-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --target beast-3 \
  --output sky_train_runner_4090.yaml \
  --execute \
  --detach-run
```

RunPod support requires both the local client and the active SkyPilot API server
environment to have `skypilot[runpod]` installed and a valid
`~/.runpod/config.toml`. As of the first RunPod catalog check on 2026-06-24,
SkyPilot resolved `RTX4090` on RunPod to `1x_RTX4090_SECURE` with 5 vCPUs,
29 GB host memory, and about `$0.690/hr`; the repo defaults for RunPod are
therefore conservative until benchmarked.

On 2026-06-24, `sky check runpod` initially failed because the active
beast-3 SkyPilot API server venv lacked the RunPod extra and server-side
RunPod config. Installing `skypilot[runpod]==0.12.3.post1` into
`/home/tsilva/.local/share/skypilot/venv` and copying the local
`~/.runpod/config.toml` to beast-3 enabled RunPod.

The same day, SkyPilot's RunPod catalog returned no T4 offerings:
`sky gpus list T4 --infra runpod --all-regions` reported `Resources 'T4' not
found on RunPod`. Do not use `runpod-t4` unless a future catalog refresh shows
T4 availability again.

RunPod L4 did smoke successfully after transient per-region capacity misses:
`rlab-runpod-l4-smoke` eventually provisioned `1x_L4_SECURE` in the US at about
`$0.390/hr`, printed `NVIDIA L4, 23034 MiB, 580.159.03`, Python `3.10.12`, and
`torch_available False` in the base image. The cluster was torn down with
`sky down -y rlab-runpod-l4-smoke`.

RunPod `docker:runpod/base:1.0.2-ubuntu2204` tasks run as `root`, expose the
SkyPilot runtime at `$HOME/skypilot-runtime/bin/python` (`/root/...`), and may
define `sudo` as an empty alias. Do not render setup commands that prefer
`sudo -n ...` just because `sudo` appears in command lookup; use direct
`apt-get` when `id -u` is `0`, and resolve the bootstrap Python from
`${SKY_RUNTIME_DIR:-$HOME}/skypilot-runtime/bin/python` with `/home/sky/...` as
only a fallback.

Modal support is wired through `rlab-compute launch --target modal-t4` for
direct launch manifests. It dispatches to `modal run
src/rlab/modal_app.py::launch_manifest`, uses the same manifest training fields
as local/SkyPilot runs, and applies the target's `gpu`, `cpu`, and `memory_mib`
defaults through Modal `with_options`. ROMs must already be uploaded to the
`rlab-data` Modal volume with `modal run src/rlab/modal_app.py::upload_roms`.

## Prebuilt Train Runtime Images

The shared Docker/OCI runtime lives in `containers/train/` and is published by
`.github/workflows/rlab-train-image.yml` to GitHub Container Registry:

```text
ghcr.io/tsilva/rlab/rlab-train:git-<short-sha>
ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>
```

Use the digest form for real runs. The image contains repo code, `uv.lock`
dependencies, Stable Retro system libraries, and `rlab-*` entrypoints. It must
not contain `.env`, ROMs, checkpoints, W&B files, or run outputs.

SkyPilot runner profiles can opt into this runtime with:

```json
{
  "image_id": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>",
  "prebuilt_image": true
}
```

2026-06-25 BEAST-3 smoke: image
`ghcr.io/tsilva/rlab/rlab-train@sha256:0016b1322b2f4ba166185de13341871945933f660cf4a2c15b873c2b4ed13fdf`
successfully ran a SkyPilot train runner on `k8s/rtx4090`, claimed
`train_job=104` from the queue, and completed W&B run
`b78_beast3_imageq_smoke_s301_20260625T191227Z`
(`https://wandb.ai/tsilva/SuperMarioBros-NES/runs/7vhhuns0`). For
prebuilt-image runner YAML, omit `workdir: .` and render file mounts as
absolute host paths; otherwise the local SkyPilot API server can resolve mounts
from its own cwd instead of the repo cwd.

2026-06-26 BEAST-3 latest-image smoke: image
`ghcr.io/tsilva/rlab/rlab-train@sha256:7c7fb9e1a88b2dd87943486555ebb87a145a3c25516d121928a72b979f322560`
successfully launched on `k8s/beast-3` via cluster `rlab-b81-image-4090`.
Kubernetes pulled the digest, started pod `rlab-b81-image-4090-922ea937-head`,
and the runner job passed container smoke with Python `3.14.6`,
`stable-retro-turbo==1.0.0.post21`, Torch `2.12.0`, CUDA available on
`NVIDIA GeForce RTX 4090`, and
`uv_lock_sha256=36f9438f6c4bdbda12f5336fa1c6b2b4fb0f59f300ea8a7cf713bfffbdd49feb`.
The first runner attempt failed because its `status_goal` used a nonexistent
`-v2` goal; after restoring the existing `mario-l11-l12-post21-image-50x50`
goal, job `2` exited successfully with no pending `v2` train jobs to claim.

2026-06-26 RunPod smoke: image
`ghcr.io/tsilva/rlab/rlab-train@sha256:10af05929528b5895c0660830497ecc78adf42e4cbb91742388b0d980315d3ca`
successfully ran on `runpod-rtx4090` via cluster
`sandbox-sb3-b79-runpod-image-best-4090` in `EU-CZ-1`. The runner first idled
because no matching train job was pending; after enqueueing smoke
`train_job=111`, it completed W&B run
`b81_runpod_image_smoke_s921_20260626T102808Z`
(`https://wandb.ai/tsilva/SuperMarioBros-NES/runs/caw950xw`) and logged the
final model as an R2-backed reference artifact. The one-off RunPod cluster was
then torn down with `sky down -y sandbox-sb3-b79-runpod-image-best-4090`.
Before launching future queue runners, confirm there is a pending train job for
the exact `profile_id`, or use `--once` so an empty queue does not leave an idle
billable pod polling.

Modal can use the same image by setting `RLAB_MODAL_IMAGE_REF` before invoking
`modal run`; `RLAB_MODAL_REGISTRY_SECRET` names an optional Modal registry
secret for private GHCR pulls.

## W&B Artifact Storage: Cloudflare R2

Use Cloudflare R2 as the default byte store for W&B model artifacts so W&B keeps
metrics, config, aliases, lineage, and reference metadata instead of storing every
checkpoint zip directly.

Current repo-local `.env` is expected to define:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_ENDPOINT_URL
AWS_REGION
CHECKPOINT_BUCKET_URI
WANDB_API_KEY
```

The active smoke-tested storage target is the bucket referenced by
`CHECKPOINT_BUCKET_URI`. As of 2026-06-17, the R2 smoke test succeeded: bucket
exists, object upload/download worked, and the smoke object was deleted.

Training support:

- `rlab.train` reads `--wandb-artifact-storage-uri`, or falls back to
  `WANDB_ARTIFACT_STORAGE_URI`, then `CHECKPOINT_BUCKET_URI`.
- When that URI is set, checkpoint/final/best model zips upload to R2/S3 and W&B
  logs reference artifacts with the existing aliases such as `latest`, `final`,
  `best`, and `step-<N>`.
- Artifact objects are stored below `<game-id>/...` under the configured bucket.
  With the current R2 bucket this means `s3://wandb/<game-id>/...`. If the
  configured URI already ends with `<game-id>`, training uses it as-is;
  otherwise it appends that game-specific segment to the configured bucket/prefix.
- Without that URI, W&B artifact behavior remains direct file upload.

Preferred SkyPilot launch pattern from this repo:

```bash
(
  set -a
  . ./.env
  set +a

  sky launch -c <cluster-name> -y <task.yaml> \
    --env AWS_REGION \
    --env AWS_S3_ENDPOINT_URL \
    --env CHECKPOINT_BUCKET_URI \
    --secret AWS_ACCESS_KEY_ID \
    --secret AWS_SECRET_ACCESS_KEY \
    --secret WANDB_API_KEY
)
```

This passes only the bucket config and secrets needed by the task. Older task
YAMLs often mount `.env` to `~/.env` and source it in `run`; that works, but
prefer `--env`/`--secret` for new tasks and remove the `.env` file mount when
editing those YAMLs.

Local R2 smoke test:

```bash
(
  set -a
  . ./.env
  set +a

  uv --cache-dir .uv-cache run python scripts/setup_r2_bucket.py \
    "$(python - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ["CHECKPOINT_BUCKET_URI"]).netloc)
PY
)" \
    --prefix "$(python - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ["CHECKPOINT_BUCKET_URI"]).path.strip("/") or "wandb/_smoke")
PY
)"
)
```

## RTX4090: beast-3

### Access

- SkyPilot infra: `k8s/beast-3`
- GPU host SSH: `ssh tsilva@beast-3`
- Kubernetes API endpoint from the Mac: `https://beast-3:6443`
- Legacy beast-3 SkyPilot dashboard/API server on LAN: `http://192.168.0.151:46580`
- Legacy beast-3 SkyPilot dashboard/API server from the current Mac/Codex network: `http://100.118.135.59:46580`
- GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
- Observed driver: `595.71.05`, reporting CUDA support up to `13.2`
- Default SkyPilot image: `docker:us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:latest`
- Task Python bootstrap path inside the standard image: `/home/sky/skypilot-runtime/bin/python`

### Mario PPO Scheduling

Benchmarked for the current near-best Mario PPO shape: `n_envs=16`, `n_steps=512`, `batch_size=512`, `n_epochs=10`, `stable-retro-turbo==1.0.0.post10`, Torch `2.12.0+cu130`, 262,144 timesteps per child, W&B logging enabled, model artifact uploads disabled.

| Shape | Aggregate wall fps | Avg GPU util | Max VRAM | Notes |
| --- | ---: | ---: | ---: | --- |
| 1 child, `env_threads=4` | 2,881 | 16.7% | 990 MiB | Single trainer severely underuses the GPU. |
| 3 children, `env_threads=4` | 5,578 | 44.3% | 2,604 MiB | Good lower-contention confirmation shape. |
| 4 children, `env_threads=4` | 5,825-5,858 | ~52% | ~3,412 MiB | Good confirmation/screening compromise. |
| 5 children, `env_threads=4` | 6,242-6,271 | 57-58% | 4,219 MiB | Best measured aggregate throughput. |
| 5 children, `env_threads=2` | 5,878 | 51.9% | not recorded here | Lower contention but about 6% slower. |
| 5 children, `env_threads=1` | 4,615 | 37.0% | not recorded here | Underfeeds rollout collection. |

Default to 5 children with `env_threads=4` for screening queues. Use 3-4
children with `env_threads=4` when individual run latency, lower CPU
contention, or easier debugging matters more than total aggregate throughput.
Do not launch more than 5 children for this Mario PPO shape unless explicitly
running a short concurrency benchmark. On 2026-06-19, a live 6-child B33
reproduction used about 10 of 12 requested CPU cores and reached about 6.1k
aggregate fps, which was not better than the recorded 5-child results; the
extra child mainly increased CPU/env contention.

2026-06-19 update: the B34 strict `100/100` confirmation shape
(`n_envs=16`, `n_steps=512`, `batch_size=512`, `n_epochs=10`, W&B artifact
uploads enabled, checkpointing every 100k) OOMKilled with three concurrent
children under the 48 GB Kubernetes memory request, while a 64 GB request was
unschedulable. Two concurrent children completed cleanly. Until memory use is
re-benchmarked or reduced, use two concurrent children for this exact
confirmation/screening shape even though older short throughput benchmarks
favored more children.

### Operational Notes

- If a single RTX4090 is already held by an `UP` SkyPilot cluster with no active managed jobs, launching a second Kubernetes cluster may fail with `Insufficient nvidia.com/gpu`. Prefer submitting to the warm cluster with `sky launch -c <existing-cluster> -y <task.yaml>` when reuse is intended.
- If the local SkyPilot CLI reports the wrong endpoint, prefer `sky api start --host 127.0.0.1`
  and `sky api info` over logging into the beast-3 API server. Beast-3 should
  be a compute target, not the default control plane.
- For Mac-side Kubernetes/SkyPilot access to beast-3, prefer the short
  hostname endpoint `https://beast-3:6443` over the hardcoded LAN IP. Do not
  use `https://beast-3.tail50040f.ts.net:6443` unless the k3s API certificate
  is regenerated with that DNS SAN.
- Use normal `sky launch -c <warm-cluster> -y <task.yaml>` for valid repeat runs. Starting a second long trainer via ad hoc `sky exec` inside an already-running task produced pathological throughput around 140 fps.
- `sky cancel` against `k8s/beast-3` jobs can fail with `PermissionError: [Errno 13] Permission denied` while trying to `os.killpg`. If that happens, identify the training process group with `sky exec <cluster> 'ps -eo pid,ppid,pgid,stat,cmd | grep <run-name>'`, terminate the process group with `kill -TERM -<pgid>`, verify no trainer remains, then run `sky down -y <cluster>`.
- The local SkyPilot CLI may not expose a useful general `sky cp`. For small artifact retrieval from Kubernetes-backed clusters, identify the pod on `beast-3` and stream files from `/home/sky/sky_workdir` with `kubectl exec ... -- cat <remote-file> > <local-file>`.
- The default interactive kube context on the server may not have the `beast-3` alias. For manual Kubernetes inspection, explicitly use `KUBECONFIG=/home/tsilva/.kube/config`.
- Local SkyPilot `0.12.3.post1` Kubernetes launches from this Mac can hang in
  `INIT` after the pod-side runtime files and Ray setup have completed. The
  working local venv patch used on 2026-06-25 switches small `.runtime_files`
  and file-mount uploads from rsync-over-`kubectl exec` to `kubectl cp`, uses
  an absolute `kubectl` binary path, expands `~/sky_logs/...` before writing
  file-mount logs, and treats a timed-out bootstrap exec as success only after
  `/tmp/apt_ssh_setup_complete`, `/tmp/ray_skypilot_installation_complete`,
  `/tmp/env_setup_complete`, and `ray status --address 127.0.0.1:6380` are
  healthy. Preserve or upstream this before reinstalling the SkyPilot venv.

### Stable Retro Runtime Notes

- As of 2026-06-25, use `stable-retro-turbo==1.0.0.post21` for new local,
  Modal, and SkyPilot training/eval work. The repo dependency pin, lockfile,
  SkyPilot launcher default, and reusable launch manifests are expected to stay
  on post21 unless a future runtime migration is explicitly benchmarked.
  Post21 adds structured native `done_on_info` payloads with previous/next
  info-variable values. Post20 added native `done_on_info` termination in
  `StableRetroNativeVecEnv`;
  Mario mixed-level training should push `life_loss` and `level_change`
  termination into native per-lane autoresets instead of Python wrapper-level
  global resets.
- On 2026-06-23, a short RTX4090 campaign speed gate retried reset-time
  Level1-1/Level1-2 probability sampling with `state_probs=0.5,0.5`,
  `stable-retro-turbo==1.0.0.post16`, `n_envs=16`, and one queued child. It
  was canceled at iteration 3 because actual PPO-loop throughput was not faster
  than historical B50: `time/fps=174`, `fps_instant=126`, while rollout-only
  throughput was `726-808 fps`. This measured the current rlab wrapper
  path, which still routed `config.states` through `MixedStateNativeVecEnv`.
  Do not use this run as evidence against stable-retro-turbo native mixed-state
  support.
- Later on 2026-06-23, B65 reran that same short queue speed gate after the
  code was changed to always pass mixed `states` and optional `state_probs`
  directly into `StableRetroNativeVecEnv`. Train job `38`, W&B run `63smvp1y`,
  completed 262,144 timesteps on `k8s/rtx4090` with final `time/fps=3307`,
  `time/fps_instant=3286`, and `time/rollout_fps=4200`. Conclusion: native
  `StableRetroNativeVecEnv` fixed-slot and reset-time `states`/`state_probs`
  sampling are the supported mixed-state path going forward; the B64 slowdown
  measured sandbox wrapper overhead, not stable-retro-turbo native mixed-state
  support.
- On 2026-06-23, local isolated testing of `stable-retro-turbo==1.0.0.post18`
  validated the active-state API for task conditioning. `initial_state_names`
  is a tuple, `active_state_indices()` returns the same read-only `int32`
  NumPy view on repeated calls, explicit sampled resets mutate the view in
  place, and automatic lane resets update it before the next observation is
  consumed. Tested modes: single state, fixed per-lane states, weighted
  sampling via `state={"Level1-1": 0.5, "Level1-2": 0.5}`, and the repo's
  current `states=["Level1-1", "Level1-2"], state_probs=[0.5, 0.5]` form.
  The documented `state_probs={"Level1-1": 0.5, "Level1-2": 0.5}` constructor
  form failed with `ValueError: state_probs requires states`; use `state={...}`
  or `states` plus `state_probs` until that API/documentation mismatch is fixed.
  In fixed per-lane mode, repeated state names produce repeated
  `initial_state_names` entries with distinct active indices, so task
  conditioning should map active names to unique task ids rather than directly
  one-hotting slot indices when lanes intentionally duplicate a state.
- Later on 2026-06-23, B66 trained five queued RTX4090 seeds with
  `stable-retro-turbo==1.0.0.post18`, native reset-time 50/50 Level1-1/Level1-2
  sampling, and SB3 `MultiInputPolicy` task conditioning from the active-state
  one-hot vector. The setup smoke passed on the remote RTX4090, and throughput
  stayed healthy at about `1180-1196` final `time/fps` per child, roughly
  `5.9k` aggregate PPO fps for five concurrent children. All five runs finished
  5,005,312 timesteps and logged final R2/W&B artifacts, but did not solve the
  mixed-level criterion: final training completion rates were seed154 `0.45`,
  seed155 `0.47`, seed156 `0.60`, seed157 `0.65`, and seed158 `0.49`. Meaningful
  peak post-warmup rates were stronger (`0.71-0.94`) but not stable. Conclusion:
  one-hot task conditioning is supported and performant, but this B50-style
  sampled mixed-start recipe still needs additional changes before confirmation
  or eval promotion.
- On 2026-06-23, the repo was updated for the post19 native start-state API.
  `StableRetroNativeVecEnv` now receives only the single `state=` constructor
  argument: a string for one start state, a list for fixed per-lane states, or a
  state-to-weight dict for reset-time sampling. The repo CLI and metadata still
  accept `--states` and `--state-probs`, but the native boundary must not pass
  removed `states=` or `state_probs=` kwargs.
- The previous default `stable-retro-turbo==1.0.0.post14` was validated for
  native-vector life-loss termination on `SuperMarioBros-Nes-v0` and remains
  useful as a historical baseline for B39/B40/B44 comparisons.
- On 2026-06-19, `stable-retro-turbo==1.0.0.post14` was validated for
  native-vector life-loss termination on `SuperMarioBros-Nes-v0`: random-action
  vector probes emitted one-slot `done`s with `life_loss=True`, `died=True`,
  `terminal_observation`, and no wrapper-level `global_reset`. This resolves the
  old first-life-loss vector reset concern for Mario when the repo uses the
  native `terminate_on_life_loss`/`life_variable="lives"` path.
- On 2026-06-19, Modal CPU eval throughput for W&B artifact
  `tsilva/SuperMarioBros-NES/b31_post12_loosekl_5m_stop100ep100_clip015_targetkl012_clippeddx_seed23_20260618_192135-checkpoint:v44`
  improved from the old single-lane profile (`129.666s` for 100 episodes,
  `0.771 eps/s`) to the validated vectorized no-life-loss path (`25.627s`,
  `3.902 eps/s`), a `5.1x` speedup. The first naive vector attempt was faster
  but semantically wrong because Python-side completion termination caused
  whole-vector resets; the validated path disables Python completion termination
  inside the VecEnv and lets the evaluator finish lanes in batches.
- A follow-up Modal cost sweep on the same artifact found the best measured
  `$ / episode` point at `cpu=1`, `memory=4096`, `n_envs=20`: `31.436s` total
  for 100 episodes, `3.181 eps/s`, about `$0.0069 / 1000 episodes` at listed
  Modal CPU and memory rates. More lanes helped up to about 20; `n_envs=24`
  regressed to `36.751s`, and `n_envs=32` regressed to `64.559s`. The fastest
  measured wall-clock point was `cpu=8`, `memory=4096`, `n_envs=16` at
  `21.644s` for 100 episodes, but about `$0.0246 / 1000 episodes`. For balanced
  speed/cost without 8 CPU, `cpu=4`, `memory=4096`, `n_envs=24` took `25.877s`
  and cost about `$0.0159 / 1000 episodes`. Prefer checkpoint training metadata
  as the source of eval environment semantics, with `eval_queue --cpu 1
  --memory-mib 4096` for cheapest eval throughput.
- `stable-retro-turbo==1.0.0.post12` returns native-vector training observations in channel-first shape `(n_envs, 4, 84, 84)`, so the repo must skip `VecTransposeImage` for that shape and only apply it to channel-last `(n_envs, 84, 84, 4)` runtimes.
- On 2026-06-18, a single RTX4090 repro of W&B run `lexxixz3` with post12, seed 24, `n_envs=16`, `env_threads=4`, `target_kl=0.04`, and a strict `100/100` terminal-episode stop trained successfully to the 5M cap but did not early-stop: final `68/100` recent completions, `189` total completions, `5,005,312` timesteps, `27m34s` progress-bar wall time, and final logged fps `3023`.
- On 2026-06-18, a matched three-seed RTX4090 batch compared `stable-retro-turbo==1.0.0.post11` and `1.0.0.post12` with seeds `23`, `24`, and `25`, 3 concurrent children, `n_envs=16`, `env_threads=4`, `target_kl=0.04`, strict `100/100` terminal-episode stop, and a 5M cap. Neither version reached the strict stop. Final recent completion rates were post11: seed23 `19/100`, seed24 `90/100`, seed25 `22/100`; post12: seed23 `6/100`, seed24 `85/100`, seed25 `55/100`. Mean final rate was post11 `0.437` vs post12 `0.487`; median final rate was post11 `0.22` vs post12 `0.55`; total completions were post11 `576` vs post12 `428`. Final logged SB3 fps averaged `1917` for post11 and `1943` for post12 in this concurrent shape, so this training workload did not show the package-level `+23.6%` throughput increase.
- On 2026-06-18, a five-seed post12 follow-up used new seeds `26`-`30`, 5 concurrent children, and the same `lexxixz3` config. Final recent completion rates were seed26 `0/100`, seed27 `32/100`, seed28 `5/100`, seed29 `90/100`, and seed30 `66/100`; total completions were `0`, `390`, `15`, `413`, and `153`. Across all eight post12 seeds tested so far (`23`-`30`), mean final rate is `0.424`, median `0.435`, max `0.90`, and total completions `1399`. The five-child batch averaged `1353` final logged fps per child, about `6766` aggregate fps, while the earlier three-child post12 batch averaged `1943` per child, about `5829` aggregate fps.

Manual inspection examples:

```bash
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl get pods -n default -o wide'
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl logs -n default <pod> -c ray-node --tail=200'
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl exec -n default <pod> -c ray-node -- bash -lc "<command>"'
```

## RTX2060: beast-2

### Access

- SkyPilot infra: `kubernetes` from the host-side beast-2 SkyPilot server
- Legacy SkyPilot SSH node-pool infra: `ssh/beast2`
- Local Mac kubeconfig context: `beast-2`, with cluster `beast-2`, authinfo
  `beast-2`, and namespace `default`; the context uses a SkyPilot SSH tunnel to
  local API endpoint `https://127.0.0.1:6443`.
- GPU host IP observed from the current Mac/Codex network: `192.168.133.26`
- SSH command: `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26`
- GPU: NVIDIA GeForce RTX 2060, 6 GB VRAM
- Observed driver: `595.71.05`
- Kubernetes: k3s `v1.35.5+k3s1` on node `beast-2`, containerd
  `2.2.3-k3s1`, GPU advertised as `RTX2060` / `nvidia.com/gpu: 1`
- NVIDIA runtime classes: `nvidia`, `nvidia-cdi`, `nvidia-legacy`
- SkyPilot server venv on host: `/home/tsilva/skypilot-server/.venv`
- Host `uv`: `/home/tsilva/.local/bin/uv`
- Host-side SkyPilot API server: localhost-only at `http://127.0.0.1:46580`; do not bind to `0.0.0.0` without explicit security approval.

As of 2026-06-26, `beast-2` did not resolve from the Mac and
`192.168.133.26` reported host down on SSH, so the local Mac kube context could
not establish its SkyPilot SSH tunnel. When reachable, use the IP plus
`HostKeyAlias=beast-2` unless hostname resolution has been re-established.
`ssh-keyscan -T 5 192.168.133.26` previously matched the existing `beast-2`
host keys in `~/.ssh/known_hosts`.

When running SkyPilot from the host-side CLI on `beast-2`, set `KUBECONFIG="$HOME/.kube/config"` first. Without it, SkyPilot or `kubectl` may try `/etc/rancher/k3s/k3s.yaml` and fail with permission denied.

Example host-side launch shape:

```bash
ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26 \
  'cd /home/tsilva/rlab && export KUBECONFIG="$HOME/.kube/config" && PATH="$HOME/.local/bin:$PATH" /home/tsilva/skypilot-server/.venv/bin/sky launch --infra kubernetes -c <cluster-name> -y <task.yaml>'
```

As of 2026-06-25, the first local training image imported into k3s is
`ghcr.io/tsilva/rlab/rlab-train:local-8332822-dirty`, digest
`sha256:873484b80a09723a8bdd78baadcc357d5bfe5f4c145c48d8eda4ae2880ddc5bf`.
It is present only in beast-2's k3s/containerd image store, not GHCR. Replace
this with a pushed immutable digest once registry publishing is approved.

SkyPilot `0.12.3.post1` hard-codes `imagePullPolicy: Always` for the
Kubernetes Ray node. For the local imported image test, the host-side template
at `/home/tsilva/skypilot-server/.venv/lib/python3.13/site-packages/sky/templates/kubernetes-ray.yml.j2`
was patched to `IfNotPresent` for the `ray-node` container, with backup
`kubernetes-ray.yml.j2.bak-20260625-local-images`. Without that patch, k3s
tries to pull the private/unpublished GHCR tag and fails with `403 Forbidden`.

The 2026-06-25 SkyPilot Kubernetes retry successfully provisioned the pod and
started Ray, but the launch remained in `INIT` because SkyPilot's Kubernetes
`rsync` helper hung while syncing `/root/.sky/.runtime_files`. The verified
B77 Docker/Kubernetes smoke therefore ran as a plain Kubernetes Job,
`rlab-b77-docker-k8s-103`, using the same imported image and hostPath ROM
mount. Re-test SkyPilot task submission before relying on `rlab-skypilot
launch-runner` for beast-2 Kubernetes queues.

### Mario PPO Scheduling

Benchmarked on 2026-06-17 with `stable-retro-turbo==1.0.0.post7`, Torch `2.12.0+cu130`, 131,072 timesteps per child, and W&B disabled to avoid upload-time contamination.

| Shape | Aggregate wall fps | Per-child wall fps | Avg GPU util | Max VRAM | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 child, `env_threads=4` | 1,248 | 1,248 | 32% | 695 MiB | Fastest single-job setting. |
| 2 children, `env_threads=4` | 1,971 | 986-993 | 56% | 1,175 MiB | Best fast-turnaround setting. |
| 3 children, `env_threads=4` | 2,473 | 824-830 | 70% | 1,655 MiB | Good compromise. |
| 4 children, `env_threads=2` | 2,661 | 665-669 | 77% | 2,135 MiB | Best measured aggregate throughput. |
| 4 children, `env_threads=4` | 2,661 | 665-672 | 75% | 2,135 MiB | Same aggregate as `env_threads=2`, no clear advantage. |

Default to 4 children with `env_threads=2` for aggregate screening throughput. Use 2 children with `env_threads=4` when individual result latency matters.

### Operational Notes

- `sky check kubernetes --verbose` from beast-2 with `KUBECONFIG="$HOME/.kube/config"` is the authoritative confirmation that the host-side SkyPilot server sees local k3s.
- Persistent setup to reuse: k3s, NVIDIA GPU operator/runtime classes, SkyPilot server venv, localhost API server, host tools, local ROM bundle, and imported prebuilt training images.
- Disposable per-job setup: SkyPilot cluster/pod creation, workdir sync, YAML `setup`, image import when the tag is new, and dataset download unless cached.
- For local imported images, use `imagePullPolicy: IfNotPresent` or publish the
  image to GHCR with registry credentials. SkyPilot's default `Always` policy
  ignores the local containerd image store.
- The RTX2060 training-validated Mario path includes `stable-retro-turbo==1.0.0.post7`; it reproduced the completed-episode stop criterion at 2,711,552 timesteps on 2026-06-15.

## Cleanup

After one-off experiments, clean up clusters unless the user explicitly wants a warm cluster left running:

```bash
sky down -y <cluster-name>
```

For `beast-2` host-side commands, run cleanup through the host SkyPilot venv and explicit kubeconfig:

```bash
ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26 \
  'cd /home/tsilva/rlab && export KUBECONFIG="$HOME/.kube/config" && PATH="$HOME/.local/bin:$PATH" /home/tsilva/skypilot-server/.venv/bin/sky down -y <cluster-name>'
```

## Related Repo Files

- `GOAL.md`: current RTX4090 scheduling decision for the active screening goal.
- root-level `sky_*.yaml`: ignored local launch files; promote reusable shapes under
  `experiments/launches/`.

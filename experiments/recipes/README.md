# Experiment Recipes

Machine-readable recipes define reusable training ingredients: environment
semantics, PPO hyperparameters, reward shaping, and logging defaults.

Use a first-class `environment` section for the MDP interface the agent acts
within. It should name the provider and `env_id`, then capture
state sampling, action mapping, preprocessing, termination/event semantics, task
conditioning, and reward shaping. `rlab` materializes this section into
`train_config` for the existing runners and records a deterministic
`environment_hash` over the canonicalized environment identity.
Use `preprocessing.obs_resize: [height, width]` for observation resizing and
`preprocessing.obs_crop: [top, right, bottom, left]` for observation cropping;
do not duplicate them as runtime-specific `observation_size` or `hud_crop_top`
fields in `environment`.

Keep optimizer and execution details such as learning rate, rollout length,
`n_envs`, `env_threads`, runtime image, and W&B naming outside `environment`;
those change the training process, not the environment identity.

Goal specs under `experiments/goals/<goal>/specs/` remain the launchable queue
documents. A YAML spec may use Hydra `defaults` to compose recipe files, set goal-specific
fields such as `environment.state`, `seeds`, `run_target`, and W&B naming, then
use `overrides` for the small delta that defines the candidate.
Queue-ready recipe fragments may also use `env`, `train`, `reward`, and
`logging` sections; `rlab` merges those into the final `train_config` before
validation. Prefer that shape when converting older specs that must preserve the
exact runner-facing payload.

Keep `experiments/goals/<goal>/goal.yaml` as the stable goal contract. Mutable
scoreboard state such as current baselines, near-misses, revalidation status,
and historical run evidence belongs in `experiments/goals/<goal>/evidence.yaml`
by convention.

At enqueue time, `rlab train --spec-file <path>.yaml` resolves every Hydra
`defaults` entry, materializes the final `train_config`, validates the existing expanded
train-spec contract, and stores composition source hashes in `spec_payload_json`.
The queue still executes a fully expanded immutable payload.

Use `profile` only for queue/eval execution lanes such as profile-locked eval
workers. Do not use it as a synonym for hyperparameter recipe; use `recipe`,
`spec`, or `lane` instead.

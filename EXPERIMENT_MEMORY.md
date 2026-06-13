# Mario PPO Experiment Memory

Goal: train PPO on `SuperMarioBros-Nes-v0` until Mario reaches maximum reward / strong level progress.

## Current Findings

- The original stable-retro scenario reward was defective for training: `scenario.json` rewards only `xscrollLo`, the low byte of scroll position, which wraps every 256 pixels.
- Training now ignores retro reward by default and uses wrapper-computed global x-position progress.
- Reward mode now defaults to SuperMarioRL-style bounded rewards: `reward = clip(terminal_or_capped_progress, -30, 30) / 30`, where progress is `min(new_global_max_x_delta, 30)`, death overrides to `-30`, and level completion overrides to `+30`. The legacy additive shaping remains available with `--reward-mode additive`.
- Reward-progress correction after level changes: `levelHi/levelLo` changes now freeze the prior level's best x-position into `completed_level_base`, reset within-level x tracking, and keep rewarding new progress on top of the global baseline. This avoids the old post-level-change tail where `xscrollHi/xscrollLo` reset to zero and the agent received no progress reward after completing a level.
- For level-1-only training, use `--terminate-on-level-change --completion-x-threshold 0`: real `levelHi/levelLo` transitions mark completion, pay the bounded positive terminal reward, and end the episode immediately instead of continuing into later levels or using a near-end x-threshold proxy.
- Episodes now terminate on first life loss by default, preventing repeated early-progress farming after death.
- Deterministic argmax eval is brittle early in PPO. Use stochastic eval/play for early policy inspection, while still tracking deterministic eval later.
- Current best corrected run: `modal_fixed_reward_gpu_250k_lr1e4_env16`.
- Best corrected W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/jtegwgl6`.
- Best corrected checkpoint reached stochastic eval mean reward `1082.7` at 250k timesteps over 10 eval episodes. The run still showed large variance and mid-run regressions, so continue using best artifacts/checkpoints instead of assuming the latest policy is best.
- More robust 50-episode evaluation in the continuation run suggests the 10-episode `1082.7` score was optimistic/noisy.
- Current best robust run: `modal_right_action_250k_lr1e4_eval50`, with 50-episode eval mean reward `765.42` at 100k timesteps.
- Training now tracks Mario-specific eval metrics: global max x-position, within-level max x-position, completion rate, death rate, death x-position histogram, and one best-episode video per eval window. Future runs should optimize for consistency and completion, not reward mean alone.
- The 500k long-horizon run did not solve level completion. It confirmed reward mean can improve while `max_x_max` and `completion_rate` do not, so select future models by completion/progress metrics and add PPO stabilization or curriculum.
- Current best resumed completion-aware run: `modal_resume_step99968_completion_bonus_125k`, best checkpoint step 99,968 reached `max_x_max=2341` over 50 stochastic eval episodes with no completions.
- Avoid using latest checkpoint by default. In recent resumed training, checkpoint 99,968 beat 124,960; selection by out-of-process checkpoint eval is necessary.
- Current default stable-retro-turbo observation pipeline: `obs_crop=(32, 0, 0, 0)`, `obs_resize_algorithm="area"`, `obs_resize=(84, 84)`, `obs_grayscale=True`, `frame_skip=4`, `frame_stack=4`, and `maxpool_last_two=True`.
- Current training rollout env now uses `StableRetroNativeVecEnv` directly instead of `StableRetroSubprocVecEnv`. Native stable-retro-turbo owns crop, resize, grayscale, frame skip, frame stack, max-pool, autoreset, and the batched HWC observation buffer; Python vector wrappers only map restricted discrete Mario actions, compute custom progress rewards/metrics, apply configured custom terminal events with a full-vector reset fallback, and transpose HWC to CHW for SB3.
- Terminal/autoreset handling has been audited for SB3 PPO: native terminal events from `StableRetroNativeVecEnv` autoreset per slot and provide `info["terminal_observation"]` with `TimeLimit.truncated=false`; `VecTransposeImage` converts those terminal observations to CHW before PPO sees them. Python custom terminal events capture the current HWC observation as `terminal_observation` before reset. Because post3 lacks per-env Python reset, any Python custom terminal triggers a full-vector reset; unaffected slots are explicitly marked `TimeLimit.truncated=true` and given terminal observations so SB3 bootstraps value targets instead of treating them as true terminals.
- Model evaluation now also uses `StableRetroNativeVecEnv` through `make_vec_envs(..., n_envs=1)` for training-loop eval, standalone `mario_ppo.evaluate`, W&B checkpoint evaluation, and best-episode policy selection. Video generation still replays the selected discrete action trace with the render-capable single env for visualization only; policy inference/eval metrics use the native vector path.
- Local stable-retro-turbo `1.0.0.post2` native-interface throughput benchmark with `StableRetroNativeVecEnv`, `copy_observations=false`, `obs_resize_algorithm="nearest"`, crop 32, grayscale 84x84, frame skip 4, 4-frame stack, and `maxpool_last_two=true`: about `957` SPS at 1 env/1 thread, `2862` SPS at 8 envs/8 threads, `2415` SPS at 16 envs/8 threads, `2484` SPS at 32 envs/8 threads, `3379` SPS at 48 envs/8 threads, and `2848` SPS at 64 envs/8 threads. A 48-env sensitivity run with 4 threads reached only `2449` SPS, so 48 envs with 8 threads was the best native setting from this sweep. Treat this as the native-interface result; the separate `make_fast_mario_env` numbers below measure the subprocess/training-env path.
- Local stable-retro-turbo `1.0.0.post2` native-interface throughput benchmark with the default `obs_resize_algorithm="area"` and otherwise the same native settings reached about `1041` SPS at 1 env/1 thread, `3577` SPS at 8 envs/8 threads on a 5k-step sample, `3602` SPS at 16 envs/8 threads on a 3k-step sample, and `3906-4126` SPS at 32 envs/8 threads across repeated 2k-step samples. A 48-env short sample reached `4029` SPS, but a longer 2k-step sample dropped to `2626` SPS; 64 envs reached `3247` SPS. Prefer 32 envs / 8 threads for sustained native-area benchmarking from this sweep.
- Follow-up source-build isolation for stable-retro-turbo `1.0.0.post2`: the installed sandbox wheel loaded `_retro` from `.venv` and only reached about `3505` SPS at 8 envs/8 threads, `3863` SPS at 16 envs/16 threads, `4022` SPS at 32 envs/32 threads, `4003` SPS at 32 envs/16 threads, and `3303` SPS at 64 envs/16 threads with native area preprocessing. Running the same benchmark against `/Users/tsilva/repos/tsilva/stable-retro-apple-silicon/build/lib.macosx-26.0-arm64-cpython-314` with explicit ROM/info/scenario paths reached `7304` SPS at 8 envs/8 threads and `8941` SPS at 32 envs/16 threads. The Python `vec_env.py` wrappers matched, but the compiled `_retro` extension hashes differed, so the direct-repo ~8-9k results do not match the currently installed wheel in this sandbox; they match the local source build.
- stable-retro-turbo `1.0.0.post3` macOS arm64 wheel is now the local pin; `uv.lock` records PyPI SHA256 `97e541d1ef047d0a4dfcead5594a20a10ed1c249dc2f61760ef6888db903cfe8`. Because only the macOS arm64 post3 wheel was published, `pyproject.toml` pins post3 for macOS arm64 and keeps post2 for Linux x86_64 until a Linux post3 wheel exists. The installed post3 `_retro` extension hash matches the local source build.
- Local stable-retro-turbo `1.0.0.post3` native direct-ROM `State.NONE` benchmark with area resize, crop 32, grayscale 84x84, frame skip 4, frame stack 4, maxpool, audio disabled, and `copy_observations=false`: about `7247` SPS at 8 envs/8 threads, `7365` SPS at 16 envs/16 threads, `6730` SPS at 32 envs/32 threads, `6717` SPS at 32 envs/16 threads on the first sample, `8038` SPS at 32 envs/16 threads on repeat, and `7561` SPS at 64 envs/16 threads. This confirms the published post3 wheel has the fast native binary path, though run-to-run variance remains.
- Important performance distinction: using the imported/default `SuperMarioBros-Nes-v0` game path or explicit ROM with `Level1-1.state` remains much slower than direct-ROM `State.NONE`; installed post3 measured about `3732` SPS at 8 envs/8 threads through the default game path and `3997` SPS at 32 envs/16 threads with explicit `Level1-1.state`. Use the `State.NONE` numbers only for native runtime comparison; use the Level1-1/default numbers for Mario PPO training-state expectations.
- Local stable-retro-turbo `1.0.0.post2` throughput benchmark with the current default observation pipeline: actual `make_fast_mario_env` training env reached about `907` SPS at 1 env, `3479` SPS at 8 envs, `3214` SPS at 16 envs, `3224` SPS at 32 envs, and `3668` SPS at 48 envs on a longer 2k-step/env sample. A 64-env short sample hit `3679` SPS, but the longer 2k-step/env sample dropped to `3027` SPS, so 48 envs is the better local throughput point from this sweep.
- Local stable-retro-turbo throughput benchmark on macOS arm64/Python 3.14 before switching max-pool/crop defaults: actual `make_fast_mario_env` training env with native resize, grayscale, frame skip, 4-frame stack, shared-memory vectorization, action wrapper, reward wrapper, and time limit reached about `1001` SPS at 1 env, `3967` SPS at 8 envs, `4193` SPS at 16 envs, `4419` SPS at 32 envs, `4707` SPS at 48 envs, and `4827` SPS at 64 envs. Lower-level native Atari-preproc diagnostic with max-pool enabled but without project wrappers reached about `1153` single-env SPS and peaked around `5252` SPS at 32 envs. Treat 48/64 results as short-run estimates; Python `resource_tracker` emits noisy shared-memory cleanup warnings.
- Modal rollout-throughput default: use `cpu=16`, `memory=32768`, and `n_envs=32` as the best-value setting for Modal Mario PPO runs. Earlier emulator package benchmarks showed about `1021` env steps/sec at roughly `$0.275` per 1M env steps. Larger settings (`cpu=32,n_envs=48` or `cpu=64,n_envs=64`) are faster in raw SPS but worse value and still much slower than local Apple Silicon rollout collection.
- Modal Linux env-only diagnosis: earlier emulator package benchmarks showed Modal Linux was already much slower in raw single-env stepping than local macOS arm64 (`raw_rgb` about `505` SPS on Modal vs `3201` SPS on Mac; native Atari preprocessing about `148` SPS on Modal vs `1102` SPS on Mac). This pointed to base emulator/core CPU performance or Linux/Modal runtime/wheel codegen, not PPO training, SB3, frame stacking, shared memory, or grayscale resize as the primary bottleneck.

## Completed Runs

### `modal_level1_real_clear_terminal_250k`

- Date: 2026-06-11.
- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-HIfWoJiXDMV9O2SwIhUyiW`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/xtd8z0xl`
- Config: fresh 250k run focused on `Level1-1`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.02`, `clip_range=0.1`, `target_kl=0.02`, bounded reward, `time_penalty=0.01`, `death_penalty=50`, `completion_x_threshold=0`, and `terminate_on_level_change=true`.
- Runtime: about 8m26s for 250,880 actual timesteps; final logged throughput about 493 fps.
- Code change: added `EnvConfig.terminate_on_level_change` and CLI/Modal/script plumbing so real `levelHi/levelLo` transitions can terminate the episode. Reset now initializes level/life tracking from reset `info`, and a wrapper smoke test confirmed level-change termination pays the bounded positive terminal reward.
- Eval command used 50 stochastic episodes per checkpoint, `completion_x_threshold=0`, `terminate_on_level_change=true`, and best-episode videos in `runs/local_evals/modal_level1_real_clear_terminal_250k/videos/`.
- Eval highlights over 50 episodes: 25k `reward_mean=20.91`, `max_x_max=1396`, `completion_rate=0.00`; 50k `21.99`, `1902`, `0.00`; 75k `21.34`, `1894`, `0.00`; 100k `21.51`, `1903`, `0.00`; 125k `21.29`, `1904`, `0.00`; 150k `20.24`, `1530`, `0.00`; 175k `22.04`, `1406`, `0.00`; 200k `23.11`, `1904`, `0.00`; 225k `27.45`, `1836`, `0.00`; 250k `24.78`, `1690`, `0.00`.
- Best by current selector: checkpoint step 199,936 (`completion_rate=0.00`, `max_x_max=1904`, `reward_mean=23.11`). Best by reward mean: checkpoint step 224,928 (`reward_mean=27.45`, `max_x_max=1836`). All checkpoints had `death_rate=1.00`.
- Lesson: removing threshold-proxy completion exposed that the current PPO setup does not achieve real level clears and does not recover the previous near-end threshold behavior. Next isolated change should improve late-level exploration/credit assignment rather than simply continuing this run: e.g. scripted-right behavior cloning, curriculum from later Level1-1 states, or explicit subtask states near known death bottlenecks.

### `modal_level1_real_clear_terminal_1m_tp01_dp50`

- Date: 2026-06-11.
- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-NkqF8ILZM5wGhyOlmO9iEF`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/05enjmxg`
- Config: fresh 1M run continuing the strict real-clear setup from `modal_level1_real_clear_terminal_250k`: `Level1-1`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.02`, `clip_range=0.1`, `target_kl=0.02`, bounded reward, `time_penalty=0.01`, `death_penalty=50`, `completion_x_threshold=0`, `terminate_on_level_change=true`, and terminal-on-life-loss enabled.
- Note: an initial same-name Modal launch accidentally used default `time_penalty=0.0` and `death_penalty=25`; it was interrupted at 1,024 timesteps before any checkpoint. The completed run used the corrected run name above.
- Runtime: about 22m59s for 1,000,448 actual timesteps; final logged throughput about 725 fps.
- Eval command used 50 stochastic episodes per checkpoint, `completion_x_threshold=0`, `terminate_on_level_change=true`, and best-episode videos in `runs/local_evals/modal_level1_real_clear_terminal_1m_tp01_dp50/videos/`.
- Eval result: some checkpoints achieved real level clears, but completion stayed sparse and regressed often. Checkpoints with clears: 24,992 (1/50), 49,984 (1/50), 74,976 (1/50), 199,936 (1/50), 399,872 (2/50), 749,760 (1/50), and 849,728 (1/50).
- Best by current selector: checkpoint step 399,872 (`completion_rate=0.04`, `completion_count=2`, `max_x_max=3098`, `reward_mean=26.39`, `death_rate=0.96`).
- Best reward mean: checkpoint step 799,744 (`reward_mean=27.68`, `max_x_max=1679`, `completion_rate=0.00`, `death_rate=1.00`).
- Local best checkpoint: `runs/wandb_artifacts/tsilva_mario-ppo_modal_level1_real_clear_terminal_1m_tp01_dp50-checkpoint_v15/ppo_mario_399872_steps.zip`.
- Local playback command: `UV_CACHE_DIR=.uv-cache uv run python -m mario_ppo.play --model runs/wandb_artifacts/tsilva_mario-ppo_modal_level1_real_clear_terminal_1m_tp01_dp50-checkpoint_v15/ppo_mario_399872_steps.zip --episodes 3 --max-steps 2500 --stochastic --action-set right --completion-x-threshold 0 --terminate-on-level-change`.
- Lesson: the 250k run was undertrained for strict real-clear success because the 1M run produced real clears, but scaling PPO alone did not create robust completion. The policy repeatedly regressed after sparse successful checkpoints, so next work should focus on improving consistency/exploration/credit assignment rather than assuming the final checkpoint is useful.

### `modal_fixed_reward_gpu_250k_lr1e4_env16`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-BdAjkwhU3VD5zZWfbqLIbW`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/jtegwgl6`
- Config: T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `learning_rate=1e-4`, stochastic eval, 250k timesteps.
- Runtime: about 12m53s for 250,880 actual timesteps, final logged throughput about 323 fps.
- Eval highlights: 50k `691.8`, 80k `867.6`, 190k `927.4`, 250k `1082.7`.
- Artifacts: checkpoints uploaded every ~25k steps plus best and final W&B artifacts.
- Lesson: lower learning rate plus longer training gave meaningful improvement, but variance remained high. Next quality runs should either evaluate with more episodes, try entropy regularization / schedule tuning, or checkpoint-select aggressively.
- Follow-up correction: this run already used PPO `ent_coef=0.01` via the train.py default. The next isolated change should be lower learning rate continuation from the best artifact plus more eval episodes, not simply "add entropy".

### `modal_continue_best_250k_lr5e5_ent01_eval50`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-EsSbYjvEbw6qBOUS1Ew35x`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/v3m19j1p`
- Config: resumed from `tsilva/mario-ppo/modal_fixed_reward_gpu_250k_lr1e4_env16-best:latest`, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `learning_rate=5e-5`, `ent_coef=0.01`, stochastic eval, 50 eval episodes, 250k continuation timesteps.
- Runtime: about 19m41s for 250,880 actual timesteps. Final logged throughput about 212 fps because 50-episode evals dominate wall time.
- Eval highlights over 50 episodes: 25k `664.38`, 50k `582.56`, 75k `554.80`, 100k `610.18`, 125k `643.38`, 150k `642.66`, 175k `620.86`, 200k `572.40`, 225k `615.50`, 250k `618.04`.
- Artifacts: checkpoints uploaded every ~25k steps plus best and final W&B artifacts.
- Lesson: continuing from the noisy best artifact at lower LR did not improve robust 50-episode performance. Next work should target policy/environment formulation rather than just longer PPO continuation.

### `modal_right_action_250k_lr1e4_eval50`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-yD7iPCC8NSpYJXMlw40rLI`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/flv71zkq`
- Config: trained from scratch with restricted `right` action set: `right`, `right_b`, `right_a`, `right_a_b`. T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `learning_rate=1e-4`, `ent_coef=0.01`, stochastic eval, 50 eval episodes, 250k timesteps.
- Runtime: about 20m11s for 250,880 actual timesteps. Final logged throughput about 206 fps with 50-episode evals.
- Eval highlights over 50 episodes: 25k `664.52`, 50k `729.26`, 75k `651.24`, 100k `765.42`, 125k `588.16`, 150k `684.92`, 175k `656.96`, 200k `618.98`, 225k `635.18`, 250k `672.08`.
- Artifacts: checkpoints uploaded every ~25k steps plus best and final W&B artifacts.
- Lesson: restricted forward action set is a clear improvement over the 7-action `simple` policy family under robust 50-episode eval. It also still suffers PPO regression, so the best artifact/checkpoint is preferable to the final model.

### Local best-of-20 playback sample from `modal_right_action_250k_lr1e4_eval50-best`

- Command: `UV_CACHE_DIR=.uv-cache uv run python scripts/record_best_episode.py --model runs/wandb_artifacts/tsilva_mario-ppo_modal_right_action_250k_lr1e4_eval50-best_latest/best_model.zip --episodes 20 --max-steps 1200 --action-set right --output runs/videos/modal_right_best_of_20.mp4 --summary-output runs/videos/modal_right_best_of_20.json`
- Result: best episode was episode 12 with reward `1376.0`, max x-position `1401`, score `20`, and `347` steps.
- Video: `runs/videos/modal_right_best_of_20.mp4`; summary JSON: `runs/videos/modal_right_best_of_20.json`.
- Lesson: stochastic sampling can produce much better individual rollouts than the 50-episode mean suggests, but many rollouts still die early around x-position `180-200`. Next training should improve consistency, not only the lucky rollout tail.

### Eval instrumentation update

- Added per-eval tracking for `max_x_mean`, `max_x_max`, `completion_rate`, `death_rate`, death x-position histograms, and best-episode videos.
- Local outputs: `runs/<run-name>/eval_metrics.jsonl` and `runs/<run-name>/eval_videos/best_episode_<timesteps>_steps.mp4`.
- W&B outputs: scalar eval metrics, `eval/death_x_pos_histogram` when deaths occur, and `eval/best_episode_video`.
- `Level1-1` completion uses stable-retro level-change info when available, otherwise the default `--completion-x-threshold 3160`.
- Implementation note: the first video-enabled Modal run (`modal_right_500k_lr5e5_longhorizon_metrics`) was interrupted after one eval because rendering all 50 eval episodes dropped throughput sharply. Eval video capture was patched to store action sequences and replay only the selected best episode.

### Workflow Update: Out-of-Process Checkpoint Eval

- Training-loop eval is now disabled by default (`--eval-freq 0`, `--eval-episodes 0`), including Modal entrypoint defaults.
- Modal training should focus on rollouts, PPO updates, checkpointing, and W&B checkpoint/final artifact upload.
- Local eval is responsible for scanning checkpoint artifacts, evaluating pending checkpoints, logging metrics back to the same W&B run, and promoting the current best checkpoint by `(completion_rate, max_x_max, reward_mean)`.
- Codex should use training-monitoring wait time to run `scripts/eval_wandb_checkpoints.py` on pending checkpoint artifacts when W&B credentials and local ROM setup are available.

### `modal_right_500k_lr5e5_longhorizon_metrics_fastvideo`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-9zn97d1K4psGaMPZgFnw8E`
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/hhvw66pq`
- Config: trained from scratch with restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=4`, `learning_rate=5e-5`, `ent_coef=0.01`, stochastic eval, 50 eval episodes, `max_episode_steps=2500`, `completion_x_threshold=3160`, 500k timesteps.
- Runtime: about 45m12s for 500,736 actual timesteps with 20 eval windows and best-episode videos.
- Eval highlights over 50 episodes: 25k `reward_mean=722.48`, `max_x_max=3093`, `completion_rate=0.02`; 50k `563.08`, `1674`, `0.00`; 75k `383.54`, `1321`, `0.00`; 100k `711.72`, `1666`, `0.00`; 125k `674.52`, `1407`, `0.00`; 150k `656.36`, `2340`, `0.00`; 200k `768.66`, `1896`, `0.00`; 250k `697.72`, `1559`, `0.00`; 325k `741.38`, `3089`, `0.02`; 375k `745.88`, `3106`, `0.02`; 425k `684.30`, `3108`, `0.04`; 475k `766.76`, `2622`, `0.00`; 500k `761.70`, `1835`, `0.00`.
- All eval windows had `death_rate=1.0`, so the completion proxy is only threshold progress before death, not a stable level clear.
- Artifacts: checkpoints uploaded every ~25k steps plus best and final W&B model artifacts. W&B synced 20 media files and 44 artifact files.
- Lesson: longer-horizon PPO with lower LR did not make robust progress. Best progress appeared at isolated evals around 25k/325k/375k/425k, but consistency stayed poor. Next run should add PPO stabilization/early stopping and likely curriculum or imitation, not more blind continuation.

### Seed Sweep: `modal_right_500k_lr5e5_longhorizon_metrics_fastvideo` step 424864

- Artifact searched: `tsilva/mario-ppo/modal_right_500k_lr5e5_longhorizon_metrics_fastvideo-checkpoint:step-424864`.
- Sweep command tested stochastic policy seeds `0..99` with restricted `right` action set, `max_episode_steps=2500`, and rank order `level_complete`, then `max_x_pos`, then `reward`.
- Best seed: `37`, `reward=2608.0`, `max_x_pos=2633`, `steps=514`, `score=30`, `level_complete=false`, `died=true`, `death_x_pos=2633`.
- Video: `runs/videos/modal_right_424864_best_seed_0_99.mp4`.
- Summary: `runs/videos/modal_right_424864_best_seed_0_99.json`.
- Lesson: even the best checkpoint found so far does not reliably complete level 1-1 under 100 stochastic seeds. It can produce interesting progress past the midpoint, but it still dies before the late-level finish. Next training should optimize for completion robustness, not just rare high `max_x_pos`.

### Objective Update: Completion-Weighted Reward

- Added `EnvConfig.completion_reward` and `completion_x_threshold` to training reward shaping. The wrapper now adds a one-time completion bonus when stable-retro reports a level change or `max_x_pos >= completion_x_threshold`.
- Completion-aware reward settings to try next: `--death-penalty 250 --completion-reward 2000 --time-penalty 0.02 --completion-x-threshold 3160`.
- Updated `MarioEvalCallback` best-model selection from reward mean to `(completion_rate, max_x_max, reward_mean)`.
- Updated best eval video selection to rank individual episodes by `(level_complete, max_x_pos, reward)`.
- Local smoke check: `smoke_completion_objective` compiled and trained for 64 timesteps; `eval_metrics.jsonl` includes `best_model_score`. A scripted low-threshold eval confirmed `completion_reward` fires.

### `modal_right_completion_reward_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-retD4UPqYoralu44TIDi7J`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/b0nvw88k`.
- Config: restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=4`, `learning_rate=5e-5`, `ent_coef=0.01`, stochastic eval, 50 eval episodes, `max_episode_steps=2500`, `death_penalty=250`, `completion_reward=2000`, `time_penalty=0.02`, `completion_x_threshold=3160`.
- Interrupted manually after the 100k eval and after the 124960-step checkpoint uploaded because the run had already regressed and entropy had collapsed.
- Eval highlights: 25k `reward_mean=367.75`, `max_x_max=2344`, `completion_rate=0.00`, `death_rate=1.00`; 50k `391.16`, `2631`, `0.00`, `1.00`; 75k `153.30`, `1316`, `0.00`, `1.00`; 100k `188.27`, `1316`, `0.00`, `1.00`.
- Entropy: started at `1.386` nats, reached as low as `0.030`, and was about `0.330` near the stop. This is true entropy collapse for the 4-action restricted policy.
- Artifacts uploaded: `modal_right_completion_reward_250k-checkpoint` aliases `step-24992`, `step-49984`, `step-74976`, `step-99968`, and `step-124960`/`latest`.
- Lesson: simply adding a large death penalty and completion bonus made PPO collapse into a narrow policy without solving completions. Next run should preserve the completion-aware model selection but use gentler shaping or stronger exploration constraints, e.g. lower `death_penalty`, smaller/no completion bonus until close to threshold, higher/scheduled `ent_coef`, lower `clip_range`, and/or `target_kl`.

### Planned Run: `modal_right_soft_completion_ent05_250k`

- Rationale: user noted PPO should be able to reach much later levels without curriculum; do not switch to curriculum yet. Try stabilizing PPO and preserving exploration while keeping restricted `right` actions.
- Config to run: T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.05`, `clip_range=0.1`, `target_kl=0.02`, `death_penalty=75`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`, stochastic eval, 50 eval episodes, 250k timesteps.
- Success signal: entropy should not collapse below about `0.25` nats early, and eval should improve beyond the 50k best of the prior completion run (`max_x_max=2631`) without regressing to `1316`.

### `modal_right_soft_completion_ent05_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-EvI78kuVUfnpoJQ4aCC5VW`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/2xzjyhcd`.
- Config: restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.05`, `clip_range=0.1`, `target_kl=0.02`, stochastic eval, 50 eval episodes, `max_episode_steps=2500`, `death_penalty=75`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`.
- Runtime: about 22m34s for 250,880 actual timesteps.
- Eval highlights over 50 episodes: 25k `reward_mean=581.98`, `max_x_mean=659.22`, `max_x_max=1320`, `completion_rate=0.00`, `death_rate=1.00`; 50k `686.32`, `763.32`, `1910`, `0.00`, `1.00`; 75k `583.03`, `660.02`, `1536`, `0.00`, `1.00`; 100k `638.98`, `716.32`, `1896`, `0.00`, `1.00`; 125k `606.04`, `683.16`, `1886`, `0.00`, `1.00`; 150k `701.15`, `778.82`, `1819`, `0.00`, `1.00`; 175k `562.10`, `639.28`, `1555`, `0.00`, `1.00`; 200k `691.79`, `769.52`, `2620`, `0.00`, `1.00`; 225k `644.38`, `722.48`, `2629`, `0.00`, `1.00`; 250k `666.23`, `744.04`, `1899`, `0.00`, `1.00`.
- Entropy: started at `1.386` nats, lowest sampled value was about `0.650`, and final was about `1.349`; the run avoided entropy collapse.
- Artifacts uploaded: checkpoints at every eval, best artifact `modal_right_soft_completion_ent05_250k-best:latest`, and final artifact `modal_right_soft_completion_ent05_250k-final:latest`. Best checkpoint by eval selector was around 224928 steps with `max_x_max=2629`.
- Lesson: PPO stabilization worked mechanically and prevented collapse, but the policy stayed too exploratory or did not learn the late-level hazard sequence. It improved robustness/mean progress but did not exceed previous rare-progress peaks or complete the level. Next run should keep `target_kl`/lower `clip_range`, but reduce entropy pressure after initial exploration or use an entropy schedule; consider returning death penalty closer to `25-50` and removing sparse completion bonus until the policy reliably reaches late level.

### `modal_right_mid_entropy_ent02_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-bOo9ZYtG3U8q418Z41MqCl`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/j1mraemq`.
- Config: restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.02`, `clip_range=0.1`, `target_kl=0.02`, stochastic eval, 50 eval episodes, `max_episode_steps=2500`, `death_penalty=50`, `completion_reward=0`, `time_penalty=0.01`, `completion_x_threshold=3160`.
- Stopped manually after the 149952-step eval and about 171k collected timesteps because the 100k completion signal did not persist and later checkpoints regressed. Modal reports the run as interrupted because the app was stopped mid-rollout.
- Eval highlights over 50 episodes: 25k `reward_mean=716.24`, `max_x_mean=771.04`, `max_x_max=2332`, `completion_rate=0.00`, `death_rate=1.00`; 50k `587.72`, `642.12`, `1555`, `0.00`, `1.00`; 75k `670.41`, `723.70`, `1405`, `0.00`, `1.00`; 100k `699.66`, `751.84`, `3097`, `0.04`, `1.00`; 125k `717.49`, `769.58`, `1557`, `0.00`, `1.00`; 150k `668.04`, `720.56`, `1830`, `0.00`, `1.00`.
- Entropy: started at `1.386` nats, lowest sampled value was about `0.919`, and last sampled value was about `1.168`; no entropy collapse.
- Artifacts uploaded: `modal_right_mid_entropy_ent02_250k-checkpoint` aliases `step-24992`, `step-49984`, `step-74976`, `step-99968`, `step-124960`, and `step-149952`/`latest`. Because the app was interrupted, W&B showed no separate `-best` or `-final` artifact; use `tsilva/mario-ppo/modal_right_mid_entropy_ent02_250k-checkpoint:step-99968` for playback.
- Lesson: this midpoint setting is the first recent run to show nonzero completion metric under 50-episode stochastic eval (`completion_rate=0.04` at 100k) and `max_x_max=3097`, close to the end threshold. However, PPO still overwrites the rare near-finish behavior after additional updates. Next run should preserve `ent_coef=0.02`, low `clip_range`, and `target_kl`, but add an explicit best-checkpoint continuation/fine-tune or early-stop strategy around completion-positive checkpoints instead of training blindly past them.

### Global Progress Fix

- Bug fixed: after `levelHi/levelLo` changed, `xscrollHi/xscrollLo` reset to low values while the wrapper kept the prior level's `max_x_pos` as the comparison baseline. This made post-level progress receive no positive reward until the new level exceeded the previous level's x-position.
- New behavior: on level change, the wrapper adds the prior level's best x-position to `completed_level_base`, resets within-level tracking, and continues reward on `global_x_pos = completed_level_base + level_x_pos`.
- New metrics: `max_x_pos` remains the global progress metric for compatibility. The wrapper also logs `level_x_pos`, `level_max_x_pos`, `global_x_pos`, `global_max_x_pos`, `completed_level_base`, `completed_level_count`, and `completion_event`. Eval now logs `eval/max_level_x_mean` and `eval/max_level_x_max`.
- Local validation: synthetic wrapper test confirmed that after a forced level reset from x=300 to x=0, the next-level step at x=80 now receives `+80` progress reward instead of only time penalty. `UV_CACHE_DIR=.uv-cache uv run python -m compileall src scripts` passed, and `smoke_global_progress_fix` trained for 1024 local timesteps.

### `modal_global_progress_fix_ent02_75k_fast_eval`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-1hGtpt07iSwM1963A6QDF5`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/ybdmxurx`.
- Config: same core PPO shape as `modal_right_mid_entropy_ent02_250k`, with restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=2`, `learning_rate=5e-5`, `ent_coef=0.02`, `clip_range=0.1`, `target_kl=0.02`, stochastic eval, `max_episode_steps=2500`, `death_penalty=50`, `completion_reward=0`, `time_penalty=0.01`, `completion_x_threshold=3160`. Eval used 20 episodes and `--no-eval-videos` to avoid video overhead.
- Runtime: about 8m56s for 75,776 actual timesteps. A prior 125k attempt with 50-episode video eval was stopped before first eval because W&B/runtime appeared stuck in the eval path.
- Eval highlights over 20 episodes: 25k `reward_mean=636.61`, `max_x_mean=690.00`, `max_x_max=1902`, `max_level_x_max=1902`, `completion_rate=0.00`, `death_rate=1.00`; 50k `641.10`, `695.30`, `1314`, `1314`, `0.00`, `1.00`; 75k `616.03`, `669.40`, `1405`, `1405`, `0.00`, `1.00`.
- Artifacts uploaded: checkpoints at `step-24992`, `step-49984`, `step-74976`/`latest`, plus `modal_global_progress_fix_ent02_75k_fast_eval-best:latest` and `modal_global_progress_fix_ent02_75k_fast_eval-final:latest`.
- Lesson: the global-progress fix is necessary for multi-level reward correctness, but it did not improve early Level1-1 learning in this short run. The policy never reached level change in eval, so the fix had no opportunity to help the measured learning curve. Do not repeat this exact 75k-from-scratch configuration as an "improvement" run; next useful test should resume/fine-tune from a completion-positive checkpoint or add an explicit level-clear bonus/termination strategy.

### `modal_resume_step99968_completion_bonus_125k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-mcV6RxHr00eF88Ktw9SXD1`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/tqd4o263`.
- Config: resumed from `tsilva/mario-ppo/modal_right_mid_entropy_ent02_250k-checkpoint:step-99968`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `ent_coef=0.01`, `clip_range=0.08`, `target_kl=0.01`, `max_episode_steps=2500`, `death_penalty=75`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`, training-loop eval disabled, checkpoint artifact eval out of process with 50 stochastic episodes.
- Runtime: about 2m39s for 125,952 actual timesteps. Training throughput was about 784 fps at the end.
- Local checkpoint eval highlights over 50 episodes: 24,992 `reward_mean=627.49`, `max_x_max=1668`, `completion_rate=0.00`; 49,984 `587.65`, `1668`, `0.00`; 74,976 `681.77`, `1900`, `0.00`; 99,968 `763.83`, `2341`, `0.00`; 124,960 `692.07`, `1667`, `0.00`. All eval windows had `death_rate=1.00`.
- Best artifact: `tsilva/mario-ppo/modal_resume_step99968_completion_bonus_125k-best:latest`, promoted from checkpoint step 99,968 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode: checkpoint step 99,968, seed 110015, `reward=2261.72`, `max_x_pos=2341`.
- Entropy stayed healthy rather than collapsing; final training log had `train/entropy_loss=-1.03799`.
- Lesson: resuming from the earlier completion-positive checkpoint with lower LR and a moderate completion bonus improved distance from early checkpoints to `x=2341`, but did not recover the previous rare completion behavior and regressed by 124,960. For the next run, resume from this run's best artifact or the original completion-positive checkpoint, keep checkpoint-eval selection, and test a change aimed at preserving rare near-finish trajectories rather than simply extending this exact fine-tune.

### `modal_resume_step99968_no_life_term_125k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-16uK9dGhBEd6ApJijKzJXq`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/xjic3vr7`.
- Config: same resume and PPO shape as `modal_resume_step99968_completion_bonus_125k`, but with `--no-terminate-on-life-loss` in training and out-of-process checkpoint eval. Resumed from `tsilva/mario-ppo/modal_right_mid_entropy_ent02_250k-checkpoint:step-99968`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `ent_coef=0.01`, `clip_range=0.08`, `target_kl=0.01`, `max_episode_steps=2500`, `death_penalty=75`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`, training-loop eval disabled.
- Runtime: about 2m43s for 125,952 actual timesteps. Training throughput ended around 764 fps.
- Training effect: rollout episode length rose to about 905 and reward to about 951 because episodes continued after life loss. This makes rollout reward non-comparable to terminal-life runs; use checkpoint eval metrics instead.
- Local checkpoint eval highlights over 50 stochastic episodes with no life-loss termination: 24,992 `reward_mean=920.86`, `max_x_mean=1153.88`, `max_x_max=2629`, `completion_rate=0.00`, `death_rate=1.00`; 49,984 `962.41`, `1187.64`, `3875`, `0.02`, `1.00`; 74,976 `1013.23`, `1235.74`, `3463`, `0.02`, `0.98`; 99,968 `954.59`, `1177.76`, `3848`, `0.02`, `1.00`; 124,960 `953.41`, `1185.16`, `2631`, `0.00`, `1.00`.
- Best artifact: `tsilva/mario-ppo/modal_resume_step99968_no_life_term_125k-best:latest`, promoted from checkpoint step 49,984 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode: checkpoint step 49,984, seed 60010, `reward=4131.32`, `max_x_pos=3875`.
- Entropy stayed healthy; final training log had `train/entropy_loss=-1.21869`.
- Lesson: disabling life-loss termination is a positive ablation for this resume setup. It produced nonzero completion signals across multiple checkpoints and much higher max progress than the matching terminal-life continuation. The cost is much slower eval and less directly comparable rollout reward. Next runs should keep no-life-loss termination disabled for this branch, use checkpoint selection aggressively, and consider two-stage eval: quick 20-episode scan for all checkpoints, then 100-episode confirmation for candidates with nonzero completion or high max-x.

### `modal_terminal_reg_from_no_life_best_75k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-UNWNUvbE9yUQ74LG6eYVTO`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/4ajol7fm`.
- Config: terminal-on-life-loss fine-tune resumed from `tsilva/mario-ppo/modal_resume_step99968_no_life_term_125k-best:latest`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=1e-5`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `death_penalty=100`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`, checkpoint every ~12.5k, training-loop eval disabled.
- Runtime: about 1m50s for 75,776 actual timesteps. Training throughput ended around 681 fps.
- Training behavior: conservative PPO settings worked mechanically. `clip_fraction` stayed at 0, `approx_kl` stayed very small, and entropy stayed healthy but drifted from about `1.29` to `1.02`.
- Local terminal-life checkpoint eval highlights over 50 stochastic episodes: 12,496 `reward_mean=655.05`, `max_x_mean=758.38`, `max_x_max=1892`, `completion_rate=0.00`, `death_rate=1.00`; 24,992 `458.89`, `561.16`, `1405`, `0.00`, `1.00`; 37,488 `626.19`, `719.22`, `3456`, `0.02`, `1.00`; 49,984 `546.07`, `648.26`, `1406`, `0.00`, `1.00`; 62,480 `688.86`, `791.12`, `2619`, `0.00`, `1.00`; 74,976 `725.54`, `827.66`, `1901`, `0.00`, `1.00`.
- Best artifact: `tsilva/mario-ppo/modal_terminal_reg_from_no_life_best_75k-best:latest`, promoted from checkpoint step 37,488 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode: checkpoint step 37,488, seed 47512, `reward=3840.14`, `max_x_pos=3456`.
- Lesson: terminal-on-life-loss polishing can preserve a rare completion signal if checkpointed frequently, but it still regresses quickly. This supports a two-phase strategy: use no-life-loss training to discover completion-capable behavior, then use short, highly regularized terminal-life fine-tunes with frequent checkpoint eval to polish. Do not run long terminal-life continuation without checkpoint selection.

### `modal_terminal_reg_world1_mix_75k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-I05s3tPNzYHK6Nuvj34Dak`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/d8flg059`.
- Config: same conservative terminal-on-life-loss fine-tune as `modal_terminal_reg_from_no_life_best_75k`, but training workers cycled through `--states Level1-1,Level1-4` by rank. Resumed from `tsilva/mario-ppo/modal_resume_step99968_no_life_term_125k-best:latest`, restricted `right` action set, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=1e-5`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `death_penalty=100`, `completion_reward=500`, `time_penalty=0.01`, `completion_x_threshold=3160`, checkpoint every ~12.5k, training-loop eval disabled.
- Runtime: about 1m47s for 75,776 actual timesteps. Training throughput ended around 697 fps.
- Available World 1 stable-retro states were only `Level1-1`, `Level1-1-99lives`, and `Level1-4`, so the ablation used `Level1-1,Level1-4` and avoided the 99-lives variant.
- Local `Level1-1` terminal-life checkpoint eval highlights over 50 stochastic episodes: 12,496 `reward_mean=562.09`, `max_x_mean=664.98`, `max_x_max=1680`, `completion_rate=0.00`, `death_rate=1.00`; 24,992 `594.49`, `697.26`, `2330`, `0.00`, `1.00`; 37,488 `649.78`, `752.48`, `1657`, `0.00`, `1.00`; 49,984 `662.35`, `764.90`, `1898`, `0.00`, `1.00`; 62,480 `709.38`, `812.30`, `1903`, `0.00`, `1.00`; 74,976 `641.46`, `744.60`, `2339`, `0.00`, `1.00`.
- Best artifact: `tsilva/mario-ppo/modal_terminal_reg_world1_mix_75k-best:latest`, promoted from checkpoint step 74,976 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode: checkpoint step 74,976, seed 85022, `reward=2232.91`, `max_x_pos=2339`.
- Lesson: mixed World 1 state training with `Level1-1,Level1-4` was a negative ablation for the current Level1-1 objective. It did not produce any Level1-1 completions and underperformed the single-state terminal polish run's best checkpoint (`completion_rate=0.02`, `max_x_max=3456`). This may still be useful later for generalization, but it diluted the fragile Level1-1 completion behavior at this stage. Do not use Level1-4 mixing as the next improvement unless eval also targets multi-level average performance.

### `modal_bounded_no_life_term_scratch_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-swoI35ygVWly2DcFxFQWn9`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/aul24jg9`.
- Config: fresh PPO from scratch with restricted `right` action set, bounded reward mode, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `progress_reward_cap=30`, `terminal_reward=30`, `reward_scale=30`, `time_penalty=0.3`, `completion_x_threshold=3160`, and `--no-terminate-on-life-loss`.
- Runtime: about 5m32s for 250,880 actual timesteps. Final throughput was about 752 fps.
- Training behavior: rollout reward improved from about `24` early to a peak around `32`, ending at `30.17`. Entropy stayed healthy, with final `train/entropy_loss=-1.29164`; no policy entropy collapse.
- Local checkpoint eval over 50 stochastic episodes with no life-loss termination:

| Step | Reward mean | Max x mean | Max x max | Completion rate | Death rate |
|---:|---:|---:|---:|---:|---:|
| 24,992 | 20.29 | 987.90 | 1548 | 0.00 | 1.00 |
| 49,984 | 29.11 | 1245.32 | 3883 | 0.02 | 1.00 |
| 74,976 | 25.72 | 1086.16 | 1918 | 0.00 | 1.00 |
| 99,968 | 28.76 | 1197.76 | 2622 | 0.00 | 1.00 |
| 124,960 | 30.96 | 1336.82 | 6208 | 0.06 | 1.00 |
| 149,952 | 32.04 | 1277.50 | 3648 | 0.02 | 1.00 |
| 174,944 | 32.50 | 1350.04 | 6210 | 0.04 | 1.00 |
| 199,936 | 29.01 | 1177.10 | 2627 | 0.00 | 1.00 |
| 224,928 | 27.13 | 1106.16 | 2350 | 0.00 | 1.00 |
| 249,920 | 27.43 | 1106.06 | 2344 | 0.00 | 1.00 |

- Best artifact: `tsilva/mario-ppo/modal_bounded_no_life_term_scratch_250k-best:latest`, promoted from checkpoint step 124,960 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode for selected checkpoint: checkpoint step 124,960, seed 134980, `reward=82.40`, `max_x_pos=6208`.
- Best local eval video: `runs/local_evals/modal_bounded_no_life_term_scratch_250k/videos/best_episode_124960_steps.mp4`.
- Lesson: the bounded SuperMarioRL-style reward plus no-life-loss training is a strong fresh-start baseline. It reached nonzero completion by 50k and peaked at 6% completion by 125k without needing a resume checkpoint. The run still regressed after the peak, so future runs should continue checkpoint selection and probably use shorter training windows or early-stop/branch from completion-positive checkpoints rather than blindly extending to final.

### `modal_bounded_gamma09_no_life_term_scratch_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-bXAJSf6RMl93zpyEQSeBeN`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/mg4mt9sm`.
- Config: one-variable ablation from `modal_bounded_no_life_term_scratch_250k`, changing only `gamma` from `0.99` to `0.9`. Kept seed `7`, restricted `right` action set, bounded reward mode, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `gae_lambda=0.95`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `progress_reward_cap=30`, `terminal_reward=30`, `reward_scale=30`, `time_penalty=0.3`, `completion_x_threshold=3160`, and `--no-terminate-on-life-loss`.
- Runtime: about 6m07s for 250,880 actual timesteps. Final throughput was about 680 fps.
- Training behavior: rollout reward rose quickly to the high 20s around 50k-150k, then drifted down and ended at `24.10`. Entropy stayed healthy, with final `train/entropy_loss=-1.37471`; no policy entropy collapse. Value fit was reasonable but not clearly better than gamma `0.99`.
- Local checkpoint eval over 50 stochastic episodes with no life-loss termination:

| Step | Reward mean | Max x mean | Max x max | Completion rate | Death rate |
|---:|---:|---:|---:|---:|---:|
| 24,992 | 22.41 | 1048.82 | 3183 | 0.02 | 1.00 |
| 49,984 | 30.80 | 1308.22 | 3917 | 0.04 | 0.98 |
| 74,976 | 24.24 | 1073.48 | 2350 | 0.00 | 1.00 |
| 99,968 | 26.97 | 1177.76 | 2623 | 0.00 | 1.00 |
| 124,960 | 24.51 | 1069.68 | 1686 | 0.00 | 1.00 |
| 149,952 | 27.18 | 1196.74 | 6166 | 0.02 | 1.00 |
| 174,944 | 29.11 | 1221.12 | 3887 | 0.02 | 1.00 |
| 199,936 | 32.28 | 1310.90 | 3874 | 0.02 | 1.00 |
| 224,928 | 28.74 | 1212.66 | 3646 | 0.02 | 1.00 |
| 249,920 | 29.04 | 1220.36 | 2621 | 0.00 | 1.00 |

- Best artifact: `tsilva/mario-ppo/modal_bounded_gamma09_no_life_term_scratch_250k-best:latest`, promoted from checkpoint step 49,984 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode for selected checkpoint: checkpoint step 49,984, seed 60027, `reward=109.07`, `max_x_pos=3917`.
- Best local eval video: `runs/local_evals/modal_bounded_gamma09_no_life_term_scratch_250k/videos/best_episode_49984_steps.mp4`.
- Comparison to gamma `0.99`: gamma `0.9` learned a completion-capable policy earlier and had a stronger 50k checkpoint (`completion_rate=0.04` vs `0.02`), but its best checkpoint did not beat gamma `0.99` overall (`0.04`/`3917` vs `0.06`/`6208`). Treat gamma `0.9` alone as not better under this PPO geometry. The short horizon may still matter when combined with longer rollouts and more epochs; test that separately instead of changing gamma alone.

### `modal_bounded_gamma09_gae1_no_life_term_scratch_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-975iZhL4wrxRHyBWpVzEnZ`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/kinupecn`.
- Config: one-variable ablation from `modal_bounded_gamma09_no_life_term_scratch_250k`, changing only `gae_lambda` from `0.95` to `1.0`. Kept seed `7`, `gamma=0.9`, restricted `right` action set, bounded reward mode, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `progress_reward_cap=30`, `terminal_reward=30`, `reward_scale=30`, `time_penalty=0.3`, `completion_x_threshold=3160`, and `--no-terminate-on-life-loss`.
- Runtime: about 7m51s for 250,880 actual timesteps. Final throughput was about 530 fps.
- Training behavior: slower than `gae_lambda=0.95` and value learning was noisier/higher-loss, as expected from less-biased higher-variance returns. Policy metrics stayed conservative: final `clip_fraction=0`, final `entropy_loss=-1.35904`, and no entropy collapse. Late rollout reward recovered to `28.21`, better than the gamma `0.9`, lambda `0.95` final reward of `24.10`.
- Local checkpoint eval over 50 stochastic episodes with no life-loss termination:

| Step | Reward mean | Max x mean | Max x max | Completion rate | Death rate |
|---:|---:|---:|---:|---:|---:|
| 24,992 | 26.12 | 1173.72 | 4162 | 0.04 | 1.00 |
| 49,984 | 23.14 | 1079.54 | 1898 | 0.00 | 1.00 |
| 74,976 | 28.57 | 1197.84 | 3481 | 0.02 | 1.00 |
| 99,968 | 27.60 | 1150.20 | 2632 | 0.00 | 1.00 |
| 124,960 | 33.31 | 1395.72 | 6178 | 0.04 | 0.98 |
| 149,952 | 31.15 | 1288.42 | 3889 | 0.04 | 1.00 |
| 174,944 | 30.70 | 1240.44 | 3641 | 0.02 | 1.00 |
| 199,936 | 29.54 | 1213.80 | 3641 | 0.02 | 1.00 |
| 224,928 | 26.97 | 1125.72 | 1905 | 0.00 | 1.00 |
| 249,920 | 28.40 | 1177.14 | 3193 | 0.02 | 1.00 |

- Best artifact: `tsilva/mario-ppo/modal_bounded_gamma09_gae1_no_life_term_scratch_250k-best:latest`, promoted from checkpoint step 124,960 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode for selected checkpoint: checkpoint step 124,960, seed 134994, `reward=85.69`, `max_x_pos=6178`.
- Best local eval video: `runs/local_evals/modal_bounded_gamma09_gae1_no_life_term_scratch_250k/videos/best_episode_124960_steps.mp4`.
- Comparison: `gae_lambda=1.0` is better than `0.95` inside the gamma `0.9` branch by max progress and mean reward (`0.04`/`6178` vs `0.04`/`3917`), but it still does not beat the gamma `0.99`, lambda `0.95` baseline by completion rate (`0.06`/`6208`). This suggests lambda `1.0` helps long-horizon progress, but gamma `0.9` may still under-discount or destabilize consistency under the current short-rollout, one-epoch geometry.

### `modal_bounded_gamma099_repro_no_life_term_scratch_250k`

- Modal run: `https://modal.com/apps/eng-tiago-silva/main/ap-G9TpSHRIEU090YKbfaTuPt`.
- W&B run: `https://wandb.ai/tsilva/mario-ppo/runs/ads2xl3b`.
- Config: exact reproduction attempt of `modal_bounded_no_life_term_scratch_250k`, with explicit seed `7`, `gamma=0.99`, `gae_lambda=0.95`, restricted `right` action set, bounded reward mode, T4 GPU, 16 envs, 16 CPU, 32 GiB, `n_steps=64`, `batch_size=256`, `n_epochs=1`, `learning_rate=2.5e-5`, `ent_coef=0.02`, `clip_range=0.05`, `target_kl=0.005`, `max_episode_steps=2500`, `progress_reward_cap=30`, `terminal_reward=30`, `reward_scale=30`, `time_penalty=0.3`, `completion_x_threshold=3160`, and `--no-terminate-on-life-loss`.
- Runtime: about 5m27s for 250,880 actual timesteps. Final throughput was about 762 fps.
- Training behavior: very close to the original baseline training curve. Final rollout reward was `30.38` vs original `30.17`; final entropy loss was `-1.30275` vs original `-1.29164`. No entropy collapse, and PPO updates stayed conservative.
- Local checkpoint eval over 50 stochastic episodes with no life-loss termination:

| Step | Reward mean | Max x mean | Max x max | Completion rate | Death rate |
|---:|---:|---:|---:|---:|---:|
| 24,992 | 20.29 | 987.90 | 1548 | 0.00 | 1.00 |
| 49,984 | 26.40 | 1149.96 | 2339 | 0.00 | 1.00 |
| 74,976 | 31.74 | 1281.58 | 5123 | 0.04 | 1.00 |
| 99,968 | 28.33 | 1166.78 | 3875 | 0.04 | 1.00 |
| 124,960 | 26.53 | 1100.64 | 1902 | 0.00 | 1.00 |
| 149,952 | 30.10 | 1270.98 | 6188 | 0.04 | 1.00 |
| 174,944 | 28.75 | 1211.74 | 6202 | 0.02 | 1.00 |
| 199,936 | 31.59 | 1243.32 | 3876 | 0.02 | 1.00 |
| 224,928 | 27.78 | 1126.06 | 3188 | 0.02 | 1.00 |
| 249,920 | 28.55 | 1136.16 | 2328 | 0.00 | 1.00 |

- Best artifact: `tsilva/mario-ppo/modal_bounded_gamma099_repro_no_life_term_scratch_250k-best:latest`, promoted from checkpoint step 149,952 by `(completion_rate, max_x_max, reward_mean)`.
- Best recorded eval episode for selected checkpoint: checkpoint step 149,952, seed 159986, `reward=84.56`, `max_x_pos=6188`.
- Best local eval video: `runs/local_evals/modal_bounded_gamma099_repro_no_life_term_scratch_250k/videos/best_episode_149952_steps.mp4`.
- Reproducibility lesson: the first checkpoint reproduced exactly (`24,992`: same reward, max-x, seed, and completion metrics), but later checkpoints diverged despite the same seed and config. The run still landed in the same performance band: original best was `completion_rate=0.06`, `max_x_max=6208`; repro best was `completion_rate=0.04`, `max_x_max=6188`. Treat single-run 50-episode completion differences of `0.02` as likely within run-to-run/eval variance unless confirmed with repeated seeds or 100-episode candidate confirmation.

## Modal Throughput Sweep

All measured on Modal T4 with W&B/eval/checkpoint disabled during 10k-step benchmarks.

| Workers | Resources | Final bar speed | Logged FPS peak | Notes |
|---:|---|---:|---:|---|
| 4 envs | T4, 4 CPU, 8 GiB | 305 it/s | 293 fps | Stable |
| 8 envs | T4, 8 CPU, 16 GiB | 385 it/s | 371 fps | Better |
| 16 envs | T4, 16 CPU, 32 GiB | 589 it/s | 557 fps | Best practical point |
| 32 envs | T4, 32 CPU, 64 GiB | n/a | n/a | Did not start promptly; stopped after pending >2 min |

Current practical Modal config:

```bash
--n-envs 16 --cpu 16 --memory 32768 --gpu T4
```

For training quality, prefer keeping rollout batch near 1024:

```bash
--n-envs 16 --n-steps 64 --batch-size 256
```

## Prior Run Command

The 250k run used:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 250000 \
  --n-envs 16 \
  --cpu 16 \
  --memory 32768 \
  --gpu T4 \
  --n-steps 64 \
  --batch-size 256 \
  --learning-rate 0.0001 \
  --run-name modal_fixed_reward_gpu_250k_lr1e4_env16 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --eval-stochastic \
  --max-episode-steps 1200 \
  --checkpoint-freq 25000 \
  --wandb \
  --wandb-project mario-ppo \
  --wandb-mode online
```

Expected playback command after a run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py \
  modal_fixed_reward_gpu_250k_lr1e4_env16 \
  --kind best \
  --stochastic
```

## Open Questions

- Whether continuing from the 250k best artifact with lower learning rate (`5e-5`) improves beyond `1082.7`.
- Whether more eval episodes (`50+`) reduce checkpoint ranking noise enough to make best-model selection more trustworthy.
- Whether `ent_coef=0.01` should be kept, reduced, or scheduled during continuation.
- Whether the restricted `right` action set should be paired with learning-rate scheduling, early stopping, or lower update pressure to avoid post-100k regression.
- Whether longer horizons plus completion/death instrumentation reveal a later-level bottleneck after early deaths around x-position `180-200` are fixed.
- Whether `target_kl`, lower `clip_range`, fewer epochs, or checkpoint selection by `completion_rate/max_x_max` prevents PPO from degrading the rare near-finish behavior.
- Whether curriculum states or scripted-right behavior cloning are needed to turn rare near-end rollouts into reliable level completion.
- Whether sticky-action/no-op stochasticity, frame-stack/history tuning, or shaped subtask curriculum improves robust progress beyond the restricted-action baseline.
- Whether deterministic eval starts improving after longer training or requires a separate later fine-tuning phase.
- Whether scripted-right pretraining / behavior cloning would accelerate early progress beyond random PPO exploration.

## 2026-06-12 Handoff: Level-1 Completion Goal

Goal: build a plain-trained policy that completes `SuperMarioBros-Nes` Level 1 more than 80% of the time, using observation preprocessing, reward shaping, and hyperparameter adjustment rather than scripted/hacky behavior.

Code changes in this branch:

- Added `terminate_on_completion` to env/config/CLI paths so threshold completion can end an episode during training/eval.
- Switched image resizing in `MarioPreprocess` to OpenCV `INTER_AREA` to avoid aliasing from manual subsampling.
- Plumbed `hud_crop_top`, `frame_skip`, and `terminate_on_completion` through training, eval, play, Modal, and helper scripts.
- Added Modal `auto_resume_latest` support to continue from the newest checkpoint in a run when no explicit resume checkpoint is supplied.

Verified before this handoff:

- `UV_CACHE_DIR=.uv-cache uv run python -m compileall src scripts`
- Low-threshold eval smoke tests with `--terminate-on-completion`, with and without `--hud-crop-top 32`.
- Tiny PPO smoke training with termination-on-completion.

Experiment outcomes:

- `modal_bounded_term_completion_scratch_500k`: old resize, no HUD crop, threshold-completion termination. Best 20-episode scan checkpoint was `449856`, `completion_rate=0.05`, `max_x=6210`, but deterministic eval was `0/20` and died near `x=204`.
- `modal_area_resize_hud32_term_completion_500k`: area resize plus `--hud-crop-top 32`. Best 20-episode scan checkpoint was `274912`, `2/20` completions, but 100-episode confirmation was only `1/100`; treat the 20-episode result as noisy.
- `modal_area_resize_hud32_skip2_term_completion_1m`: frame skip 2 ablation looked worse by live rollout reward around 200k and was stopped.
- `modal_area_hud32_274912_exploit_ent001_150k`: low-LR/low-entropy fine-tune from checkpoint `274912`. Best 20-episode scan was checkpoint `50000`, `1/20` completions; not an improvement.
- `modal_area_hud32_term90_gae1_rollout256_2m`: terminal reward 90, `gae_lambda=1.0`, larger rollout. Stopped around 1.27M because rollout reward degraded to about 19-20; negative ablation.
- `modal_area_hud32_baseline_seed11_3m`: baseline seed-11 run with area resize, HUD crop, threshold-completion termination, no life-loss termination. Stopped by user request at about `2,952,192` timesteps out of 3M. Live reward was not collapsed: peaked around `33.7` near 1.75M, had another good phase around 2.80M (`~32.7`), and was about `31.1` when stopped. Follow-up Modal volume check confirmed remote artifacts exist, including checkpoints every 50k from `50,000` through `2,950,000`, but this run still has not been downloaded or checkpoint-scanned locally.

Current best confirmed result:

- Best confirmed policy remains `modal_area_resize_hud32_term_completion_500k` checkpoint `274912` at `1/100` stochastic completions with the current eval protocol. This is far below the 80% target.

Resume checklist:

1. Download the confirmed remote volume data:

```bash
UV_CACHE_DIR=.uv-cache uv run modal volume get --force mario-ppo-data /runs/modal_area_hud32_baseline_seed11_3m runs/modal_volume_downloads
```

2. Run a 20-episode stochastic checkpoint scan. Prioritize checkpoints near 1.0M, 1.35-1.45M, 1.70-1.80M, 2.75-2.85M, and the last saved checkpoint.
3. Only run 100-episode confirmation if a checkpoint shows a real signal in the 20-episode scan. Earlier `2/20` did not hold up.

Main lesson:

- Live rollout reward in the low 30s and occasional 20-episode completions are not enough. This setup can produce rare clears, but completion estimates are noisy and the policy is not robust. The next useful work is checkpoint selection/confirmation on the seed-11 run, then a more structural change if completion remains under roughly 10%.

## 2026-06-12 SkyPilot RTX 4090 Run

- Added `sky_mario_score_4090.yaml` for the home-network SkyPilot provider `ssh/rtx4090`.
- Sky API server: `http://192.168.0.151:46580`; cluster dashboard: `http://192.168.0.151:46580/dashboard/clusters/mario-ppo-4090`.
- Cluster/job: `mario-ppo-4090`, Sky job `1`, task `mario-score-4090`.
- Run name: `sky_score_style_simple_maxpool_5m_seed23`.
- Config: score-based shaping, simple action set, frame skip 4, max-pool last two skipped frames, `gamma=0.9`, `gae_lambda=1.0`, lr `1e-4`, entropy `0.01`, clip `0.2`, 10 epochs, `terminal_reward=50`, `reward_scale=10`, `terminate_on_completion`, 5M timesteps, 16 envs.
- Setup succeeded: `SuperMarioBros-Nes-v0` imported on the remote, PyTorch reported CUDA available on `NVIDIA GeForce RTX 4090`.
- Early training status: job reached `RUNNING`; first PPO logs showed about `1277-1468 fps`, reward rising from `37.4` at 8k steps to `63.4` at 49k steps.
- Useful commands:

```bash
/Users/tsilva/repos/tsilva/sandbox-skypilot/.venv/bin/sky queue mario-ppo-4090
/Users/tsilva/repos/tsilva/sandbox-skypilot/.venv/bin/sky logs mario-ppo-4090 1
/Users/tsilva/repos/tsilva/sandbox-skypilot/.venv/bin/sky cancel mario-ppo-4090 1
```

## 2026-06-12 Upstream Viet PPO Modal Reproduction

- User asked to run `vietnh1009/Super-mario-bros-PPO-pytorch` itself on Modal to test whether the upstream implementation works and estimate how many timesteps it needs to solve Level 1-1.
- Cloned upstream repo into `/Users/tsilva/repos/tsilva/sandbox-sb3/viet-super-mario-ppo`.
- Added a Modal compatibility harness around the upstream env/model/PPO code. Operational changes only: old dependency stack pinned for Modal, headless eval, real `num_global_steps` stopping, checkpoint saves, JSON metrics, worker cleanup, and deployed async launcher. PPO settings match upstream defaults: simple action set, `gamma=0.9`, `tau/GAE=1.0`, lr `1e-4`, `beta=0.01`, clip `0.2`, 10 epochs, 8 envs, 512 local steps.
- Smoke run `viet_level1_smoke_8192` completed on Modal T4. It trained to 8,192 policy-decision timesteps and produced checkpoint/eval metrics, confirming the upstream repo can run headlessly after compatibility fixes.
- Long deployed run started via deployed Modal app `viet-mario-ppo`, app id `ap-f17d64qgIJxeqYkHcnbQKl`, function call `fc-01KTYS8P75TVHHRYC3G0NKTXYJ`, run name `viet_level1_repro_5m_seed123`, target `5,000,000` policy-decision timesteps, checkpoints every 50k, 20-episode stochastic scans, 100-episode confirmation if completion rate reaches 80%.
- First checkpoint scan completed at step `53,248`: `completion_rate=0.00` over 20 stochastic evals, `max_x_max=1130`, `max_x_mean=691.25`, `reward_mean=49.35`, `death_rate=1.00`. Training resumed and had reached at least `77,824` steps in logs.
- Later scans showed steady progress but no clears through `352,256` steps. First nonzero completion appeared at step `401,408`: `completion_rate=0.05` over 20 stochastic evals, `completion_count=1`, `max_x_max=3161`, `max_x_mean=1863.35`, `reward_mean=173.42`, `death_rate=0.95`. This confirms upstream-style PPO can rarely clear Level 1-1 by about 400k timesteps, but it is far below the 80% solve threshold.
- Best result through about 1M timesteps: checkpoint `802,816` reached `completion_rate=0.30` over 20 stochastic evals (`6/20`), `max_x_mean=2371.75`, and `reward_mean=226.47`. Nearby checkpoints were lower: `851,968` was `5/20`, `901,120` was `2/20`, `950,272` was `2/20`, and `1,003,520` was `2/20`. Lesson so far: upstream PPO clearly works better than the local SB3 attempts for Level1-1 under this eval protocol, but it has not robustly solved the level by 1M timesteps and still needs checkpoint selection/longer training.
- Final 5M result: run completed cleanly with 100 eval scans. No checkpoint reached the `>=16/20` (`80%`) solve threshold, so no 100-episode confirmation was triggered. Best checkpoint was `4,603,904` steps at `14/20` stochastic clears (`70%`), `reward_mean=276.39`, `max_x_mean=2828.3`. Second-best was `4,751,360` at `14/20` with lower reward/position; other strong checkpoints included `3,403,776` at `13/20`, `2,953,216` at `12/20`, and final `5,001,216` at `12/20`.
- Answer to the upstream-baseline question: the repo works and learns Level 1-1 substantially better than our local SB3 setup, with first rare clears around `401,408` steps, `6/20` by `802,816`, `12/20` by `2,953,216`, and best `14/20` by `4,603,904`. Under this eval protocol it did not robustly solve Level 1-1 within 5M timesteps.
- Main lesson: upstream PPO's fixed entropy bonus (`beta=0.01`, no entropy schedule) is enough to discover the level-completion behavior, but training is non-monotonic and remains unreliable. Best-checkpoint selection is mandatory, final checkpoint quality is not sufficient, and an 80%+ solve likely needs either more seeds/steps, deterministic/temperature-controlled evaluation, entropy/coefficient scheduling, or a structural improvement to exploration/reward/curriculum.
- Useful follow-up commands:

```bash
/tmp/modal-cli-venv/bin/modal app logs ap-f17d64qgIJxeqYkHcnbQKl
/tmp/modal-cli-venv/bin/modal app list
/tmp/modal-cli-venv/bin/modal volume get --force viet-mario-ppo-data /runs/viet_level1_repro_5m_seed123/metrics.jsonl /tmp/viet_level1_metrics.jsonl
/tmp/modal-cli-venv/bin/modal volume ls viet-mario-ppo-data /runs/viet_level1_repro_5m_seed123/checkpoints
```

## 2026-06-13 Local Apple Silicon MPS Training Throughput

- `torch==2.12.0` macOS arm64 reported MPS unavailable on macOS `26.5.1` even though MPS was built into the wheel. Pinning local Apple Silicon to `torch==2.11.0` fixed actual `mps:0` tensor execution.
- SB3's `device="auto"` resolves to CPU even when MPS is available, so local code now resolves `auto` as CUDA, then MPS, then CPU before passing the device to SB3.
- Training defaults were updated for the local machine: `n_envs=64`, `n_steps=128`, `batch_size=1024`, `n_epochs=2`, `device=auto -> mps`.
- Stable-retro-turbo native preprocessing remains: HUD crop `(32, 0, 0, 0)`, resize `84x84` with `area`, grayscale, frame skip `4`, frame stack `4`, max-pool last two frames, audio disabled for benchmarks.
- Benchmark sweep on `SuperMarioBros-Nes-v0`, `action_set=right`, native vec env, MPS:
  - `16 envs`, `128 steps`, `batch 1024`, `1 epoch`: `935.4` env steps/s.
  - Same shape on CPU: `483.5` env steps/s.
  - `32 envs`, `128 steps`, `batch 1024`, `1 epoch`: best short run `1488.2` env steps/s; longer 5-update run `1066.0` env steps/s.
  - `64 envs`, `128 steps`, `batch 1024`, `1 epoch`: best short run `1588.9` env steps/s; longer 5-update run `1410.1` env steps/s.
  - `64 envs`, `128 steps`, `batch 1024`, `2 epochs`: `1470.4` env steps/s with `8192` env steps per PPO update.
  - `64 envs`, `128 steps`, `batch 1024`, `4 epochs`: `991.5` env steps/s.
  - `64 envs`, `64 steps`, `batch 1024`, `2 epochs`: `1089.8` env steps/s.
  - `96 envs`, `128 steps`, `batch 1024`, `2 epochs`: `1006.0` env steps/s.
- Current recommendation: use `64 envs / 128 n_steps / batch 1024 / 2 PPO epochs / MPS` for local training. It keeps more PPO reuse than `1 epoch` while staying close to the best observed throughput.

## 2026-06-13 Local MPS Native 1M Run

- Completed `local_mps_native_1m` on Apple MPS with `StableRetroNativeVecEnv`, `action_set=right`, `n_envs=64`, `n_steps=128`, `batch_size=1024`, `n_epochs=2`, and the standardized stable-retro-turbo native preprocessing.
- Command: `STABLE_RETRO_DISABLE_AUDIO=1 uv run python -m mario_ppo.train --timesteps 1000000 --run-name local_mps_native_1m --action-set right`
- Final SB3 boundary step: `1,007,616` env steps after `123` rollout iterations and `244` PPO update epochs. Wall-clock progress bar reported `0:14:11`; final cumulative training fps was `1177`.
- Final rollout log: `ep_len_mean=86.3`, `ep_rew_mean=14.0`, `approx_kl=0.0107`, `clip_fraction=0.0781`, `entropy_loss=-1.28`, `explained_variance=0.886`, `value_loss=2.28`.
- Artifacts: final model at `runs/local_mps_native_1m/final_model.zip`; checkpoints at approximately every 100k steps from `99,968` through `999,680`. Run directory size after training: about `240M`.
- Quick deterministic CPU eval of the final model over 5 episodes with `--max-steps 2500 --action-set right`: `completion_rate=0.0`, `death_rate=1.0`, `max_x_mean=max_x_max=325`, `reward_mean=9.83`. This final checkpoint is not useful as a solved policy; checkpoint scanning may still be needed, but the run mainly validates local MPS/native training throughput.

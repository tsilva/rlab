# SuperMarioBros-NES Level 1 Winning Training Recipe

Last updated: 2026-06-22

This is the current preferred training recipe for `SuperMarioBros-Nes-v0`
`Level1-1` screening on the RTX4090. The current incumbent is B55
`lowkl_lrdecay`, which refined the earlier B33/B40 recipe by lowering PPO update
pressure and adding late learning-rate decay. It is preferred over the older
B40 note because both B55 seeds reached the strict `100/100` stop.

## Recipe

```text
stable-retro-turbo: 1.0.0.post16
game: SuperMarioBros-Nes-v0
state: Level1-1
n_envs: 16
env_threads: 4
torch_num_threads: 1
n_steps: 512
batch_size: 512
n_epochs: 10
learning_rate: 1.5e-4 -> 1.0e-4 over 4M
ent_coef: 0.01 -> 0.0001 over 4M
gamma: 0.9
gae_lambda: 1.0
clip_range: 0.15
target_kl: 0.16
vf_coef: 1.0
adam_eps: 1e-8
normalize_advantage: false
reward_mode: score
score_progress_clipped: false
terminal_reward: 50
death_penalty: 25
reward_scale: 10
action_set: simple
frame_skip: 4
max_pool_frames: false
obs_resize_algorithm: area
observation_size: 84
hud_crop_top: 32
max_episode_steps: 4500
completion_x_threshold: 0
terminate_on_life_loss: true
terminate_on_level_change: false
terminate_on_completion: true
eval_freq: 0
eval_episodes: 0
checkpoint_freq: 100000
timesteps: 5,000,000
stop_completion_episode_window: 100
stop_completion_rate_threshold: 1.0
```

Use 5 concurrent child trainings on `k8s/rtx4090` for throughput screening.
Recent recipe-search batches used 6 concurrent child trainings for 3-arm /
2-seed comparisons; that is acceptable for search continuity, but the measured
default throughput shape in `INSTANCES.md` remains 5 children.
Level completion is detected from stable-retro `levelHi`/`levelLo` changes, not
from an x-position threshold.

## Evidence

B55 `lowkl_lrdecay`, `stable-retro-turbo==1.0.0.post16`, seeds `108-109`:

```text
wandb_group: b55-level1-1-recipe-search-lowkl-refine-3arms-2seeds-6parallel-20260621_194855
seed108 qt3h08mc final 100/100, stop step 4,273,200, total completions 2,622
seed109 actk7fw5 final 100/100, stop step 4,471,008, total completions 1,506
mean_stop_step: 4,372,104
total_completions: 4,128
```

B57 control repeat of the same B55 recipe, seeds `116-117`, did not beat B55
but supports that the recipe is strong and still seed-variable:

```text
wandb_group: b57-b55-level1-1-targetkl-bracket-3arms-2seeds-6parallel-20260622_083007
seed116 c916vdl6 final 96/100 at 5,005,312, total completions 1,698
seed117 8j5mlmu0 final 100/100, stop step 4,141,584, total completions 1,981
```

Decision: B55 `lowkl_lrdecay` is the current incumbent because it has paired
strict `100/100` success. Rank future recipes by paired `1.0`, then lower mean
stop step, then higher total completions, then lower seed variance.

## Historical Baseline

B40 no-maxpool, `stable-retro-turbo==1.0.0.post14`, seeds `42-46`:

```text
wandb_group: b40-b33-targetkl020-post14-no-maxpool-5parallel-20260619_190307
seed42 yo9p05i3 final 39/100, peak 83/100 at 2,726,112
seed43 ibdjtkba final 65/100, peak 88/100 at 4,442,944
seed44 8dea4isi final 24/100, peak 30/100 at 4,765,136
seed45 aoc0pevt final 56/100, peak 68/100 at 4,741,392
seed46 6ugxz6bs final 100/100, peak 100/100 at 3,881,520
mean_final: 56.8/100
median_final: 56/100
best_final: 100/100
best_peak: 100/100
aggregate_fps: 6429
```

B39 maxpool comparison, same post14 target-KL 0.20 recipe, seeds `37-41`:

```text
wandb_group: b39-b33-targetkl020-post14-5parallel-20260619_171334
mean_final: 70.2/100
median_final: 84/100
best_final: 90/100
best_peak: 93/100
aggregate_fps: 6095
```

B40 was the previous recorded winning screening recipe. It produced one strict
`100/100` winner, but it was weaker on paired reproducibility than B55.

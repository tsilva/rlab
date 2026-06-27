# SuperMarioBros-NES Level1-2 Teacher Recipe

Last updated: 2026-06-26

This is the current starting recipe for `SuperMarioBros-Nes-v0` `Level1-2`
100/100 work. It captures the historical B46 Level1-2 teacher as evidence, but
does not treat that single run as a confirmed baseline. New comparable launches
should revalidate the recipe under the active post21 runtime.

## Recipe

```text
stable-retro-turbo: 1.0.0.post21 for new launches; historical evidence came from a post14-era same-hparams run
game: SuperMarioBros-Nes-v0
state: Level1-2
n_envs: 16
env_threads: 4
torch_num_threads: 1
n_steps: 512
batch_size: 512
n_epochs: 10
learning_rate: fixed 1.5e-4
ent_coef: 0.01 -> 0.0003 over 2M
gamma: 0.9
gae_lambda: 1.0
clip_range: 0.15
target_kl: 0.20
vf_coef: 1.0
adam_eps: 1e-8
normalize_advantage: false
reward_mode: score
progress_reward_cap: 30.0
progress_reward_scale: 1.0
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
info_events_json: '{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}'
done_on_events: life_loss,level_change
eval_freq: 0
eval_episodes: 0
checkpoint_freq: 100000
timesteps: 5,000,000
```

Use `info_events_json` to define observed info-variable events and
`done_on_events` to choose which observed events terminate episodes. Do not use
the legacy `done_on_info_json` key in new launch specs.

Level completion is detected from stable-retro `levelHi`/`levelLo` changes, not
from an x-position threshold. For Level1-2, the training stop metric is
`train/info/level_complete/from/0-1/rate`.

## Evidence

B46 `b44_level1_2_same_hparams`, seed `60`:

```text
wandb_run_id: 5fpk7ki3
run_name: b46_b44_level1_2_same_hparams_5m_stop100ep100_seed60_20260620_124408
local_artifact: runs/wandb_artifacts/tsilva_SuperMarioBros-NES_b46_b44_level1_2_same_hparams_5m_stop100ep100_seed60_20260620_124408-checkpoint_latest/ppo_supermariobros-nes-v0_5000000_steps.zip
```

This run is useful teacher evidence, and it was later used as the Level1-2
teacher for Level1-1/Level1-2 policy distillation. It is not yet enough to
declare a Level1-2 population baseline; confirmation needs fresh seeds under
the current goal contract.

## Promotion Rule

Rank future Level1-2 recipes by five-seed strict `100/100` success count, then
lower median stop step, then lower worst successful seed stop step, then higher
external checkpoint eval completion rate.

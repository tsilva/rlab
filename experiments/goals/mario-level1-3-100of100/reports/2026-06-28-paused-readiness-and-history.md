# 2026-06-28 Paused Readiness And History

Goal: `mario-level1-3-100of100`

Status: paused by operator instruction. Do not relaunch until the operator
explicitly says to resume.

## Live State

Read-only queue check:

```text
UV_CACHE_DIR=.uv-cache uv run --frozen rlab-queue status --goal mario-level1-3-100of100
```

Current queue state shows five failed train jobs, no active train jobs, no
active eval jobs, and no eval results for this goal.

W&B read-only refresh shows the interrupted B117-B121 runs remain `crashed`:

| Run | W&B id | State | Global step | Level1-3 rate | Level1-3 count |
| --- | --- | --- | ---: | ---: | ---: |
| `b117_l13_b55post21_s80_20260628T095535Z` | `rmx94up8` | `crashed` | `941248` | `0` | `0` |
| `b118_l13_b55complete25_s80_20260628T095550Z` | `ctfz60j2` | `crashed` | `913920` | `0` | `0` |
| `b119_l13_b55slowent_s80_20260628T095605Z` | `3dd9af5y` | `crashed` | `864464` | `0` | `0` |
| `b120_l13_b46style_s80_20260628T095621Z` | `pk4cfmcg` | `crashed` | `851968` | `0` | `0` |
| `b121_l13_b55slowent_complete25_s80_20260628T100401Z` | `7a1mvmtl` | `crashed` | `114064` | `0` | `0` |

These interrupted runs are stale partial evidence only. They do not prove any
recipe failed under the 5M-step contract.

## Resume Batch Readiness

The five checked-in resume specs load successfully with
`rlab.job_queue.load_spec_document`:

- `specs/b117-b55post21-l13-screen.json`
- `specs/b118-b55complete25-l13-screen.json`
- `specs/b119-b55slowent-l13-screen.json`
- `specs/b120-b46style-l13-screen.json`
- `specs/b121-b55slowent-complete25-l13-screen.json`

Focused unittest guard passed:

```text
UV_CACHE_DIR=.uv-cache uv run --frozen python -m unittest \
  tests.test_job_queue_runner.JobQueueTests.test_level1_3_specs_configure_goal_metric_early_stop \
  tests.test_job_queue_runner.JobQueueTests.test_checked_in_goal_specs_match_train_spec_schema
```

Result: `Ran 2 tests ... OK`.

All five specs use:

- target `rtx4090`
- state `Level1-3`
- seed `80`
- cap `5000000`
- early stop metric `train/info/level_complete/rate/min/last`
- early stop threshold `> 0.99`

## Historical Level1-3 Negative Control

W&B contains an older Level1-3 batch not captured in this goal folder:
`b49-b47-post16-level1-3-5parallel-20260620_182227`.

This is not current-contract evidence because it predates the active post21
goal metrics, but it is useful context:

| Seed | Run id | State | Steps | `train/completion_episodes_total` |
| ---: | --- | --- | ---: | ---: |
| `72` | `1y37913x` | `finished` | `5005312` | `0` |
| `73` | `0t4zts36` | `finished` | `5005312` | `0` |
| `74` | `sao6h9dx` | `finished` | `5005312` | `0` |
| `75` | `vp57i3jb` | `finished` | `5005312` | `0` |
| `76` | `1fug3bq2` | `finished` | `5005312` | `0` |

B49's relevant config shape:

- `state=Level1-3`
- `learning_rate=0.00015`, effectively fixed
- `target_kl=0.20`
- `ent_coef=0.01`, `ent_coef_final=0.0003`,
  `ent_coef_schedule_timesteps=2000000`
- `completion_reward=0`
- `terminal_reward=50`, `death_penalty=25`
- `reward_mode=score`
- `terminate_on_completion=true`
- `timesteps=5000000`

Interpretation: the historical fixed-LR, higher-KL, no-completion-bonus shape
was not enough for Level1-3 across five seeds. That makes current B120 useful as
a post21 diagnostic, but not the best prior for success. The cleaner bets remain
B118/B121 for true completion reinforcement and B119/B121 for slower entropy.

## Next Action When Resumed

When the operator says to resume, relaunch the same five checked-in specs before
designing replacement arms. After the resumed jobs finish, rank by the goal
contract:

1. Peak `train/info/level_complete/rate/min/last`
2. Peak `train/info/level_complete/rate/mean/last`
3. Out-of-process eval completion metrics

If a resumed candidate reaches the strict source-attempt window, freeze that
recipe and run the goal confirmation seeds `80,81,82,83,84` plus out-of-process
eval before declaring success. If none of B117-B121 records real Level1-3
completions by the cap, design the next batch from legal reward and
hyperparameter levers only.

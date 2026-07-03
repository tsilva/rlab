from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def stub_module(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


stub_module("rlab.config_validation", load_goal_contract=lambda *_args, **_kwargs: {})
stub_module("rlab.dotenv", load_env_file=lambda *_args, **_kwargs: None)
stub_module(
    "rlab.env",
    EnvConfig=object,
    assert_rom_imported=lambda *_args, **_kwargs: None,
    resolve_env_config=lambda config: config,
)
stub_module(
    "rlab.env_config_aliases",
    normalize_provider_env_config_aliases=lambda config, **_kwargs: config,
)
stub_module("rlab.eval_runner", evaluate_model_episodes=lambda *_args, **_kwargs: ({}, None))
stub_module("rlab.json_utils", json_safe=lambda value: value)
stub_module("rlab.model_sources", download_artifact_ref_source=lambda *_args, **_kwargs: None)
stub_module("rlab.seeds", DEFAULT_EVAL_SEED=10000, validate_eval_seed=lambda seed: seed)
stub_module(
    "rlab.wandb_leaders",
    CHECKPOINT_PRIMARY_ORDER="-summary_metrics.leader/checkpoint/completion_rate",
    checkpoint_leader=lambda run: run,
    checkpoint_summary_filter=lambda: {},
    rank_checkpoint_leaders=lambda leaders: list(leaders),
    wandb_runs=lambda **_kwargs: [],
)
stub_module("rlab.wandb_utils", DEFAULT_WANDB_PROJECT_PATH="tsilva/SuperMarioBros-NES")


class ReleaseModelCardTests(unittest.TestCase):
    def test_model_card_uses_hf_video_preview_without_inline_embed(self) -> None:
        from rlab.release import HuggingFaceReleaseConfig, write_model_card

        goal = {
            "goal_id": "Level1-1",
            "title": "Sample-efficient Level1-1 completion",
            "train": {
                "environment": {
                    "env_config": {
                        "game": "SuperMarioBros-Nes-v0",
                        "state": "Level1-1",
                    },
                },
            },
        }
        release = HuggingFaceReleaseConfig(
            owner="tsilva",
            repo="SuperMarioBros-NES_Level1-1",
            card_template="stable-retro-sb3",
            checkpoint_filename="ppo_supermariobros-nes-v0_{checkpoint_step}_steps.zip",
            preview_filename="replay.mp4",
            include_youtube_preview=True,
        )
        leader = SimpleNamespace(
            goal_slug="Level1-1",
            spec_slug="b55",
            run_id="abc123",
            run_name="b272-l11-b55-s1-20260703T000000Z",
            url="https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abc123",
            completion_rate=1.0,
            completion_rate_mean=1.0,
            max_x_max=3161.0,
            reward_mean=1234.5,
            steps_to_completion_goal=500000.0,
            checkpoint_step=500000,
            artifact_ref="tsilva/SuperMarioBros-NES/example-checkpoint:step-500000",
            eval_source="post_train_inline",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README.md"

            write_model_card(
                path=path,
                goal=goal,
                release=release,
                leader=leader,
                metrics={"episodes": 100},
                checkpoint_filename="ppo_supermariobros-nes-v0_500000_steps.zip",
            )

            content = path.read_text(encoding="utf-8")

        self.assertNotIn("<video", content)
        self.assertNotIn("resolve/main/replay.mp4", content)
        self.assertIn("| `replay.mp4` | Representative preview episode |", content)


if __name__ == "__main__":
    unittest.main()

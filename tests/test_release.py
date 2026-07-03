from __future__ import annotations

import tempfile
import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def make_stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def import_release_card_symbols():
    stubs = {
        "rlab.config_validation": make_stub_module(
            "rlab.config_validation",
            load_goal_contract=lambda *_args, **_kwargs: {},
        ),
        "rlab.dotenv": make_stub_module("rlab.dotenv", load_env_file=lambda *_args, **_kwargs: None),
        "rlab.env": make_stub_module(
            "rlab.env",
            EnvConfig=object,
            assert_rom_imported=lambda *_args, **_kwargs: None,
            resolve_env_config=lambda config: config,
        ),
        "rlab.env_config_aliases": make_stub_module(
            "rlab.env_config_aliases",
            normalize_provider_env_config_aliases=lambda config, **_kwargs: config,
        ),
        "rlab.eval_runner": make_stub_module(
            "rlab.eval_runner",
            evaluate_model_episodes=lambda *_args, **_kwargs: ({}, None),
        ),
        "rlab.json_utils": make_stub_module("rlab.json_utils", json_safe=lambda value: value),
        "rlab.model_sources": make_stub_module(
            "rlab.model_sources",
            download_artifact_ref_source=lambda *_args, **_kwargs: None,
        ),
        "rlab.seeds": make_stub_module(
            "rlab.seeds",
            DEFAULT_EVAL_SEED=10000,
            validate_eval_seed=lambda seed: seed,
        ),
        "rlab.wandb_leaders": make_stub_module(
            "rlab.wandb_leaders",
            CHECKPOINT_PRIMARY_ORDER="-summary_metrics.leader/checkpoint/completion_rate",
            checkpoint_leader=lambda run: run,
            checkpoint_summary_filter=lambda: {},
            rank_checkpoint_leaders=lambda leaders: list(leaders),
            wandb_runs=lambda **_kwargs: [],
        ),
        "rlab.wandb_utils": make_stub_module(
            "rlab.wandb_utils",
            DEFAULT_WANDB_PROJECT_PATH="tsilva/SuperMarioBros-NES",
        ),
    }
    module_names = [*stubs, "rlab.release"]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    rlab_package = sys.modules.get("rlab")
    saved_attrs = {}
    if rlab_package is not None:
        for name in stubs:
            attr = name.split(".", 1)[1]
            if hasattr(rlab_package, attr):
                saved_attrs[attr] = getattr(rlab_package, attr)
    try:
        sys.modules.pop("rlab.release", None)
        sys.modules.update(stubs)
        module = importlib.import_module("rlab.release")
        return module.HuggingFaceReleaseConfig, module.write_model_card, module.copy_release_files
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if rlab_package is not None:
            for name in stubs:
                attr = name.split(".", 1)[1]
                if attr in saved_attrs:
                    setattr(rlab_package, attr, saved_attrs[attr])
                elif hasattr(rlab_package, attr):
                    delattr(rlab_package, attr)


class ReleaseModelCardTests(unittest.TestCase):
    def test_model_card_uses_hf_video_preview_without_inline_embed(self) -> None:
        HuggingFaceReleaseConfig, write_model_card, _copy_release_files = import_release_card_symbols()

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
            checkpoint_filename="model.zip",
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
                checkpoint_filename="model.zip",
            )

            content = path.read_text(encoding="utf-8")

        at_a_glance = content.split("## Preview", 1)[0]

        self.assertNotIn("<video", content)
        self.assertNotIn("resolve/main/replay.mp4", content)
        self.assertIn(
            "trained with [`rlab`](https://github.com/tsilva/rlab).",
            content,
        )
        self.assertIn(
            "| Source project | [`rlab`](https://github.com/tsilva/rlab) |",
            content,
        )
        self.assertIn(
            "current [`rlab`](https://github.com/tsilva/rlab) checkpoint promotion contract",
            content,
        )
        self.assertIn("uv tool install --from git+https://github.com/tsilva/rlab rlab", content)
        self.assertIn("rlab import-roms ~/roms --game SuperMarioBros-Nes-v0", content)
        self.assertIn("rlab play hf://tsilva/SuperMarioBros-NES_Level1-1", content)
        self.assertIn("rlab eval hf://tsilva/SuperMarioBros-NES_Level1-1", content)
        self.assertNotIn("hf download", content)
        self.assertIn("## Preview", content)
        self.assertIn("## Evaluation Results", content)
        self.assertIn("## Environment Details", content)
        self.assertIn("## Architecture", content)
        self.assertIn("## Training Recipe", content)
        self.assertIn("| Preview video | `replay.mp4` |", content)
        self.assertIn("| Component | Value |", content)
        self.assertIn("| Limitation | Detail |", content)
        self.assertIn("| Environment | `SuperMarioBros-Nes-v0`, state `Level1-1` |", content)
        self.assertIn("| Input | 4 stacked grayscale `84 x 84` frames, channel-first |", content)
        self.assertIn("| Output | Discrete action over the `simple` action set |", content)
        self.assertIn("| Runtime | Environment provider | `Stable Retro` |", content)
        self.assertIn("| Runtime | Environment id | `SuperMarioBros-Nes-v0` |", content)
        self.assertIn(
            "| Observation | Preprocessing | crop top `32` px, grayscale, resize to `84 x 84` |",
            content,
        )
        self.assertIn("| Action | Action set | `simple` |", content)
        self.assertIn("| Reward | Reward shaping | reward_mode=`score` |", content)
        self.assertIn("| Termination | Done conditions | goal-specific termination |", content)
        self.assertNotIn("| W&B run |", at_a_glance)
        self.assertNotIn("| W&B artifact |", at_a_glance)
        self.assertIn("| W&B run |", content)
        self.assertIn("| W&B artifact |", content)
        self.assertNotIn("For the original W&B artifact", content)
        self.assertNotIn("rlab leaders checkpoints", content)
        self.assertNotIn("Release staging re-evaluated", content)
        self.assertNotIn("Validate on the Eval Profile", content)
        self.assertNotIn("stable_retro_ppo.evaluate", content)
        self.assertNotIn("Expected summary fields", content)
        self.assertNotIn("leaderboard", content.lower())
        self.assertIn(
            "| `replay.mp4` | Hugging Face reinforcement-learning widget preview",
            content,
        )

    def test_copy_release_files_rewrites_model_metadata_filename(self) -> None:
        _config, _write_model_card, copy_release_files = import_release_card_symbols()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ppo_supermariobros-nes-v0_500000_steps.zip"
            source.write_bytes(b"model")
            source.with_suffix(".metadata.json").write_text(
                '{"filename": "ppo_supermariobros-nes-v0_500000_steps.zip"}\n',
                encoding="utf-8",
            )
            stage_dir = root / "stage"
            stage_dir.mkdir()

            checkpoint_path, metadata_path = copy_release_files(
                stage_dir=stage_dir,
                model_path=source,
                checkpoint_filename="model.zip",
            )

            self.assertEqual(checkpoint_path.name, "model.zip")
            self.assertIsNotNone(metadata_path)
            assert metadata_path is not None
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{\n  "filename": "model.zip"\n}\n',
            )


if __name__ == "__main__":
    unittest.main()

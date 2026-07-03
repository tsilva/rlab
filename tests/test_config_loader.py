from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rlab.config_loader import load_composed_mapping, template_context_from_path


class ConfigLoaderTests(unittest.TestCase):
    def test_template_context_does_not_infer_state_from_goal_path(self) -> None:
        path = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-2/specs/base.yaml")

        context = template_context_from_path(path)

        self.assertEqual(context["goal_id"], "Level1-2")
        self.assertNotIn("state", context)

    def test_load_composed_mapping_merges_parent_chain_and_tracks_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.yaml"
            middle = root / "layers" / "middle.yaml"
            child = root / "goals" / "child.yaml"
            middle.parent.mkdir()
            child.parent.mkdir()
            base.write_text(
                """
name: base
settings:
  keep: parent
  replace: parent
items:
- parent-list
""",
                encoding="utf-8",
            )
            middle.write_text(
                """
defaults:
- ../base@_global_
- _self_
settings:
  replace: middle
  middle_only: true
""",
                encoding="utf-8",
            )
            child.write_text(
                """
defaults:
- ../layers/middle@_global_
- _self_
name: child
settings:
  child_only: true
items:
- child-list
""",
                encoding="utf-8",
            )

            composed = load_composed_mapping(child, cycle_label="test")

        self.assertEqual(
            composed.document,
            {
                "name": "child",
                "settings": {
                    "keep": "parent",
                    "replace": "middle",
                    "middle_only": True,
                    "child_only": True,
                },
                "items": ["child-list"],
            },
        )
        self.assertEqual(composed.sources, (base.resolve(), middle.resolve(), child.resolve()))

    def test_load_composed_mapping_rejects_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.yaml"
            second = root / "second.yaml"
            first.write_text(
                "defaults:\n- second@_global_\n- _self_\nname: first\n",
                encoding="utf-8",
            )
            second.write_text(
                "defaults:\n- first@_global_\n- _self_\nname: second\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cyclic Hydra defaults chain"):
                load_composed_mapping(first, cycle_label="test")

    def test_load_composed_mapping_composes_fragment_into_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fragment = root / "fragments" / "env.yaml"
            goal = root / "goals" / "goal.yaml"
            fragment.parent.mkdir()
            goal.parent.mkdir()
            fragment.write_text(
                """
env_provider: stable-retro-turbo
game: SuperMarioBros-Nes-v0
frame_skip: 4
""",
                encoding="utf-8",
            )
            goal.write_text(
                """
defaults:
- ../fragments/env@train.environment.env_config
- _self_
train:
  environment:
    env_config:
      state: Level1-1
""",
                encoding="utf-8",
            )

            composed = load_composed_mapping(goal, cycle_label="test")

        self.assertEqual(
            composed.document["train"]["environment"]["env_config"],
            {
                "env_provider": "stable-retro-turbo",
                "game": "SuperMarioBros-Nes-v0",
                "frame_skip": 4,
                "state": "Level1-1",
            },
        )
        self.assertEqual(composed.sources, (fragment.resolve(), goal.resolve()))


if __name__ == "__main__":
    unittest.main()

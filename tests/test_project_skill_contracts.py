from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SKILL = REPO_ROOT / ".codex/skills/launch-experiment/SKILL.md"


def test_launch_skill_describes_goal_owned_recipe_location() -> None:
    skill = LAUNCH_SKILL.read_text()
    assert "one recipe under `experiments/recipes/`" not in skill
    assert "selected goal's `recipes/` directory" in skill
    assert "`experiments/recipes/_presets/`" in skill

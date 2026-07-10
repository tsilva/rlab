from __future__ import annotations

from types import SimpleNamespace

from rlab.play import StepOverControls


def test_step_over_controls_advance_once_on_space_press() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=32), keydown_type=1, keyup_type=2, step_key=32
    )
    controls.handle_event(
        SimpleNamespace(type=2, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert controls.consume_step()
    assert not controls.consume_step()


def test_step_over_controls_keep_advancing_while_space_is_held() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert controls.consume_step()
    assert controls.consume_step()

    controls.handle_event(
        SimpleNamespace(type=2, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert not controls.consume_step()


def test_step_over_controls_ignore_other_keys() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=13), keydown_type=1, keyup_type=2, step_key=32
    )

    assert not controls.consume_step()

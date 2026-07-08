from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from rlab.env_wrappers import SuperMarioBrosNesProgressInfoWrapper
from rlab.event_payloads import event_payloads
from rlab.fused_vec import FusedVectorHooks, HookStep, VectorInfoView
from rlab.targets import target_for_game


class SuperMarioBrosNesFusedHooks(FusedVectorHooks):
    """Fused Mario action mapping and reward/progress semantics."""

    hook_id = "super_mario_bros_nes"

    def __init__(self, config: Any, native_env: gym.vector.VectorEnv):
        super().__init__(config, native_env)
        if config.game != "SuperMarioBros-Nes-v0":
            raise ValueError("super_mario_bros_nes fused hooks require SuperMarioBros-Nes-v0")
        if config.task_conditioning:
            raise ValueError("super_mario_bros_nes fused hooks do not support task_conditioning yet")
        if config.no_progress_timeout_steps or config.max_episode_steps:
            # The current vector progress wrapper records those custom done flags
            # but does not feed them back into SB3's dones. Keep v1 parity tight.
            pass
        target = target_for_game(config.game)
        action_masks = target.action_masks_for_set(config.action_set)
        if not action_masks:
            raise ValueError("super_mario_bros_nes fused hooks require a discrete action_set")
        self.action_names = target.action_names_for_set(config.action_set)
        self.action_masks = np.stack(action_masks).astype(np.uint8)
        self._action_buffer = np.empty((int(native_env.num_envs), self.action_masks.shape[1]), dtype=np.uint8)
        self._action_space = gym.spaces.Discrete(len(self.action_masks))
        self.num_envs = int(native_env.num_envs)
        self.keys = self._progress_keys(config)

        self.level_x_pos = np.zeros(self.num_envs, dtype=np.int64)
        self.level_max_x_pos = np.zeros(self.num_envs, dtype=np.int64)
        self.completed_level_base = np.zeros(self.num_envs, dtype=np.int64)
        self.max_global_x_pos = np.zeros(self.num_envs, dtype=np.int64)
        self.curr_score = np.zeros(self.num_envs, dtype=np.int64)
        self.prev_lives = np.zeros(self.num_envs, dtype=np.int64)
        self.has_prev_lives = np.zeros(self.num_envs, dtype=bool)
        self.current_level_hi = np.zeros(self.num_envs, dtype=np.int64)
        self.current_level_lo = np.zeros(self.num_envs, dtype=np.int64)
        self.has_current_level = np.zeros(self.num_envs, dtype=bool)
        self.completed_level_count = np.zeros(self.num_envs, dtype=np.int64)
        self.current_level_completion_awarded = np.zeros(self.num_envs, dtype=bool)
        self.completed = np.zeros(self.num_envs, dtype=bool)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_progress_step = np.zeros(self.num_envs, dtype=np.int64)
        self.previous_event_values: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    @staticmethod
    def _progress_keys(config: Any) -> dict[str, str]:
        values = dict(SuperMarioBrosNesProgressInfoWrapper.default_keys)
        for spec in getattr(config, "env_wrappers", ()):
            if spec.get("id") == SuperMarioBrosNesProgressInfoWrapper.wrapper_id:
                values.update(spec.get("kwargs", {}))
        return values

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def map_actions(self, actions: Any) -> np.ndarray:
        action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        if action_indices.shape[0] != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} action ids, got {action_indices.shape[0]}"
            )
        self._action_buffer[...] = self.action_masks[action_indices]
        return self._action_buffer

    def on_reset(self, infos: VectorInfoView) -> None:
        del infos
        for index in range(self.num_envs):
            self._reset_lane(index, {})

    def shape_step(
        self,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: VectorInfoView,
    ) -> HookStep:
        dones = np.logical_or(terminations, truncations)
        if not bool(np.any(dones)) and infos.is_mapping:
            return self._shape_vector_step(rewards, infos)
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)
        lane_infos: dict[int, dict[str, Any]] = {}
        reset_infos: dict[int, dict[str, Any]] = {}
        for index in range(self.num_envs):
            info = (
                infos.lane_mapping("final_info", index)
                if bool(dones[index]) and infos.has_lane("final_info", index)
                else infos.lane(index)
            )
            reward, progress_info = self._shape_lane(index, float(rewards[index]), info, bool(dones[index]))
            shaped_rewards[index] = reward
            if bool(dones[index]):
                reset_infos[index] = infos.reset_info(index)
                lane_infos[index] = progress_info
            elif progress_info.get("info_events"):
                lane_infos[index] = progress_info
        if self.config.clip_rewards:
            shaped_rewards = np.sign(shaped_rewards).astype(np.float32)
        for index, reset_info in reset_infos.items():
            self._reset_lane(index, reset_info)
        return HookStep(shaped_rewards, lane_infos)

    def _shape_vector_step(self, rewards: np.ndarray, infos: VectorInfoView) -> HookStep:
        x_pos = (
            infos.array(self.keys["xscroll_hi_key"], 0).astype(np.int64) * 256
            + infos.array(self.keys["xscroll_lo_key"], 0).astype(np.int64)
        )
        score = infos.array(self.keys["score_key"], 0).astype(np.int64)
        level_hi = infos.array(self.keys["level_hi_key"], 0).astype(np.int64)
        level_lo = infos.array(self.keys["level_lo_key"], 0).astype(np.int64)
        lives_values = infos.array(self.keys["lives_key"], 0)
        has_lives = np.asarray(
            [infos.has_lane(self.keys["lives_key"], index) for index in range(self.num_envs)],
            dtype=bool,
        )
        lives = lives_values.astype(np.int64)

        event_payloads_by_lane = self._vector_event_payloads(infos)
        life_loss_event = self.keys["life_loss_event"]
        level_change_event = self.keys["level_change_event"]
        died = np.asarray(
            [
                life_loss_event in event_payloads_by_lane[index]
                or bool(infos.value(life_loss_event, index, False))
                for index in range(self.num_envs)
            ],
            dtype=bool,
        )
        died |= self.has_prev_lives & has_lives & (lives < self.prev_lives)
        self.prev_lives[has_lives] = lives[has_lives]
        self.has_prev_lives |= has_lives

        missing_level = ~self.has_current_level
        self.current_level_hi[missing_level] = level_hi[missing_level]
        self.current_level_lo[missing_level] = level_lo[missing_level]
        self.has_current_level[missing_level] = True
        native_level_changed = np.asarray(
            [level_change_event in payloads for payloads in event_payloads_by_lane],
            dtype=bool,
        )
        level_changed = native_level_changed | (level_hi != self.current_level_hi) | (
            level_lo != self.current_level_lo
        )
        level_changed &= ~missing_level
        completed_level = level_changed & ~died
        completion_event = completed_level & ~self.current_level_completion_awarded
        if np.any(completed_level):
            self.completed_level_base[completed_level] += self.level_max_x_pos[completed_level]
            self.completed_level_count[completed_level] += 1
            self.current_level_hi[completed_level] = level_hi[completed_level]
            self.current_level_lo[completed_level] = level_lo[completed_level]
            self.level_max_x_pos[completed_level] = 0
            self.current_level_completion_awarded[completed_level] = False

        self.level_x_pos = x_pos
        self.level_max_x_pos = np.maximum(self.level_max_x_pos, x_pos)
        global_x_pos = self.completed_level_base + self.level_x_pos
        global_max_x_pos = self.completed_level_base + self.level_max_x_pos
        progress_delta = np.maximum(0, global_max_x_pos - self.max_global_x_pos)
        self.max_global_x_pos = np.maximum(self.max_global_x_pos, global_max_x_pos)
        self.completed |= completion_event

        score_delta = np.maximum(0, score - self.curr_score)
        self.curr_score = score
        progressed = progress_delta > self.config.no_progress_min_delta
        self.last_progress_step[progressed] = self.episode_steps[progressed]
        shaped_rewards = self._reward_values_vector(
            rewards.astype(np.float32),
            progress_delta.astype(np.float32),
            score_delta.astype(np.float32),
            completion_event,
            died,
        )
        self.episode_steps += 1

        lane_infos: dict[int, dict[str, Any]] = {}
        for index, lane_event_payloads in enumerate(event_payloads_by_lane):
            progress_info = self._progress_info(
                index=index,
                native_reward=float(rewards[index]),
                progress_delta=int(progress_delta[index]),
                score_delta=int(score_delta[index]),
                completion_event=bool(completion_event[index]),
                died=bool(died[index]),
                done=False,
                global_x_pos=int(global_x_pos[index]),
                level_hi=int(level_hi[index]),
                level_lo=int(level_lo[index]),
                level_changed=bool(level_changed[index]),
            )
            if lane_event_payloads:
                progress_info["info_events"] = lane_event_payloads
            lane_infos[index] = progress_info
        if self.config.clip_rewards:
            shaped_rewards = np.sign(shaped_rewards).astype(np.float32)
        return HookStep(shaped_rewards.astype(np.float32), lane_infos)

    def _vector_event_payloads(self, infos: VectorInfoView) -> list[dict[str, Any]]:
        payloads = [event_payloads(infos.value("done_on_info", index)) for index in range(self.num_envs)]
        current_values_by_name: dict[str, list[Any]] = {}
        for name, rule in self.config.info_events.items():
            key_or_keys, _op = rule
            if isinstance(key_or_keys, str):
                current_values_by_name[name] = [
                    infos.value(key_or_keys, index, None) for index in range(self.num_envs)
                ]
            else:
                current_values_by_name[name] = [
                    tuple(infos.value(key, index, None) for key in key_or_keys)
                    for index in range(self.num_envs)
                ]
        for index in range(self.num_envs):
            current_values: dict[str, Any] = {}
            for name, rule in self.config.info_events.items():
                key_or_keys, op = rule
                current = current_values_by_name[name][index]
                if isinstance(key_or_keys, tuple) and any(value is None for value in current):
                    continue
                if current is None:
                    continue
                current_values[name] = current
                if name in payloads[index] or name not in self.previous_event_values[index]:
                    continue
                previous = self.previous_event_values[index][name]
                if self._event_rule_fired(previous, current, op):
                    payloads[index][name] = {
                        "op": op,
                        "keys": key_or_keys,
                        "prev": previous,
                        "next": current,
                    }
            self.previous_event_values[index].update(current_values)
        return payloads

    def _reset_lane(self, index: int, info: Mapping[str, Any] | None = None) -> None:
        info = info or {}
        self.level_x_pos[index] = 0
        self.level_max_x_pos[index] = 0
        self.completed_level_base[index] = 0
        self.max_global_x_pos[index] = 0
        self.curr_score[index] = self._int_info(info, self.keys["score_key"], 0)
        lives = info.get(self.keys["lives_key"])
        self.has_prev_lives[index] = lives is not None
        self.prev_lives[index] = int(lives) if lives is not None else 0
        has_level = self.keys["level_hi_key"] in info or self.keys["level_lo_key"] in info
        self.has_current_level[index] = has_level
        self.current_level_hi[index] = self._int_info(info, self.keys["level_hi_key"], 0)
        self.current_level_lo[index] = self._int_info(info, self.keys["level_lo_key"], 0)
        self.completed_level_count[index] = 0
        self.current_level_completion_awarded[index] = False
        self.completed[index] = False
        self.episode_steps[index] = 0
        self.last_progress_step[index] = 0
        self.previous_event_values[index] = self._event_values(info)

    @staticmethod
    def _int_info(info: Mapping[str, Any], key: str, default: int = 0) -> int:
        value = info.get(key, default)
        if isinstance(value, np.generic):
            return int(value.item())
        return int(value)

    def _event_values(self, info: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, rule in self.config.info_events.items():
            key_or_keys, _op = rule
            if isinstance(key_or_keys, str):
                if key_or_keys in info:
                    values[name] = info[key_or_keys]
                continue
            if all(key in info for key in key_or_keys):
                values[name] = tuple(info[key] for key in key_or_keys)
        return values

    @staticmethod
    def _event_rule_fired(previous: Any, current: Any, op: str) -> bool:
        if previous is None or current is None:
            return False
        if op == "change":
            return current != previous
        if op == "increase":
            return current > previous
        if op == "decrease":
            return current < previous
        return False

    def _annotate_info_events(self, index: int, info: Mapping[str, Any]) -> dict[str, Any]:
        payloads = event_payloads(info.get("done_on_info"))
        previous_values = self.previous_event_values[index]
        current_values = self._event_values(info)
        for name, rule in self.config.info_events.items():
            if name in payloads:
                continue
            if name not in previous_values or name not in current_values:
                continue
            key_or_keys, op = rule
            previous = previous_values[name]
            current = current_values[name]
            if not self._event_rule_fired(previous, current, op):
                continue
            payloads[name] = {
                "op": op,
                "keys": key_or_keys,
                "prev": previous,
                "next": current,
            }
        self.previous_event_values[index].update(current_values)
        return payloads

    def _shape_lane(
        self,
        index: int,
        native_reward: float,
        info: Mapping[str, Any],
        done: bool,
    ) -> tuple[float, dict[str, Any]]:
        config = self.config
        event_payloads = self._annotate_info_events(index, info)
        x_pos = (
            self._int_info(info, self.keys["xscroll_hi_key"], 0) * 256
            + self._int_info(info, self.keys["xscroll_lo_key"], 0)
        )
        lives = info.get(self.keys["lives_key"])
        level_hi = self._int_info(info, self.keys["level_hi_key"], 0)
        level_lo = self._int_info(info, self.keys["level_lo_key"], 0)
        life_loss_event = self.keys["life_loss_event"]
        level_change_event = self.keys["level_change_event"]

        died = life_loss_event in event_payloads or bool(info.get(life_loss_event, False))
        if self.has_prev_lives[index] and lives is not None and int(lives) < self.prev_lives[index]:
            died = True
        if lives is not None:
            self.prev_lives[index] = int(lives)
            self.has_prev_lives[index] = True

        if not self.has_current_level[index]:
            self.current_level_hi[index] = level_hi
            self.current_level_lo[index] = level_lo
            self.has_current_level[index] = True
        native_level_changed = level_change_event in event_payloads
        level_changed = (
            native_level_changed
            or level_hi != self.current_level_hi[index]
            or level_lo != self.current_level_lo[index]
        )
        level_completion_event = False
        if level_changed and not died:
            self.completed_level_base[index] += self.level_max_x_pos[index]
            self.completed_level_count[index] += 1
            level_completion_event = not self.current_level_completion_awarded[index]
            self.current_level_hi[index] = level_hi
            self.current_level_lo[index] = level_lo
            self.level_max_x_pos[index] = 0
            self.current_level_completion_awarded[index] = False

        self.level_x_pos[index] = x_pos
        self.level_max_x_pos[index] = max(self.level_max_x_pos[index], x_pos)
        global_x_pos = self.completed_level_base[index] + self.level_x_pos[index]
        global_max_x_pos = self.completed_level_base[index] + self.level_max_x_pos[index]
        progress_delta = max(0, int(global_max_x_pos - self.max_global_x_pos[index]))
        self.max_global_x_pos[index] = max(self.max_global_x_pos[index], global_max_x_pos)

        completion_event = bool(level_completion_event)
        if completion_event:
            self.completed[index] = True

        score = self._int_info(info, self.keys["score_key"], 0)
        score_delta = max(0, int(score - self.curr_score[index]))
        self.curr_score[index] = score
        if progress_delta > config.no_progress_min_delta:
            self.last_progress_step[index] = self.episode_steps[index]

        shaped_reward, reward_info = self._reward_components(
            native_reward=native_reward,
            progress_delta=progress_delta,
            score_delta=score_delta,
            completion_event=completion_event,
            died=died,
            done=done,
        )
        self.episode_steps[index] += 1
        progress_info = {
            "x_pos": int(global_x_pos),
            "max_x_pos": int(self.max_global_x_pos[index]),
            "level_x_pos": int(self.level_x_pos[index]),
            "level_max_x_pos": int(self.level_max_x_pos[index]),
            "completed_level_base": int(self.completed_level_base[index]),
            "global_x_pos": int(global_x_pos),
            "global_max_x_pos": int(self.max_global_x_pos[index]),
            "progress_delta": int(progress_delta),
            "level_id": f"{level_hi}-{level_lo}",
            self.keys["level_hi_key"]: int(level_hi),
            self.keys["level_lo_key"]: int(level_lo),
            "level_changed": bool(level_changed),
            "completed_level_count": int(self.completed_level_count[index]),
            "level_complete": bool(completion_event),
            "completion_event": bool(completion_event),
            "score_delta": int(score_delta),
            "died": bool(died),
            **reward_info,
        }
        if event_payloads:
            progress_info["info_events"] = event_payloads
        if died:
            progress_info["death_x_pos"] = int(self.max_global_x_pos[index])
            progress_info["death_level_x_pos"] = int(self.level_max_x_pos[index])
        return shaped_reward, progress_info

    def _progress_info(
        self,
        *,
        index: int,
        native_reward: float,
        progress_delta: int,
        score_delta: int,
        completion_event: bool,
        died: bool,
        done: bool,
        global_x_pos: int,
        level_hi: int,
        level_lo: int,
        level_changed: bool,
    ) -> dict[str, Any]:
        _reward, reward_info = self._reward_components(
            native_reward=native_reward,
            progress_delta=progress_delta,
            score_delta=score_delta,
            completion_event=completion_event,
            died=died,
            done=done,
        )
        progress_info = {
            "x_pos": int(global_x_pos),
            "max_x_pos": int(self.max_global_x_pos[index]),
            "level_x_pos": int(self.level_x_pos[index]),
            "level_max_x_pos": int(self.level_max_x_pos[index]),
            "completed_level_base": int(self.completed_level_base[index]),
            "global_x_pos": int(global_x_pos),
            "global_max_x_pos": int(self.max_global_x_pos[index]),
            "progress_delta": int(progress_delta),
            "level_id": f"{level_hi}-{level_lo}",
            self.keys["level_hi_key"]: int(level_hi),
            self.keys["level_lo_key"]: int(level_lo),
            "level_changed": bool(level_changed),
            "completed_level_count": int(self.completed_level_count[index]),
            "level_complete": bool(completion_event),
            "completion_event": bool(completion_event),
            "score_delta": int(score_delta),
            "died": bool(died),
            **reward_info,
        }
        if died:
            progress_info["death_x_pos"] = int(self.max_global_x_pos[index])
            progress_info["death_level_x_pos"] = int(self.level_max_x_pos[index])
        return progress_info

    def _reward_values_vector(
        self,
        native_rewards: np.ndarray,
        progress_delta: np.ndarray,
        score_delta: np.ndarray,
        completion_event: np.ndarray,
        died: np.ndarray,
    ) -> np.ndarray:
        config = self.config
        progress_reward = np.minimum(progress_delta, float(config.progress_reward_cap))
        native_component = native_rewards if config.use_retro_reward else np.zeros_like(native_rewards)
        if config.reward_mode == "native":
            return native_rewards.astype(np.float32)
        if config.reward_mode == "bounded":
            raw = progress_reward.copy()
            raw[completion_event] = float(config.terminal_reward)
            raw[died] = -float(config.terminal_reward)
            divisor = float(config.reward_scale or 1.0)
            return (raw / divisor - float(config.time_penalty)).astype(np.float32)
        if config.reward_mode == "baseline":
            raw = native_rewards + score_delta / 40.0
            raw[completion_event] += float(config.terminal_reward)
            death_mask = died
            raw[death_mask] -= float(config.terminal_reward)
            divisor = float(config.reward_scale or 1.0)
            return (raw / divisor - float(config.time_penalty)).astype(np.float32)
        if config.reward_mode == "score":
            progress_component = (
                progress_reward if config.score_progress_clipped else progress_delta
            )
            shaped = (
                native_component
                + float(config.progress_reward_scale) * progress_component
                + 0.01 * score_delta
            )
            shaped[completion_event] += float(config.completion_reward)
            shaped[died] -= float(config.death_penalty)
            return (shaped - float(config.time_penalty)).astype(np.float32)
        shaped = native_component + float(config.progress_reward_scale) * progress_delta
        shaped[completion_event] += float(config.completion_reward)
        shaped[died] -= float(config.death_penalty)
        return (shaped - float(config.time_penalty)).astype(np.float32)

    def _reward_components(
        self,
        *,
        native_reward: float,
        progress_delta: int,
        score_delta: int,
        completion_event: bool,
        died: bool,
        done: bool,
    ) -> tuple[float, dict[str, Any]]:
        config = self.config
        progress_reward = min(float(progress_delta), config.progress_reward_cap)
        native_reward_component = float(native_reward) if config.use_retro_reward else 0.0
        progress_component = float(progress_reward)
        progress_reward_component = 0.0
        score_reward_component = 0.0
        completion_reward_component = 0.0
        death_penalty_component = 0.0
        if config.reward_mode == "native":
            raw_reward = float(native_reward)
            shaped_reward = raw_reward
            native_reward_component = float(native_reward)
        elif config.reward_mode == "bounded":
            raw_reward = progress_reward
            if completion_event:
                raw_reward = config.terminal_reward
            if died:
                raw_reward = -config.terminal_reward
            reward_divisor = config.reward_scale if config.reward_scale else 1.0
            shaped_reward = raw_reward / reward_divisor
            if completion_event:
                completion_reward_component = config.terminal_reward / reward_divisor
            elif died:
                death_penalty_component = -config.terminal_reward / reward_divisor
            else:
                progress_reward_component = float(progress_reward) / reward_divisor
        elif config.reward_mode == "baseline":
            raw_reward = float(native_reward) + float(score_delta) / 40.0
            if completion_event:
                raw_reward += config.terminal_reward
            elif died or done:
                raw_reward -= config.terminal_reward
            reward_divisor = config.reward_scale if config.reward_scale else 1.0
            shaped_reward = raw_reward / reward_divisor
            native_reward_component = float(native_reward) / reward_divisor
            score_reward_component = (float(score_delta) / 40.0) / reward_divisor
            if completion_event:
                completion_reward_component = config.terminal_reward / reward_divisor
            elif died or done:
                death_penalty_component = -config.terminal_reward / reward_divisor
        elif config.reward_mode == "score":
            progress_component = (
                progress_reward if config.score_progress_clipped else float(progress_delta)
            )
            progress_reward_component = config.progress_reward_scale * progress_component
            score_reward_component = 0.01 * float(score_delta)
            shaped_reward = (
                native_reward_component
                + progress_reward_component
                + score_reward_component
            )
            if completion_event:
                completion_reward_component = config.completion_reward
                shaped_reward += completion_reward_component
            if died:
                death_penalty_component = -config.death_penalty
                shaped_reward += death_penalty_component
            raw_reward = shaped_reward
        else:
            progress_component = float(progress_delta)
            progress_reward_component = config.progress_reward_scale * progress_component
            shaped_reward = native_reward_component + progress_reward_component
            if completion_event:
                completion_reward_component = config.completion_reward
                shaped_reward += completion_reward_component
            if died:
                death_penalty_component = -config.death_penalty
                shaped_reward += death_penalty_component
            raw_reward = shaped_reward
        shaped_reward -= config.time_penalty
        return float(shaped_reward), {
            "completion_bonus": config.completion_reward if completion_event else 0.0,
            "reward_mode": config.reward_mode,
            "progress_reward": float(progress_reward),
            "progress_component": float(progress_component),
            "native_reward_component": float(native_reward_component),
            "progress_reward_component": float(progress_reward_component),
            "score_progress_clipped": config.score_progress_clipped,
            "score_delta": int(score_delta),
            "score_reward_component": float(score_reward_component),
            "completion_reward_component": float(completion_reward_component),
            "death_penalty_component": float(death_penalty_component),
            "time_penalty_component": -config.time_penalty,
            "terminal_reward": (
                -config.terminal_reward
                if died
                else config.terminal_reward
                if completion_event
                else 0.0
            ),
            "raw_reward": float(raw_reward),
            "clipped_reward": float(raw_reward),
            "reward_scale": config.reward_scale,
            "time_penalty": config.time_penalty,
            "shaped_reward": float(shaped_reward),
        }

#!/usr/bin/env python3
"""Single-file PPO trainer for SuperMarioBros-NES Level1-1.

Requires: torch, supermariobrosnes-turbo.  No rlab or stable-baselines imports.
"""

import math
import os
import time
from collections import deque

import torch
from torch import nn
from torch.distributions import Categorical
from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv

GAME = "SuperMarioBros-Nes-v0"
STATE = "Level1-1"
N_ENVS = 16
ENV_THREADS = 4
TOTAL_STEPS = 5_000_000
N_STEPS = 512
BATCH = 512
EPOCHS = 10
GAMMA = 0.9
GAE_LAMBDA = 1.0
LR0, LR1, LR_DECAY_STEPS = 1.5e-4, 1.0e-4, 4_000_000
ENT0, ENT1, ENT_DECAY_STEPS = 0.01, 1.0e-4, 4_000_000
CLIP = 0.15
TARGET_KL = 0.16
VF_COEF = 1.0
ADAM_EPS = 1e-8
SEED = int(os.environ.get("SEED", "1"))
MODEL_PATH = os.environ.get("MODEL_PATH", "mario_level1_1_ppo.pt")

# NES button order: B, -, SELECT, START, UP, DOWN, LEFT, RIGHT, A
ACTIONS = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # noop
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
        [1, 0, 0, 0, 0, 0, 0, 1, 0],  # right+B
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right+A
        [1, 0, 0, 0, 0, 0, 0, 1, 1],  # right+A+B
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    ],
    dtype=torch.int8,
)
ACTIONS_NP = ACTIONS.numpy()
N_ACTIONS = len(ACTIONS)


class MarioTracker:
    def __init__(self):
        self.reset({})

    def reset(self, info):
        self.level_max_x = 0
        self.max_global_x = 0
        self.completed_base = 0
        self.curr_score = int(info.get("score", 0))
        self.prev_lives = int(info["lives"]) if "lives" in info else None
        self.level = self._level(info)
        self.completed_level = False

    @staticmethod
    def _level(info):
        return (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))

    @staticmethod
    def _events(info):
        ev = info.get("info_events") or info.get("done_on_info") or {}
        if isinstance(ev, dict):
            return ev
        if isinstance(ev, (list, tuple, set)):
            return {str(x): {} for x in ev}
        return {str(ev): {}} if ev else {}

    def shape(self, info):
        events = self._events(info)
        level = self._level(info)
        lives = info.get("lives")
        died = "life_loss" in events
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
        if lives is not None:
            self.prev_lives = int(lives)

        changed = "level_change" in events or (self.level is not None and level != self.level)
        complete = bool(changed and not died and not self.completed_level)
        if changed and not died:
            self.completed_base += self.level_max_x
            self.level_max_x = 0
            self.level = level
            self.completed_level = False
        if complete:
            self.completed_level = True

        x = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        self.level_max_x = max(self.level_max_x, x)
        global_x = self.completed_base + self.level_max_x
        progress_delta = max(0, global_x - self.max_global_x)
        self.max_global_x = max(self.max_global_x, global_x)
        score = int(info.get("score", 0))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score

        reward = float(progress_delta) + 0.01 * float(score_delta)
        if died:
            reward -= 25.0
        info["level_complete"] = complete
        info["died"] = died
        info["max_x_pos"] = int(self.max_global_x)
        info["progress_delta"] = int(progress_delta)
        info["shaped_reward"] = reward
        return reward


class NaturePPO(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.pi = nn.Linear(512, n_actions)
        self.v = nn.Linear(512, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, obs):
        z = self.cnn(obs.float() / 255.0)
        return self.pi(z), self.v(z).squeeze(-1)

    def act_value(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def schedule(start, end, step, horizon):
    p = min(max(step / horizon, 0.0), 1.0)
    return start + (end - start) * p


def make_env():
    env = SuperMarioBrosNesTurboVecEnv(
        GAME,
        state=STATE,
        num_envs=N_ENVS,
        num_threads=ENV_THREADS,
        render_mode="rgb_array",
        obs_resize=(84, 84),
        obs_crop=(32, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        maxpool_last_two=False,
        sticky_action_prob=0.0,
        obs_copy="safe_view",
        obs_layout="chw",
        done_on={"life_loss": None, "level_change": None},
    )
    env.seed(SEED)
    return env


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    env = make_env()
    trackers = [MarioTracker() for _ in range(N_ENVS)]
    obs = env.reset()
    for i, info in enumerate(getattr(env, "reset_infos", [{} for _ in range(N_ENVS)])):
        trackers[i].reset(info)

    model = NaturePPO(N_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR0, eps=ADAM_EPS)
    completes = deque(maxlen=100)
    steps = 0
    t0 = time.time()
    b_obs = torch.empty((N_STEPS, N_ENVS, 4, 84, 84), dtype=torch.uint8)
    b_act = torch.empty((N_STEPS, N_ENVS), dtype=torch.long)
    b_logp = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_rew = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_done = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_val = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    adv = torch.empty_like(b_rew)

    while steps < TOTAL_STEPS:
        lr = schedule(LR0, LR1, steps, LR_DECAY_STEPS)
        ent_coef = schedule(ENT0, ENT1, steps, ENT_DECAY_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        for t in range(N_STEPS):
            obs_t = torch.as_tensor(obs, device=device)
            with torch.inference_mode():
                act, logp, _, val = model.act_value(obs_t)
            next_obs, _native_rew, done, infos = env.step(ACTIONS_NP[act.cpu().numpy()])

            for i, info in enumerate(infos):
                b_rew[t, i] = trackers[i].shape(info)
                if bool(done[i]):
                    completes.append(1 if info.get("level_complete") else 0)
                    reset_info = info.get("reset_info", {}) if isinstance(info, dict) else {}
                    trackers[i].reset(reset_info if isinstance(reset_info, dict) else {})

            b_obs[t].copy_(torch.as_tensor(obs, dtype=torch.uint8))
            b_act[t].copy_(act.cpu())
            b_logp[t].copy_(logp.cpu())
            b_done[t].copy_(torch.as_tensor(done, dtype=torch.float32))
            b_val[t].copy_(val.cpu())
            obs = next_obs
            steps += N_ENVS

        with torch.inference_mode():
            _, last_val = model(torch.as_tensor(obs, device=device))
        last_val = last_val.cpu()
        last_gae = torch.zeros(N_ENVS)
        for t in reversed(range(N_STEPS)):
            next_nonterminal = 1.0 - b_done[t]
            next_value = last_val if t == N_STEPS - 1 else b_val[t + 1]
            delta = b_rew[t] + GAMMA * next_value * next_nonterminal - b_val[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + b_val

        flat_obs = b_obs.reshape(-1, 4, 84, 84).to(device)
        flat_act = b_act.flatten().to(device)
        flat_old_logp = b_logp.flatten().to(device)
        flat_adv = adv.flatten().to(device)
        flat_ret = ret.flatten().to(device)
        approx_kl = torch.tensor(0.0)

        for _ in range(EPOCHS):
            order = torch.randperm(flat_act.numel(), device=device)
            for start in range(0, order.numel(), BATCH):
                idx = order[start : start + BATCH]
                logits, value = model(flat_obs[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat_act[idx])
                entropy = dist.entropy().mean()
                old_logp = flat_old_logp[idx]
                ratio = (logp - old_logp).exp()
                a = flat_adv[idx]
                pg_loss = -torch.min(a * ratio, a * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
                v_loss = 0.5 * (flat_ret[idx] - value).pow(2).mean()
                loss = pg_loss + VF_COEF * v_loss - ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.inference_mode():
                    approx_kl = ((ratio - 1) - (logp - old_logp)).mean().cpu()
            if approx_kl > 1.5 * TARGET_KL:
                break

        rate = sum(completes) / len(completes) if completes else 0.0
        fps = int(steps / max(time.time() - t0, 1e-9))
        print(
            f"steps={steps} fps={fps} lr={lr:.2e} ent={ent_coef:.2e} "
            f"kl={float(approx_kl):.3f} clears={sum(completes)}/{len(completes)} rate={rate:.2%}",
            flush=True,
        )
        torch.save({"model": model.state_dict(), "steps": steps, "clear_rate_last_100": rate}, MODEL_PATH)
        if len(completes) == 100 and sum(completes) == 100:
            print(f"solved: 100/100 clears at {steps} steps; saved {MODEL_PATH}", flush=True)
            break
    env.close()


if __name__ == "__main__":
    main()

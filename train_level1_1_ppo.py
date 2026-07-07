#!/usr/bin/env python3
"""Single-file PPO trainer for SuperMarioBros-NES Level1-1.

This file is intentionally written as a learning script.  It avoids rlab and
stable-baselines so the whole PPO loop is visible in one place:

1. Run Mario in many copies of the emulator at the same time.
2. Ask the policy network which action to take in each copy.
3. Save one rollout of observations, actions, rewards, dones, and value guesses.
4. Turn those rewards into "advantages": how much better an action was than the
   critic expected.
5. Update the policy with PPO's clipped objective, update the critic with value
   regression, and repeat.

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

# -----------------------------
# Hardcoded experiment settings
# -----------------------------
#
# PPO has lots of knobs.  These mirror the reliable rlab Level1-1 recipe.  The
# point of hardcoding them is pedagogical: when you read the training loop below,
# every number that matters is visible near the top of the file.

# Stable Retro game and state names.  A "state" is the starting save-state.
GAME = "SuperMarioBros-Nes-v0"
STATE = "Level1-1"

# We train 16 Mario emulator lanes in parallel.  This gives PPO a batch of
# experience faster than running one Mario at a time.
N_ENVS = 16
ENV_THREADS = 4

# Stop no later than 5M environment steps.  Each step here means one action sent
# to each active emulator lane after native frame-skip.
TOTAL_STEPS = 5_000_000

# PPO collects N_STEPS per env, so one rollout/update contains:
# 512 * 16 = 8192 transitions.
N_STEPS = 512

# Minibatch size for gradient descent over that rollout.  8192 / 512 = 16
# minibatches per epoch.
BATCH = 512

# PPO reuses the same rollout for multiple optimization passes.  This is sample
# efficient, but too many epochs can overfit to stale data.
EPOCHS = 10

# Discount factor.  A reward one step in the future is worth GAMMA times as much
# as a reward now.  Lower gamma makes Mario care more about near-term progress.
GAMMA = 0.9

# GAE smoothing.  1.0 means use the full discounted multi-step error; lower
# values trade some bias for lower variance.
GAE_LAMBDA = 1.0

# Learning rate starts a little higher and decays as the policy becomes useful.
LR0, LR1, LR_DECAY_STEPS = 1.5e-4, 1.0e-4, 4_000_000

# Entropy bonus starts high to encourage exploration, then decays so the policy
# can become more deterministic once it discovers winning behavior.
ENT0, ENT1, ENT_DECAY_STEPS = 0.01, 1.0e-4, 4_000_000

# PPO clipping limit.  0.15 means an update cannot easily make an action more
# than 15 percent more or less likely than it was during rollout collection.
CLIP = 0.15

# If the new policy drifts too far from the rollout policy, stop the PPO epochs
# early.  This is another guardrail against destructive policy jumps.
TARGET_KL = 0.16

# Weight on the critic/value loss.  The critic learns "how good is this state?"
# and is used to compute advantages.
VF_COEF = 1.0
ADAM_EPS = 1e-8

# These two can be overridden without editing the file:
#   SEED=2 MODEL_PATH=run2.pt python train_level1_1_ppo.py
SEED = int(os.environ.get("SEED", "1"))
MODEL_PATH = os.environ.get("MODEL_PATH", "mario_level1_1_ppo.pt")

# The environment expects a 9-button NES mask, but PPO is easier with a small
# discrete action space.  ACTIONS maps action index -> actual NES buttons.
#
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
    """Convert raw emulator info into learning rewards and success signals.

    The native environment tells us low-level game RAM values such as x-scroll,
    score, lives, and level id.  PPO only sees scalar rewards, so this tracker
    implements reward shaping:

    - reward rightward progress,
    - reward score increases a little,
    - penalize death,
    - mark a level change as completion.

    One tracker is kept per emulator lane because each lane is in a different
    episode and needs its own previous x-position, score, lives, and level.
    """

    def __init__(self):
        self.reset({})

    def reset(self, info):
        # Highest x-position reached in the current level attempt.
        self.level_max_x = 0
        # Highest global x-position reached across any completed level segments.
        self.max_global_x = 0
        # If a sequence has multiple levels, completed_base would carry distance
        # from earlier levels.  For Level1-1 it usually stays 0, but keeping it
        # makes "level changed" accounting explicit.
        self.completed_base = 0
        # Used to reward only score increases, not the absolute score number.
        self.curr_score = int(info.get("score", 0))
        # Used to detect death from lives decreasing, even if the event payload
        # does not explicitly say "life_loss".
        self.prev_lives = int(info["lives"]) if "lives" in info else None
        # Current level id as raw RAM values.  Level1-1 is usually (0, 0).
        self.level = self._level(info)
        # Prevents double-counting one level-clear event.
        self.completed_level = False

    @staticmethod
    def _level(info):
        # The game stores world/stage as two RAM variables.
        return (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))

    @staticmethod
    def _events(info):
        # supermariobrosnes-turbo reports why an episode ended in done_on_info.
        # This helper accepts a few possible shapes so the rest of the code can
        # simply ask "is 'life_loss' present?".
        ev = info.get("info_events") or info.get("done_on_info") or {}
        if isinstance(ev, dict):
            return ev
        if isinstance(ev, (list, tuple, set)):
            return {str(x): {} for x in ev}
        return {str(ev): {}} if ev else {}

    def shape(self, info):
        # "Events" are important because this env is configured to end episodes
        # on either life loss or level change.
        events = self._events(info)
        level = self._level(info)
        lives = info.get("lives")

        # Death detection.  Prefer the explicit event, but also detect lives
        # decreasing because it is a robust backup.
        died = "life_loss" in events
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
        if lives is not None:
            self.prev_lives = int(lives)

        # A level change without death means Mario cleared the level.
        changed = "level_change" in events or (self.level is not None and level != self.level)
        complete = bool(changed and not died and not self.completed_level)
        if changed and not died:
            self.completed_base += self.level_max_x
            self.level_max_x = 0
            self.level = level
            self.completed_level = False
        if complete:
            self.completed_level = True

        # xscrollHi/xscrollLo are two bytes of horizontal position.  Combining
        # them gives one monotonically increasing position within the level.
        x = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        self.level_max_x = max(self.level_max_x, x)
        global_x = self.completed_base + self.level_max_x

        # Reward only new progress.  If Mario moves left, repeats an old area,
        # or jitters in place, progress_delta is 0.
        progress_delta = max(0, global_x - self.max_global_x)
        self.max_global_x = max(self.max_global_x, global_x)

        # Score can reward useful events like stomping enemies or collecting
        # items.  It is scaled down so x-progress remains the main signal.
        score = int(info.get("score", 0))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score

        # This is the reward PPO actually optimizes.
        reward = float(progress_delta) + 0.01 * float(score_delta)
        if died:
            reward -= 25.0

        # Store human-readable diagnostics back into info.  The training loop
        # uses level_complete for the 100/100 early stop.
        info["level_complete"] = complete
        info["died"] = died
        info["max_x_pos"] = int(self.max_global_x)
        info["progress_delta"] = int(progress_delta)
        info["shaped_reward"] = reward
        return reward


class NaturePPO(nn.Module):
    """One neural network with two heads: policy and value.

    - The CNN trunk reads the 4 stacked 84x84 grayscale frames.
    - The policy head outputs logits for the 7 discrete Mario actions.
    - The value head estimates expected future reward from the current state.

    PPO is an actor-critic method: the policy is the actor, the value function is
    the critic.  They share visual features but have separate final heads.
    """

    def __init__(self, n_actions):
        super().__init__()
        # "NatureCNN" architecture from Atari RL: three conv layers, then a
        # 512-unit feature vector.  Input is channel-first: (4, 84, 84).
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
        # Orthogonal initialization is the SB3-style default for PPO policies.
        # Small policy-head scale starts the action distribution near uniform,
        # which helps early exploration.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, obs):
        # Observations arrive as uint8 pixels in [0, 255].  Neural nets train on
        # floats, so normalize to roughly [0, 1].
        z = self.cnn(obs.float() / 255.0)
        return self.pi(z), self.v(z).squeeze(-1)

    def act_value(self, obs):
        # Categorical(logits=...) represents a probability distribution over the
        # 7 discrete actions.  Sampling makes rollout behavior stochastic.
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        # We save log_prob(action) because PPO later asks:
        # "How much more/less likely is this same action under the updated
        # policy compared with the policy that collected it?"
        return action, dist.log_prob(action), dist.entropy(), value


def schedule(start, end, step, horizon):
    """Linear decay from start to end over horizon environment steps."""

    p = min(max(step / horizon, 0.0), 1.0)
    return start + (end - start) * p


def make_env():
    """Create the vectorized Mario environment.

    Important preprocessing choices:
    - obs_resize=(84, 84): small image for fast CNN training.
    - obs_crop=(32, 0, 0, 0) with mask mode: hide the HUD but keep geometry.
    - frame_skip=4: one chosen action is repeated for 4 emulator frames.
    - frame_stack=4: policy sees short-term motion, not just one still image.
    - done_on life_loss and level_change: episode ends on death or clear.
    """

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
    # Torch seed makes network init and action sampling reproducible enough for
    # learning/debugging.  GPU kernels may still have nondeterminism.
    torch.manual_seed(SEED)

    # Keep PyTorch CPU thread use low so it does not fight the emulator threads.
    torch.set_num_threads(1)

    # Prefer GPU.  On Apple Silicon, MPS can run this, but RTX/CUDA is the target
    # for full-speed training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    env = make_env()

    # Each env lane needs its own progress/reward tracker.
    trackers = [MarioTracker() for _ in range(N_ENVS)]
    obs = env.reset()
    for i, info in enumerate(getattr(env, "reset_infos", [{} for _ in range(N_ENVS)])):
        trackers[i].reset(info)

    # Create actor-critic network and Adam optimizer.
    model = NaturePPO(N_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR0, eps=ADAM_EPS)

    # Rolling 100 completed episodes.  When all 100 are clears, we stop.
    completes = deque(maxlen=100)
    steps = 0
    t0 = time.time()

    # Rollout buffers.  PPO is "on-policy": collect a fresh rollout, update from
    # it, then throw it away and collect a new one.  Buffers are preallocated to
    # avoid reallocating big tensors every update.
    b_obs = torch.empty((N_STEPS, N_ENVS, 4, 84, 84), dtype=torch.uint8)
    b_act = torch.empty((N_STEPS, N_ENVS), dtype=torch.long)
    b_logp = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_rew = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_done = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    b_val = torch.empty((N_STEPS, N_ENVS), dtype=torch.float32)
    adv = torch.empty_like(b_rew)

    while steps < TOTAL_STEPS:
        # Decay learning rate and entropy coefficient according to the recipe.
        lr = schedule(LR0, LR1, steps, LR_DECAY_STEPS)
        ent_coef = schedule(ENT0, ENT1, steps, ENT_DECAY_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        # ------------------
        # 1. Collect rollout
        # ------------------
        #
        # During rollout, gradients are disabled.  We are only using the current
        # policy to choose actions and record the data needed for PPO.
        for t in range(N_STEPS):
            obs_t = torch.as_tensor(obs, device=device)
            with torch.inference_mode():
                act, logp, _, val = model.act_value(obs_t)

            # Convert discrete action indices to NES button masks and step all
            # 16 env lanes together.
            next_obs, _native_rew, done, infos = env.step(ACTIONS_NP[act.cpu().numpy()])

            for i, info in enumerate(infos):
                # Replace native reward with our shaped Mario learning signal.
                b_rew[t, i] = trackers[i].shape(info)
                if bool(done[i]):
                    # done means either death or level clear.  Track clears only.
                    completes.append(1 if info.get("level_complete") else 0)

                    # Native vector env auto-resets done lanes.  reset_info is
                    # the first info dict of the new episode, so reset our
                    # Python-side tracker to match the lane's new episode.
                    reset_info = info.get("reset_info", {}) if isinstance(info, dict) else {}
                    trackers[i].reset(reset_info if isinstance(reset_info, dict) else {})

            # Save exactly what PPO needs:
            # - observation before the action,
            # - action sampled,
            # - log-prob under the old policy,
            # - shaped reward,
            # - done flag,
            # - critic value estimate before seeing the reward.
            b_obs[t].copy_(torch.as_tensor(obs, dtype=torch.uint8))
            b_act[t].copy_(act.cpu())
            b_logp[t].copy_(logp.cpu())
            b_done[t].copy_(torch.as_tensor(done, dtype=torch.float32))
            b_val[t].copy_(val.cpu())
            obs = next_obs
            steps += N_ENVS

        # ------------------------------------
        # 2. Compute advantages with GAE-Lambda
        # ------------------------------------
        #
        # The value head predicts future reward.  Advantage answers:
        #   "Was the action better or worse than the critic expected?"
        #
        # Positive advantage -> make action more likely.
        # Negative advantage -> make action less likely.
        #
        # GAE computes this from temporal-difference errors:
        #   delta_t = reward_t + gamma * V(next_state) - V(state_t)
        #
        # and then smooths deltas backward through time.
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

        # Flatten rollout from [time, env] into one big training table.
        flat_obs = b_obs.reshape(-1, 4, 84, 84).to(device)
        flat_act = b_act.flatten().to(device)
        flat_old_logp = b_logp.flatten().to(device)
        flat_adv = adv.flatten().to(device)
        flat_ret = ret.flatten().to(device)
        approx_kl = torch.tensor(0.0)

        # -----------------------------
        # 3. PPO policy/value update
        # -----------------------------
        #
        # The key PPO idea is conservative policy improvement.  Instead of
        # freely maximizing advantage, PPO compares new action probability to
        # old action probability:
        #
        #   ratio = exp(new_log_prob - old_log_prob)
        #
        # If ratio is 1.10, the action is now 10 percent more likely.
        # If ratio is 0.80, the action is now 20 percent less likely.
        #
        # PPO clips this ratio to [1 - CLIP, 1 + CLIP] so a single update cannot
        # move the policy too far from the behavior that collected the rollout.
        for _ in range(EPOCHS):
            # Shuffle indices so each minibatch sees a different mix of times
            # and env lanes.
            order = torch.randperm(flat_act.numel(), device=device)
            for start in range(0, order.numel(), BATCH):
                idx = order[start : start + BATCH]
                logits, value = model(flat_obs[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat_act[idx])
                entropy = dist.entropy().mean()
                old_logp = flat_old_logp[idx]

                # Probability ratio between new policy and rollout policy.
                ratio = (logp - old_logp).exp()
                a = flat_adv[idx]

                # Actor loss.  The min() implements PPO clipping:
                # use the clipped objective whenever the unclipped objective
                # would reward a too-large policy move.
                pg_loss = -torch.min(a * ratio, a * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()

                # Critic loss.  The value head learns to predict rollout returns.
                v_loss = 0.5 * (flat_ret[idx] - value).pow(2).mean()

                # Entropy bonus rewards randomness.  Early in training this helps
                # Mario keep trying varied jump/run timings.
                loss = pg_loss + VF_COEF * v_loss - ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping prevents rare huge gradients from exploding
                # the update.
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

                with torch.inference_mode():
                    # Approximate KL is a drift meter: how far did the updated
                    # policy move away from the rollout policy?
                    approx_kl = ((ratio - 1) - (logp - old_logp)).mean().cpu()

            # If KL is too high, stop reusing this rollout and collect fresh
            # experience.  This is one of the recipe's stability tricks.
            if approx_kl > 1.5 * TARGET_KL:
                break

        # -----------------------------
        # 4. Logging, saving, early stop
        # -----------------------------
        rate = sum(completes) / len(completes) if completes else 0.0
        fps = int(steps / max(time.time() - t0, 1e-9))
        print(
            f"steps={steps} fps={fps} lr={lr:.2e} ent={ent_coef:.2e} "
            f"kl={float(approx_kl):.3f} clears={sum(completes)}/{len(completes)} rate={rate:.2%}",
            flush=True,
        )

        # Save a plain torch checkpoint.  Loading it later means recreating the
        # same NaturePPO class and calling load_state_dict().
        torch.save({"model": model.state_dict(), "steps": steps, "clear_rate_last_100": rate}, MODEL_PATH)

        # Match the practical recipe gate: latest 100 terminal episodes all
        # cleared Level1-1.
        if len(completes) == 100 and sum(completes) == 100:
            print(f"solved: 100/100 clears at {steps} steps; saved {MODEL_PATH}", flush=True)
            break

    env.close()


if __name__ == "__main__":
    main()

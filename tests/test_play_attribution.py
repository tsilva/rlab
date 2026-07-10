from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlab.play import render_obs_stack
from rlab.play_attribution import (
    ActionLogProbForward,
    PolicyActionAttributor,
    actor_image_feature_extractor,
    find_last_conv2d,
)


class TinyImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.float() / 255.0)


class TinyDictExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractors = nn.ModuleDict({"image": TinyImageExtractor(), "task": nn.Identity()})


class TinyPolicy(nn.Module):
    def __init__(self, *, dict_obs: bool = False, preserve_obs_dtype: bool = False):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.pi_features_extractor = TinyDictExtractor() if dict_obs else TinyImageExtractor()
        self.action_net = nn.Linear(2, 2)
        self.value_net = nn.Linear(2, 1)
        self.dict_obs = dict_obs
        self.preserve_obs_dtype = preserve_obs_dtype

    def obs_to_tensor(self, observation):
        dtype = None if self.preserve_obs_dtype else torch.float32
        if isinstance(observation, dict):
            return {
                key: torch.as_tensor(value, dtype=dtype) for key, value in observation.items()
            }, True
        return torch.as_tensor(observation, dtype=dtype), True

    def image_features(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            return self.pi_features_extractor.extractors["image"](obs["image"])
        return self.pi_features_extractor(obs)

    def evaluate_actions(self, obs, actions):
        features = self.image_features(obs)
        logits = self.action_net(features)
        log_probs = torch.log_softmax(logits, dim=1)
        selected = actions.reshape(-1, 1).long()
        log_prob = log_probs.gather(1, selected).squeeze(1)
        values = self.value_net(features)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        return values, log_prob, entropy


def test_last_conv_selection_uses_final_conv_layer() -> None:
    extractor = TinyImageExtractor()

    assert find_last_conv2d(extractor) is extractor.net[2]


def test_actor_image_feature_extractor_handles_dict_image_branch() -> None:
    policy = TinyPolicy(dict_obs=True)

    assert actor_image_feature_extractor(policy) is policy.pi_features_extractor.extractors["image"]


def test_action_log_prob_forward_returns_gradient_scalar_and_preserves_task() -> None:
    policy = TinyPolicy(dict_obs=True)
    obs = {
        "image": np.ones((1, 4, 84, 84), dtype=np.float32),
        "task": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    forward = ActionLogProbForward(policy, obs, np.array([1]))
    image = forward.image_tensor.detach().requires_grad_(True)

    output = forward(image)
    output.sum().backward()

    assert output.shape == (1,)
    assert output.requires_grad
    assert image.grad is not None
    assert torch.equal(forward.fixed_obs["task"], torch.as_tensor(obs["task"]))


def test_action_log_prob_forward_converts_uint8_image_to_float_for_gradients() -> None:
    policy = TinyPolicy(preserve_obs_dtype=True)
    obs = np.ones((1, 4, 84, 84), dtype=np.uint8)

    forward = ActionLogProbForward(policy, obs, np.array([1]))
    image = forward.image_tensor.detach().requires_grad_(True)
    output = forward(image)
    output.sum().backward()

    assert forward.image_tensor.dtype == torch.float32
    assert image.grad is not None


def test_gradcam_returns_normalized_spatial_heatmap() -> None:
    policy = TinyPolicy()
    model = SimpleNamespace(policy=policy)
    attributor = PolicyActionAttributor(model)
    obs = np.random.default_rng(3).integers(0, 255, size=(1, 4, 84, 84), dtype=np.uint8)

    heatmap = attributor.attribute("gradcam", obs, np.array([1]))

    assert heatmap.shape == (84, 84)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_occlusion_returns_normalized_spatial_heatmap() -> None:
    policy = TinyPolicy()
    model = SimpleNamespace(policy=policy)
    attributor = PolicyActionAttributor(model, occlusion_window=28, occlusion_stride=28)
    obs = np.random.default_rng(4).integers(0, 255, size=(1, 4, 84, 84), dtype=np.uint8)

    heatmap = attributor.attribute("occlusion", obs, np.array([0]))

    assert heatmap.shape == (84, 84)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_render_obs_stack_accepts_optional_heatmap_without_resizing_layout() -> None:
    frames = [np.full((84, 84, 1), value, dtype=np.uint8) for value in (10, 50, 90, 130)]
    plain = render_obs_stack(frames, scale=2)
    heatmap = np.zeros((84, 84), dtype=np.float32)
    heatmap[20:40, 30:50] = 1.0

    overlay = render_obs_stack(frames, scale=2, heatmap=heatmap, heatmap_opacity=0.5)

    assert plain.shape == overlay.shape == (168, 672, 3)
    assert plain.dtype == overlay.dtype == np.uint8
    assert not np.array_equal(plain, overlay)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AttributionMode = Literal["gradcam", "occlusion"]


class AttributionError(RuntimeError):
    """Raised when a loaded policy cannot support visual attribution."""


def find_last_conv2d(module: nn.Module) -> nn.Conv2d:
    last_conv: nn.Conv2d | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise AttributionError("policy image feature extractor has no Conv2d layer for Grad-CAM")
    return last_conv


def actor_image_feature_extractor(policy: Any) -> nn.Module:
    extractor = getattr(policy, "pi_features_extractor", None)
    if extractor is None:
        extractor = getattr(policy, "features_extractor", None)
    if extractor is None:
        raise AttributionError("policy has no actor feature extractor")

    extractors = getattr(extractor, "extractors", None)
    if isinstance(extractors, Mapping) or isinstance(extractors, nn.ModuleDict):
        image_extractor = extractors["image"] if "image" in extractors else None
        if isinstance(image_extractor, nn.Module):
            return image_extractor
        raise AttributionError("task-conditioned policy feature extractor has no 'image' branch")
    return extractor


def _action_tensor(
    action: Any, *, device: torch.device, batch_size: int, action_shape: tuple
) -> torch.Tensor:
    arr = np.asarray(action)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    tensor = torch.as_tensor(arr, device=device)
    if np.issubdtype(arr.dtype, np.integer):
        tensor = tensor.long()
    expected_shape = (batch_size, *action_shape)
    if tensor.shape != expected_shape:
        try:
            tensor = tensor.reshape(expected_shape)
        except RuntimeError as exc:
            raise AttributionError(
                f"selected action shape {tuple(tensor.shape)} cannot be reshaped to {expected_shape}"
            ) from exc
    return tensor


class ActionLogProbForward(nn.Module):
    def __init__(self, policy: Any, model_obs: np.ndarray | dict[str, np.ndarray], action: Any):
        super().__init__()
        self.policy = policy
        obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
        self.image_key: str | None = None
        if isinstance(obs_tensor, Mapping):
            if "image" not in obs_tensor:
                raise AttributionError("dict observation is missing required 'image' key")
            image_tensor = obs_tensor["image"]
            fixed: dict[str, torch.Tensor] = {}
            for key, value in obs_tensor.items():
                if key == "image":
                    continue
                fixed[key] = value.detach()
            self.fixed_obs = fixed
            self.image_key = "image"
        elif isinstance(obs_tensor, torch.Tensor):
            image_tensor = obs_tensor
            self.fixed_obs = None
        else:
            raise AttributionError(
                f"unsupported policy observation tensor type {type(obs_tensor)!r}"
            )

        if image_tensor.ndim != 4:
            raise AttributionError(
                f"expected batched image observation tensor with 4 dims, got {tuple(image_tensor.shape)}"
            )
        self.image_tensor = image_tensor.detach().float()
        action_shape = tuple(getattr(getattr(policy, "action_space", None), "shape", ()))
        self.action_tensor = _action_tensor(
            action,
            device=self.image_tensor.device,
            batch_size=int(self.image_tensor.shape[0]),
            action_shape=action_shape,
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.fixed_obs is None:
            obs: torch.Tensor | dict[str, torch.Tensor] = image_tensor
        else:
            obs = dict(self.fixed_obs)
            obs["image"] = image_tensor
        _values, log_prob, _entropy = self.policy.evaluate_actions(obs, self.action_tensor)
        return log_prob.reshape(-1)


def _spatial_heatmap(attribution: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    if attribution.ndim == 4:
        attribution = attribution[0]
    if attribution.ndim == 3:
        positive = attribution.clamp_min(0.0)
        if torch.count_nonzero(positive).item() == 0:
            positive = attribution.abs()
        heatmap = positive.mean(dim=0)
    elif attribution.ndim == 2:
        heatmap = attribution.clamp_min(0.0)
        if torch.count_nonzero(heatmap).item() == 0:
            heatmap = attribution.abs()
    else:
        raise AttributionError(f"unsupported attribution tensor shape {tuple(attribution.shape)}")

    heatmap = heatmap.detach().float()[None, None, ...]
    if tuple(heatmap.shape[-2:]) != output_size:
        heatmap = F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)
    arr = heatmap[0, 0].cpu().numpy()
    arr = arr - float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= 0.0 or not np.isfinite(max_value):
        return np.zeros(output_size, dtype=np.float32)
    return (arr / max_value).astype(np.float32)


@dataclass
class PolicyActionAttributor:
    model: Any
    occlusion_window: int = 12
    occlusion_stride: int = 6

    def __post_init__(self) -> None:
        self.policy = self.model.policy
        self.image_extractor = actor_image_feature_extractor(self.policy)
        self.gradcam_layer = find_last_conv2d(self.image_extractor)

    def attribute(
        self,
        mode: AttributionMode,
        model_obs: np.ndarray | dict[str, np.ndarray],
        action: Any,
    ) -> np.ndarray:
        from captum.attr import LayerGradCam, Occlusion

        forward = ActionLogProbForward(self.policy, model_obs, action)
        image = forward.image_tensor.detach().requires_grad_(mode == "gradcam")
        output_size = tuple(int(dim) for dim in image.shape[-2:])

        was_training = bool(getattr(self.policy, "training", False))
        self.policy.eval()
        try:
            if mode == "gradcam":
                attr = LayerGradCam(forward, self.gradcam_layer).attribute(
                    image,
                    relu_attributions=True,
                )
            elif mode == "occlusion":
                channels = int(image.shape[1])
                height = int(image.shape[2])
                width = int(image.shape[3])
                window = min(self.occlusion_window, height, width)
                stride = min(self.occlusion_stride, window)
                attr = Occlusion(forward).attribute(
                    image,
                    sliding_window_shapes=(channels, window, window),
                    strides=(channels, stride, stride),
                    baselines=0,
                )
            else:
                raise AttributionError(f"unknown attribution mode {mode!r}")
        finally:
            if was_training:
                self.policy.train()
        return _spatial_heatmap(attr, output_size)

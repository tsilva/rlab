---
name: upload-checkpoint
description: Composite rlab release workflow for trained checkpoints. Use when the user asks to upload, publish, release, or promote a checkpoint/model to Hugging Face, especially RL checkpoints with gameplay preview videos. Coordinates Hugging Face model-card publishing with video preview requirements and YouTube upload of the same preview video.
---

# Upload Checkpoint

## Contract

Publish a trained checkpoint as a Hugging Face model repo with a model card and preview video, then upload the same preview video to YouTube.

This is a composite skill. When executing the workflow, also load and follow:

- `$model-card-author` at `/Users/tsilva/.codex/skills/model-card-author/SKILL.md`
- `$upload-youtube-video` at `.codex/skills/upload-youtube-video/SKILL.md`

Do not treat the Hugging Face and YouTube steps as alternatives. For visual RL checkpoints, both are part of the checkpoint upload unless the user explicitly opts out.

## Required Inputs

Resolve these from the user request, eval database, W&B artifact metadata, local artifacts, or generated summaries:

- checkpoint identity: run, seed, checkpoint step/timestep, artifact path, or local file
- target Hugging Face model repo name and owner
- environment/game, level/task, algorithm, source project, and training framework
- eval result: completion/win rate, eval count, max progress, reward mean, and eval profile when available
- representative preview episode video, or enough information to generate one

If a required fact is ambiguous and cannot be safely inferred from source artifacts, ask one concise question before publishing.

## Workflow

1. Gather source evidence.
   - Verify the checkpoint file or artifact exists.
   - Verify reported metrics against the eval database, W&B metadata, model metadata, or generated eval summaries.
   - Keep generated staging artifacts under ignored locations such as `runs/`.

2. Prepare the Hugging Face model repo with `$model-card-author`.
   - Build or update the model card using that skill's structure and writing rules.
   - For RL checkpoints with visual behavior, include a browser-safe `replay.mp4` in the model repo root.
   - Encode/verify `replay.mp4` as H.264/AVC, `yuv420p`, faststart, with valid duration and frames.
   - Let Hugging Face's reinforcement-learning widget render the root `replay.mp4` as the model page preview; do not also embed the same video in the README body unless the widget is unavailable.
   - Upload the checkpoint, README, metadata, and `replay.mp4` to Hugging Face.
   - Verify the live Hugging Face card and remote raw video after upload.

3. Upload the same preview video to YouTube with `$upload-youtube-video`.
   - Use the exact same representative video uploaded to Hugging Face unless it must be re-encoded for YouTube compatibility.
   - Default playlist: `rlab`.
   - Default privacy: `public`.
   - Use the title shape from `$upload-youtube-video`:
     `<env>, <level>, <algorithm>, <win-rate> win rate`
   - Use the description shape from `$upload-youtube-video`:

```text
PPO policy checkpoint completing <env> <level>, trained with `rlab`.

Model: https://huggingface.co/<owner>/<repo>
rlab: https://github.com/tsilva/rlab
```

4. Cross-link and verify.
   - Ensure the YouTube description links to the direct Hugging Face model URL.
   - If the model card should mention YouTube, update the card after the YouTube URL exists, then re-upload and verify.
   - Save upload results under the staging directory, for example `runs/hf_upload/<repo>/youtube_upload_result.json`.
   - Report the Hugging Face model URL, Hugging Face commit URL when available, YouTube URL, playlist URL when available, and exact local staging paths.

## Defaults

- Hugging Face preview filename: `replay.mp4`.
- YouTube playlist: `rlab`.
- YouTube privacy: `public`.
- YouTube links order: `Model:` first, then `rlab:`.
- Model-card preview video is required for visual/interactive checkpoints unless the user explicitly opts out.

## Safety

- Do not print or expose Hugging Face tokens, YouTube OAuth client secrets, OAuth refresh tokens, W&B credentials, or R2/S3 credentials.
- Do not overwrite generated videos or model-card assets silently if labels, metrics, or task names are wrong; fix or regenerate them first.
- Do not claim a win/completion rate unless it is backed by source evidence.
- Do not move detailed YouTube formatting rules into this skill. Keep those in `$upload-youtube-video`; this skill only composes the release workflow.
- Do not move detailed model-card writing or HF video-preview rules into this skill. Keep those in `$model-card-author`; this skill only requires that they are followed.

## Validation Checklist

Before final response, verify:

- model card uploaded and readable on Hugging Face
- checkpoint artifact present in the HF repo
- `replay.mp4` present in the HF repo root and browser-safe
- YouTube video uploaded from the same preview video
- YouTube title, description, playlist, and privacy match `$upload-youtube-video`
- all reported URLs are live or were returned by the relevant upload API/CLI

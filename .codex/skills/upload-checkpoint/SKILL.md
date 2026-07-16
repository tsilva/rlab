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
- materialized goal contract and current model metadata; rlab generates the Hugging Face repo id
- environment/game, level/task, algorithm, source project, and training framework
- eval result: completion/win rate, eval count, max progress, reward mean, and eval profile when available
- representative preview episode video, or enough information to generate one

If a required fact is ambiguous and cannot be safely inferred from source artifacts, ask one concise question before publishing.

## Workflow

1. Gather source evidence.
   - Verify the checkpoint file or artifact exists.
   - Verify reported metrics against the eval database, W&B metadata, model metadata, or generated eval summaries.
   - Keep generated staging artifacts under ignored locations such as `runs/`.

2. Resolve the deterministic publication identity.
   - Run the repository-owned `scripts/prepare_huggingface_release.py` with
     `--identity-only` before creating or moving any repository.
   - The only supported namespace is `tsilva`.
   - The generated name is
     `<game-family>_<goal>_<policy-variant>_<algorithm>`; do not accept a manual repo name.
   - Provider, run, seed, recipe, environment hash, and runtime versions belong in metadata,
     not in the repository name.

3. Prepare the generated model card and preview with `$model-card-author` as the quality standard.
   - Do not hand-author `README.md`. The repository-owned release helper renders it from the
     release manifest, model document, recipe, and verified evaluation evidence.
   - For RL checkpoints with visual behavior, include a browser-safe `replay.mp4` in the model repo root.
   - Encode/verify `replay.mp4` as H.264/AVC, `yuv420p`, faststart, with valid duration and frames.
   - Let Hugging Face's reinforcement-learning widget render the root `replay.mp4` as the model page preview; do not also embed the same video in the README body unless the widget is unavailable.

4. Upload the same preview video to YouTube with `$upload-youtube-video`.
   - Use the exact same representative video uploaded to Hugging Face unless it must be re-encoded for YouTube compatibility.
   - Defer title, description, tags, thumbnail, playlist, and privacy details to that skill so
     this composite workflow cannot drift from the YouTube publication contract.

5. Build and publish the immutable release bundle.
   - Supply the YouTube URL and stochastic evaluation evidence to
     `scripts/prepare_huggingface_release.py`; it fails closed on deterministic evaluation,
     invalid video, non-portable paths, inconsistent identity, or a nonstandard file set.
   - The bundle must contain exactly `.gitattributes`, `README.md`, `LICENSE`, `model.zip`,
     `model.json`, `recipe.json`, `release_manifest.json`, and `replay.mp4`.
   - Upload the bundle in one Hugging Face commit, then create the next sequential immutable
     `vN` tag. `main` is the latest promoted release.
   - Do not publish a new current release until stochastic evaluation and replay evidence exist.

6. Cross-link and verify.
   - Ensure the YouTube description links to the direct generated Hugging Face model URL.
   - The model card does not duplicate the YouTube video; the release manifest records its URL.
   - Run `scripts/audit_huggingface_release.py <repo-id> --revision <vN>` after publication. It
     verifies the live generated card, exact remote file set, manifest hashes, main/tag commit,
     replay encoding, and Collection membership through the Hugging Face API.
   - Save upload results under the staging directory, for example `runs/hf_upload/<repo>/youtube_upload_result.json`.
   - Report the Hugging Face model URL, Hugging Face commit URL when available, YouTube URL, playlist URL when available, and exact local staging paths.

## One-time Legacy Migration

- Generate the complete old-to-new mapping with
  `scripts/migrate_huggingface_legacy_repos.py` before
  changing the Hub. The legacy algorithm and crop behavior must be supplied explicitly from
  repository and source-history evidence.
- Review the collision-free mapping and retain its printed SHA-256. Applying it requires that
  exact digest, authenticated ownership, and empty destination names.
- Tag each historical source revision as `legacy-deterministic` before moving it. A move does
  not make that historical revision a schema-v1 release.
- Do not create `v1` until the checkpoint has new stochastic evaluation evidence, a validated
  representative replay, and the exact eight-file release bundle.
- Create or update the game-family Collection only after all repository moves succeed.
- For the already-moved Mario repositories, use
  `scripts/repair_huggingface_legacy_cards.py` to generate the canonical legacy cards and fill
  Collection membership. Review the dry-run digest before applying it; the repair changes only
  `README.md` and Collection membership and never creates a `v1` tag.

## Contract Ownership

- Repository code owns the namespace, naming schema, license, exact release file set, generated
  card structure, evidence validation, and Hub audit.
- `$upload-youtube-video` owns YouTube metadata and upload defaults.
- `$model-card-author` owns general model-card quality guidance.
- This skill owns orchestration: gather evidence, invoke those contracts in order, and report the
  verified publication result.

## Safety

- Do not print or expose Hugging Face tokens, YouTube OAuth client secrets, OAuth refresh tokens, W&B credentials, or R2/S3 credentials.
- Do not overwrite generated videos or model-card assets silently if labels, metrics, or task names are wrong; fix or regenerate them first.
- Do not accept manual repository names, silently truncate long names, publish an unknown game
  family, or infer an algorithm from a filename.
- Do not claim a win/completion rate unless it is backed by source evidence.
- Do not move detailed YouTube formatting rules into this skill. Keep those in `$upload-youtube-video`; this skill only composes the release workflow.
- Do not move detailed model-card writing or HF video-preview rules into this skill. Keep those in `$model-card-author`; this skill only requires that they are followed.

## Validation Checklist

Before final response, verify:

- model card uploaded and readable on Hugging Face
- generated repo id matches `model.json` and `release_manifest.json`
- remote repo contains exactly the standardized eight-file allowlist
- immutable `vN` tag points to the verified release commit
- checkpoint artifact present in the HF repo
- `replay.mp4` present in the HF repo root and browser-safe
- YouTube video uploaded from the same preview video
- YouTube title, description, playlist, and privacy match `$upload-youtube-video`
- all reported URLs are live or were returned by the relevant upload API/CLI

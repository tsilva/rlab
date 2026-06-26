---
name: upload-youtube-video
description: Upload or update YouTube model preview videos from rlab, especially RL gameplay previews associated with Hugging Face model cards. Use when the user asks to upload a local video to YouTube, update an existing video description/title/tags/privacy, add a video to a playlist, produce OAuth authorization flow instructions, or ensure model-card preview videos have clean clickable model links.
---

# Upload YouTube Video

## Contract

Publish a local preview video to the user's YouTube account, connect it to the associated model page, and leave enough metadata for future agents to update or verify it.

Prefer the repo uploader script at `scripts/upload_youtube_video.py`. It uses `.secret/youtube_client_secret.json` and stores the OAuth token at `.secret/youtube_token.json`; do not print secret or token contents.

## Before Uploading

1. Confirm the local video path, title, model page URL, and privacy status. Default privacy to `public` unless the user asks otherwise.
2. For model preview videos, prefer adding them to the `rlab` playlist unless the user asks for a different playlist.
3. Check the model page URL is direct and fully qualified. Prefer `https://huggingface.co/<owner>/<repo>` over redirect/short domains such as `https://hf.co/...`.
4. If the user needs OAuth again, run the uploader with `--no-browser` only when appropriate and relay the printed authorization URL. After the user says they authorized it, continue the waiting process; do not restart unless the callback failed.

## Description Format

Use this title shape for RL/model preview videos:

```text
<env>, <level>, <algorithm>, <win-rate> win rate
```

Example:

```text
SuperMarioBros-NES-v0, Level1-1, PPO, 100% win rate
```

Use this shape for RL/model preview videos:

```text
PPO policy checkpoint completing <env> <level>, trained with `rlab`.

Model: https://huggingface.co/<owner>/<repo>
rlab: https://github.com/tsilva/rlab
```

Rules:

- Keep titles compact and scannable: environment, level, algorithm, and headline outcome.
- Start with a concise human-readable description of what the video shows.
- Mention that the checkpoint was trained with `rlab` in the first sentence when true.
- Put the `Model:` link before the `rlab:` link.
- Keep the link labels short and plain: `rlab:` and `Model:`.
- Avoid shorteners and redirect domains for YouTube descriptions; they can be visually truncated or treated suspiciously by YouTube.
- Expect YouTube to visually ellipsize long links in collapsed views even when the actual link is correct and clickable.
- Include extra eval claims only when the user asks for them or when they are needed for the video description and are backed by the current model card, eval database, W&B artifact metadata, or generated summary files.

## Upload Command

Use the script from the repo root:

```bash
python3 scripts/upload_youtube_video.py <video.mp4> \
  --title "<env>, <level>, <algorithm>, <win-rate> win rate" \
  --human-description "PPO policy checkpoint completing <env> <level>, trained with \`rlab\`." \
  --model-page $'Model: https://huggingface.co/<owner>/<repo>\nrlab: https://github.com/tsilva/rlab' \
  --playlist-title rlab \
  --privacy-status public \
  --output runs/<artifact-dir>/youtube_upload_result.json
```

If the OAuth callback cannot open a browser or the user asks for the OAuth URL:

```bash
python3 scripts/upload_youtube_video.py <video.mp4> ... --no-browser
```

Relay the URL printed by the script and wait for the user to authorize.

## Updating an Existing Video

Use `--video-id` to update metadata without re-uploading:

```bash
python3 scripts/upload_youtube_video.py \
  --video-id <youtube-video-id> \
  --title "<env>, <level>, <algorithm>, <win-rate> win rate" \
  --human-description "PPO policy checkpoint completing <env> <level>, trained with \`rlab\`." \
  --model-page $'Model: https://huggingface.co/<owner>/<repo>\nrlab: https://github.com/tsilva/rlab' \
  --privacy-status public \
  --output runs/<artifact-dir>/youtube_upload_result.json \
  --no-browser
```

The update path preserves existing title and tags unless replacements are passed.

## Verification

After upload or update:

1. Read the JSON output and report the YouTube URL, playlist URL when present, and the final description shape.
2. Verify the model link is direct `https://huggingface.co/...` and appears on a `Model:` line.
3. Run `python3 -m py_compile scripts/upload_youtube_video.py` after changing the uploader script.
4. If the user reports link issues and the stored description is correct, distinguish visual truncation from actual clickability. For clickability issues, check account verification, Shorts vs normal video, direct URL formatting, redirect domains, and whether the issue is only in a collapsed/mobile view.

## Repo Rules

When YouTube upload policy or description conventions change, encode detailed rules here first. Keep `AGENTS.md` as a short trigger pointer to this skill, not the detailed source of truth.

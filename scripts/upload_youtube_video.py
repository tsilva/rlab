from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import secrets
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any


YOUTUBE_UPLOAD_SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
)
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
YOUTUBE_THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
DEFAULT_THUMBNAIL_TIME_SECONDS = 10.0


class OAuthRequestError(RuntimeError):
    def __init__(self, message: str, *, code: int, body: str) -> None:
        super().__init__(message)
        self.code = code
        self.body = body

    @property
    def is_invalid_grant(self) -> bool:
        return "invalid_grant" in self.body


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    server: "OAuthCallbackServer"

    def do_GET(self) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.server.auth_code = query.get("code", [None])[0]
        self.server.auth_error = query.get("error", [None])[0]
        self.server.auth_state = query.get("state", [None])[0]
        if self.server.auth_code:
            body = b"YouTube upload authorization complete. You can close this tab."
        else:
            body = b"YouTube upload authorization failed. Return to the terminal."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class OAuthCallbackServer(http.server.HTTPServer):
    auth_code: str | None = None
    auth_error: str | None = None
    auth_state: str | None = None


def load_client_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("installed") or payload.get("web")
    if not config:
        raise SystemExit(f"OAuth client secret must contain 'installed' or 'web': {path}")
    return config


def post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise OAuthRequestError(
            f"OAuth request failed: HTTP {exc.code}: {body}",
            code=exc.code,
            body=body,
        ) from exc


def request_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=UTF-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            return {} if not body.strip() else json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"YouTube API request failed: HTTP {exc.code}: {body}") from exc


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def authorize(client_config: dict[str, Any], token_path: Path, *, no_browser: bool) -> dict[str, Any]:
    token_uri = client_config["token_uri"]
    client_id = client_config["client_id"]
    client_secret = client_config.get("client_secret", "")
    state = secrets.token_urlsafe(24)

    server = OAuthCallbackServer(("127.0.0.1", 0), OAuthCallbackHandler)
    host, port = server.server_address
    redirect_uri = f"http://{host}:{port}/"
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(YOUTUBE_UPLOAD_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    print("Open this URL to authorize YouTube upload access:", flush=True)
    print(auth_url, flush=True)
    if not no_browser:
        webbrowser.open(auth_url)

    server.handle_request()
    if server.auth_error:
        raise SystemExit(f"OAuth authorization failed: {server.auth_error}")
    if not server.auth_code or server.auth_state != state:
        raise SystemExit("OAuth authorization failed: missing code or state mismatch")

    try:
        token = post_form(
            token_uri,
            {
                "code": server.auth_code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    except OAuthRequestError as exc:
        raise SystemExit(str(exc)) from exc
    token["client_id"] = client_id
    token["client_secret"] = client_secret
    token["token_uri"] = token_uri
    save_json(token_path, token)
    return token


def refresh_token(token: dict[str, Any]) -> dict[str, Any]:
    if "refresh_token" not in token:
        return token
    try:
        refreshed = post_form(
            token["token_uri"],
            {
                "client_id": token["client_id"],
                "client_secret": token.get("client_secret", ""),
                "refresh_token": token["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
    except OAuthRequestError as exc:
        if exc.is_invalid_grant:
            raise
        raise SystemExit(str(exc)) from exc
    token.update(refreshed)
    return token


def access_token(client_config: dict[str, Any], token_path: Path, *, no_browser: bool) -> str:
    if token_path.is_file():
        token = json.loads(token_path.read_text(encoding="utf-8"))
        try:
            token = refresh_token(token)
        except OAuthRequestError as exc:
            if not exc.is_invalid_grant:
                raise SystemExit(str(exc)) from exc
            print(
                "Stored YouTube OAuth token expired or was revoked; starting authorization flow.",
                flush=True,
            )
            token_path.unlink(missing_ok=True)
            token = authorize(client_config, token_path, no_browser=no_browser)
        save_json(token_path, token)
    else:
        token = authorize(client_config, token_path, no_browser=no_browser)
    value = token.get("access_token")
    if not value:
        raise SystemExit("OAuth token does not contain access_token")
    return str(value)


def start_resumable_upload(
    *,
    token: str,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    privacy_status: str,
) -> str:
    metadata = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    data = json.dumps(metadata).encode("utf-8")
    upload_url = f"{UPLOAD_URL}?{urllib.parse.urlencode({'uploadType': 'resumable', 'part': 'snippet,status'})}"
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    request = urllib.request.Request(
        upload_url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(data)),
            "X-Upload-Content-Length": str(video_path.stat().st_size),
            "X-Upload-Content-Type": mime_type,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            location = response.headers.get("Location")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Could not start YouTube upload: HTTP {exc.code}: {body}") from exc
    if not location:
        raise SystemExit("YouTube did not return a resumable upload URL")
    return location


def upload_video(upload_url: str, token: str, video_path: Path) -> dict[str, Any]:
    data = video_path.read_bytes()
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    request = urllib.request.Request(
        upload_url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": mime_type,
            "Content-Length": str(len(data)),
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"YouTube upload failed: HTTP {exc.code}: {body}") from exc


def video_duration_seconds(video_path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def thumbnail_seek_time(video_path: Path, requested_seconds: float) -> float:
    duration = video_duration_seconds(video_path)
    if duration is None or duration <= 0:
        return max(0.0, requested_seconds)
    return max(0.0, min(requested_seconds, max(0.0, duration - 0.25)))


def extract_thumbnail(
    *,
    video_path: Path,
    thumbnail_path: Path,
    seconds: float,
) -> Path:
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    seek_seconds = thumbnail_seek_time(video_path, seconds)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seek_seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(thumbnail_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required to generate YouTube thumbnails") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise SystemExit(f"Could not generate thumbnail from {video_path}: {stderr}") from exc
    if not thumbnail_path.is_file() or thumbnail_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg did not write thumbnail: {thumbnail_path}")
    return thumbnail_path


def upload_thumbnail(*, token: str, video_id: str, thumbnail_path: Path) -> dict[str, Any]:
    data = thumbnail_path.read_bytes()
    mime_type = mimetypes.guess_type(thumbnail_path.name)[0] or "image/jpeg"
    url = f"{YOUTUBE_THUMBNAIL_URL}?{urllib.parse.urlencode({'videoId': video_id})}"
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": mime_type,
            "Content-Length": str(len(data)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return {} if not body.strip() else json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Could not upload YouTube thumbnail: HTTP {exc.code}: {body}") from exc


def build_description(
    *,
    human_description: str,
    model_page: str,
    details: str,
) -> str:
    parts = [part.strip() for part in (human_description, model_page, details) if part.strip()]
    return "\n\n".join(parts)


def find_or_create_playlist(
    *,
    token: str,
    title: str,
    privacy_status: str,
) -> str:
    page_token = None
    while True:
        params = {"part": "snippet", "mine": "true", "maxResults": "50"}
        if page_token:
            params["pageToken"] = page_token
        payload = request_json(
            f"{YOUTUBE_API_URL}/playlists?{urllib.parse.urlencode(params)}",
            token=token,
        )
        for item in payload.get("items", []):
            if item.get("snippet", {}).get("title") == title:
                return str(item["id"])
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    created = request_json(
        f"{YOUTUBE_API_URL}/playlists?{urllib.parse.urlencode({'part': 'snippet,status'})}",
        token=token,
        method="POST",
        payload={
            "snippet": {
                "title": title,
                "description": "Reinforcement learning lab videos and model previews.",
            },
            "status": {"privacyStatus": privacy_status},
        },
    )
    return str(created["id"])


def add_video_to_playlist(*, token: str, playlist_id: str, video_id: str) -> dict[str, Any]:
    page_token = None
    while True:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": "50",
        }
        if page_token:
            params["pageToken"] = page_token
        payload = request_json(
            f"{YOUTUBE_API_URL}/playlistItems?{urllib.parse.urlencode(params)}",
            token=token,
        )
        for item in payload.get("items", []):
            if item.get("contentDetails", {}).get("videoId") == video_id:
                return {"already_present": True, "playlist_item_id": item.get("id")}
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    item = request_json(
        f"{YOUTUBE_API_URL}/playlistItems?{urllib.parse.urlencode({'part': 'snippet'})}",
        token=token,
        method="POST",
        payload={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            },
        },
    )
    return {"already_present": False, "playlist_item_id": item.get("id")}


def get_video_metadata(*, token: str, video_id: str) -> dict[str, Any]:
    params = {"part": "snippet,status", "id": video_id}
    payload = request_json(
        f"{YOUTUBE_API_URL}/videos?{urllib.parse.urlencode(params)}",
        token=token,
    )
    items = payload.get("items", [])
    if not items:
        raise SystemExit(f"YouTube video not found or not accessible: {video_id}")
    return dict(items[0])


def update_video_metadata(
    *,
    token: str,
    video_id: str,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    privacy_status: str,
) -> dict[str, Any]:
    existing = get_video_metadata(token=token, video_id=video_id)
    existing_status = existing.get("status", {})
    payload = {
        "id": video_id,
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
            "embeddable": existing_status.get("embeddable", True),
            "publicStatsViewable": existing_status.get("publicStatsViewable", True),
            **({"license": existing_status["license"]} if "license" in existing_status else {}),
        },
    }
    return request_json(
        f"{YOUTUBE_API_URL}/videos?{urllib.parse.urlencode({'part': 'snippet,status'})}",
        token=token,
        method="PUT",
        payload=payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a local video to YouTube.")
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument("--client-secret", type=Path, default=Path(".secret/youtube_client_secret.json"))
    parser.add_argument("--token", type=Path, default=Path(".secret/youtube_token.json"))
    parser.add_argument("--title")
    parser.add_argument("--video-id", help="Update metadata for this existing YouTube video instead of uploading.")
    parser.add_argument(
        "--human-description",
        help=(
            "First paragraph of the YouTube description. Use a plain-language summary "
            "of what the viewer is seeing."
        ),
    )
    parser.add_argument(
        "--model-page",
        help="Associated model page URL. Written after the human description on its own paragraph.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Additional details appended after the human description and model page.",
    )
    parser.add_argument("--tags", default="", help="Comma-separated tags.")
    parser.add_argument("--category-id", help="20 is Gaming.")
    parser.add_argument(
        "--playlist-title",
        help="Find or create this playlist and add the uploaded video to it.",
    )
    parser.add_argument("--privacy-status", choices=["private", "public", "unlisted"])
    parser.add_argument(
        "--thumbnail",
        type=Path,
        help="Custom thumbnail image to upload after the video is available.",
    )
    parser.add_argument(
        "--thumbnail-time",
        type=float,
        default=DEFAULT_THUMBNAIL_TIME_SECONDS,
        help=(
            "Seconds into the local video to capture for the default custom thumbnail. "
            "Defaults to 10 seconds."
        ),
    )
    parser.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="Do not generate or upload a custom thumbnail.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/youtube_upload_result.json"))
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.client_secret.is_file():
        raise SystemExit(f"client secret not found: {args.client_secret}")

    client_config = load_client_config(args.client_secret)
    token = access_token(client_config, args.token, no_browser=args.no_browser)
    description_provided = any(
        value is not None and value != ""
        for value in (args.human_description, args.model_page, args.description)
    )
    description = build_description(
        human_description=args.human_description or args.description,
        model_page=args.model_page or "",
        details="" if args.human_description else "",
    )
    if args.human_description:
        description = build_description(
            human_description=args.human_description,
            model_page=args.model_page or "",
            details=args.description,
        )
    if args.video_id:
        existing = get_video_metadata(token=token, video_id=args.video_id)
        existing_snippet = existing.get("snippet", {})
        existing_status = existing.get("status", {})
        tags = (
            [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            if args.tags
            else list(existing_snippet.get("tags", []))
        )
        result = update_video_metadata(
            token=token,
            video_id=args.video_id,
            title=args.title or existing_snippet.get("title", args.video_id),
            description=description if description_provided else existing_snippet.get("description", ""),
            tags=tags,
            category_id=args.category_id or existing_snippet.get("categoryId", "20"),
            privacy_status=args.privacy_status or existing_status.get("privacyStatus", "unlisted"),
        )
        result["youtube_url"] = f"https://www.youtube.com/watch?v={args.video_id}"
        if not args.no_thumbnail and (args.thumbnail or args.video):
            thumbnail_path = args.thumbnail
            if thumbnail_path is None:
                if not args.video or not args.video.is_file():
                    raise SystemExit("local video is required to generate thumbnail for --video-id")
                thumbnail_path = args.output.with_name(f"{args.output.stem}_thumbnail.jpg")
                thumbnail_path = extract_thumbnail(
                    video_path=args.video,
                    thumbnail_path=thumbnail_path,
                    seconds=float(args.thumbnail_time),
                )
            elif not args.thumbnail.is_file():
                raise SystemExit(f"thumbnail not found: {args.thumbnail}")
            result["thumbnail"] = {
                "path": str(thumbnail_path),
                "time_seconds": None if args.thumbnail else float(args.thumbnail_time),
                "upload": upload_thumbnail(
                    token=token,
                    video_id=args.video_id,
                    thumbnail_path=thumbnail_path,
                ),
            }
        save_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    if not args.video:
        raise SystemExit("video is required when --video-id is not provided")
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    if not args.title:
        raise SystemExit("--title is required when uploading a new video")
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    upload_url = start_resumable_upload(
        token=token,
        video_path=args.video,
        title=args.title,
        description=description,
        tags=tags,
        category_id=args.category_id or "20",
        privacy_status=args.privacy_status or "unlisted",
    )
    result = upload_video(upload_url, token, args.video)
    video_id = result.get("id")
    if video_id:
        result["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}"
        if args.playlist_title:
            playlist_id = find_or_create_playlist(
                token=token,
                title=args.playlist_title,
                privacy_status=args.privacy_status or "unlisted",
            )
            playlist_result = add_video_to_playlist(
                token=token,
                playlist_id=playlist_id,
                video_id=str(video_id),
            )
            result["playlist"] = {
                "title": args.playlist_title,
                "id": playlist_id,
                "url": f"https://www.youtube.com/playlist?list={playlist_id}",
                **playlist_result,
            }
        if not args.no_thumbnail:
            thumbnail_path = args.thumbnail
            if thumbnail_path is None:
                thumbnail_path = args.output.with_name(f"{args.output.stem}_thumbnail.jpg")
                thumbnail_path = extract_thumbnail(
                    video_path=args.video,
                    thumbnail_path=thumbnail_path,
                    seconds=float(args.thumbnail_time),
                )
            elif not thumbnail_path.is_file():
                raise SystemExit(f"thumbnail not found: {thumbnail_path}")
            result["thumbnail"] = {
                "path": str(thumbnail_path),
                "time_seconds": None if args.thumbnail else float(args.thumbnail_time),
                "upload": upload_thumbnail(
                    token=token,
                    video_id=str(video_id),
                    thumbnail_path=thumbnail_path,
                ),
            }
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

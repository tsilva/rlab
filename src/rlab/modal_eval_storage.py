from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlparse

from rlab.file_utils import file_sha256 as _file_sha256


file_sha256 = _file_sha256


def _strip_env_file_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class ObjectNotFound(FileNotFoundError):
    pass


def object_store_base_uri(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    uri = _strip_env_file_quotes(
        values.get("MODAL_EVAL_STORAGE_URI", "") or values.get("CHECKPOINT_BUCKET_URI", "")
    )
    if not uri:
        raise RuntimeError("MODAL_EVAL_STORAGE_URI or CHECKPOINT_BUCKET_URI must be set")
    return uri.rstrip("/")


class ObjectStore:
    def __init__(self, base_uri: str):
        self.base_uri = str(base_uri).rstrip("/")
        parsed = urlparse(self.base_uri)
        if parsed.scheme not in {"s3", "file"}:
            raise ValueError("Modal eval object storage must use s3:// or file://")
        self.scheme = parsed.scheme
        self._client = None

    def uri(self, key: str) -> str:
        return f"{self.base_uri}/{quote(key.strip('/'), safe='/._-')}"

    def _file_path(self, key_or_uri: str) -> Path:
        value = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        parsed = urlparse(value)
        if parsed.scheme != "file":
            raise ValueError(f"expected file URI, got {value}")
        return Path(unquote(parsed.path))

    def _s3_parts(self, key_or_uri: str) -> tuple[str, str]:
        return _parse_s3_uri(key_or_uri if "://" in key_or_uri else self.uri(key_or_uri))

    def _s3_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            endpoint = _strip_env_file_quotes(
                os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3", "")
            )
            kwargs: dict[str, Any] = {
                "config": Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=5,
                    read_timeout=15,
                    retries={"max_attempts": 2},
                )
            }
            if endpoint:
                kwargs["endpoint_url"] = endpoint
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def put_bytes(
        self,
        key_or_uri: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        create_only: bool = True,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            path = self._file_path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb" if create_only else "wb") as handle:
                    handle.write(payload)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise RuntimeError(f"immutable object already exists with different content: {uri}")
            return uri
        bucket, key = self._s3_parts(uri)
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": payload,
            "ContentType": content_type,
            "Metadata": dict(metadata or {}),
        }
        if create_only:
            kwargs["IfNoneMatch"] = "*"
        try:
            self._s3_client().put_object(**kwargs)
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if create_only and status == 412:
                existing = self.get_bytes(uri)
                if existing != payload:
                    raise RuntimeError(
                        f"immutable object already exists with different content: {uri}"
                    ) from exc
            else:
                raise
        return uri

    def put_json(self, key_or_uri: str, value: Mapping[str, Any], *, create_only: bool = True) -> str:
        payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
        return self.put_bytes(
            key_or_uri,
            payload,
            content_type="application/json",
            create_only=create_only,
        )

    def put_json_conditional(
        self,
        key_or_uri: str,
        value: Mapping[str, Any],
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> tuple[str, str]:
        """Conditionally replace one small JSON object and return its URI/ETag.

        ROM game pointers use this instead of an unguarded last-writer-wins
        update.  The file backend uses the payload SHA-256 as its ETag analogue.
        """

        if if_none_match and if_match is not None:
            raise ValueError("if_none_match and if_match are mutually exclusive")
        payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            import fcntl

            path = self._file_path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.parent / f".{path.name}.lock"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                current = path.read_bytes() if path.is_file() else None
                current_etag = hashlib.sha256(current).hexdigest() if current is not None else None
                if if_none_match and current is not None:
                    raise RuntimeError(f"conditional object create failed: {uri}")
                if if_match is not None and current_etag != if_match.strip('"'):
                    raise RuntimeError(f"conditional object replace failed: {uri}")
                fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
                temporary = Path(name)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    directory_fd = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    temporary.unlink(missing_ok=True)
            return uri, hashlib.sha256(payload).hexdigest()

        bucket, key = self._s3_parts(uri)
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": payload,
            "ContentType": "application/json",
        }
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        try:
            result = self._s3_client().put_object(**kwargs)
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status in {409, 412}:
                raise RuntimeError(f"conditional object update failed: {uri}") from exc
            raise
        etag = str(result.get("ETag") or "").strip('"')
        if not etag:
            etag = str(self.head(uri).get("etag") or "")
        return uri, etag

    def put_file(
        self,
        key_or_uri: str,
        path: Path,
        *,
        sha256: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            return self.put_bytes(
                uri,
                path.read_bytes(),
                content_type=content_type,
                metadata={"sha256": sha256},
            )
        bucket, key = self._s3_parts(uri)
        with path.open("rb") as handle:
            try:
                self._s3_client().put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=handle,
                    ContentType=content_type,
                    Metadata={"sha256": sha256},
                    IfNoneMatch="*",
                )
            except Exception as exc:
                response = getattr(exc, "response", {})
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status != 412:
                    raise
        head = self.head(uri)
        if int(head["size"]) != path.stat().st_size:
            raise RuntimeError(f"uploaded object size mismatch: {uri}")
        remote_sha = str(head.get("metadata", {}).get("sha256") or "")
        if remote_sha != sha256:
            raise RuntimeError(f"uploaded object hash metadata mismatch: {uri}")
        return uri

    def get_bytes(self, key_or_uri: str) -> bytes:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            try:
                return self._file_path(uri).read_bytes()
            except FileNotFoundError as exc:
                raise ObjectNotFound(uri) from exc
        bucket, key = self._s3_parts(uri)
        try:
            return self._s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = response.get("Error", {}).get("Code")
            if status == 404 or code in {"NoSuchKey", "404"}:
                raise ObjectNotFound(uri) from exc
            raise

    def get_json(self, key_or_uri: str) -> dict[str, Any]:
        value = json.loads(self.get_bytes(key_or_uri))
        if not isinstance(value, dict):
            raise ValueError(f"object must contain a JSON mapping: {key_or_uri}")
        return value

    def get_json_optional(self, key_or_uri: str) -> dict[str, Any] | None:
        try:
            return self.get_json(key_or_uri)
        except ObjectNotFound:
            return None

    def head(self, key_or_uri: str) -> dict[str, Any]:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            try:
                stat = self._file_path(uri).stat()
            except FileNotFoundError as exc:
                raise ObjectNotFound(uri) from exc
            payload = self._file_path(uri).read_bytes()
            return {
                "size": stat.st_size,
                "metadata": {},
                "etag": hashlib.sha256(payload).hexdigest(),
            }
        bucket, key = self._s3_parts(uri)
        try:
            result = self._s3_client().head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                raise ObjectNotFound(uri) from exc
            raise
        return {
            "size": int(result["ContentLength"]),
            "metadata": dict(result.get("Metadata") or {}),
            "etag": str(result.get("ETag") or "").strip('"'),
        }

    def presign_get(self, key_or_uri: str, *, expires_seconds: int) -> str:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            return uri
        bucket, key = self._s3_parts(uri)
        return self._s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=max(1, int(expires_seconds)),
        )

    def presign_put(
        self,
        key_or_uri: str,
        *,
        expires_seconds: int,
        content_type: str = "application/json",
        cache_control: str | None = None,
    ) -> str:
        uri = key_or_uri if "://" in key_or_uri else self.uri(key_or_uri)
        if self.scheme == "file":
            return uri
        bucket, key = self._s3_parts(uri)
        params = {
            "Bucket": bucket,
            "Key": key,
            "ContentType": content_type,
            "IfNoneMatch": "*",
        }
        if cache_control:
            params["CacheControl"] = cache_control
        return self._s3_client().generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=max(1, int(expires_seconds)),
        )


def write_downloaded_file(url: str, destination: Path) -> Path:
    parsed = urlparse(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                if parsed.scheme == "file":
                    with Path(unquote(parsed.path)).open("rb") as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                else:
                    import urllib.request

                    with urllib.request.urlopen(url, timeout=60) as response:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            raise RuntimeError(f"object download failed: {type(exc).__name__}") from exc
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return destination

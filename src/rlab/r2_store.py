from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


class ObjectNotFound(FileNotFoundError):
    pass


class ConditionalWriteConflict(RuntimeError):
    pass


PUBLIC_OBJECT_USER_AGENT = "rlab-public-client/1.0 (+https://github.com/tsilva/rlab)"


def public_object_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        str(url),
        headers={"User-Agent": PUBLIC_OBJECT_USER_AGENT},
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _clean_env(value: str | None) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


@dataclass(frozen=True)
class BucketConfig:
    uri: str
    endpoint_url: str = ""
    region: str = "auto"
    access_key_id: str = ""
    secret_access_key: str = ""
    public_base_url: str = ""

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        environment: Mapping[str, str] | None = None,
        public: bool = False,
    ) -> BucketConfig:
        values = os.environ if environment is None else environment
        normalized = prefix.rstrip("_").upper()
        config = cls(
            uri=_clean_env(values.get(f"{normalized}_URI")),
            endpoint_url=_clean_env(values.get(f"{normalized}_ENDPOINT_URL")),
            region=_clean_env(values.get(f"{normalized}_REGION")) or "auto",
            access_key_id=_clean_env(values.get(f"{normalized}_ACCESS_KEY_ID")),
            secret_access_key=_clean_env(values.get(f"{normalized}_SECRET_ACCESS_KEY")),
            public_base_url=(
                _clean_env(values.get(f"{normalized}_PUBLIC_BASE_URL")).rstrip("/")
                if public
                else ""
            ),
        )
        config.validate(public=public)
        return config

    def validate(self, *, public: bool = False) -> None:
        parsed = urlparse(self.uri)
        if parsed.scheme not in {"s3", "file"} or not parsed.netloc and parsed.scheme == "s3":
            raise ValueError("R2 bucket URI must use s3://bucket[/prefix] or file:///path")
        if parsed.scheme == "s3":
            if not self.endpoint_url:
                raise ValueError("R2 S3 configuration requires an endpoint URL")
            if not self.access_key_id or not self.secret_access_key:
                raise ValueError("R2 S3 configuration requires explicit credentials")
        if public and not self.public_base_url:
            raise ValueError("public model storage requires a public base URL")


@dataclass(frozen=True)
class RunStorageConfig:
    control: BucketConfig
    evaluation: BucketConfig
    models: BucketConfig

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> RunStorageConfig:
        result = cls(
            control=BucketConfig.from_env("RLAB_CONTROL_R2", environment=environment),
            evaluation=BucketConfig.from_env("RLAB_EVAL_R2", environment=environment),
            models=BucketConfig.from_env(
                "RLAB_MODELS_R2",
                environment=environment,
                public=True,
            ),
        )
        access_keys = [
            config.access_key_id
            for config in (result.control, result.evaluation, result.models)
            if urlparse(config.uri).scheme == "s3"
        ]
        if len(access_keys) != len(set(access_keys)):
            raise ValueError("control, eval, and model R2 buckets require separate credentials")
        return result

    def manifest_locations(self) -> dict[str, Any]:
        return {
            "control": self.control.uri,
            "evaluation": self.evaluation.uri,
            "models": self.models.uri,
            "public_models_base_url": self.models.public_base_url,
        }


class R2Bucket:
    def __init__(self, config: BucketConfig):
        self.config = config
        self.config.validate(public=bool(config.public_base_url))
        self.base_uri = config.uri.rstrip("/")
        self.scheme = urlparse(self.base_uri).scheme
        self._client = None

    def uri(self, key: str) -> str:
        return f"{self.base_uri}/{quote(key.strip('/'), safe='/._-')}"

    def key_from_uri(self, uri: str) -> str:
        parsed_base = urlparse(self.base_uri)
        parsed = urlparse(uri)
        if parsed.scheme != parsed_base.scheme or parsed.netloc != parsed_base.netloc:
            raise ValueError(f"object URI is outside configured bucket: {uri}")
        base_prefix = parsed_base.path.strip("/")
        object_path = unquote(parsed.path).strip("/")
        if base_prefix:
            if object_path == base_prefix:
                return ""
            if not object_path.startswith(base_prefix + "/"):
                raise ValueError(f"object URI is outside configured prefix: {uri}")
            object_path = object_path[len(base_prefix) + 1 :]
        return object_path

    def public_url(self, key: str) -> str:
        if not self.config.public_base_url:
            raise RuntimeError("bucket has no public base URL")
        return f"{self.config.public_base_url.rstrip('/')}/{quote(key.strip('/'), safe='/._-')}"

    def _file_path(self, key: str) -> Path:
        parsed = urlparse(self.uri(key))
        return Path(unquote(parsed.path))

    def _s3_parts(self, key: str) -> tuple[str, str]:
        parsed = urlparse(self.uri(key))
        return parsed.netloc, parsed.path.lstrip("/")

    def _s3_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=5,
                    read_timeout=30,
                    retries={"max_attempts": 2},
                ),
            )
        return self._client

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        cache_control: str | None = None,
        metadata: Mapping[str, str] | None = None,
        create_only: bool = True,
        if_match: str | None = None,
    ) -> str:
        if create_only and if_match is not None:
            raise ValueError("create_only and if_match are mutually exclusive")
        if self.scheme == "file":
            return self._put_file_bytes(
                key,
                payload,
                create_only=create_only,
                if_match=if_match,
            )
        bucket, object_key = self._s3_parts(key)
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": payload,
            "ContentType": content_type,
            "Metadata": dict(metadata or {}),
        }
        if cache_control:
            kwargs["CacheControl"] = cache_control
        if create_only:
            kwargs["IfNoneMatch"] = "*"
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        try:
            result = self._s3_client().put_object(**kwargs)
        except Exception as exc:
            status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if status not in {409, 412}:
                raise
            if create_only:
                try:
                    existing = self.get_bytes(key)
                except ObjectNotFound:
                    pass
                else:
                    if existing == payload:
                        return str(self.head(key)["etag"])
            raise ConditionalWriteConflict(f"conditional write failed: {self.uri(key)}") from exc
        return str(result.get("ETag") or "").strip('"') or str(self.head(key)["etag"])

    def _put_file_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        create_only: bool,
        if_match: str | None,
    ) -> str:
        import fcntl

        path = self._file_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / f".{path.name}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = path.read_bytes() if path.is_file() else None
            current_etag = hashlib.sha256(current).hexdigest() if current is not None else None
            if create_only and current is not None:
                if current == payload:
                    return str(current_etag)
                raise ConditionalWriteConflict(f"conditional create failed: {self.uri(key)}")
            if if_match is not None and current_etag != if_match.strip('"'):
                raise ConditionalWriteConflict(f"conditional replace failed: {self.uri(key)}")
            fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary = Path(temporary_name)
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
        return hashlib.sha256(payload).hexdigest()

    def put_json(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        create_only: bool = True,
        if_match: str | None = None,
        cache_control: str | None = None,
    ) -> str:
        return self.put_bytes(
            key,
            _canonical_json(value),
            content_type="application/json",
            cache_control=cache_control,
            create_only=create_only,
            if_match=if_match,
        )

    def put_file(
        self,
        key: str,
        path: Path,
        *,
        sha256: str,
        content_type: str = "application/octet-stream",
        cache_control: str | None = None,
    ) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"local file hash mismatch for {path}")
        etag = self.put_bytes(
            key,
            path.read_bytes(),
            content_type=content_type,
            cache_control=cache_control,
            metadata={"sha256": sha256},
            create_only=True,
        )
        head = self.head(key)
        if int(head["size"]) != path.stat().st_size:
            raise RuntimeError(f"uploaded object size mismatch: {self.uri(key)}")
        if self.scheme == "s3" and str(head["metadata"].get("sha256") or "") != sha256:
            raise RuntimeError(f"uploaded object hash metadata mismatch: {self.uri(key)}")
        return etag

    def get_bytes(self, key: str) -> bytes:
        if self.scheme == "file":
            try:
                return self._file_path(key).read_bytes()
            except FileNotFoundError as exc:
                raise ObjectNotFound(self.uri(key)) from exc
        bucket, object_key = self._s3_parts(key)
        try:
            return self._s3_client().get_object(Bucket=bucket, Key=object_key)["Body"].read()
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = response.get("Error", {}).get("Code")
            if status == 404 or code in {"NoSuchKey", "404"}:
                raise ObjectNotFound(self.uri(key)) from exc
            raise

    def get_json(self, key: str) -> dict[str, Any]:
        value = json.loads(self.get_bytes(key))
        if not isinstance(value, dict):
            raise ValueError(f"object must contain a JSON mapping: {self.uri(key)}")
        return value

    def get_json_optional(self, key: str) -> dict[str, Any] | None:
        try:
            return self.get_json(key)
        except ObjectNotFound:
            return None

    def head(self, key: str) -> dict[str, Any]:
        if self.scheme == "file":
            try:
                payload = self._file_path(key).read_bytes()
            except FileNotFoundError as exc:
                raise ObjectNotFound(self.uri(key)) from exc
            return {
                "size": len(payload),
                "metadata": {},
                "etag": hashlib.sha256(payload).hexdigest(),
            }
        bucket, object_key = self._s3_parts(key)
        try:
            result = self._s3_client().head_object(Bucket=bucket, Key=object_key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                raise ObjectNotFound(self.uri(key)) from exc
            raise
        return {
            "size": int(result["ContentLength"]),
            "metadata": dict(result.get("Metadata") or {}),
            "etag": str(result.get("ETag") or "").strip('"'),
        }

    def iter_keys(self, prefix: str) -> Iterator[str]:
        normalized = prefix.strip("/")
        if self.scheme == "file":
            base = self._file_path(normalized)
            if base.is_file():
                yield normalized
                return
            root = self._file_path("")
            if not base.exists():
                return
            for path in sorted(item for item in base.rglob("*") if item.is_file()):
                if path.name.startswith(".") and path.name.endswith(".lock"):
                    continue
                yield path.relative_to(root).as_posix()
            return
        bucket, object_prefix = self._s3_parts(normalized)
        paginator = self._s3_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=object_prefix):
            for row in page.get("Contents") or []:
                full_key = str(row["Key"])
                base_prefix = urlparse(self.base_uri).path.lstrip("/").rstrip("/")
                if base_prefix and full_key.startswith(base_prefix + "/"):
                    full_key = full_key[len(base_prefix) + 1 :]
                yield full_key

    def delete(self, key: str, *, if_match: str | None = None) -> None:
        if self.scheme == "file":
            if if_match is not None and str(self.head(key)["etag"]) != if_match.strip('"'):
                raise ConditionalWriteConflict(f"conditional delete failed: {self.uri(key)}")
            self._file_path(key).unlink(missing_ok=False)
            return
        bucket, object_key = self._s3_parts(key)
        kwargs = {"Bucket": bucket, "Key": object_key}
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        try:
            self._s3_client().delete_object(**kwargs)
        except Exception as exc:
            status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if status in {409, 412}:
                raise ConditionalWriteConflict(
                    f"conditional delete failed: {self.uri(key)}"
                ) from exc
            raise

    def copy_within(self, source_key: str, destination_key: str) -> str:
        source_head = self.head(source_key)
        if self.scheme == "file":
            payload = self.get_bytes(source_key)
            destination_etag = self.put_bytes(
                destination_key,
                payload,
                create_only=True,
            )
        else:
            bucket, source_object_key = self._s3_parts(source_key)
            destination_bucket, destination_object_key = self._s3_parts(
                destination_key
            )
            if destination_bucket != bucket:
                raise ValueError("copy_within requires one bucket")
            existing = None
            try:
                existing = self.head(destination_key)
            except ObjectNotFound:
                pass
            if existing is not None:
                if (
                    int(existing["size"]) != int(source_head["size"])
                    or dict(existing["metadata"]) != dict(source_head["metadata"])
                ):
                    raise ConditionalWriteConflict(
                        f"copy destination conflicts: {self.uri(destination_key)}"
                    )
                destination_etag = str(existing["etag"])
            else:
                try:
                    result = self._s3_client().copy_object(
                        Bucket=bucket,
                        Key=destination_object_key,
                        CopySource={
                            "Bucket": bucket,
                            "Key": source_object_key,
                        },
                        CopySourceIfMatch=str(source_head["etag"]),
                        MetadataDirective="COPY",
                    )
                except Exception as exc:
                    status = getattr(exc, "response", {}).get(
                        "ResponseMetadata", {}
                    ).get("HTTPStatusCode")
                    if status in {409, 412}:
                        raise ConditionalWriteConflict(
                            f"conditional copy failed: {self.uri(source_key)}"
                        ) from exc
                    raise
                destination_etag = str(
                    dict(result.get("CopyObjectResult") or {}).get("ETag") or ""
                ).strip('"')
        copied = self.head(destination_key)
        if (
            int(copied["size"]) != int(source_head["size"])
            or (
                self.scheme == "s3"
                and dict(copied["metadata"]) != dict(source_head["metadata"])
            )
        ):
            raise RuntimeError(
                f"copied object failed verification: {self.uri(destination_key)}"
            )
        return destination_etag or str(copied["etag"])

    def presign_get(self, key: str, *, expires_seconds: int) -> str:
        if self.scheme == "file":
            return self.uri(key)
        bucket, object_key = self._s3_parts(key)
        return self._s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=max(1, int(expires_seconds)),
        )

    def presign_put(
        self,
        key: str,
        *,
        expires_seconds: int,
        content_type: str = "application/json",
    ) -> str:
        if self.scheme == "file":
            return self.uri(key)
        bucket, object_key = self._s3_parts(key)
        return self._s3_client().generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": object_key,
                "ContentType": content_type,
                "IfNoneMatch": "*",
            },
            ExpiresIn=max(1, int(expires_seconds)),
        )

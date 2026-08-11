# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from typing import Any

import requests

from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._common import (
    _DEFAULT_MAX_REQUEST_SIZE,
    _load_session_from_envvar,
)
from opentelemetry.sdk.environment_variables import (
    _OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER,
    OTEL_EXPORTER_OTLP_CERTIFICATE,
    OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE,
    OTEL_EXPORTER_OTLP_CLIENT_KEY,
    OTEL_EXPORTER_OTLP_COMPRESSION,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_EXPORTER_OTLP_HEADERS,
    OTEL_EXPORTER_OTLP_TIMEOUT,
    OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE,
    OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE,
    OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY,
    OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
    OTEL_EXPORTER_OTLP_TRACES_HEADERS,
    OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
    OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED,
)
from opentelemetry.util.re import parse_env_headers

_logger = logging.getLogger(__name__)

DEFAULT_COMPRESSION = Compression.NoCompression
DEFAULT_ENDPOINT = "http://localhost:4318/"
DEFAULT_TRACES_EXPORT_PATH = "v1/traces"
DEFAULT_TIMEOUT = 10  # in seconds


@dataclass(frozen=True)
class _OTLPHTTPSpanExporterConfig:
    endpoint: str
    certificate_file: str | bool
    client_key_file: str | None
    client_certificate_file: str | None
    headers: Mapping[str, str]
    timeout: float
    compression: Compression
    session: requests.Session
    max_request_size: int
    internal_metrics_enabled: bool


def _resolve_config_from_env(
    *,
    endpoint: str | None,
    certificate_file: str | None,
    client_key_file: str | None,
    client_certificate_file: str | None,
    headers: dict[str, str] | None,
    timeout: float | None,
    compression: Compression | None,
    session: requests.Session | None,
    max_request_size: int | None,
) -> _OTLPHTTPSpanExporterConfig:
    resolved_endpoint = endpoint or environ.get(
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
        _append_trace_path(environ.get(OTEL_EXPORTER_OTLP_ENDPOINT, DEFAULT_ENDPOINT)),
    )
    resolved_certificate_file = certificate_file or environ.get(
        OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE,
        environ.get(OTEL_EXPORTER_OTLP_CERTIFICATE, True),
    )
    resolved_client_key_file = client_key_file or environ.get(
        OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY,
        environ.get(OTEL_EXPORTER_OTLP_CLIENT_KEY),
    )
    resolved_client_certificate_file = client_certificate_file or environ.get(
        OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE,
        environ.get(OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE),
    )
    headers_string = environ.get(
        OTEL_EXPORTER_OTLP_TRACES_HEADERS,
        environ.get(OTEL_EXPORTER_OTLP_HEADERS, ""),
    )
    resolved_headers = headers or parse_env_headers(
        headers_string,
        liberal=True,
    )
    resolved_timeout = timeout or float(
        environ.get(
            OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
            environ.get(OTEL_EXPORTER_OTLP_TIMEOUT, DEFAULT_TIMEOUT),
        )
    )
    resolved_compression = compression or _compression_from_env()
    resolved_session = (
        session
        or _load_session_from_envvar(_OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER)
        or requests.Session()
    )
    resolved_max_request_size = _DEFAULT_MAX_REQUEST_SIZE if max_request_size is None else max_request_size
    resolved_internal_metrics_enabled = _internal_metrics_enabled()

    return _OTLPHTTPSpanExporterConfig(
        endpoint=resolved_endpoint,
        certificate_file=resolved_certificate_file,
        client_key_file=resolved_client_key_file,
        client_certificate_file=resolved_client_certificate_file,
        headers=resolved_headers,
        timeout=resolved_timeout,
        compression=resolved_compression,
        session=resolved_session,
        max_request_size=resolved_max_request_size,
        internal_metrics_enabled=resolved_internal_metrics_enabled,
    )


def _resolve_declarative_config(
    config: Mapping[str, Any],
) -> _OTLPHTTPSpanExporterConfig:
    tls = config.get("tls") or {}
    client_key_file = tls.get("key_file")
    client_certificate_file = tls.get("cert_file")

    timeout_millis: int | None = config.get("timeout")
    timeout = timeout_millis / 1000.0 if timeout_millis else DEFAULT_TIMEOUT
    compression = (config.get("compression") or "none").lower()

    return _OTLPHTTPSpanExporterConfig(
        endpoint=config.get("endpoint") or _append_trace_path(DEFAULT_ENDPOINT),
        certificate_file=tls.get("ca_file") or True,
        client_key_file=client_key_file,
        client_certificate_file=client_certificate_file,
        headers=_declarative_headers(config),
        timeout=timeout,
        compression=Compression(compression),
        session=requests.Session(),
        max_request_size=_DEFAULT_MAX_REQUEST_SIZE,
        internal_metrics_enabled=False,
    )


def _declarative_headers(config: Mapping[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in (config.get("headers_list") or "").split(","):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            headers[key.strip()] = value.strip()
        elif item:
            _logger.warning(
                "Invalid header pair in headers_list (missing '='): %s",
                item,
            )
    for header in config.get("headers") or ():
        headers[header["name"]] = header.get("value") or ""
    return headers


def _compression_from_env() -> Compression:
    compression = (
        environ.get(
            OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
            environ.get(OTEL_EXPORTER_OTLP_COMPRESSION, "none"),
        )
        .lower()
        .strip()
    )
    return Compression(compression)


def _internal_metrics_enabled() -> bool:
    return environ.get(OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED, "").strip().lower() == "true"


def _append_trace_path(endpoint: str) -> str:
    if endpoint.endswith("/"):
        return endpoint + DEFAULT_TRACES_EXPORT_PATH
    return endpoint + f"/{DEFAULT_TRACES_EXPORT_PATH}"

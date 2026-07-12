"""Shared utilities for video generation and rendering routes."""

import ipaddress
import socket
from typing import Tuple
from urllib.parse import urlparse

ALLOWED_URL_SCHEMES = {"http", "https"}


def get_frame_config(aspect_ratio: str) -> Tuple[Tuple[int, int], float]:
    """Return (frame_size, frame_width) for a given aspect ratio string."""
    if aspect_ratio == "9:16":
        return (1080, 1920), 8.0
    if aspect_ratio == "1:1":
        return (1080, 1080), 8.0
    return (3840, 2160), 14.22


def assert_public_http_url(url: str) -> None:
    """Raise ValueError if *url* is not a safe, public http(s) URL.

    Guards against SSRF: rejects non-http(s) schemes and hostnames that
    resolve to loopback, private, link-local, or otherwise reserved
    addresses (e.g. cloud metadata endpoints, internal services).
    """
    if not isinstance(url, str) or not url:
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"URL scheme must be one of {sorted(ALLOWED_URL_SCHEMES)}")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"URL resolves to a disallowed address: {ip}")

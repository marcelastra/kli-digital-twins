"""Local LLM provider: Mistral (or any local model) with a hard no-egress guard.

ORIGINAL WORK (Kravis Leadership Institute). Not derived from either upstream
repository. See ATTRIBUTION.md.

-------------------------------------------------------------------------------
WHY LOCAL
-------------------------------------------------------------------------------

Digital twins are built from participants' own survey answers. Sending those to
a third-party API means transmitting human-subjects data off-site, which is a
disclosure your IRB protocol and consent language may not cover. Running the
model locally removes the question entirely.

The upstream pipelines are OpenAI-first. This module replaces that layer with a
local server, and enforces -- not merely documents -- that requests stay on the
machine.

-------------------------------------------------------------------------------
THE GUARD
-------------------------------------------------------------------------------

:func:`assert_local_endpoint` rejects any endpoint that is not demonstrably
loopback, before a single request is made. It resolves the hostname and checks
that *every* resolved address is loopback, so a name that happens to point
somewhere else cannot slip through.

The guard can be lifted only by setting ``allow_remote_endpoint: true`` in the
config -- deliberate, visible in version control, and recorded in the run
manifest written by :mod:`kli.run_simulation`. There is no environment variable
for it, because an env var would not show up in your appendix.

A caveat worth stating plainly: this guard constrains *this pipeline's* model
calls. It is not a system-level egress control, and it cannot stop a local
server that has itself been configured to proxy to a remote API. If you need an
auditable guarantee rather than a strong default, block outbound traffic at the
host or network level as well.

-------------------------------------------------------------------------------
SETUP (Ollama)
-------------------------------------------------------------------------------

    # install from https://ollama.com, then:
    ollama pull mistral          # ~4.1 GB, Mistral 7B Instruct
    ollama serve                 # usually already running

    # verify:
    curl http://localhost:11434/api/tags

Config::

    LLM_config:
      provider: local
      model_name: mistral
      base_url: http://localhost:11434
      temperature: 0.0

``format: "json"`` is sent on every request, so Ollama constrains decoding to
valid JSON. That removes the malformed-output failure mode that the vendored
``postprocess_responses.py`` handles with a bare ``split("```json")``.

Other servers: anything exposing Ollama's ``/api/chat`` works as-is. For an
OpenAI-compatible local server (llama.cpp, LM Studio, vLLM) set
``api_style: openai``; the endpoint guard applies identically.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "LocalModelConfig",
    "LocalModelClient",
    "assert_local_endpoint",
    "RemoteEndpointBlocked",
    "LocalModelError",
]

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"


class RemoteEndpointBlocked(RuntimeError):
    """Raised when a configured endpoint is not on this machine.

    Participant data would leave the host. Resolve the configuration rather
    than catching this.
    """


class LocalModelError(RuntimeError):
    """Raised when the local model server cannot be reached or errors out."""


def _resolved_addresses(host: str) -> list[str]:
    """Resolve a hostname to every address it maps to.

    Args:
        host: Hostname or literal IP.

    Returns:
        List of resolved address strings.

    Raises:
        RemoteEndpointBlocked: If the hostname cannot be resolved -- an
            unresolvable host cannot be shown to be local, so it is refused.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise RemoteEndpointBlocked(
            f"Cannot resolve host {host!r}, so it cannot be verified as local. "
            "Refusing to send participant data to an unverifiable endpoint."
        ) from exc
    return sorted({info[4][0] for info in infos})


def assert_local_endpoint(base_url: str, *, allow_remote: bool = False) -> None:
    """Refuse any endpoint that is not loopback.

    Args:
        base_url: The configured server URL.
        allow_remote: Set only from ``allow_remote_endpoint: true`` in the
            config. Lifts the guard and permits data to leave the machine.

    Raises:
        RemoteEndpointBlocked: If the endpoint is not loopback and the guard is
            in force.
        ValueError: If ``base_url`` has no host component.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"base_url {base_url!r} has no host component.")

    if allow_remote:
        return

    for address in _resolved_addresses(host):
        try:
            is_loopback = ipaddress.ip_address(address).is_loopback
        except ValueError as exc:
            raise RemoteEndpointBlocked(
                f"Endpoint {base_url!r} resolved to {address!r}, which is not a "
                "usable IP address; cannot verify it is local."
            ) from exc

        if not is_loopback:
            raise RemoteEndpointBlocked(
                f"Endpoint {base_url!r} resolves to {address}, which is not "
                "loopback. Participant survey responses would leave this "
                "machine.\n"
                "If this is intended, set 'allow_remote_endpoint: true' under "
                "LLM_simulation in the study config. That choice is recorded in "
                "the run manifest and should be disclosed in your methods."
            )


@dataclass
class LocalModelConfig:
    """Settings for a local model server.

    Attributes:
        model_name: Model tag as the server knows it, e.g. ``"mistral"``.
        base_url: Server root. Must be loopback unless the guard is lifted.
        api_style: ``"ollama"`` for ``/api/chat``, ``"openai"`` for
            ``/v1/chat/completions``.
        temperature: Sampling temperature. Use 0.0 for the deterministic arm.
        max_tokens: Response cap. Generous by default: a full question chunk's
            JSON skeleton can be long.
        max_retries: Attempts per request before giving up.
        request_timeout: Seconds per request. Local generation on CPU is slow;
            the default is deliberately high.
        allow_remote_endpoint: Lifts the no-egress guard. Keep False.
        extra_options: Passed through to the server's options block.
    """

    model_name: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_style: str = "ollama"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_retries: int = 5
    request_timeout: float = 900.0
    allow_remote_endpoint: bool = False
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_style not in {"ollama", "openai"}:
            raise ValueError(
                f"api_style must be 'ollama' or 'openai', got {self.api_style!r}"
            )
        assert_local_endpoint(
            self.base_url, allow_remote=self.allow_remote_endpoint
        )

    def manifest(self) -> dict[str, Any]:
        """Serializable record of these settings, for the run manifest."""
        return {
            "provider": "local",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_style": self.api_style,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "endpoint_is_loopback": not self.allow_remote_endpoint,
            "allow_remote_endpoint": self.allow_remote_endpoint,
        }


class LocalModelClient:
    """Minimal client for a local chat model, forcing JSON output.

    No third-party HTTP dependency: uses ``urllib`` so the no-egress claim does
    not rest on a client library's behavior.
    """

    def __init__(self, config: LocalModelConfig) -> None:
        """Initialize the client.

        Args:
            config: Validated local model settings.
        """
        self.config = config

    # -- plumbing ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Guard again at call time: a mutated config should not be able to
        # redirect traffic after construction.
        assert_local_endpoint(
            self.config.base_url, allow_remote=self.config.allow_remote_endpoint
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LocalModelError(
                f"{url} returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LocalModelError(
                f"Cannot reach the local model server at {url} ({exc.reason}).\n"
                "Is it running? For Ollama:\n"
                "    ollama serve\n"
                f"    ollama pull {self.config.model_name}\n"
                f"    curl {self.config.base_url}/api/tags"
            ) from exc

    def health_check(self) -> dict[str, Any]:
        """Confirm the server is up and the model is present.

        Returns:
            ``{"reachable": bool, "model_available": bool, "models": [...]}``.
        """
        if self.config.api_style == "ollama":
            url = f"{self.config.base_url}/api/tags"
        else:
            url = f"{self.config.base_url}/v1/models"

        assert_local_endpoint(
            self.config.base_url, allow_remote=self.config.allow_remote_endpoint
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return {"reachable": False, "model_available": False, "error": str(exc)}

        if self.config.api_style == "ollama":
            names = [m.get("name", "") for m in data.get("models", [])]
        else:
            names = [m.get("id", "") for m in data.get("data", [])]

        wanted = self.config.model_name
        available = any(
            name == wanted or name.split(":")[0] == wanted.split(":")[0]
            for name in names
        )
        return {"reachable": True, "model_available": available, "models": names}

    # -- generation -------------------------------------------------------

    def complete_json(
        self, *, system: str, user: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a JSON object response.

        Retries on transport errors and on output that is not a JSON object.
        JSON-mode decoding makes the latter rare, but a model can still return
        a JSON *array* or a bare string, which is not usable here.

        Args:
            system: System instruction.
            user: The assembled prompt.

        Returns:
            ``(parsed_object, meta)``. ``meta`` carries ``attempts``,
            ``elapsed_seconds``, and any token counts the server reports.

        Raises:
            LocalModelError: If no attempt produced a JSON object.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self.config.api_style == "ollama":
            path = "/api/chat"
            payload: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
                "stream": False,
                # Constrains decoding to valid JSON.
                "format": "json",
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **self.config.extra_options,
                },
            }
        else:
            path = "/v1/chat/completions"
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
                **self.config.extra_options,
            }

        started = time.monotonic()
        last_error: str | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw = self._post(path, payload)
            except LocalModelError as exc:
                last_error = str(exc)
                if attempt == self.config.max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
                continue

            if self.config.api_style == "ollama":
                content = (raw.get("message") or {}).get("content", "")
                usage = {
                    "prompt_tokens": raw.get("prompt_eval_count"),
                    "completion_tokens": raw.get("eval_count"),
                }
            else:
                choices = raw.get("choices") or [{}]
                content = (choices[0].get("message") or {}).get("content", "")
                usage = raw.get("usage", {}) or {}

            parsed, error = _parse_json_object(content)
            if parsed is not None:
                return parsed, {
                    "attempts": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    **usage,
                }

            last_error = error
            if attempt < self.config.max_retries:
                time.sleep(min(2**attempt, 30))

        raise LocalModelError(
            f"{self.config.model_name} did not return a JSON object after "
            f"{self.config.max_retries} attempts. Last problem: {last_error}"
        )


def _parse_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse model output into a JSON object.

    Tries the whole string, then a fenced block, then the outermost brace pair.
    This is deliberately more forgiving than upstream's single
    ``split("```json")``, but it never guesses at values -- it only locates the
    object.

    Args:
        content: Raw model output.

    Returns:
        ``(parsed, None)`` on success, ``(None, reason)`` on failure.
    """
    if not content or not content.strip():
        return None, "empty response"

    text = content.strip()

    candidates = [text]
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                candidates.append(chunk)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None

    try:
        import json_repair  # type: ignore[import-not-found]
    except ImportError:
        return None, "output was not a JSON object (json_repair not installed)"

    try:
        repaired = json_repair.loads(text)
    except Exception as exc:  # noqa: BLE001
        return None, f"output was not a JSON object; repair failed ({exc})"

    if isinstance(repaired, dict):
        return repaired, None
    return None, f"output parsed as {type(repaired).__name__}, not an object"

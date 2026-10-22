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

import http.client
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
    "verified_peer_addresses",
    "ContextWindowExceeded",
    "estimate_tokens",
    "DEFAULT_NUM_CTX",
    "LocalModelClient",
    "assert_local_endpoint",
    "RemoteEndpointBlocked",
    "LocalModelError",
]

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"

#: Context window to request, in tokens. Set explicitly and deliberately.
#:
#: Ollama does NOT default to the model's own capability -- it applies its own
#: default (2048 in older builds, 4096 in newer ones) and, when a prompt
#: exceeds it, SILENTLY DISCARDS THE OVERFLOW. No error, no warning, no field
#: in the response.
#:
#: For this pipeline that failure is particularly severe, because truncation
#: drops the OLDEST tokens and the persona is the first thing in the prompt.
#: An over-long prompt therefore yields a fluent, well-formatted answer from a
#: model that never saw the persona -- a no-persona control masquerading as a
#: twin, with nothing in the output to distinguish it.
#:
#: 8192 is a safe default for Mistral 7B (32k capable). Raise it for longer
#: personas, but note that Ollama allocates KV cache for the full window, so
#: a large num_ctx costs memory whether or not you use it.
DEFAULT_NUM_CTX = 8192

#: Characters per token, used to size a prompt before sending it.
#:
#: Deliberately pessimistic. English averages roughly 4 characters per token,
#: but survey text is full of punctuation, digits, option markers and JSON
#: scaffolding, all of which tokenize less efficiently. Estimating 3.0 means we
#: over-count and refuse slightly early, which is the safe direction: refusing
#: a prompt that would have fit costs a config change, while accepting one that
#: does not fit costs a silently persona-free dataset.
CHARS_PER_TOKEN = 3.0


class ContextWindowExceeded(RuntimeError):
    """Raised when a prompt is too large for the configured context window.

    Raised *before* the request is sent. The alternative -- letting the server
    truncate -- produces a plausible-looking answer from a model that never saw
    the start of the prompt.
    """


def estimate_tokens(text: str) -> int:
    """Conservatively estimate the token count of a string.

    A character heuristic rather than a real tokenizer, because the tokenizer
    depends on the model and we do not want a hard dependency for a safety
    check. It over-estimates, which is the direction that fails safe.

    Args:
        text: The text to size.

    Returns:
        An estimated token count.
    """
    return int(len(text) / CHARS_PER_TOKEN) + 1


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


# ---------------------------------------------------------------------------
# Hardened transport
# ---------------------------------------------------------------------------
#
# Checking that the URL's hostname is loopback is necessary but NOT sufficient.
# Two ways a request to http://localhost:11434 can still leave the machine:
#
#   1. PROXY ENVIRONMENT VARIABLES. urllib.request.urlopen() builds an opener
#      with a ProxyHandler that reads http_proxy / HTTPS_PROXY. Those are
#      exempted for loopback only when no_proxy happens to name it. On a
#      managed laptop or VPN where http_proxy is set and no_proxy is not,
#      urllib routes the request -- participant data included -- to the proxy,
#      while a hostname-only check still reports the endpoint as local.
#      Verified: with http_proxy set and no_proxy unset,
#      urllib.request.proxy_bypass("localhost") returns False.
#
#   2. NAME RESOLUTION. A hostname that resolves to loopback at check time
#      could resolve elsewhere at connect time.
#
# Both are closed here. ProxyHandler({}) disables proxying regardless of the
# environment, and the connection classes below verify the socket's ACTUAL peer
# address after connecting, which is the ground truth no configuration can
# misreport. The peer address is recorded so the run manifest can state where
# the data really went rather than where it was supposed to go.


#: Peer addresses observed on verified connections, newest last. Read by the
#: run manifest.
_VERIFIED_PEERS: list[str] = []


def _assert_peer_is_loopback(sock: Any, host: str) -> None:
    """Verify an established socket's remote address is loopback.

    Args:
        sock: A connected socket.
        host: The hostname requested, for the error message.

    Raises:
        RemoteEndpointBlocked: If the peer is not loopback. The socket is
            closed before raising, so nothing is sent over it.
    """
    try:
        peer = sock.getpeername()[0]
    except OSError as exc:  # pragma: no cover - platform dependent
        sock.close()
        raise RemoteEndpointBlocked(
            f"Could not determine the peer address for {host!r}; refusing to "
            "send participant data over an unverifiable connection."
        ) from exc

    try:
        is_loopback = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        is_loopback = False

    if not is_loopback:
        sock.close()
        raise RemoteEndpointBlocked(
            f"Connection to {host!r} was established to {peer}, which is NOT "
            "loopback. The connection has been closed and nothing was sent.\n"
            "This usually means an HTTP proxy is configured in the environment "
            "(http_proxy / HTTPS_PROXY), or the hostname resolves off-machine."
        )

    if not _VERIFIED_PEERS or _VERIFIED_PEERS[-1] != peer:
        _VERIFIED_PEERS.append(peer)


class _LoopbackHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that refuses to proceed unless the peer is loopback."""

    def connect(self) -> None:  # noqa: D102 - inherited
        super().connect()
        _assert_peer_is_loopback(self.sock, self.host)


class _LoopbackHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that refuses to proceed unless the peer is loopback."""

    def connect(self) -> None:  # noqa: D102 - inherited
        super().connect()
        _assert_peer_is_loopback(self.sock, self.host)


class _LoopbackHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler using the loopback-verified HTTP connection."""

    def http_open(self, req):  # noqa: D102, ANN001, ANN201 - inherited
        return self.do_open(_LoopbackHTTPConnection, req)


class _LoopbackHTTPSHandler(urllib.request.HTTPSHandler):
    """urllib handler using the loopback-verified HTTPS connection."""

    def https_open(self, req):  # noqa: D102, ANN001, ANN201 - inherited
        return self.do_open(_LoopbackHTTPSConnection, req)


def _build_loopback_opener() -> urllib.request.OpenerDirector:
    """Build an opener that cannot proxy and cannot leave the machine.

    ``ProxyHandler({})`` is the critical piece: an empty proxy map means no
    proxy can be applied no matter what the environment says. The loopback
    handlers are defense in depth, verifying the realized connection rather
    than trusting configuration.

    Returns:
        An opener safe for sending participant data to a local model.
    """
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _LoopbackHTTPHandler(),
        _LoopbackHTTPSHandler(),
    )


def verified_peer_addresses() -> list[str]:
    """Peer addresses this process has actually connected to, in order.

    Recorded in the run manifest so a write-up can state where requests went,
    rather than only where they were configured to go.
    """
    return list(_VERIFIED_PEERS)


@dataclass
class LocalModelConfig:
    """Settings for a local model server.

    Attributes:
        model_name: Model tag as the server knows it, e.g. ``"mistral"``.
        base_url: Server root. Must be loopback unless the guard is lifted.
        api_style: ``"ollama"`` for ``/api/chat``, ``"openai"`` for
            ``/v1/chat/completions``.
        temperature: Sampling temperature. Use 0.0 for the deterministic arm.
        max_tokens: Cap on the GENERATED response, not the context window.
            These are different numbers and conflating them is a common error:
            upstream's ``max_tokens: 16384`` is an output cap and says nothing
            about how much input the model can see.
        num_ctx: Total context window in tokens, covering prompt and response
            together. Sent explicitly so the server cannot apply its own
            smaller default and silently truncate. See :data:`DEFAULT_NUM_CTX`.
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
    max_tokens: int = 4096
    num_ctx: int = DEFAULT_NUM_CTX
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
            "num_ctx": self.num_ctx,
            "endpoint_is_loopback": not self.allow_remote_endpoint,
            "allow_remote_endpoint": self.allow_remote_endpoint,
            "proxy_disabled": not self.allow_remote_endpoint,
            "verified_peer_addresses": verified_peer_addresses(),
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

    def check_prompt_fits(self, *, system: str, user: str) -> dict[str, Any]:
        """Size a prompt against the context window before sending it.

        Args:
            system: System instruction.
            user: Assembled user prompt.

        Returns:
            ``{"estimated_prompt_tokens", "num_ctx", "max_tokens",
            "headroom_tokens"}``.

        Raises:
            ContextWindowExceeded: If the prompt plus the reserved response
                space would not fit.
        """
        prompt_tokens = estimate_tokens(system) + estimate_tokens(user)
        reserved = self.config.max_tokens
        available = self.config.num_ctx - reserved

        if prompt_tokens > available:
            raise ContextWindowExceeded(
                f"Prompt is about {prompt_tokens:,} tokens, but only "
                f"{available:,} are available "
                f"(num_ctx {self.config.num_ctx:,} minus {reserved:,} reserved "
                "for the response).\n"
                "\n"
                "Refusing to send it. If this were sent, the server would "
                "silently discard the overflow from the START of the prompt -- "
                "which is the persona -- and return a confident answer from a "
                "model that never saw it.\n"
                "\n"
                "Fix by any of:\n"
                "  - lower 'questions_per_call' so each call carries fewer "
                "questions\n"
                f"  - raise 'num_ctx' (Mistral 7B supports up to 32768; costs "
                "memory)\n"
                "  - lower 'max_tokens' if responses are comfortably shorter\n"
                "  - use a shorter persona variant"
            )

        return {
            "estimated_prompt_tokens": prompt_tokens,
            "num_ctx": self.config.num_ctx,
            "max_tokens": reserved,
            "headroom_tokens": available - prompt_tokens,
        }

    def _opener(self) -> urllib.request.OpenerDirector:
        """Build the opener for this client.

        For a local endpoint this is the hardened, proxy-free, loopback-verified
        opener (:func:`_build_loopback_opener`). Only when the guard has been
        deliberately lifted does it fall back to the stock opener, which honors
        the environment's proxy settings like any ordinary HTTP client.
        """
        if self.config.allow_remote_endpoint:
            return urllib.request.build_opener()
        return _build_loopback_opener()

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
            with self._opener().open(
                request, timeout=self.config.request_timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except RemoteEndpointBlocked:
            raise
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
            with self._opener().open(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except RemoteEndpointBlocked:
            raise
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
        # Fail before sending rather than let the server truncate silently.
        self.check_prompt_fits(system=system, user=user)

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
                    # Sent on every request. Without it the server applies its
                    # own default and truncates without saying so.
                    "num_ctx": self.config.num_ctx,
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

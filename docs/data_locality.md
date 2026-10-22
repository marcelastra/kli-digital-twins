# Keeping participant data on the machine

What this pipeline guarantees about data locality, how it is enforced, and —
importantly — what it does **not** cover.

## The claim

Participant survey responses are sent only to a model server running on the
same machine. No participant data is transmitted to any third-party service.

## How it is enforced

Three independent layers, each of which would have to fail for data to leave.

### 1. Endpoint check (configuration)

`assert_local_endpoint()` resolves the configured `base_url` and requires
**every** resolved address to be loopback. A hostname that cannot be resolved is
treated as remote, not as local: unverifiable is not the same as safe. The check
runs at client construction and again before every request, so mutating the
config mid-run cannot redirect traffic.

### 2. Proxy-free transport

**This layer was added in response to external review and closes a real hole.**

An endpoint check alone is insufficient. `urllib.request.urlopen()` builds an
opener whose `ProxyHandler` reads `http_proxy` / `HTTPS_PROXY` from the
environment. Loopback is exempted *only* when `no_proxy` happens to name it. On
a managed laptop or VPN where a proxy is set and `no_proxy` is not, urllib
routes the request — participant data included — through the proxy, while a
hostname-only check still reports the endpoint as local and the run manifest
still records `endpoint_is_loopback: true`.

This is verifiable in a few lines:

```python
os.environ["http_proxy"] = "http://proxy.example.com:8080"
urllib.request.proxy_bypass("localhost")   # -> False
```

The fix is `urllib.request.build_opener(urllib.request.ProxyHandler({}))`. An
empty proxy map means no proxy can be applied regardless of the environment.
The pipeline builds its own opener this way and never calls `urlopen()`.

### 3. Peer-address verification (the realized connection)

Configuration can be wrong and names can resolve differently at connect time
than at check time. So after each socket connects — and before anything is
written to it — the pipeline reads the socket's **actual peer address** and
requires it to be loopback. A non-loopback peer causes the socket to be closed
and the request abandoned, with nothing sent.

This is the ground truth. It cannot be misreported by configuration, and it
closes both the proxy case and DNS rebinding.

Verified peer addresses are recorded in `run_manifest.json`, so a write-up can
state where requests actually went rather than where they were meant to go.

## Lifting the guard

Set `allow_remote_endpoint: true` under `LLM_simulation`. This is deliberately:

- **in the config**, so it appears in version control and code review — there is
  no environment variable for it, because an environment variable would not show
  up in a methods section;
- **recorded** in every run manifest (`allow_remote_endpoint`,
  `proxy_disabled`, `verified_peer_addresses`);
- **all-or-nothing** — it disables the hardened transport as well as the
  endpoint check, because a partial guarantee is worse than a clear one.

If this has ever been set during a run whose data you report, the no-egress
statement in your methods section is not accurate as written and must be
revised.

## What this does NOT cover

Stating the limits plainly, because an overstated guarantee is worse than a
narrow one.

- **It constrains this pipeline's model calls only.** It is not a system-level
  egress control. Other software on the machine is unaffected.
- **It cannot inspect the local server.** If Ollama — or whatever is listening
  on that port — has itself been configured to forward requests upstream, the
  data reaches it over loopback and leaves from there. The guarantee is "this
  pipeline sends only to localhost", not "localhost keeps it".
- **It does not cover the rest of the project.** Opening a response file in a
  cloud-synced folder, pasting an excerpt into a web tool, or committing
  `data/` to a remote repository all bypass it entirely. `.gitignore` excludes
  `data/`, `**/raw_data/` and run outputs — keep it that way.
- **It is not an IRB control by itself.** It is a strong technical default that
  supports a locality claim. For an auditable guarantee, pair it with host- or
  network-level egress blocking during data collection.

## Verifying it yourself

```bash
# the guard rejects non-local endpoints
python -c "
from kli.providers.local import assert_local_endpoint, RemoteEndpointBlocked
for url in ['http://localhost:11434', 'https://api.openai.com']:
    try:
        assert_local_endpoint(url); print(f'{url}: allowed')
    except RemoteEndpointBlocked:
        print(f'{url}: blocked')
"

# the transport ignores proxy environment variables
http_proxy=http://proxy.example:8080 python -c "
from kli.providers.local import _build_loopback_opener
for h in _build_loopback_opener().handlers:
    if type(h).__name__ == 'ProxyHandler':
        print('proxies:', h.proxies)   # {}
"

# the full test suite for this behaviour
python -m pytest tests/test_security_and_pilot.py -v -k "Proxy or Peer"
```

## For the methods section

> All model inference was performed locally. The implementation resolves the
> configured model endpoint and requires every resolved address to be loopback,
> disables HTTP proxying irrespective of environment configuration, and
> verifies the actual peer address of each established connection before
> transmitting, closing the connection if it is not loopback. The realized peer
> addresses are recorded for each run. No participant survey responses were
> transmitted to any third-party service.

Add, if applicable: *"Host-level egress restrictions were additionally in place
during data collection."*

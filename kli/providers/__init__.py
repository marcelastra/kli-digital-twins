"""Model providers for the kli pipeline.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

Only a local provider is supplied. Upstream is OpenAI-first; this project runs
the model on the researcher's own machine so that participant survey responses
are never transmitted off-site. See kli/providers/local.py.
"""

from kli.providers.local import (
    LocalModelClient,
    LocalModelConfig,
    LocalModelError,
    RemoteEndpointBlocked,
    assert_local_endpoint,
)

__all__ = [
    "LocalModelClient",
    "LocalModelConfig",
    "LocalModelError",
    "RemoteEndpointBlocked",
    "assert_local_endpoint",
]

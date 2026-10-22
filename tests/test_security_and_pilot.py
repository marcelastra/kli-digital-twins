"""Tests for the no-egress guarantee, context-window handling, and pilot metrics.

ORIGINAL WORK (Kravis Leadership Institute).

The security tests reproduce a specific reported failure: a hostname-only
loopback check passes while an environment proxy silently routes the request
off-machine. That was a real hole in an earlier version of this code, found in
external review, and these tests exist so it cannot reopen.

Run with:  python -m pytest tests/test_security_and_pilot.py -v
"""

from __future__ import annotations

import os
import random
import urllib.request
from typing import Any

import pytest

from kli.pilot_metrics import (
    compute_item_agreement,
    compute_stability,
    extract_item_values,
)
from kli.providers.local import (
    CHARS_PER_TOKEN,
    ContextWindowExceeded,
    LocalModelClient,
    LocalModelConfig,
    RemoteEndpointBlocked,
    _assert_peer_is_loopback,
    _build_loopback_opener,
    estimate_tokens,
)


class _FakeSocket:
    """Stand-in for a connected socket with a fixed peer address."""

    def __init__(self, peer: str) -> None:
        self.peer = peer
        self.closed = False

    def getpeername(self) -> tuple[str, int]:
        return (self.peer, 11434)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def proxied_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with an HTTP proxy configured and no localhost exemption.

    This is the configuration that defeated the original hostname-only check:
    common on managed laptops and VPNs.
    """
    for key in list(os.environ):
        if "proxy" in key.lower():
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", "http://proxy.example.com:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example.com:8080")


class TestProxyBypassIsClosed:
    """The reported vulnerability: env proxies routing 'local' traffic away."""

    def test_the_vulnerability_is_real(self, proxied_environment: None) -> None:
        """Documents why the fix is needed.

        With a proxy set and no_proxy unset, urllib does NOT treat localhost as
        exempt, so the stock opener would send participant data to the proxy.
        """
        assert urllib.request.proxy_bypass("localhost") is False
        stock = urllib.request.build_opener()
        proxies: dict[str, str] = {}
        for handler in stock.handlers:
            if type(handler).__name__ == "ProxyHandler":
                proxies.update(handler.proxies)
        assert proxies, "stock opener would proxy -- this is the hole"

    def test_hardened_opener_has_no_proxies(self, proxied_environment: None) -> None:
        """Our opener ignores the environment entirely."""
        opener = _build_loopback_opener()
        for handler in opener.handlers:
            if type(handler).__name__ == "ProxyHandler":
                assert handler.proxies == {}

    def test_hardened_opener_installs_verifying_handlers(self) -> None:
        names = {type(h).__name__ for h in _build_loopback_opener().handlers}
        assert "_LoopbackHTTPHandler" in names
        assert "_LoopbackHTTPSHandler" in names

    def test_client_uses_the_hardened_opener_when_guarded(
        self, proxied_environment: None
    ) -> None:
        client = LocalModelClient(LocalModelConfig())
        for handler in client._opener().handlers:
            if type(handler).__name__ == "ProxyHandler":
                assert handler.proxies == {}

    def test_client_uses_the_stock_opener_only_when_guard_is_lifted(
        self, proxied_environment: None
    ) -> None:
        """Lifting the guard is a deliberate, recorded choice."""
        client = LocalModelClient(
            LocalModelConfig(
                base_url="https://api.example.com", allow_remote_endpoint=True
            )
        )
        proxies: dict[str, str] = {}
        for handler in client._opener().handlers:
            if type(handler).__name__ == "ProxyHandler":
                proxies.update(handler.proxies)
        assert proxies, "with the guard lifted, normal proxy behaviour applies"


class TestPeerVerification:
    """Second layer: verify the realized connection, not the configuration."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.0.53", "::1"])
    def test_loopback_peers_allowed(self, peer: str) -> None:
        sock = _FakeSocket(peer)
        _assert_peer_is_loopback(sock, "localhost")
        assert not sock.closed

    @pytest.mark.parametrize("peer", ["93.184.216.34", "10.0.0.5", "192.168.1.9"])
    def test_non_loopback_peers_blocked(self, peer: str) -> None:
        sock = _FakeSocket(peer)
        with pytest.raises(RemoteEndpointBlocked, match="NOT loopback"):
            _assert_peer_is_loopback(sock, "localhost")

    def test_socket_is_closed_before_raising(self) -> None:
        """Nothing may be sent over a connection that failed verification."""
        sock = _FakeSocket("93.184.216.34")
        with pytest.raises(RemoteEndpointBlocked):
            _assert_peer_is_loopback(sock, "localhost")
        assert sock.closed is True

    def test_unreadable_peer_is_treated_as_unsafe(self) -> None:
        class Unreadable(_FakeSocket):
            def getpeername(self) -> tuple[str, int]:
                raise OSError("no peer")

        sock = Unreadable("x")
        with pytest.raises(RemoteEndpointBlocked, match="unverifiable"):
            _assert_peer_is_loopback(sock, "localhost")
        assert sock.closed is True

    def test_manifest_records_the_guarantee(self) -> None:
        manifest = LocalModelConfig().manifest()
        assert manifest["endpoint_is_loopback"] is True
        assert manifest["proxy_disabled"] is True
        assert "verified_peer_addresses" in manifest


class TestContextWindow:
    """Prompts that would be silently truncated are refused instead."""

    def test_estimate_is_conservative(self) -> None:
        """Over-estimating is the safe direction."""
        text = "a" * 1000
        assert estimate_tokens(text) >= 1000 / CHARS_PER_TOKEN

    def test_prompt_within_the_window_passes(self) -> None:
        client = LocalModelClient(LocalModelConfig(num_ctx=8192, max_tokens=2048))
        info = client.check_prompt_fits(system="short", user="x" * 3000)
        assert info["headroom_tokens"] > 0
        assert info["num_ctx"] == 8192

    def test_oversized_prompt_is_refused_before_sending(self) -> None:
        client = LocalModelClient(LocalModelConfig(num_ctx=2048, max_tokens=512))
        with pytest.raises(ContextWindowExceeded, match="Refusing to send"):
            client.check_prompt_fits(system="s", user="x" * 100_000)

    def test_error_names_the_remedies(self) -> None:
        client = LocalModelClient(LocalModelConfig(num_ctx=2048, max_tokens=512))
        with pytest.raises(ContextWindowExceeded) as excinfo:
            client.check_prompt_fits(system="s", user="x" * 100_000)
        message = str(excinfo.value)
        assert "questions_per_call" in message
        assert "num_ctx" in message

    def test_response_space_is_reserved(self) -> None:
        """A prompt that fits the window but leaves no room to answer is refused."""
        client = LocalModelClient(LocalModelConfig(num_ctx=4096, max_tokens=3000))
        with pytest.raises(ContextWindowExceeded):
            client.check_prompt_fits(system="s", user="x" * 5000)

    def test_num_ctx_is_recorded(self) -> None:
        assert LocalModelConfig(num_ctx=16384).manifest()["num_ctx"] == 16384


def _doc(answers: list[Any], qtype: str = "MC") -> dict[str, Any]:
    """A minimal response document with the given answers."""
    questions = []
    for i, value in enumerate(answers):
        questions.append(
            {
                "QuestionID": f"QID{i}",
                "QuestionType": qtype,
                "QuestionText": "q",
                "Options": [str(n) for n in range(1, 6)],
                "Settings": {"Selector": "SAVR"},
                "Answers": {"SelectedByPosition": value},
            }
        )
    return {
        "Metadata": {},
        "Elements": [
            {
                "ElementType": "Block",
                "BlockName": "A",
                "BlockID": "BL_A",
                "Questions": questions,
            }
        ],
        "EmbeddedData": {},
    }


class TestItemAgreement:
    """Within-pair comparison: meaningful at n = 3, unlike a correlation."""

    def test_identical_answers_give_perfect_agreement(self) -> None:
        humans = {"p1": _doc([1, 2, 3]), "p2": _doc([4, 5, 1])}
        result = compute_item_agreement(humans, dict(humans))
        assert result.summary["exact_match_rate"] == pytest.approx(1.0)
        assert result.summary["mean_abs_deviation"] == pytest.approx(0.0)

    def test_disagreement_is_counted_per_item(self) -> None:
        humans = {"p1": _doc([1, 1, 1, 1])}
        twins = {"p1": _doc([1, 1, 2, 2])}
        result = compute_item_agreement(humans, twins)
        assert result.summary["exact_match_rate"] == pytest.approx(0.5)
        assert result.summary["mean_abs_deviation"] == pytest.approx(0.5)

    def test_within_one_rate_is_more_forgiving_than_exact(self) -> None:
        humans = {"p1": _doc([3, 3, 3])}
        twins = {"p1": _doc([2, 3, 4])}
        result = compute_item_agreement(humans, twins)
        assert result.summary["exact_match_rate"] == pytest.approx(1 / 3)
        assert result.summary["within_one_rate"] == pytest.approx(1.0)

    def test_works_with_three_participants(self) -> None:
        """The pilot case: few people, many items."""
        rng = random.Random(1)
        humans = {f"p{i}": _doc([rng.randint(1, 5) for _ in range(40)]) for i in range(3)}
        twins = {f"p{i}": _doc([rng.randint(1, 5) for _ in range(40)]) for i in range(3)}
        result = compute_item_agreement(humans, twins)
        assert result.n_comparisons == 120
        assert result.summary["n_participants"] == 3

    def test_reports_worst_items_first(self) -> None:
        humans = {"p1": _doc([1, 1]), "p2": _doc([1, 1])}
        twins = {"p1": _doc([1, 5]), "p2": _doc([1, 5])}
        result = compute_item_agreement(humans, twins)
        assert result.per_item.iloc[0]["item"] == "QID1"

    def test_requires_shared_participants(self) -> None:
        with pytest.raises(ValueError, match="both"):
            compute_item_agreement({"a": _doc([1])}, {"b": _doc([1])})


class TestStability:
    """Run-to-run variability: a property of the model, not of the sample."""

    def test_identical_runs_are_fully_stable(self) -> None:
        doc = _doc([1, 2, 3])
        result = compute_stability({"p1": [doc, doc, doc]})
        assert result.summary["modal_rate"] == pytest.approx(1.0)
        assert result.summary["fully_consistent_rate"] == pytest.approx(1.0)
        assert result.summary["mean_sd"] == pytest.approx(0.0)

    def test_varying_runs_reduce_the_modal_rate(self) -> None:
        result = compute_stability(
            {"p1": [_doc([1]), _doc([2]), _doc([3]), _doc([4])]}
        )
        assert result.summary["modal_rate"] == pytest.approx(0.25)
        assert result.summary["fully_consistent_rate"] == pytest.approx(0.0)

    def test_flags_unstable_items(self) -> None:
        result = compute_stability(
            {"p1": [_doc([1]), _doc([2]), _doc([3]), _doc([4]), _doc([5])]}
        )
        assert any("unstable" in w for w in result.warnings)

    def test_warns_when_everything_is_identical(self) -> None:
        """Usually means temperature 0, where stability is not informative."""
        doc = _doc([1, 2, 3])
        result = compute_stability({"p1": [doc, doc, doc]})
        assert any("temperature 0" in w for w in result.warnings)

    def test_requires_two_replicates(self) -> None:
        with pytest.raises(ValueError, match="at least 2 replicates"):
            compute_stability({"p1": [_doc([1])]})

    def test_records_replicate_count(self) -> None:
        result = compute_stability({"p1": [_doc([1])] * 5})
        assert result.n_replicates == 5


class TestItemExtraction:
    def test_multi_part_questions_split_into_items(self) -> None:
        """A matrix row is its own item, not averaged into the block."""
        doc = {
            "Elements": [
                {
                    "ElementType": "Block",
                    "BlockName": "A",
                    "Questions": [
                        {
                            "QuestionID": "QID1",
                            "QuestionType": "Matrix",
                            "Rows": ["a", "b", "c"],
                            "Columns": ["1", "2", "3"],
                            "Settings": {"Selector": "Likert"},
                            "Answers": {"SelectedByPosition": [1, 2, 3]},
                        }
                    ],
                }
            ]
        }
        values = extract_item_values(doc)
        assert values == {"QID1_1": 1, "QID1_2": 2, "QID1_3": 3}

    def test_single_answer_questions_keep_a_bare_key(self) -> None:
        assert set(extract_item_values(_doc([4]))) == {"QID0"}

    def test_blanked_answers_are_omitted(self) -> None:
        doc = _doc([1])
        doc["Elements"][0]["Questions"][0]["Answers"] = None
        assert extract_item_values(doc) == {}

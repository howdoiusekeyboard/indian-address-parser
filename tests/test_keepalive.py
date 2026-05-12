"""Tests for the HF Space keepalive decision and helpers.

Only the pure, side-effect-free pieces are tested here. The HTTP and GitHub-
Actions glue is verified by the workflow's manual `force_ping` dispatch.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from datetime import UTC, datetime, timedelta

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / ".github" / "scripts"))

import keepalive  # noqa: E402

NOW = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "elapsed_hours,target_hours,expected",
    [
        (0.0, 46.5, False),
        (1.0, 46.5, False),
        (30.0, 46.5, False),
        (46.0, 46.5, False),
        (46.4999, 46.5, False),
        (46.5, 46.5, True),
        (46.6, 46.5, True),
        (47.0, 46.0, True),
        (100.0, 46.5, True),
    ],
)
def test_should_ping_threshold(elapsed_hours: float, target_hours: float, expected: bool) -> None:
    last_ping = NOW - timedelta(hours=elapsed_hours)
    assert keepalive.should_ping(NOW, last_ping, target_hours) is expected


def test_should_ping_no_history_fires() -> None:
    assert keepalive.should_ping(NOW, None, 46.5) is True


def test_should_ping_future_timestamp_fails_open() -> None:
    future = NOW + timedelta(hours=1)
    assert keepalive.should_ping(NOW, future, 46.5) is True


def test_pick_target_hours_in_range() -> None:
    rng = random.Random(0)
    for _ in range(2000):
        value = keepalive.pick_target_hours(rng)
        assert keepalive.MIN_INTERVAL_HOURS <= value < keepalive.MAX_INTERVAL_HOURS


def test_pick_target_hours_covers_range() -> None:
    rng = random.Random(0)
    samples = [keepalive.pick_target_hours(rng) for _ in range(2000)]
    assert min(samples) < keepalive.MIN_INTERVAL_HOURS + 0.05
    assert max(samples) > keepalive.MAX_INTERVAL_HOURS - 0.05


def test_pick_user_agent_from_pool() -> None:
    rng = random.Random(0)
    for _ in range(50):
        assert keepalive.pick_user_agent(rng) in keepalive.USER_AGENTS


def test_pick_accept_language_from_pool() -> None:
    rng = random.Random(0)
    for _ in range(50):
        assert keepalive.pick_accept_language(rng) in keepalive.ACCEPT_LANGUAGES


def test_build_headers_includes_required_browserish_fields() -> None:
    headers = keepalive.build_headers("UA/1.0", "en-US,en;q=0.9")
    assert headers["User-Agent"] == "UA/1.0"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    for key in (
        "Accept",
        "Accept-Encoding",
        "Sec-Fetch-Site",
        "Sec-Fetch-Mode",
        "Sec-Fetch-Dest",
        "Upgrade-Insecure-Requests",
    ):
        assert key in headers


def test_load_last_ping_missing_file(tmp_path: pathlib.Path) -> None:
    assert keepalive.load_last_ping(tmp_path / "nope.json") is None


def test_load_last_ping_empty_payload(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"last_ping": None}), encoding="utf-8")
    assert keepalive.load_last_ping(state) is None


def test_load_last_ping_corrupt_json(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("not-json{{", encoding="utf-8")
    assert keepalive.load_last_ping(state) is None


def test_load_last_ping_invalid_iso(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"last_ping": "not-a-date"}), encoding="utf-8")
    assert keepalive.load_last_ping(state) is None


def test_load_and_save_round_trip(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state.json"
    keepalive.save_last_ping(state, NOW)
    loaded = keepalive.load_last_ping(state)
    assert loaded == NOW


def test_load_last_ping_normalizes_to_utc(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state.json"
    naive_offset = "2026-05-11T17:30:00+05:30"
    state.write_text(json.dumps({"last_ping": naive_offset}), encoding="utf-8")
    loaded = keepalive.load_last_ping(state)
    assert loaded is not None
    assert loaded.tzinfo == UTC
    assert loaded == datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)

"""HF Space keepalive ping with a randomized 46-47 hour interval.

Hourly GitHub Actions cron runs this script. The script reads the last-ping
timestamp from a committed state file, picks a fresh random target interval
in [46h, 47h), and only fires an HTTP request to the Space when the elapsed
time has crossed the target. This produces a true uniform-random ping
distribution in [46h, 47h) — well clear of HF's 48h sleep timer — without
trying to express that interval in cron itself.

Stdlib-only so the workflow needs no `pip install` step.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL",
    "https://howdoiuse-keyboard-indian-address-parser.hf.space",
)
STATE_FILE = pathlib.Path(__file__).resolve().parent.parent / "keepalive_state.json"

MIN_INTERVAL_HOURS = 46.0
MAX_INTERVAL_HOURS = 47.0
MAX_JITTER_SECONDS = 1800
HTTP_TIMEOUT_SECONDS = 180.0

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
]

ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.8,hi;q=0.6",
]


def should_ping(
    now: datetime,
    last_ping: datetime | None,
    target_hours: float,
) -> bool:
    """Decide whether to fire a ping this run.

    Fail-open: a missing or future-dated last_ping causes a ping, so a corrupt
    or never-initialized state file always wakes the Space.
    """
    if last_ping is None:
        return True
    elapsed = now - last_ping
    if elapsed < timedelta(0):
        return True
    return elapsed >= timedelta(hours=target_hours)


def pick_target_hours(rng: random.Random | None = None) -> float:
    rng = rng or secrets.SystemRandom()
    return MIN_INTERVAL_HOURS + rng.random() * (MAX_INTERVAL_HOURS - MIN_INTERVAL_HOURS)


def pick_user_agent(rng: random.Random | None = None) -> str:
    rng = rng or secrets.SystemRandom()
    return rng.choice(USER_AGENTS)


def pick_accept_language(rng: random.Random | None = None) -> str:
    rng = rng or secrets.SystemRandom()
    return rng.choice(ACCEPT_LANGUAGES)


def build_headers(user_agent: str, accept_language: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": accept_language,
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Connection": "close",
    }


def load_last_ping(state_path: pathlib.Path) -> datetime | None:
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("last_ping")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def save_last_ping(state_path: pathlib.Path, when: datetime) -> None:
    state_path.write_text(
        json.dumps({"last_ping": when.isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def ping_space(
    url: str, headers: dict[str, str], timeout: float = HTTP_TIMEOUT_SECONDS
) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(
        req, timeout=timeout
    ) as resp:  # noqa: S310 — fixed scheme, headers controlled
        code = int(resp.status)
        head = resp.read(256)
        try:
            preview = head.decode("utf-8", errors="replace").splitlines()[0]
        except IndexError:
            preview = ""
        return code, preview


def emit_github_output(**pairs: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for key, value in pairs.items():
            safe = str(value).replace("\n", " ")
            fh.write(f"{key}={safe}\n")


def main() -> int:
    now = datetime.now(UTC)
    last_ping = load_last_ping(STATE_FILE)
    rng = secrets.SystemRandom()
    target_hours = pick_target_hours(rng)
    force = os.environ.get("FORCE_PING", "").lower() == "true"

    print(f"[keepalive] now={now.isoformat()}")
    print(f"[keepalive] last_ping={last_ping.isoformat() if last_ping else 'never'}")
    print(f"[keepalive] target_interval={target_hours:.4f}h")
    if last_ping:
        elapsed_h = (now - last_ping).total_seconds() / 3600
        print(f"[keepalive] elapsed={elapsed_h:.4f}h")

    if not force and not should_ping(now, last_ping, target_hours):
        print("[keepalive] decision: skip")
        emit_github_output(pinged="false")
        return 0

    print(f"[keepalive] decision: ping{' (forced)' if force else ''}")

    max_jitter = int(os.environ.get("KEEPALIVE_MAX_JITTER_SECONDS", str(MAX_JITTER_SECONDS)))
    jitter_s = rng.randint(0, max(0, max_jitter))
    print(f"[keepalive] pre-ping jitter: {jitter_s}s")
    if jitter_s:
        time.sleep(jitter_s)

    ua = pick_user_agent(rng)
    al = pick_accept_language(rng)
    headers = build_headers(ua, al)
    print(f"[keepalive] UA={ua[:60]}...")
    print(f"[keepalive] Accept-Language={al}")

    try:
        code, preview = ping_space(HF_SPACE_URL, headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[keepalive] ERROR: request failed: {exc}", file=sys.stderr)
        emit_github_output(pinged="false", error=str(exc)[:200])
        return 1

    print(f"[keepalive] HTTP {code} body[0]={preview!r}")

    if not (200 <= code < 300):
        print(f"[keepalive] ERROR: non-2xx status {code}", file=sys.stderr)
        emit_github_output(pinged="false", error=f"status={code}")
        return 1

    save_last_ping(STATE_FILE, now)
    emit_github_output(pinged="true", new_timestamp=now.isoformat())
    print(f"[keepalive] state saved: last_ping={now.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

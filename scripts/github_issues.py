"""
github_issues.py — Backend-less issue tracking for locale reports.

Reports are tracked as GitHub Issues so that both my own reports and other
people's reports live in one transparent, traceable place — without running a
custom backend or database.

- Creating a report builds a *prefilled* "new issue" URL. The user reviews and
  submits it on GitHub (no token ever leaves the machine / browser).
- Listing reports reads the public Issues API (read-only, no token required for
  public repositories).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import os
import time
import urllib.parse
import urllib.request

from scripts.utils import setup_logging

logger = setup_logging("github_issues")

# Configurable so forks/self-hosted can override without code changes.
GITHUB_REPO = os.environ.get("CIBOBUONO_GH_REPO", "lucaosti/cibobuono")
LOCALE_REPORT_LABEL = "locale-report"

_API_BASE = "https://api.github.com"
_HTML_BASE = "https://github.com"

# Short-lived cache to stay well under the unauthenticated rate limit.
_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 120.0


def new_issue_url(title: str, body: str, labels: list[str] | None = None) -> str:
    """Build a GitHub 'new issue' URL with prefilled fields.

    The user confirms and submits on GitHub — no API token needed.
    """
    params = {"title": title, "body": body}
    if labels:
        params["labels"] = ",".join(labels)
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"{_HTML_BASE}/{GITHUB_REPO}/issues/new?{query}"


def report_issue_url(
    locale_name: str,
    reason: str,
    video_id: str = "",
    youtube_url: str = "",
) -> str:
    """Prefilled issue URL for reporting an incorrect locale."""
    title = f"[locale] {locale_name or 'segnalazione'}"
    lines = [
        f"**Locale segnalato:** {locale_name or '(non specificato)'}",
        "",
        f"**Motivo:** {reason or '(non specificato)'}",
        "",
    ]
    if youtube_url:
        lines.append(f"**Video:** {youtube_url}")
    elif video_id:
        lines.append(f"**Video ID:** {video_id}")
    lines += ["", "---", "_Report created from the CiboBuono dashboard._"]
    return new_issue_url(title, "\n".join(lines), [LOCALE_REPORT_LABEL])


def list_reports(limit: int = 50, state: str = "all") -> list[dict]:
    """Read locale-report issues (mine + everyone's) from the public API."""
    key = f"{state}:{LOCALE_REPORT_LABEL}"
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1][:limit]

    params = urllib.parse.urlencode(
        {
            "labels": LOCALE_REPORT_LABEL,
            "state": state,
            "per_page": min(max(limit, 1), 100),
            "sort": "created",
            "direction": "desc",
        }
    )
    url = f"{_API_BASE}/repos/{GITHUB_REPO}/issues?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "cibobuono-dashboard",
        },
    )
    items: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        for it in raw:
            if "pull_request" in it:  # the issues API also returns PRs
                continue
            items.append(
                {
                    "number": it.get("number"),
                    "title": it.get("title", ""),
                    "state": it.get("state", ""),
                    "url": it.get("html_url", ""),
                    "user": (it.get("user") or {}).get("login", ""),
                    "created_at": it.get("created_at", ""),
                    "comments": it.get("comments", 0),
                }
            )
        _cache[key] = (now, items)
    except Exception as e:
        logger.warning("GitHub issues fetch failed: %s", e)
        if cached:
            return cached[1][:limit]
    return items[:limit]


def issues_page_url(state: str = "open") -> str:
    label = urllib.parse.quote(LOCALE_REPORT_LABEL)
    return f"{_HTML_BASE}/{GITHUB_REPO}/issues?q=is%3Aissue+label%3A{label}+state%3A{state}"

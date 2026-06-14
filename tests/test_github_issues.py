"""Tests for github_issues: URL building and in-memory cache logic."""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
from unittest.mock import MagicMock, patch
import urllib.parse


class TestNewIssueUrl:
    def test_title_and_body_encoded(self):
        from scripts.github_issues import new_issue_url

        url = new_issue_url("Test title", "Test body")
        assert "title=Test+title" in url or "title=Test%20title" in url
        assert "body=" in url

    def test_labels_included_when_given(self):
        from scripts.github_issues import new_issue_url

        url = new_issue_url("T", "B", labels=["locale-report", "bug"])
        assert "labels=" in url
        assert "locale-report" in url

    def test_no_labels_param_when_omitted(self):
        from scripts.github_issues import new_issue_url

        url = new_issue_url("T", "B")
        assert "labels=" not in url

    def test_url_points_to_correct_repo(self):
        from scripts.github_issues import new_issue_url, GITHUB_REPO

        url = new_issue_url("T", "B")
        assert GITHUB_REPO in url
        assert "issues/new" in url


class TestReportIssueUrl:
    def test_contains_locale_name(self):
        from scripts.github_issues import report_issue_url

        url = report_issue_url("Da Remo", "Nome sbagliato")
        assert "Da+Remo" in url or "Da%20Remo" in url or "Da Remo" in urllib.parse.unquote(url)

    def test_contains_youtube_url_when_provided(self):
        from scripts.github_issues import report_issue_url

        url = report_issue_url("Da Remo", "motivo", youtube_url="https://youtu.be/abc123")
        decoded = urllib.parse.unquote(url)
        assert "youtu.be/abc123" in decoded

    def test_falls_back_to_video_id(self):
        from scripts.github_issues import report_issue_url

        url = report_issue_url("Da Remo", "motivo", video_id="abc123")
        decoded = urllib.parse.unquote(url)
        assert "abc123" in decoded

    def test_locale_report_label_applied(self):
        from scripts.github_issues import report_issue_url, LOCALE_REPORT_LABEL

        url = report_issue_url("Da Remo", "motivo")
        assert LOCALE_REPORT_LABEL in urllib.parse.unquote(url)


class TestListReports:
    def _fake_response(self, items: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(items).encode("utf-8")
        return mock_resp

    def test_returns_issues_from_api(self):
        from scripts import github_issues as gi

        gi._cache.clear()
        items = [
            {"number": 1, "title": "Wrong locale", "state": "open",
             "html_url": "https://github.com/x/y/issues/1",
             "user": {"login": "user1"}, "created_at": "2024-01-01", "comments": 0},
        ]
        with patch("urllib.request.urlopen", return_value=self._fake_response(items)):
            result = gi.list_reports()
        assert len(result) == 1
        assert result[0]["number"] == 1
        gi._cache.clear()

    def test_pulls_requests_filtered_out(self):
        from scripts import github_issues as gi

        gi._cache.clear()
        items = [
            {"number": 1, "title": "Issue", "state": "open",
             "html_url": "https://github.com/x/y/issues/1",
             "user": {"login": "user1"}, "created_at": "2024-01-01", "comments": 0},
            {"number": 2, "title": "PR", "state": "open", "pull_request": {},
             "html_url": "https://github.com/x/y/pull/2",
             "user": {"login": "user2"}, "created_at": "2024-01-01", "comments": 0},
        ]
        with patch("urllib.request.urlopen", return_value=self._fake_response(items)):
            result = gi.list_reports()
        assert all("pull_request" not in r for r in result)
        assert len(result) == 1
        gi._cache.clear()

    def test_returns_cached_result_on_second_call(self):
        from scripts import github_issues as gi

        gi._cache.clear()
        items = [{"number": 99, "title": "Cached", "state": "open",
                  "html_url": "u", "user": {"login": "u"}, "created_at": "x", "comments": 0}]
        call_count = {"n": 0}

        def _fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            return self._fake_response(items)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            gi.list_reports()
            gi.list_reports()

        assert call_count["n"] == 1  # second call used cache
        gi._cache.clear()

    def test_returns_stale_cache_on_network_error(self):
        from scripts import github_issues as gi
        import time

        gi._cache.clear()
        items = [{"number": 5, "title": "Stale", "state": "open",
                  "html_url": "u", "user": {"login": "u"}, "created_at": "x", "comments": 0}]
        key = f"all:{gi.LOCALE_REPORT_LABEL}"  # matches default state="all"
        gi._cache[key] = (time.time() - 9999, items)  # expired cache

        with patch("urllib.request.urlopen", side_effect=Exception("network down")):
            result = gi.list_reports()

        assert len(result) == 1
        assert result[0]["number"] == 5
        gi._cache.clear()

    def test_limit_respected(self):
        from scripts import github_issues as gi

        gi._cache.clear()
        items = [
            {"number": i, "title": f"Issue {i}", "state": "open",
             "html_url": "u", "user": {"login": "u"}, "created_at": "x", "comments": 0}
            for i in range(20)
        ]
        with patch("urllib.request.urlopen", return_value=self._fake_response(items)):
            result = gi.list_reports(limit=5)
        assert len(result) == 5
        gi._cache.clear()


class TestIssuesPageUrl:
    def test_state_encoded_in_url(self):
        from scripts.github_issues import issues_page_url

        url = issues_page_url("closed")
        assert "closed" in url

    def test_label_in_url(self):
        from scripts.github_issues import issues_page_url, LOCALE_REPORT_LABEL

        url = issues_page_url()
        assert LOCALE_REPORT_LABEL in urllib.parse.unquote(url)

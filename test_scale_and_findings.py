#!/usr/bin/env python3
"""
Regression tests for the scaling fixes and the takeover / CORS checks.

The scaling tests exist because the failure they guard against is silent: the
tool stayed correct as the dataset grew, it just got slower and heavier until
it stopped responding. Each test pins the property that broke.
"""

import json
import os
import sqlite3
import sys
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

with patch('main.ensure_dirs'), \
     patch('main.init_database'), \
     patch('main.migrate_json_to_sqlite'):
    import main  # noqa: E402


# --------------------------------------------------------------------------
# Job snapshots
# --------------------------------------------------------------------------

def _make_job(domain, status, log_lines=400):
    return {
        "domain": domain,
        "thread": None,
        "started": "2026-01-01T00:00:00+00:00",
        "queued_at": "2026-01-01T00:00:00+00:00",
        "wordlist": "",
        "skip_nikto": False,
        "interval": 30,
        "status": status,
        "message": "",
        "progress": 0,
        "last_update": "2026-01-01T00:00:00+00:00",
        "steps": main.init_job_steps(False),
        "logs": [{"ts": "t", "source": "s", "text": "x" * 200} for _ in range(log_lines)],
    }


class TestJobSnapshotScaling:
    def setup_method(self):
        self._jobs = dict(main.RUNNING_JOBS)
        self._done = dict(main.COMPLETED_JOBS)
        self._queue = list(main.JOB_QUEUE)
        main.RUNNING_JOBS.clear()
        main.COMPLETED_JOBS.clear()
        main.JOB_QUEUE.clear()

    def teardown_method(self):
        main.RUNNING_JOBS.clear(); main.RUNNING_JOBS.update(self._jobs)
        main.COMPLETED_JOBS.clear(); main.COMPLETED_JOBS.update(self._done)
        main.JOB_QUEUE.clear(); main.JOB_QUEUE.extend(self._queue)

    def test_queued_jobs_are_not_shipped_in_the_job_snapshot(self):
        """
        The dashboard filters queued jobs out of the job list, so sending them —
        each with a full step tree and log tail — was pure waste. With a large
        backlog it was the single biggest item in the response.
        """
        main.RUNNING_JOBS["live.com"] = _make_job("live.com", "running")
        for i in range(2000):
            domain = f"q{i}.com"
            main.RUNNING_JOBS[domain] = _make_job(domain, "queued")
            main.JOB_QUEUE.append(domain)

        snapshot = main.snapshot_running_jobs()

        assert [job["domain"] for job in snapshot] == ["live.com"]
        assert all(job["status"] != "queued" for job in snapshot)

    def test_snapshot_payload_stays_small_with_a_large_backlog(self):
        for i in range(5000):
            domain = f"q{i}.com"
            main.RUNNING_JOBS[domain] = _make_job(domain, "queued")
            main.JOB_QUEUE.append(domain)
        main.RUNNING_JOBS["live.com"] = _make_job("live.com", "running")

        payload = json.dumps({
            "running_jobs": main.snapshot_running_jobs(),
            "queued_jobs": main.job_queue_snapshot(),
            "queued_total": main.count_queued_jobs(),
        })
        # Before the fix this was hundreds of megabytes.
        assert len(payload) < 2_000_000, f"payload grew to {len(payload)} bytes"

    def test_queue_snapshot_is_capped_but_total_is_reported(self):
        for i in range(1000):
            domain = f"q{i}.com"
            main.RUNNING_JOBS[domain] = _make_job(domain, "queued", log_lines=0)
            main.JOB_QUEUE.append(domain)

        assert len(main.job_queue_snapshot()) == main.MAX_QUEUE_IN_SNAPSHOT
        # The UI still needs the real depth to display.
        assert main.count_queued_jobs() == 1000

    def test_job_logs_are_tail_capped(self):
        main.RUNNING_JOBS["live.com"] = _make_job("live.com", "running", log_lines=5000)
        logs = main.snapshot_running_jobs()[0]["logs"]
        assert len(logs) == main.MAX_LOGS_IN_SNAPSHOT

    def test_completed_jobs_are_capped_and_newest_first(self):
        for i in range(200):
            main.COMPLETED_JOBS[f"d{i}.com_{i}"] = {
                **_make_job(f"d{i}.com", "completed", log_lines=0),
                "completed_at": f"2026-01-01T00:00:{i:02d}+00:00",
            }
        snapshot = main.snapshot_running_jobs()
        assert len(snapshot) <= main.MAX_JOBS_IN_SNAPSHOT
        completed_times = [job["completed_at"] for job in snapshot]
        assert completed_times == sorted(completed_times, reverse=True)

    def test_queue_refuses_work_beyond_the_cap(self):
        """An unbounded queue exhausts memory long before the backlog drains."""
        # A worker only counts as busy while its thread is alive, so the cap
        # only applies once every slot is genuinely occupied.
        release = threading.Event()
        busy = threading.Thread(target=release.wait, daemon=True)
        busy.start()
        running = _make_job("running.com", "running", log_lines=0)
        running["thread"] = busy
        main.RUNNING_JOBS["running.com"] = running
        for i in range(main.MAX_QUEUED_JOBS):
            domain = f"q{i}.com"
            main.RUNNING_JOBS[domain] = _make_job(domain, "queued", log_lines=0)
            main.JOB_QUEUE.append(domain)

        # start_pipeline_job re-derives MAX_RUNNING_JOBS from config, so the
        # patched value only survives if that re-derivation is stubbed too.
        # _start_job_thread is stubbed because letting a job through here would
        # launch real recon subprocesses.
        with patch.object(main, "apply_concurrency_limits"), \
             patch.object(main, "MAX_RUNNING_JOBS", 1), \
             patch.object(main, "_start_job_thread") as start_thread, \
             patch.object(main, "request_active_jobs_persist"):
            ok, message = main.start_pipeline_job("overflow.com", None, False, None)

        assert ok is False
        assert "Queue is full" in message
        assert "overflow.com" not in main.RUNNING_JOBS
        start_thread.assert_not_called()
        release.set()
        busy.join(timeout=5)


# --------------------------------------------------------------------------
# State scoping and dirty tracking
# --------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "recon.db"
    with patch.object(main, "DB_FILE", db_file), patch.object(main, "DB_CONN", None):
        main.init_database()
        yield db_file
        conn = main.DB_CONN
        if conn is not None:
            conn.close()
        main.DB_CONN = None


def _seed(domains):
    state = {"targets": {}}
    for domain, subs in domains.items():
        target = main.ensure_target_state(state, domain)
        main.add_subdomains_to_state(state, domain, subs, "test")
    with patch.object(main, "request_dashboard_refresh", lambda: None):
        main.save_state(state)


class TestScopedState:
    def test_scoped_load_returns_only_that_target(self, temp_db):
        _seed({"a.com": ["x.a.com"], "b.com": ["y.b.com"]})
        scoped = main.load_state("a.com")
        assert set(scoped["targets"]) == {"a.com"}
        assert scoped["_scope"] == "a.com"
        # The full load still sees everything.
        assert set(main.load_state()["targets"]) == {"a.com", "b.com"}

    def test_saving_a_scoped_state_does_not_delete_other_targets(self, temp_db):
        """
        save_state() writes whatever targets the state holds. A scoped state
        must therefore leave every other target untouched, not treat it as
        deleted.
        """
        _seed({"a.com": ["x.a.com"], "b.com": ["y.b.com"]})
        scoped = main.load_state("a.com")
        scoped["targets"]["a.com"]["subdomains"]["new.a.com"] = main.make_subdomain_entry()
        with patch.object(main, "request_dashboard_refresh", lambda: None):
            main.save_state(scoped)

        full = main.load_state()
        assert set(full["targets"]) == {"a.com", "b.com"}
        assert "new.a.com" in full["targets"]["a.com"]["subdomains"]
        assert "y.b.com" in full["targets"]["b.com"]["subdomains"]

    def test_unchanged_subdomains_are_not_rewritten(self, temp_db):
        _seed({"a.com": [f"h{i}.a.com" for i in range(50)]})
        state = main.load_state("a.com")

        written = []
        real_db = main.get_db()

        class SpyCursor:
            def __init__(self, inner):
                self._inner = inner

            def executemany(self, sql, rows):
                if "INTO subdomains" in sql:
                    written.extend(rows)
                return self._inner.executemany(sql, rows)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        class SpyDb:
            def cursor(self):
                return SpyCursor(real_db.cursor())

            def __getattr__(self, name):
                return getattr(real_db, name)

        state["targets"]["a.com"]["subdomains"]["h7.a.com"]["probe"] = 1
        with patch.object(main, "get_db", lambda: SpyDb()), \
             patch.object(main, "request_dashboard_refresh", lambda: None):
            main.save_state(state)

        # Exactly one row touched, out of 50 in the target.
        assert [row[1] for row in written] == ["h7.a.com"]

    def test_changes_still_persist(self, temp_db):
        _seed({"a.com": ["h1.a.com", "h2.a.com"]})
        state = main.load_state("a.com")
        state["targets"]["a.com"]["subdomains"]["h1.a.com"]["probe"] = "kept"
        with patch.object(main, "request_dashboard_refresh", lambda: None):
            main.save_state(state)
        reloaded = main.load_state("a.com")
        assert reloaded["targets"]["a.com"]["subdomains"]["h1.a.com"]["probe"] == "kept"

    def test_removing_a_subdomain_still_deletes_it(self, temp_db):
        _seed({"a.com": ["h1.a.com", "h2.a.com"]})
        state = main.load_state("a.com")
        del state["targets"]["a.com"]["subdomains"]["h2.a.com"]
        with patch.object(main, "request_dashboard_refresh", lambda: None):
            main.save_state(state)
        assert set(main.load_state("a.com")["targets"]["a.com"]["subdomains"]) == {"h1.a.com"}

    def test_state_export_is_json_serialisable(self, temp_db):
        """load_state() carries internal keys that must not reach /api/state."""
        _seed({"a.com": ["h1.a.com"]})
        exported = main.public_state(main.load_state())
        assert "_digests" not in exported and "_scope" not in exported
        json.dumps(exported)  # would raise if internals leaked


class TestHistoryBuffering:
    def test_buffered_lines_are_visible_to_readers(self, temp_db):
        main.append_domain_history("a.com", {"ts": "t", "source": "s", "text": "buffered line"})
        # Readers flush first, so a line is never invisible just because it is
        # still in the buffer.
        entries = main.load_domain_history("a.com")
        assert any(e.get("text") == "buffered line" for e in entries)

    def test_flush_is_idempotent(self, temp_db):
        main.append_domain_history("a.com", {"ts": "t", "source": "s", "text": "one"})
        main.flush_domain_history()
        main.flush_domain_history()
        texts = [e.get("text") for e in main.load_domain_history("a.com")]
        assert texts.count("one") == 1


# --------------------------------------------------------------------------
# Subdomain takeover
# --------------------------------------------------------------------------

class TestTakeoverDetection:
    def test_matches_known_services(self):
        assert main.match_takeover_service(["shop.myshopify.com"])["service"] == "Shopify"
        assert main.match_takeover_service(["user.github.io"])["service"] == "GitHub Pages"
        assert main.match_takeover_service(["bucket.s3.amazonaws.com"])["service"] == "AWS/S3"

    def test_ignores_unrelated_and_empty_cnames(self):
        assert main.match_takeover_service(["cdn.cloudflare.net"]) is None
        assert main.match_takeover_service([]) is None

    def test_live_service_is_not_reported(self):
        """
        A CNAME to a *live* Shopify store also points at shopify.com. Reporting
        that is exactly the unverified scanner output that gets reports closed,
        so the fingerprint has to be present before anything is claimed.
        """
        with patch.object(main, "_takeover_probe", return_value="<html>Our store</html>"):
            assert main.check_subdomain_takeover("shop.x.com", ["shop.myshopify.com"]) is None

    def test_dangling_service_is_confirmed(self):
        body = "Sorry, this shop is currently unavailable"
        with patch.object(main, "_takeover_probe", return_value=body):
            finding = main.check_subdomain_takeover("shop.x.com", ["shop.myshopify.com"])
        assert finding["confirmed"] is True
        assert finding["severity"] == "HIGH"
        assert finding["service"] == "Shopify"

    def test_unreachable_host_is_a_lead_not_a_finding(self):
        with patch.object(main, "_takeover_probe", return_value=None):
            finding = main.check_subdomain_takeover("shop.x.com", ["shop.myshopify.com"])
        assert finding["confirmed"] is False
        assert finding["severity"] == "INFO"

    def test_scan_persists_and_clears_findings(self, temp_db):
        _seed({"a.com": ["shop.a.com"]})
        state = main.load_state("a.com")
        state["targets"]["a.com"]["subdomains"]["shop.a.com"]["httpx"] = {
            "url": "https://shop.a.com", "status_code": 404, "cname": ["shop.myshopify.com"],
        }
        with patch.object(main, "request_dashboard_refresh", lambda: None):
            main.save_state(state)

        body = "Sorry, this shop is currently unavailable"
        with patch.object(main, "_takeover_probe", return_value=body), \
             patch.object(main, "request_dashboard_refresh", lambda: None):
            result = main.run_takeover_scan("a.com", {})
        assert result["summary"]["confirmed"] == 1
        stored = main.load_state("a.com")["targets"]["a.com"]
        assert stored["subdomains"]["shop.a.com"]["takeover"]["confirmed"] is True

        # Re-running after the delegation is fixed must clear the stale finding.
        with patch.object(main, "_takeover_probe", return_value="<html>live</html>"), \
             patch.object(main, "request_dashboard_refresh", lambda: None):
            main.run_takeover_scan("a.com", {})
        stored = main.load_state("a.com")["targets"]["a.com"]
        assert "takeover" not in stored["subdomains"]["shop.a.com"]


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------

def _cors_responder(mapping):
    def probe(url, origin, timeout=main.CORS_PROBE_TIMEOUT):
        return mapping.get(origin)
    return probe


class TestCorsDetection:
    ATTACKER = "https://brutsec-cors-probe.example.com"

    def test_reflected_origin_with_credentials_is_exploitable(self):
        responder = _cors_responder({
            self.ATTACKER: {"allow_origin": self.ATTACKER, "allow_credentials": "true"}})
        with patch.object(main, "_cors_probe", responder):
            finding = main.check_cors_misconfiguration("https://a.x.com", "a.x.com")
        assert finding["exploitable"] is True
        assert finding["severity"] == "HIGH"

    def test_reflected_origin_without_credentials_is_informational(self):
        """Without credentials an attacker only reads what anyone could read."""
        responder = _cors_responder({
            self.ATTACKER: {"allow_origin": self.ATTACKER, "allow_credentials": ""}})
        with patch.object(main, "_cors_probe", responder):
            finding = main.check_cors_misconfiguration("https://a.x.com", "a.x.com")
        assert finding["exploitable"] is False
        assert finding["severity"] == "INFO"

    def test_wildcard_with_credentials_is_not_claimed_as_exploitable(self):
        """Browsers refuse `*` plus credentials, so it is a smell, not a read."""
        responder = _cors_responder({
            self.ATTACKER: {"allow_origin": "*", "allow_credentials": "true"}})
        with patch.object(main, "_cors_probe", responder):
            finding = main.check_cors_misconfiguration("https://a.x.com", "a.x.com")
        assert finding["exploitable"] is False
        assert finding["severity"] == "LOW"

    def test_correct_configuration_produces_no_finding(self):
        responder = _cors_responder({
            self.ATTACKER: {"allow_origin": "https://a.x.com", "allow_credentials": "true"}})
        with patch.object(main, "_cors_probe", responder):
            assert main.check_cors_misconfiguration("https://a.x.com", "a.x.com") is None

    def test_absent_cors_headers_produce_no_finding(self):
        with patch.object(main, "_cors_probe", _cors_responder({})):
            assert main.check_cors_misconfiguration("https://a.x.com", "a.x.com") is None

    def test_unanchored_suffix_allowlist_is_caught(self):
        suffix_origin = "https://a.x.com.brutsec-cors-probe.example.com"
        responder = _cors_responder({
            suffix_origin: {"allow_origin": suffix_origin, "allow_credentials": "true"}})
        with patch.object(main, "_cors_probe", responder):
            finding = main.check_cors_misconfiguration("https://a.x.com", "a.x.com")
        assert finding["exploitable"] is True
        assert finding["issue"] == "origin suffix not anchored"

    def test_null_origin_is_probed(self):
        responder = _cors_responder({
            "null": {"allow_origin": "null", "allow_credentials": "true"}})
        with patch.object(main, "_cors_probe", responder):
            finding = main.check_cors_misconfiguration("https://a.x.com", "a.x.com")
        assert finding["exploitable"] is True
        assert finding["issue"] == "null origin allowed"


class TestPipelineWiring:
    def test_new_steps_are_part_of_the_pipeline(self):
        assert "takeover" in main.PIPELINE_STEPS
        assert "cors" in main.PIPELINE_STEPS
        # They run after httpx, whose CNAME/URL output they consume.
        assert main.PIPELINE_STEPS.index("takeover") > main.PIPELINE_STEPS.index("httpx")
        assert main.PIPELINE_STEPS.index("cors") > main.PIPELINE_STEPS.index("httpx")

    def test_new_steps_are_toggleable(self):
        cfg = main.default_config()
        assert cfg["enable_takeover_scan"] is True
        assert cfg["enable_cors_scan"] is True

    def test_httpx_requests_cname(self):
        """The takeover check reads the CNAME chain httpx records."""
        with tempfile.TemporaryDirectory() as tmp:
            subs = Path(tmp) / "subs.txt"
            subs.write_text("a.x.com\n")
            captured = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                return False

            with patch.object(main, "ensure_tool_installed", return_value=True), \
                 patch.object(main, "run_subprocess", fake_run), \
                 patch.object(main, "DATA_DIR", Path(tmp)):
                main.httpx_scan(subs, "x.com")
        assert "-cname" in captured["cmd"]

"""Regression tests for the POST /ingest timestamp window.

The X-Webhook-Timestamp header used to be parsed with a bare ``float()``
before the +/-300s skew check. That accepts non-finite values:

- ``NaN`` parses successfully and poisons the window check -- every
  comparison against a NaN is false, so ``abs(now - nan) > tolerance`` is
  false and an unsigned/forged request proceeded to HMAC verification (and,
  with a valid signature, could be persisted);
- ``Infinity``/``-Infinity`` and decimal overflow (``1e999`` parses to
  ``inf``) are equally non-finite.

These tests pin the contract:

- the timestamp must be a *finite* decimal Unix-seconds value -- NaN, +/-
  Infinity (in any capitalization) and values that overflow float all answer
  400, before signature verification and before anything is persisted;
- a finite value is still judged against the +/-300s window (inside -> the
  request proceeds, outside -> 401), including both boundary sides;
- the original header text (after the endpoint's surrounding-whitespace
  strip) is what participates in the HMAC: no numeric normalization (e.g.
  ``1700000000`` -> ``1700000000.0``) happens, so existing senders stay
  signature-compatible;
- the signed happy path still ingests (201) and replays idempotently (200).

Pure stdlib + the app's own FastAPI test client; run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
import time
import unittest

from fastapi import HTTPException

# A private temp database must be selected before app.config (and hence
# app.main) is first imported, since config reads the environment at import.
_TMP_ROOT = tempfile.mkdtemp(prefix="wr-ingest-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP_ROOT, "default.db"))
os.environ.setdefault("INGEST_SECRET", "test-ingest-secret")
# Delivery targets never accept a connection (port 1); the tests only assert
# task creation, never actual delivery.
os.environ.setdefault("WORKER_COUNT", "2")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
import app.main as main  # noqa: E402

SECRET = "test-ingest-secret"
HOOK_URL = "http://127.0.0.1:1/hook"
# Stay a safe margin inside/outside the window; the skew comparison uses the
# live clock, so the exact +/-300s edge is pinned on the parser helper while
# these end-to-end cases avoid timing flake.
WINDOW_TOLERANCE = config.INGEST_TIMESTAMP_TOLERANCE


def _sign(timestamp: str, webhook_id: str, body: bytes) -> str:
    signed = timestamp.encode("utf-8") + b"." + webhook_id.encode("utf-8") + b"." + body
    digest = hmac.new(SECRET.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class ParseIngestTimestampTests(unittest.TestCase):
    """Unit boundary contract for the finite-decimal parser."""

    def test_finite_decimal_values_pass(self):
        for raw in ("1700000000", "1700000000.25", "0", "-1", " 1700000000 ", "1e3"):
            with self.subTest(raw=raw):
                self.assertTrue(math.isfinite(main._parse_ingest_timestamp(raw)))

    def test_nan_is_rejected(self):
        for raw in ("NaN", "nan", "+nan", "-NaN"):
            with self.subTest(raw=raw):
                with self.assertRaises(HTTPException) as ctx:
                    main._parse_ingest_timestamp(raw)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_infinities_are_rejected(self):
        for raw in ("Infinity", "-Infinity", "+Infinity", "inf", "-inf"):
            with self.subTest(raw=raw):
                with self.assertRaises(HTTPException) as ctx:
                    main._parse_ingest_timestamp(raw)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_overflow_is_rejected(self):
        # float('1e999') silently becomes inf rather than raising; it is still
        # non-finite and must answer 400, in both directions.
        for raw in ("1e999", "-1e999", "9" * 400):
            with self.subTest(raw=raw[:12]):
                self.assertEqual(math.isfinite(float(raw)), False)
                with self.assertRaises(HTTPException) as ctx:
                    main._parse_ingest_timestamp(raw)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_garbage_is_rejected(self):
        for raw in ("not-a-number", "0x1p4", ""):
            with self.subTest(raw=raw):
                with self.assertRaises(HTTPException) as ctx:
                    main._parse_ingest_timestamp(raw)
                self.assertEqual(ctx.exception.status_code, 400)


class IngestTimestampWindowAPITests(unittest.TestCase):
    """End-to-end behaviour through the real FastAPI lifespan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        config.DB_PATH = os.path.join(self.tmp.name, "api.db")
        # Pin the secret on the live module too: another test module imported
        # before this one can have caused app.config to cache INGEST_SECRET as
        # None (config reads the environment once, at first import).
        self._prev_secret = config.INGEST_SECRET
        config.INGEST_SECRET = SECRET
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.client.__exit__(None, None, None)
        config.INGEST_SECRET = self._prev_secret
        self.tmp.cleanup()

    def _create_sub(self) -> str:
        resp = self.client.post(
            "/subscriptions", json={"target_url": HOOK_URL, "secret": "s3cr3t"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _ingest(self, timestamp: str, webhook_id: str, body: bytes, *,
                signature: str | None = None, set_signature: bool = True) -> object:
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Id": webhook_id,
        }
        if set_signature:
            headers["X-Webhook-Signature"] = (
                signature if signature is not None else _sign(timestamp, webhook_id, body)
            )
        return self.client.post("/ingest", headers=headers, content=body)

    # ------------------------------------------------------------------
    # the original bypass
    # ------------------------------------------------------------------
    def test_nan_timestamp_is_400_even_with_valid_signature(self):
        sid = self._create_sub()
        webhook_id = "evt-nan"
        body = json.dumps({"subscription_ids": [sid], "n": 1}).encode("utf-8")
        resp = self._ingest("NaN", webhook_id, body)
        self.assertEqual(resp.status_code, 400, resp.text)
        # Rejected before persistence: the id is unknown and no event exists.
        self.assertEqual(self.client.get(f"/events/{webhook_id}").status_code, 404)

    def test_non_finite_timestamps_are_all_400(self):
        sid = self._create_sub()
        for raw in ("Infinity", "-Infinity", "+inf", "1e999", "-1e999", "nan"):
            webhook_id = f"evt-nonfinite-{raw.replace('-', 'm')}"
            body = json.dumps({"subscription_ids": [sid], "ts": raw}).encode("utf-8")
            with self.subTest(raw=raw):
                resp = self._ingest(raw, webhook_id, body)
                self.assertEqual(resp.status_code, 400, (raw, resp.text))
                self.assertEqual(
                    self.client.get(f"/events/{webhook_id}").status_code, 404
                )

    def test_non_finite_timestamp_fails_before_signature_check(self):
        # Even a signature computed over the NaN header must not rescue the
        # request -- finiteness is validated first, so it is 400 (malformed
        # input), not 401 (bad signature).
        sid = self._create_sub()
        webhook_id = "evt-nan-order"
        body = json.dumps({"subscription_ids": [sid]}).encode("utf-8")
        resp = self._ingest("NaN", webhook_id, body, signature=_sign("NaN", webhook_id, body))
        self.assertEqual(resp.status_code, 400, resp.text)

    # ------------------------------------------------------------------
    # existing window contract is preserved for finite values
    # ------------------------------------------------------------------
    def test_finite_timestamp_inside_window_is_accepted(self):
        sid = self._create_sub()
        now = int(time.time())
        for offset in (0, WINDOW_TOLERANCE - 2, -(WINDOW_TOLERANCE - 2)):
            webhook_id = f"evt-inside-{offset}"
            ts = str(now + int(offset))
            body = json.dumps({"subscription_ids": [sid], "offset": offset}).encode("utf-8")
            with self.subTest(offset=offset):
                resp = self._ingest(ts, webhook_id, body)
                self.assertEqual(resp.status_code, 201, resp.text)

    def test_finite_timestamp_outside_window_is_401(self):
        sid = self._create_sub()
        now = int(time.time())
        for offset in (WINDOW_TOLERANCE + 2, -(WINDOW_TOLERANCE + 2)):
            webhook_id = f"evt-outside-{offset}"
            ts = str(now + int(offset))
            body = json.dumps({"subscription_ids": [sid], "offset": offset}).encode("utf-8")
            with self.subTest(offset=offset):
                resp = self._ingest(ts, webhook_id, body)
                self.assertEqual(resp.status_code, 401, resp.text)
                self.assertEqual(
                    self.client.get(f"/events/{webhook_id}").status_code, 404
                )

    def test_unparseable_timestamp_is_400(self):
        sid = self._create_sub()
        body = json.dumps({"subscription_ids": [sid]}).encode("utf-8")
        resp = self._ingest("not-a-number", "evt-garbage", body)
        self.assertEqual(resp.status_code, 400, resp.text)

    # ------------------------------------------------------------------
    # signature compatibility: the original header text signs the request
    # ------------------------------------------------------------------
    def test_signature_uses_raw_header_text_without_numeric_normalization(self):
        sid = self._create_sub()
        now = int(time.time())
        # The endpoint signs the header text it receives verbatim, never a
        # numerically normalized form. Equivalent decimal spellings must each
        # be signed as themselves: "1700000000.000" reaching the HMAC proves the
        # server did not rewrite it to "1700000000.0"/"1700000000" first.
        # (Surrounding whitespace is stripped by HTTP before the header
        # reaches the app, so that case is signed over the stripped text.)
        for idx, raw in enumerate((str(now), f"{now}.000")):
            webhook_id = f"evt-raw-{idx}"
            body = json.dumps({"subscription_ids": [sid], "raw": raw}).encode("utf-8")
            with self.subTest(raw=raw):
                resp = self._ingest(raw, webhook_id, body)
                self.assertEqual(resp.status_code, 201, resp.text)

        stripped = str(now)
        body = json.dumps({"subscription_ids": [sid], "raw": "ws"}).encode("utf-8")
        resp = self._ingest(f"  {stripped}  ", "evt-raw-ows", body,
                            signature=_sign(stripped, "evt-raw-ows", body))
        self.assertEqual(resp.status_code, 201, resp.text)

        # A signature over the normalized text must NOT match a header carrying
        # the integer spelling (and vice versa): the bytes really differ.
        webhook_id = "evt-normalized-mismatch"
        body = json.dumps({"subscription_ids": [sid]}).encode("utf-8")
        resp = self._ingest(
            str(now), webhook_id, body, signature=_sign(f"{now}.0", webhook_id, body)
        )
        self.assertEqual(resp.status_code, 401, resp.text)

    def test_bad_signature_is_401(self):
        sid = self._create_sub()
        ts = str(int(time.time()))
        body = json.dumps({"subscription_ids": [sid]}).encode("utf-8")
        resp = self._ingest(ts, "evt-badsig", body, signature="sha256=" + "0" * 64)
        self.assertEqual(resp.status_code, 401, resp.text)

    # ------------------------------------------------------------------
    # end-to-end happy path + idempotent replay
    # ------------------------------------------------------------------
    def test_signed_ingest_creates_tasks_and_replays_idempotently(self):
        sid = self._create_sub()
        ts = str(int(time.time()))
        webhook_id = "evt-happy"
        body = json.dumps({"subscription_ids": [sid], "type": "order.paid"}).encode("utf-8")

        first = self._ingest(ts, webhook_id, body)
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(len(first.json()["created"]), 1)

        # Exact replay (same id, same raw body, same signature) returns 200
        # with the *stored first receipt* (its created entries preserved), and
        # never creates another task: the event has exactly one target.
        replay = self._ingest(ts, webhook_id, body)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())

        status = self.client.get(f"/events/{webhook_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(len(status.json()["targets"]), 1)


if __name__ == "__main__":
    unittest.main()

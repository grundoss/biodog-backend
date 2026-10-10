import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

import main

SYNTH = {
    "situation_title": "T", "panksepp": "PLAY", "panksepp_label": "PLAY", "arousal": 70, "valence": 20,
    "thought": "Odore nuovo!", "sensory": {"smell": "a", "sight": "b", "hearing": "c", "touch": "d"},
    "human_body_language": {"voice": "calma", "posture": "di fianco"}, "explanation": "spiegazione",
    "steps": ["uno", "due"], "forbidden": ["mai"],
}


class FakeDB:
    def __init__(self):
        self.subs = {}
        self.usage = {}

    def get_subscription(self, user_id):
        return self.subs.get(user_id)

    def get_usage(self, user_id, bucket):
        return self.usage.get((user_id, bucket), 0)

    def increment(self, user_id, bucket):
        self.usage[(user_id, bucket)] = self.usage.get((user_id, bucket), 0) + 1
        return self.usage[(user_id, bucket)]

    def upsert(self, user_id, fields):
        self.subs.setdefault(user_id, {}).update(fields)

    def update_by_stripe_id(self, sub_id, fields):
        rows = [s for s in self.subs.values() if s.get("stripe_subscription_id") == sub_id]
        for s in rows:
            s.update(fields)
        return rows


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    users = {"tok-free": {"id": "u-free", "email": "f@x.it"}, "tok-pro": {"id": "u-pro", "email": "p@x.it"}}
    monkeypatch.setattr(main, "_fetch_supabase_user", lambda token: users.get(token))
    monkeypatch.setattr(main, "_get_subscription", fake.get_subscription)
    monkeypatch.setattr(main, "_get_usage", fake.get_usage)
    monkeypatch.setattr(main, "_increment_usage", fake.increment)
    monkeypatch.setattr(main, "_upsert_subscription", fake.upsert)
    monkeypatch.setattr(main, "_update_subscription_by_stripe_id", fake.update_by_stripe_id)
    monkeypatch.setattr(main, "_call_gemini_api", lambda key, prompt: (main._normalize_synthesis(SYNTH), None))
    monkeypatch.setattr(main, "_token_cache", {})
    monkeypatch.setattr(main, "_guest_hits", {})
    fake.subs["u-pro"] = {"is_active": True, "plan_tier": "pro", "stripe_customer_id": "cus_pro", "stripe_subscription_id": "sub_pro"}
    return fake


@pytest.fixture
def client():
    return TestClient(main.app)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


BODY = {"user_text": "salta addosso", "lang": "it"}


def test_decide_quota():
    assert main.decide_quota("free", 0, 0, 0, False) == (True, "trial", None)
    assert main.decide_quota("free", 2, 0, 0, False) == (False, None, "trial_exhausted")
    assert main.decide_quota("free", 2, 0, 0, True) == (True, "sunday", None)
    assert main.decide_quota("free", 2, 0, 1, True) == (False, None, "trial_exhausted")
    assert main.decide_quota("premium", 0, 99, 0, False) == (True, "month", None)
    assert main.decide_quota("premium", 0, 100, 0, False) == (False, None, "monthly_limit")
    assert main.decide_quota("pro", 99, 999, 9, False) == (True, None, None)


def test_free_user_gets_two_trials_then_paywall(db, client, monkeypatch):
    monkeypatch.setattr(main, "_local_now", lambda: main.datetime(2026, 10, 7, 12))  # mercoledì
    for expected_remaining in (1, 0):
        r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("tok-free"))
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["remaining"] == expected_remaining
        assert r.json()["neural_synthesis"]["steps"] == ["uno", "due"]
    r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("tok-free"))
    assert r.status_code == 402
    assert r.json()["detail"]["code"] == "trial_exhausted"


def test_sunday_token(db, client, monkeypatch):
    monkeypatch.setattr(main, "_local_now", lambda: main.datetime(2026, 10, 11, 9))  # domenica
    db.usage[("u-free", "trial")] = 2
    r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("tok-free"))
    assert r.status_code == 200
    assert r.json()["sunday_token_used"] is True
    r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("tok-free"))
    assert r.status_code == 402


def test_engine_failure_does_not_consume_trial(db, client, monkeypatch):
    monkeypatch.setattr(main, "_call_gemini_api", lambda key, prompt: (None, "boom"))
    r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("tok-free"))
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "engine_unavailable"
    assert db.usage == {}


def test_guest_gets_one_locked_preview(db, client):
    r = client.post("/api/v1/umwelt/transduce", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["locked"] is True
    assert data["neural_synthesis"]["steps"] == []
    assert data["neural_synthesis"]["thought"] == "Odore nuovo!"
    r = client.post("/api/v1/umwelt/transduce", json=BODY)
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "auth_required"


def test_invalid_token_rejected(db, client):
    r = client.post("/api/v1/umwelt/transduce", json=BODY, headers=auth("forged"))
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "invalid_session"


def test_morphology_only_for_pro(db, client, monkeypatch):
    prompts = []
    monkeypatch.setattr(main, "_call_gemini_api", lambda key, prompt: (prompts.append(prompt), (SYNTH, None))[1])
    body = {**BODY, "snout": "flat"}
    client.post("/api/v1/umwelt/transduce", json=body, headers=auth("tok-free"))
    client.post("/api/v1/umwelt/transduce", json=body, headers=auth("tok-pro"))
    assert "Cranio/Skull: normal" in prompts[0]
    assert "Cranio/Skull: flat" in prompts[1]


def test_video_requires_pro(db, client, monkeypatch):
    monkeypatch.setattr(main, "_call_gemini_api_video", lambda *a: (main._normalize_synthesis(SYNTH), None))
    files = {"video": ("v.mp4", b"1234", "video/mp4")}
    assert client.post("/api/v1/umwelt/transduce-video", files=files).status_code == 401
    r = client.post("/api/v1/umwelt/transduce-video", files=files, headers=auth("tok-free"))
    assert r.status_code == 402 and r.json()["detail"]["code"] == "pro_required"
    r = client.post("/api/v1/umwelt/transduce-video", files=files, headers=auth("tok-pro"))
    assert r.status_code == 200, r.text


def test_portal_uses_own_customer_only(db, client, monkeypatch):
    created = {}

    class FakePortal:
        @staticmethod
        def create(customer, return_url):
            created["customer"] = customer
            return type("S", (), {"url": "https://portal"})()

    monkeypatch.setattr(main, "STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setattr(main.stripe.billing_portal, "Session", FakePortal)
    assert client.post("/api/v1/stripe/create-portal-session").status_code == 401
    r = client.post("/api/v1/stripe/create-portal-session", json={"customer_id": "cus_someone_else"}, headers=auth("tok-pro"))
    assert r.status_code == 200
    assert created["customer"] == "cus_pro"
    r = client.post("/api/v1/stripe/create-portal-session", headers=auth("tok-free"))
    assert r.status_code == 404


def test_checkout_refuses_missing_price_and_double_subscription(db, client, monkeypatch):
    monkeypatch.setattr(main, "STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setattr(main, "STRIPE_PRICE_ID_PREMIUM", "price_premium")
    monkeypatch.setattr(main, "STRIPE_PRICE_ID_PRO", "")
    r = client.post("/api/v1/stripe/create-checkout-session", json={"tier": "pro"}, headers=auth("tok-free"))
    assert r.status_code == 500 and r.json()["detail"]["code"] == "price_not_configured"
    r = client.post("/api/v1/stripe/create-checkout-session", json={"tier": "premium"}, headers=auth("tok-pro"))
    assert r.status_code == 409


def _signed(payload: dict, secret: str):
    body = json.dumps(payload)
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
    return body, {"Stripe-Signature": f"t={ts},v1={sig}", "Content-Type": "application/json"}


def test_webhook_cancellation_deactivates(db, client, monkeypatch):
    monkeypatch.setattr(main, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    event = {"id": "evt_1", "object": "event", "type": "customer.subscription.deleted",
             "data": {"object": {"id": "sub_pro", "object": "subscription", "status": "canceled"}}}
    body, headers = _signed(event, "whsec_test")
    r = client.post("/api/v1/stripe/webhook", content=body, headers=headers)
    assert r.status_code == 200, r.text
    assert db.subs["u-pro"]["is_active"] is False


def test_webhook_past_due_and_plan_change(db, client, monkeypatch):
    monkeypatch.setattr(main, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(main, "STRIPE_PRICE_ID_STANDARD", "price_std")
    event = {"id": "evt_2", "object": "event", "type": "customer.subscription.updated",
             "data": {"object": {"id": "sub_pro", "object": "subscription", "status": "past_due",
                                 "items": {"data": [{"price": {"id": "price_std"}}]}}}}
    body, headers = _signed(event, "whsec_test")
    assert client.post("/api/v1/stripe/webhook", content=body, headers=headers).status_code == 200
    assert db.subs["u-pro"]["is_active"] is False
    assert db.subs["u-pro"]["plan_tier"] == "standard"


def test_webhook_checkout_activates(db, client, monkeypatch):
    monkeypatch.setattr(main, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    event = {"id": "evt_3", "object": "event", "type": "checkout.session.completed",
             "data": {"object": {"object": "checkout.session", "client_reference_id": "u-free",
                                 "metadata": {"plan_tier": "pro"}, "customer": "cus_new", "subscription": "sub_new",
                                 "customer_details": {"email": "f@x.it"}}}}
    body, headers = _signed(event, "whsec_test")
    assert client.post("/api/v1/stripe/webhook", content=body, headers=headers).status_code == 200
    assert db.subs["u-free"]["is_active"] is True and db.subs["u-free"]["plan_tier"] == "pro"


def test_webhook_bad_signature(db, client, monkeypatch):
    monkeypatch.setattr(main, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    body, headers = _signed({"type": "checkout.session.completed"}, "wrong")
    assert client.post("/api/v1/stripe/webhook", content=body, headers=headers).status_code == 400


def test_normalize_synthesis_handles_garbage():
    s = main._normalize_synthesis({"thought": "ok", "arousal": "300", "valence": None, "steps": ["a", 3, ""]})
    assert s["arousal"] == 100 and s["valence"] == 0 and s["steps"] == ["a"]
    with pytest.raises(ValueError):
        main._normalize_synthesis({"arousal": 5})

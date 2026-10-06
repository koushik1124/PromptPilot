"""
API Integration Tests for PromptPilot FastAPI Backend
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("[OK] /health passed")


def test_analyze():
    res = client.post(
        "/api/analyze",
        json={"prompt": "Write a Python script to parse JSON files and sort keys."},
    )
    assert res.status_code == 200
    data = res.json()
    assert "score" in data
    assert "is_direct" in data
    print(f"[OK] /api/analyze passed: {data['score']} points")


def test_enhance_direct():
    res = client.post(
        "/api/enhance",
        json={
            "prompt": "make a python script to read csv and remove duplicates",
            "mode": "code",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["decision"] == "unchanged"
    assert data["enhancement_overhead_tokens"] == 0
    print("[OK] /api/enhance (direct pre-gate) passed")


def test_enhance_fluff():
    res = client.post(
        "/api/enhance",
        json={
            "prompt": "Hello! Could you please help me write a python script to read csv and remove duplicates? Thank you!",
            "mode": "code",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    print(f"[OK] /api/enhance (conversational) passed: {data['decision']}")


def test_evaluate():
    res = client.post(
        "/api/evaluate",
        json={
            "original": "make a python script to read csv and remove duplicates",
            "enhanced": "make a python script to read csv and remove duplicates",
            "enhancement_overhead_tokens": 0,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["estimated_effectiveness"] == "Optimal (No Change Needed)"
    assert data["net_tokens_saved"] == 0
    print(f"[OK] /api/evaluate passed: {data['estimated_effectiveness']}")


if __name__ == "__main__":
    print("\nRunning API Integration Tests...")
    test_health()
    test_analyze()
    test_enhance_direct()
    test_enhance_fluff()
    test_evaluate()
    print("All API tests passed successfully!\n")
